from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding, SeqEmbedding

import transformers

from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.llama_config = GPT2Config.from_pretrained('./models/pretrained_model/gpt2/')
        # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = configs.llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True


        self.llama = GPT2Model.from_pretrained(
            './models/pretrained_model/gpt2/',
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=self.llama_config,
            # load_in_4bit=True 
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            './models/pretrained_model/gpt2/',
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
            # 'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结预训练模型
        for param in self.llama.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.seq_embedding = SeqEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        

        self.word_embeddings = self.llama.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.d_llm = self.word_embeddings.shape[1]

        # 可训练的模型
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or 1:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.out_Linear = nn.Linear(configs.enc_in, configs.c_out)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or 1:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 提示词工程
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llama.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # 数据预训练
        n_vars = x_enc.shape[-1]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()                         # [batch_size, nvars, seq_len]
        # x_mark_enc = x_mark_enc.permute(0, 2, 1).contiguous()               # [batch_size, nvars, seq_len]
        enc_out, n_vars, x_mask = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out, x_mask = self.seq_embedding(x_enc.to(torch.bfloat16), x_mark_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        

        # 数据和prompt一起进入骨干网络训练
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        # 反patch
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        
        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')

        # 特征11->1
        dec_out = self.out_Linear(dec_out.to(torch.bfloat16))
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, x_mask=None):    # x_mask: [batch_size, seq_len, nvars]
        B, L, _ = target_embedding.shape                                                    # [batch_size, seq_len, d_model]
        S, _ = source_embedding.shape                                                       # [n_tokens, 768]
        H = self.n_heads                                                                    # [n_heads]

        # # 掩去<pad>
        # if x_mask is not None:
        #     x_mask = ~x_mask.to(torch.bool)
        #     target_embedding = target_embedding.masked_fill_(x_mask, float('-inf'))

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)        # [B, seq_len, n_heads, d_ff]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)             # [n_tokens, n_heads, d_ff]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)             # [n_tokens, n_heads, d_ff]

        out = self.reprogramming(target_embedding, source_embedding, value_embedding, x_mask)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding, x_mask=None):  # x_mask: [batch_size, seq_len, nvars]
        B, L, H, E = target_embedding.shape         # [B, seq_len, n_heads, d_ff]

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)         # [B, n_heads, seq_len, n_tokens]
        scores = scale * scores

        # 掩码
        # if x_mask is not None:
        #     x_mask = x_mask.unsqueeze(1)[:,:,:,0].unsqueeze(-1)
        #     x_mask = ~x_mask.to(torch.bool)
        #      scores = scores.masked_fill(x_mask, float('-inf'))

        scores = torch.softmax(scores, dim=-1)

        A = self.dropout(scores)
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
