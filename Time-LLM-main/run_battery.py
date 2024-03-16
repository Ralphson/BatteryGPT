import time
import random
import numpy as np
import os
import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, BatteryGPT, BatteryGPT_mask
from log import set_logger


from data_provider.data_factory import data_provider
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from utils.losses import smape_loss, mase_loss, mape_loss, Metrics



os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"



if __name__=="__main__":
    filetime = time.strftime('%Y%m%d%H%M', time.localtime())
    parser = argparse.ArgumentParser(description='Time-LLM')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='Battery', help='model id')
    parser.add_argument('--model_comment', type=str, required=False, default='on_local', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=False, default='BatteryGPTv0',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--filetime', type=str, default=filetime, help='file start time')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='batdata', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/my', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv', help='data file')
    parser.add_argument('--drop_bid', type=int, default=0)
    parser.add_argument('--cutting_rate', type=float, default=1.2)
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./cache/', help='location of model checkpoints')
    parser.add_argument('--logger', type=str, default='./logs', help='log folder')
    parser.add_argument('--on_server', type=bool, default=False)

    # forecasting task
    parser.add_argument('--seq_limit', type=int, default=48, help='raw sequence length')
    parser.add_argument('--seq_len', type=int, default=18, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=9, help='start token length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=11, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()

    # 初始化随机数
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 初始化多卡设置
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    if args.on_server:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    else:
        accelerator = Accelerator(mixed_precision='bf16')

    # 初始化logger设置
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, 
        args.itr)
    set_logger(args.logger + '/' + setting + '/' + filetime + '.log')
    print('Args in experiment:')
    print(args)

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).bfloat16()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).bfloat16()
        elif args.model == 'BatteryGPTv0':
            model = BatteryGPT.Model(args).bfloat16()
        elif args.model == 'BatteryGPTv1':
            model = BatteryGPT_mask.Model(args).bfloat16()
        else:
            model = TimeLLM.Model(args).bfloat16()

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, filetime=filetime)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        if args.model == 'BatteryGPTv0':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        mae_metric = Metrics()

        args.frequency_map = {
            'Yearly': 1,
            'Quarterly': 4,
            'Monthly': 12,
            'Weekly': 1,
            'Daily': 1,
            'Hourly': 24
        }
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                # 这里nvars包含最后一维的电能
                batch_x = batch_x.float().to(accelerator.device)                # [bs, seq_len, nvars]
                batch_y = batch_y.float().to(accelerator.device)                # [bs, label_len+pred_len, 1]
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                        raise NotImplementedError

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]   # 截掉label部分
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]   # 截掉label部分

                    batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:]

                    # loss = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)
                    if args.on_server:
                        loss = criterion(outputs, batch_y.to(torch.bfloat16))       # TODO：修改metric，mse不行
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_metrics = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Test MSE Loss: {4:.7f} Test MAE Loss: {5:.7f} Test RMSE Loss: {6:.7f} Test MAPE Loss: {7:.7f} Test MSPE Loss: {8:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], test_metrics[4]))

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.wait_for_everyone()
    # if accelerator.is_local_main_process:
    #     path = './checkpoints'  # unique checkpoint saving path
    #     del_files(path)  # delete checkpoint files
    #     accelerator.print('success delete checkpoints')