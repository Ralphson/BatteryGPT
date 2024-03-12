model_name=GPT4TS

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --gpt_layer 3 \
  --train_epochs 30 \
  --batch_size 8 \
  --d_model 768 \
  --patch_size 1 \
  --stride 1 \
  --des 'Exp' \
  --itr 3 \
  --mlp 1 \
  --learning_rate 0.0005
  
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 30 \
  --gpt_layer 3 \
  --batch_size 8 \
  --d_model 768 \
  --patch_size 1 \
  --stride 1 \
  --des 'Exp' \
  --itr 3 \
  --mlp 1 \
  --learning_rate 0.0005
  
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 30 \
  --gpt_layer 3 \
  --batch_size 8 \
  --d_model 768 \
  --patch_size 1 \
  --stride 1 \
  --des 'Exp' \
  --itr 3 \
  --mlp 1 \
  --learning_rate 0.0005
  
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 30 \
  --train_epochs 8 \
  --gpt_layer 3 \
  --batch_size 8 \
  --d_model 768 \
  --patch_size 1 \
  --stride 1 \
  --des 'Exp' \
  --itr 3 \
  --mlp 1 \
  --learning_rate 0.0005
