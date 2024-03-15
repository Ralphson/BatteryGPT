model_name=BatteryGPT
learning_rate=0.01
on_server=true


llama_layers=32
d_model=32
d_ff=128
des='Exp'
comment='BatteryGPT-Masked_battery'

# --multi_gpu 
# 正式训练
num_workers=10
master_port=00097
num_process=1
batch_size=8
train_epochs=100
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_mask_battery.py \
  --on_server $on_server \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/my \
  --data_path trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv \
  --model_id Battery \
  --model $model_name \
  --data masked_battery \
  --enc_in 11 \
  --dec_in 7 \
  --c_out 1 \
  --des $des \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --num_workers $num_workers






