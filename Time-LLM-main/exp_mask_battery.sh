# 总参数
learning_rate=0.01
train_epochs=100
llama_layers=32

# 服务器设置  --multi_gpu 
on_server=true
num_workers=16
master_port=00097
num_process=1

# mask-BatteryGPT
task_name='base'      # 从上到下显示到setting中
model_id='mask'
model='BatteryGPTv1'
data='mbatdata'
des='Exp'
comment='on_server' #
d_model=32
d_ff=128
batch_size=8
seq_len=24
label_len=12
pred_len=36
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_mask_battery.py \
  --on_server $on_server \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/my \
  --data $data \
  --data_path trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv \
  --model_id $model_id \
  --model $model \
  --data $data \
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
  --num_workers $num_workers \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len

# 延迟30s
sleep 30s

# 关机
shutdown -h now
