# 总参数
learning_rate=0.01
train_epochs=100
llama_layers=32

# 服务器设置  --multi_gpu 
on_server=true
num_workers=16
master_port=00097
num_process=1

# b3
task_name='base'      # 从上到下显示到setting中
model_id='mask_from0'
model='BatteryGPTv0'
data='batdata_from_0'
des='Exp'
comment='on_server' #
d_model=32
d_ff=128
batch_size=8
seq_len=24
label_len=12
pred_len=48
scale_data=0         # 不归一化
cal_mask=1           # mask数据应该计算掩码损失
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
  --pred_len $pred_len \
  --scale_data $scale_data \
  --cal_mask $cal_mask
  

echo '>>>>>>>>>>>>>>>finished'
# 延迟120s
sleep 120s
echo '<<<<<<<<<<<<<<<finished'

# 关机
shutdown -h now
