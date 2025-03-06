#!/bin/bash
# for metamath, global_batch_size = 4 * 2 * 8 = 64

export NCCL_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

base_path={your_base_path_here}

# Qwen2.5-3B, Llama3.1-8B
model_name_or_path=${base_path}/BiRM/models/Qwen2.5-7B
save_generator_id=qwen2.5_7b_math
task_name=${save_generator_id}_generator

if [[ $save_generator_id == *"llama"* ]]; then
    cp ${base_path}/BiRM/utils/constants_llama.py ${base_path}/BiRM/utils/constants.py
    echo "llama series"
elif [[ $save_generator_id == *"qwen"* ]]; then
    cp ${base_path}/BiRM/utils/constants_qwen.py ${base_path}/BiRM/utils/constants.py
    echo "qwen series"
else
    echo "warning: constants.py unable to be set"
    exit 1
fi

save_dir=${base_path}/BiRM/models/generators/${save_generator_id}/
mkdir -p ${save_dir}
echo ${task_name}

logging_dir=${base_path}/BiRM/A_running_log/generator_training
mkdir -p ${logging_dir}

accelerate launch \
  --config_file ${base_path}/BiRM/configs/zero2.yaml \
  --main_process_port=40650 \
  ${base_path}/BiRM/train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset gsm8k \
  --data_dir ${base_path}/BiRM/data/MetaMathQA-MATH \
  --target_set train \
  --save_dir ${save_dir} \
  --num_train_epoches 2 \
  --eval_steps 100000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_steps 100000 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ${base_path}/BiRM/wandb \
  --logging_steps 8 \
  --seed 42 \
  > ${logging_dir}/${task_name}.log 2>&1 

