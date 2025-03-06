#!/bin/bash

export NCCL_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

n_solution=15
generator_id=qwen2.5_7b_math # qwen2.5_3b_math, llama3_8b_math
save_verifier_id=n${n_solution}_prm

base_path={your_base_path_here}

if [[ $generator_id == *"llama"* ]]; then
    cp ${base_path}/BiRM/utils/constants_llama.py ${base_path}/BiRM/utils/constants.py
    echo "llama series"
elif [[ $generator_id == *"qwen"* ]]; then
    cp ${base_path}/BiRM/utils/constants_qwen.py ${base_path}/BiRM/utils/constants.py
    echo "qwen series"
else
    echo "warning: constants.py unable to be set"
    exit 1
fi

checkpoint_dir=${base_path}/BiRM/models/generators/${generator_id}
final_id=${generator_id}_${save_verifier_id}
task_name=${final_id}
echo ${task_name}

save_dir=${base_path}/BiRM/models/verifiers/${final_id}
mkdir -p ${save_dir}

logging_dir=${base_path}/BiRM/A_running_log/verifier
mkdir -p ${logging_dir}

accelerate launch \
  --config_file ${base_path}/BiRM/configs/zero2.yaml \
  --main_process_port=40665 \
  ${base_path}/BiRM/train_verifier.py \
  --model_name_or_path ${checkpoint_dir} \
  --data_dir ${base_path}/BiRM/data/Qwen2.5_7b_PRM \
  --target_set train \
  --process True \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --dedup True \
  --per_problem_sampling_solution ${n_solution} \
  --loss_level token \
  --loss_on_llm True \
  --num_train_epoches 1 \
  --eval_steps 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_epoches 1 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ${base_path}/BiRM/wandb \
  --logging_steps 20 \
  --seed 42 \
  > ${logging_dir}/${task_name}.log 2>&1 