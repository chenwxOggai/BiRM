#!/bin/bash

export NCCL_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

base_path={your_base_path_here}

generator_id=qwen2.5_7b_math
n_solutions=512

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

model_name_or_path=${base_path}/BiRM/models/generators/${generator_id}

for task in math500 gsm8k gaokao2023; do
    mkdir -p ${base_path}/BiRM/data/${task}-512
done

logging_dir=${base_path}/BiRM/A_running_log/generate_n
mkdir -p ${logging_dir}


dataset=math500
for seed in 42; do
  task_name=${generator_id}_${dataset}_n${n_solutions}_for_verify_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40655 \
    --config_file ${base_path}/BiRM/configs/zero1.yaml \
    ${base_path}/BiRM/generate_n_solutions.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --data_dir ${base_path}/BiRM/data/${dataset} \
    --output_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
    --metric_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator \
    --target_set test \
    --n_solutions ${n_solutions} \
    --batch_size 64 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_new_tokens 2048 \
    --seed ${seed} \
    > ${logging_dir}/${task_name}.log 2>&1
done

wait

dataset=gsm8k
for seed in 42; do
  task_name=${generator_id}_${dataset}_n${n_solutions}_for_verify_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40655 \
    --config_file ${base_path}/BiRM/configs/zero1.yaml \
    ${base_path}/BiRM/generate_n_solutions.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --data_dir ${base_path}/BiRM/data/${dataset} \
    --output_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
    --metric_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator \
    --target_set test \
    --n_solutions ${n_solutions} \
    --batch_size 165 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_new_tokens 500 \
    --seed ${seed} \
    > ${logging_dir}/${task_name}.log 2>&1
done


dataset=gaokao2023
for seed in 42; do
  task_name=${generator_id}_${dataset}_n${n_solutions}_for_verify_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40655 \
    --config_file ${base_path}/BiRM/configs/zero1.yaml \
    ${base_path}/BiRM/generate_n_solutions.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --data_dir ${base_path}/BiRM/data/${dataset} \
    --output_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
    --metric_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator \
    --target_set test \
    --n_solutions ${n_solutions} \
    --batch_size 30 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_length 4096 \
    --seed ${seed} \
    > ${logging_dir}/${task_name}.log 2>&1
done