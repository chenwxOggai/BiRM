#!/bin/bash

export NCCL_DEBUG=OFF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

base_path={your_base_path_here}

generator_id=qwen2.5_7b_math
model_name_or_path=${base_path}/BiRM/models/generators/${generator_id}

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

logging_dir=${base_path}/BiRM/A_running_log/greedy_eval
mkdir -p ${logging_dir}

#################### launch vLLM first #####################
########### set seed=42 in utils/vllm_utils.py #############

dataset=math500
accelerate launch \
  --main_process_port=40655 \
  --config_file ${base_path}/BiRM/configs/zero1.yaml \
  ${base_path}/BiRM/generate_n_solutions.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --data_dir ${base_path}/BiRM/data/${dataset} \
  --output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --metric_output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --target_set test \
  --batch_size 64 \
  --do_sample False \
  --max_new_tokens 2048 \
  --seed 42 \
  > ${logging_dir}/${generator_id}_${dataset}.log 2>&1 

wait

dataset=gsm8k
accelerate launch \
  --main_process_port=40655 \
  --config_file ${base_path}/BiRM/configs/zero1.yaml \
  ${base_path}/BiRM/generate_n_solutions.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --data_dir ${base_path}/BiRM/data/${dataset}\
  --output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --metric_output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --target_set test \
  --batch_size 64 \
  --do_sample False \
  --max_new_tokens 500 \
  --seed 42 \
  > ${logging_dir}/${generator_id}_${dataset}.log 2>&1  

wait

dataset=gaokao2023
accelerate launch \
  --main_process_port=40655 \
  --config_file ${base_path}/BiRM/configs/zero1.yaml \
  ${base_path}/BiRM/generate_n_solutions.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --data_dir ${base_path}/BiRM/data/${dataset}\
  --output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --metric_output_dir ${base_path}/BiRM/eval_results/${dataset} \
  --target_set test \
  --batch_size 30 \
  --do_sample False \
  --max_new_tokens 4096 \
  --seed 42 \
  > ${logging_dir}/${generator_id}_${dataset}.log 2>&1  
