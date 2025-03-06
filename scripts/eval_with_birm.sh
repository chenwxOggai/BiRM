#!/bin/bash

export NCCL_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

generator_id=qwen2.5_7b_math
verifier_id=n15_birm
final_id=${generator_id}_${verifier_id}

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

verifier_model_name_or_path=${base_path}/BiRM/models/verifiers/${final_id}
echo ${verifier_model_name_or_path}

logging_dir=${base_path}/BiRM/A_running_log/BoN
mkdir -p ${logging_dir}


#################### BiRM #####################

seed=42
dataset=gsm8k
agg_mode=mean

verifier_model_name_or_path=${base_path}/BiRM/models/verifiers/${final_id}
echo ${verifier_model_name_or_path}

task_name=test_eval_with_verifier_${final_id}_${dataset}_${agg_mode}_seed${seed}_512
echo ${task_name}

accelerate launch \
  --main_process_port=40665 \
  --config_file ${base_path}/BiRM/configs/zero2.yaml \
  ${base_path}/BiRM/eval_with_verifier_bi.py \
  --model_name_or_path ${verifier_model_name_or_path} \
  --data_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
  --verifier_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/verifier \
  --generator_metric_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator_with_verifier \
  --generator_id ${generator_id} \
  --target_set test \
  --batch_size 48 \
  --seed ${seed} \
  --agg_mode ${agg_mode} \
  > ${logging_dir}/${task_name}.log 2>&1

wait


seed=42
dataset=math500
agg_mode=mean

task_name=test_eval_with_verifier_${final_id}_${dataset}_${agg_mode}_seed${seed}_512
echo ${task_name}

accelerate launch \
  --main_process_port=40665 \
  --config_file ${base_path}/BiRM/configs/zero2.yaml \
  ${base_path}/BiRM/eval_with_verifier_bi.py \
  --model_name_or_path ${verifier_model_name_or_path} \
  --data_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
  --verifier_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/verifier \
  --generator_metric_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator_with_verifier \
  --generator_id ${generator_id} \
  --target_set test \
  --batch_size 24 \
  --seed ${seed} \
  --agg_mode ${agg_mode} \
  > ${logging_dir}/${task_name}.log 2>&1

wait


seed=42
dataset=gaokao2023
agg_mode=mean

task_name=test_eval_with_verifier_${final_id}_${dataset}_${agg_mode}_seed${seed}_512
echo ${task_name}

accelerate launch \
  --main_process_port=40665 \
  --config_file ${base_path}/BiRM/configs/zero2.yaml \
  ${base_path}/BiRM/eval_with_verifier_bi.py \
  --model_name_or_path ${verifier_model_name_or_path} \
  --data_dir ${base_path}/BiRM/data/${dataset}-512/model_generation \
  --verifier_output_dir ${base_path}/BiRM/eval_results/${dataset}-512/verifier \
  --generator_metric_dir ${base_path}/BiRM/eval_results/${dataset}-512/generator_with_verifier \
  --generator_id ${generator_id} \
  --target_set test \
  --batch_size 8 \
  --seed ${seed} \
  --agg_mode ${agg_mode} \
  > ${logging_dir}/${task_name}.log 2>&1