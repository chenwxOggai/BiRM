export NCCL_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

base_path={your_base_path_here}

generator_id=qwen2.5_7b_math

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

logging_dir=${base_path}/BiRM/A_running_log/beam_search
mkdir -p ${logging_dir}

##########################################################
########################## GSM8K #########################
##########################################################

verifier_id=n15_birm

model_name_or_path=${base_path}/BiRM/models/generators/${generator_id}
verifier_model_name_or_path=${base_path}/BiRM/models/verifiers/${generator_id}_${verifier_id}


n_sampling_steps=20

for n_beam in 20 10 5 4 2 1; do
for seed in 41 42 43; do
  task_name=gsm8k_total${n_sampling_steps}_beam${n_beam}_${generator_id}_${verifier_id}_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40550 \
    --config_file ${base_path}/BiRM/configs/zero2.yaml \
    ${base_path}/BiRM/eval_beam_search.py \
    --model_name_or_path ${model_name_or_path} \
    --birm_verifier_model_name_or_path ${verifier_model_name_or_path} \
    --dataset gsm8k \
    --data_dir ${base_path}/BiRM/data/gsm8k \
    --output_dir ${base_path}/BiRM/eval_results/gsm8k/beam_search \
    --target_set test \
    --inference_mode beam \
    --batch_size 20 \
    --vs_batch_size 10 \
    --n_beam ${n_beam} \
    --n_sampling_steps ${n_sampling_steps} \
    --max_n_step 10 \
    --max_step_length 100 \
    --dedup_mode 0 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_new_tokens 500 \
    --seed ${seed} \
    --agg_mode mean \
    --beta 1.0 \
    > ${logging_dir}/${task_name}.log 2>&1
done
done


##########################################################
####################### math500 ##########################
##########################################################

verifier_id=n15_birm

model_name_or_path=${base_path}/BiRM/models/generators/${generator_id}
verifier_model_name_or_path=${base_path}/BiRM/models/verifiers/${generator_id}_${verifier_id}

n_sampling_steps=20

for n_beam in 20 10 5 4 2 1; do
for seed in 41 42 43; do
  task_name=math500_total${n_sampling_steps}_beam${n_beam}_${generator_id}_${verifier_id}_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40550 \
    --config_file ${base_path}/BiRM/configs/zero2.yaml \
    ${base_path}/BiRM/eval_beam_search.py \
    --model_name_or_path ${model_name_or_path} \
    --birm_verifier_model_name_or_path ${verifier_model_name_or_path} \
    --dataset math500 \
    --data_dir ${base_path}/BiRM/data/math500 \
    --output_dir ${base_path}/BiRM/eval_results/math500/beam_search \
    --target_set test \
    --inference_mode beam \
    --batch_size 10 \
    --vs_batch_size 5 \
    --n_beam ${n_beam} \
    --n_sampling_steps ${n_sampling_steps} \
    --max_n_step 30 \
    --max_step_length 500 \
    --dedup_mode 0 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_new_tokens 2048 \
    --seed ${seed} \
    --agg_mode mean \
    --beta 2.5 \
    > ${logging_dir}/${task_name}.log 2>&1
done
done


##########################################################
########################## gaokao ########################
##########################################################

verifier_id=n15_birm

model_name_or_path=${base_path}/BiRM/models/generators/${generator_id}
verifier_model_name_or_path=${base_path}/BiRM/models/verifiers/${generator_id}_${verifier_id}


n_sampling_steps=20

for n_beam in 20 10 5 4 2 1; do
for seed in 41 42 43; do
  task_name=gaokao2023_total${n_sampling_steps}_beam${n_beam}_${generator_id}_${verifier_id}_seed${seed}
  echo ${task_name}

  accelerate launch \
    --main_process_port=40550 \
    --config_file ${base_path}/BiRM/configs/zero2.yaml \
    ${base_path}/BiRM/eval_beam_search.py \
    --model_name_or_path ${model_name_or_path} \
    --birm_verifier_model_name_or_path ${verifier_model_name_or_path} \
    --dataset gaokao2023 \
    --data_dir ${base_path}/BiRM/data/gaokao2023 \
    --output_dir ${base_path}/BiRM/eval_results/gaokao2023/beam_search \
    --target_set test \
    --inference_mode beam \
    --batch_size 10 \
    --vs_batch_size 4 \
    --n_beam ${n_beam} \
    --n_sampling_steps ${n_sampling_steps} \
    --max_n_step 30 \
    --max_step_length 500 \
    --dedup_mode 0 \
    --do_sample True \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 1.0 \
    --max_new_tokens 4096 \
    --seed ${seed} \
    --agg_mode mean \
    --beta 2.0 \
    > ${logging_dir}/${task_name}.log 2>&1
done
done