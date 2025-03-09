#!/bin/bash

# input file
# ~/BiRM/eval_results/math500-512/verifier/test/responses_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42_1.0.jsonl

dataset=math500
result_name="v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42_1.0"
python ./shuffle_BoN_parallel.py --dataset $dataset --result_name $result_name