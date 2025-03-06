#!/bin/bash

base_path={your_base_path_here}

model_name_or_path=${base_path}/BiRM/models/generators/qwen2.5_3b_math
# model_name_or_path=${base_path}/BiRM/models/generators/llama3_8b_math

if [[ $model_name_or_path == *"llama"* ]]; then
    cp ${base_path}/BiRM/utils/constants_llama.py ${base_path}/BiRM/utils/constants.py
elif [[ $model_name_or_path == *"qwen"* ]]; then
    cp ${base_path}/BiRM/utils/constants_qwen.py ${base_path}/BiRM/utils/constants.py
else
    echo "warning: constants.py unable to be set"
    exit 1
fi


# Create and start each session
for i in {0..7}; do
    # tmux kill-session -t vllm$i  # Close the specified session
    tmux new-session -d -s vllm$i  # Start the session in the background using the -d option
    tmux send-keys -t vllm$i "export CUDA_VISIBLE_DEVICES=$i" C-m  # Send the corresponding command to each session

    port=$((36100 + i))  # Calculate the port number, from 36100 to 36107
    command="python -m vllm.entrypoints.openai.api_server --model ${model_name_or_path} --gpu_memory_utilization=0.4 --max_model_len=8192 --port $port"
    tmux send-keys -t vllm$i "$command" C-m

    # tmux send-keys -t vllm$i C-c  # Send Ctrl+C to vllm0 through vllm7
done

