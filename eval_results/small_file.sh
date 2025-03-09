#!/bin/bash

# Set the Python script path
PYTHON_SCRIPT="./small_file.py"

# List of file paths (replace with the absolute paths of each JSONL file)
FILE_LIST=(
    'xxx'
    '~/BiRM/eval_results/math500-512/verifier/test/responses_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42.jsonl'

)

# Iterate through the list of file paths
for file in "${FILE_LIST[@]}"; do
    if [[ -f "$file" ]]; then
        echo "Processing file: $file"
        python3 "$PYTHON_SCRIPT" "$file"
    else
        echo "File does not exist: $file"
    fi
done

echo "All files have been processed!"