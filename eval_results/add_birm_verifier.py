import json

# different beta for different tasks
betas = [0.5, 0.67, 1.0, 1.5, 2.0]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                data.append(json.loads(line))
        except:
            print(f"ErrorErrorError {line}")
    return data

def calculate_vscores(data, beta):
    """
    Computes the BiRM vscores using the given beta value.

    The BiRM vscore is calculated as:
        BiRM_vscores = outcome_vscores + beta * process_vscores
    """
    for item in data:
        for output in item['outputs']:
            outcome_vscores = output['outcome_vscores'] # value score, outcome-supervised or math-shepherd
            process_vscores = output['process_vscores'] # reward score
            combined_vscores = [outcome + beta * process for outcome, process in zip(outcome_vscores, process_vscores)]
            output['vscores'] = combined_vscores
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("Loading the data...")
    data = read_jsonl('# ~/BiRM/eval_results/math500-512/verifier/test/responses_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42.jsonl')

    for beta in betas:
        combined_data = calculate_vscores(data, beta)
        output_file = f'~/BiRM/eval_results/math500-512/verifier/test/responses_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42_{beta}.jsonl'
        save_jsonl(combined_data, output_file)
        print(f'File {output_file} has been saved.')

if __name__ == '__main__':
    main()