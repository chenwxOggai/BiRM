import json
import random
import argparse
import csv
import numpy as np
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from typing import List, Dict

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_best_of_n_accuracy(data: List[Dict], n: int, selected_indices: List[int]) -> float:
    """
    Computes the Best-of-N Accuracy for a given N and selected indices.

    Args:
        data: List of dataset entries.
        n: The number of selected solutions.
        selected_indices: The indices of selected outputs.

    Returns:
        The Best-of-N accuracy value.
    """
    correct_count = 0
    total_samples = len(data)
    
    for sample in data:
        outputs = sample['outputs']
        # Select outputs based on the given indices
        selected_outputs = [outputs[i] for i in selected_indices]
        
        # Choose the best output based on the first element of vscores
        best_output = max(selected_outputs, key=lambda x: x['vscores'][0])
        
        # Check if the best output's label is true
        if best_output['label']:
            correct_count += 1
    
    accuracy = correct_count / total_samples
    return accuracy

def process_seed(args) -> Dict[int, float]:
    """
    Processes a single seed for accuracy calculation.

    Args:
        args: A tuple containing (seed, data, n_range).

    Returns:
        A dictionary mapping N to its calculated accuracy.
    """
    seed, data, n_range = args
    random.seed(seed)
    np.random.seed(seed)
    accuracies = {}
    
    for n in n_range:
        # Generate randomly selected indices (shared across all samples)
        selected_indices = np.random.choice(512, n, replace=False).tolist()
        accuracy = calculate_best_of_n_accuracy(data, n, selected_indices)
        accuracies[n] = accuracy
    
    return accuracies

def save_results_to_csv(results: Dict[int, Dict[str, float]], output_file: str):
    """
    Saves the computed results to a CSV file.

    Args:
        results: A dictionary mapping N to its average accuracy and standard deviation.
        output_file: Path to save the CSV file.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_solution', 'avg_acc', 'std_dev'])  # Write header
        for n, metrics in results.items():
            writer.writerow([n, metrics['avg_acc'], metrics['std_dev']])

def main(dataset, result_name):
    """
    Main function to compute and save Best-of-N Accuracy results.

    Args:
        dataset: Name of the dataset.
        result_name: Name of the result file.
    """
    # example
    # file_path = f'~/BiRM/eval_results/math500-512/verifier/test/responses_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42_1.0.jsonl'  # Input file path
    # output_file = f'~/BiRM/eval_results/math500-512/shuffle_BoN_math500_v(qwen2.5_7b_math_n15_birm_math-shepherd-soft)_g(qwen2.5_7b_math)_mean_seed42_1.0.csv'  # Output file path

    file_path = f'~/BiRM/eval_results/{dataset}-512/verifier/test/responses_{result_name}.jsonl'  # input file
    output_file = f'~/BiRM/eval_results/${dataset}-512/shuffle_BoN_{dataset}_{result_name}.csv'  # output file


    print("Loading the data...")
    data = read_jsonl(file_path)
    print("Start to shuffle...")
    
    K = 5  # Number of different seeds
    n_range = range(1, 513)  # Range of N values
    accuracies = {n: [] for n in n_range}  # Dictionary to store accuracy for each N
    
    # Prepare task parameters
    tasks = [(seed, data, n_range) for seed in range(42, 42 + K)]
    
    # Parallelize computation using tqdm's process_map
    results = process_map(process_seed, tasks, max_workers=cpu_count(), chunksize=1)
    
    # Aggregate results
    for result in results:
        for n, acc in result.items():
            accuracies[n].append(acc)
    
    # Compute the average accuracy and standard deviation for each N
    final_results = {}
    for n, acc_list in accuracies.items():
        avg_acc = np.mean(acc_list)  # Compute average accuracy
        std_dev = np.std(acc_list)   # Compute standard deviation
        final_results[n] = {'avg_acc': avg_acc, 'std_dev': std_dev}
    
    # Save results to CSV file
    save_results_to_csv(final_results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--result_name', type=str, required=True, help="Name of the result file")

    args = parser.parse_args()
    
    main(args.dataset, args.result_name)