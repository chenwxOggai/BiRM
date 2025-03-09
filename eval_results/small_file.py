import json
import argparse

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process each line
    modified_lines = []
    for line in lines:
        data = json.loads(line)
        for output in data['outputs']:
            # Retain only the first value in vscores, outcome_vscores, and process_vscores
            if 'vscores' in output and len(output['vscores']) > 0:
                output['vscores'] = [output['vscores'][0]]
            if 'outcome_vscores' in output and len(output['outcome_vscores']) > 0:
                output['outcome_vscores'] = [output['outcome_vscores'][0]]
            if 'process_vscores' in output and len(output['process_vscores']) > 0:
                output['process_vscores'] = [output['process_vscores'][0]]
        modified_lines.append(json.dumps(data, ensure_ascii=False))

    # Write the modified data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in modified_lines:
            file.write(line + '\n')

    print(f"File {file_path} has been successfully modified and overwritten.")

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Process a JSONL file, retaining only the first value of vscores, outcome_vscores, and process_vscores.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file to be processed")
    args = parser.parse_args()

    # Process the file
    process_file(args.file_path)