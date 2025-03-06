import os
import torch
from typing import Sequence, Dict, Any
import transformers
from dataclasses import dataclass

from utils.datasets import read_jsonl, left_pad_sequences
from utils.math500.decoding import extract_answer


def get_examples(data_dir, split):
    read_file = {
        'test': 'test.jsonl',
    }[split]
    
    path = os.path.join(data_dir, read_file)
    examples = read_jsonl(path)

    # MetaMath , The asnwer is: xxx
    if split in ('test'):
        for ex in examples:
            ex.update(question=ex["question"] + "\n")
    else:
        pass


    # 500 test examples
    print(f"{len(examples)} {split} examples")
    return examples


def make_test_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass, inference_args: dataclass) -> Dict:
    test_dataset = TestGeneratorDataset(
        tokenizer=tokenizer, 
        data_dir=data_args.data_dir,
        target_set=data_args.target_set
    )
    return test_dataset


def make_test_split_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass, inference_args: dataclass) -> Dict:
    test_dataset = TestSplitGeneratorDataset(
        tokenizer=tokenizer, 
        data_dir=data_args.data_dir,
        target_set=data_args.target_set
    )
    return test_dataset


class TestGeneratorDataset(torch.utils.data.Dataset):
    """Left Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/MATH-500', 
        target_set: str = None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.pad_token_id = tokenizer.pad_token_id


        print("+ [Dataset] Loading Testing Data")
        self.examples = get_examples(data_dir, target_set)

        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        gts_str = [extract_answer(ans) for ans in ans_str]

        print("+ [Dataset] Tokenizing Testing Data")
        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False).input_ids

        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.ans_str = ans_str
        self.gts_str = gts_str

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        # Max tokens: 574
        print(f"Max tokens: {self.max_len}")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        qn_str = self.qns_str[idx]
        ans_str = self.ans_str[idx]
        gt = self.gts_str[idx]

        input_ids = torch.tensor(qn_tokens)
        return dict(
            idx=idx, 
            input_ids=input_ids, 
            input=qn_str,
            question=qn_str,
            reference=gt,
            record_data=dict(answer=ans_str, ground_truth=gt),
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        idx, input_ids, input, question, reference, record_data = tuple([instance[key] for instance in instances] for key in ("idx", "input_ids", "input", "question", "reference", "record_data"))
        # {
        #    'answer': ['answer 1', 'answer 2', 'answer 3'],
        #    'ground_truth': ['gt 1', 'gt 2', 'gt 3']
        # }
        record_data = {k: [instance[k] for instance in record_data] for k in record_data[0].keys()}

        input_ids, attention_mask = left_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        
        # left padding
        return dict(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input=input,
            question=question,
            reference=reference,
            record_data=record_data,
        )


class TestSplitGeneratorDataset(torch.utils.data.Dataset):
    """Left Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/MATH-500', 
        target_set: str = None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.pad_token_id = tokenizer.pad_token_id


        print("+ [Dataset] Loading Testing Data")
        self.examples = get_examples(data_dir, target_set)


        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        gts_str = [extract_answer(ans) for ans in ans_str]
        # idx
        q_idx_str = [ex['q_idx'] for ex in self.examples]
        response_idx_str = [ex['response_idx'] for ex in self.examples]
        step_idx_str = [ex['step_idx'] for ex in self.examples]

        print("+ [Dataset] Tokenizing Testing Data")
        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False).input_ids

        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.ans_str = ans_str
        self.gts_str = gts_str
        self.q_idx_str = q_idx_str
        self.response_idx_str = response_idx_str
        self.step_idx_str = step_idx_str

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        # Max tokens: 574
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        qn_str = self.qns_str[idx]
        ans_str = self.ans_str[idx]
        gt = self.gts_str[idx]
        q_idx = self.q_idx_str[idx]
        response_idx = self.response_idx_str[idx]
        step_idx = self.step_idx_str[idx]

        input_ids = torch.tensor(qn_tokens)
        return dict(
            idx=idx, 
            q_idx=q_idx,
            response_idx=response_idx,
            step_idx=step_idx,
            input_ids=input_ids, 
            input=qn_str,
            question=qn_str,
            reference=gt,
            record_data=dict(answer=ans_str, ground_truth=gt),
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        idx, q_idx, response_idx, step_idx, input_ids, input, question, reference, record_data = tuple([instance[key] for instance in instances] for key in ("idx", "q_idx", "response_idx", "step_idx", "input_ids", "input", "question", "reference", "record_data"))
        # {
        #    'answer': ['answer 1', 'answer 2', 'answer 3'],
        #    'ground_truth': ['gt 1', 'gt 2', 'gt 3']
        # }
        record_data = {k: [instance[k] for instance in record_data] for k in record_data[0].keys()}

        input_ids, attention_mask = left_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        
        # left padding
        return dict(
            idx=idx,
            q_idx=q_idx,
            response_idx=response_idx,
            step_idx=step_idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input=input,
            question=question,
            reference=reference,
            record_data=record_data,
        )

