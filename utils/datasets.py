import json
import os
import re
import torch
import torch.nn.functional as F
from typing import Sequence, List, Dict, Union
import transformers
from dataclasses import dataclass
import pathlib
from torch.utils.data import DataLoader
from utils.constants import IGNORE_INDEX, MODEL_NEWLINE


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_few_shot_prompt(data_dir, prompt_file):
    with open(os.path.join(data_dir, prompt_file), 'r') as f:
        prompt = f.read()
    return prompt.replace('{', '{{').replace('}', '}}').replace('{{test_question}}', '{test_question}')


def get_model_solutions(data_dir, generator_id, target_set, process: bool = False):
    # data_dir + target_set => dataset_path
    data_dir = os.path.join(data_dir, target_set)

    if process:
        files_pattern = f'responses_n*_{generator_id}_process.jsonl'
    else:
        files_pattern = f'responses_n*_{generator_id}.jsonl'

    response_files = [str(x) for x in pathlib.Path(data_dir).glob(files_pattern)]
    if not response_files:
        raise ValueError(f'Fail to find {files_pattern} in {data_dir}')


    ordering_and_response_path = []
    for response_file in response_files:
        regex_match = re.match(r".*responses_n([0-9]+)", response_file)
        if regex_match is not None:
            ordering_and_response_path.append((int(regex_match.group(1)), response_file))

    responses_sorted = sorted(ordering_and_response_path)
    responses_sorted = [response[1] for response in responses_sorted]
    read_file = responses_sorted[-1]

    examples = read_jsonl(read_file)
    print(f"{len(examples)} {target_set} examples, each with {len(examples[0]['outputs'])} solutions")
    return examples


def make_training_dataloaders(
    data_module: Dict[str, torch.utils.data.Dataset],
    training_args: dataclass = None,
) -> Dict:
    train_dataloader = DataLoader(
                            data_module['train_dataset'], 
                            batch_size=training_args.per_device_train_batch_size, 
                            shuffle=True, 
                            drop_last=False, 
                            collate_fn=data_module['train_dataset'].collate_fn, 
                        )
    if data_module['val_dataset'] is not None:
        val_dataloader = DataLoader(
                            data_module['val_dataset'], 
                            batch_size=training_args.per_device_eval_batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_module['val_dataset'].collate_fn, 
                        )
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader


def make_testing_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)


def make_training_verifier_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    if data_args.bi_process:
        dataset_class = BiVerifierDataset
    elif data_args.process:
        dataset_class = ProcessVerifierDataset
    else:
        dataset_class = VerifierDataset

    train_dataset = dataset_class(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir, 
                        target_set=data_args.target_set, 
                        generator_id=data_args.generator_id, 
                        per_problem_sampling_solution=data_args.per_problem_sampling_solution, 
                        loss_level=data_args.loss_level,
                        loss_on_llm=data_args.loss_on_llm,
                        dedup=data_args.dedup,
                    )
    
    val_dataset = None

    if data_args.val_target_set is not None:
        val_dataset = dataset_class(
                            tokenizer=tokenizer, 
                            data_dir=data_args.data_dir, 
                            target_set=data_args.val_target_set, 
                            generator_id=data_args.generator_id, 
                            per_problem_sampling_solution=-1, 
                            loss_level=data_args.loss_level,
                            loss_on_llm=data_args.loss_on_llm,
                        )
    return dict(train_dataset=train_dataset, val_dataset=val_dataset)


def make_test_verifier_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    test_dataset = VerifierDataset(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir,
                        target_set=data_args.target_set,
                        generator_id=data_args.generator_id, 
                        per_problem_sampling_solution=-1, 
                    )
    return test_dataset


# ORM
class VerifierDataset(torch.utils.data.Dataset):
    """Right Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k/model_generation', 
        target_set: str = None,
        generator_id: str = None, 
        per_problem_sampling_solution: str = None, 
        loss_level: str = 'token', 
        loss_on_llm: bool = False,
        dedup: bool = False
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ('token', 'step')

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # idx，input，question, answer, ground_truth, outputs(response, response_answer, label)
        self.examples = get_model_solutions(data_dir, generator_id, target_set)
        # assert len(self.examples[0]['outputs']) >= per_problem_sampling_solution

        if per_problem_sampling_solution != -1:
            for example in self.examples:
                if len(example['outputs']) >= per_problem_sampling_solution:
                    example['outputs'] = example['outputs'][:per_problem_sampling_solution]
                else:
                    print(f"Warning: {len(example['outputs'])} solutions for problem {example['idx']} is less than per_problem_sampling_solution {per_problem_sampling_solution}")
                    continue
        else:
            per_problem_sampling_solution = len(self.examples[0]['outputs'])
        

        if dedup:
            for ex in self.examples:
                dedup_outputs = []
                responses = set()
                for output in ex['outputs']:
                    if output['response'] in responses:
                        continue
                    responses.add(output['response'])
                    dedup_outputs.append(output)
                ex['outputs'] = dedup_outputs

        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)]
        
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [[outputs["response"] for outputs in ex["outputs"]] for ex in self.examples]
        # [True, False, True, ..., True]
        v_classes = [[outputs["label"] for outputs in ex["outputs"]] for ex in self.examples]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        v_classes = self._flatten(v_classes)

        print(f"All solutions_str: {len(solutions_str)}")

        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        solutions_tokens = tokenizer(solutions_str, padding=False, add_special_tokens=False).input_ids

        self.indices1 = indices1
        self.indices2 = indices2
        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.solutions_str = solutions_str
        self.solutions_tokens = solutions_tokens
        self.v_classes = v_classes

        self.n_question = len(self.examples)
        self.per_problem_sampling_solution = per_problem_sampling_solution

        # Number of examples = 74356 with #deduplication = 374
        print("Loading OVM dataset...")
        print(f'Number of examples = {len(qns_str)} with #deduplication = {self.n_question * self.per_problem_sampling_solution - len(qns_str)}')

        # question + solution + eos
        self.max_len = max([
                len(self.qns_tokens[i]) + len(self.solutions_tokens[i]) + 1
                for i in range(len(self.solutions_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")


    def __len__(self):
        return len(self.solutions_tokens)


    def _flatten(self, ls):
        return [item for sublist in ls for item in sublist]

    # idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class
    # tensor: input_ids, labels, v_labels
    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        v_class = self.v_classes[idx]

        # input_ids, attention_mask
        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]
        
        masks = (
            ([0] * len(qn_tokens))
            + ([1] * len(sol_tokens))
            + ([1])
        )

        # create language modeling labels
        if self.loss_on_llm:
            labels = input_ids
            # -100
            labels = mask_labels(labels, masks)

        # create verifier labels
        if self.loss_level == 'token':
            # [1, 1, 1, ..., 1, 0]
            v_labels = [int(v_class)] * (len(input_ids) - 1) + [-100]
            # -100
            v_labels = mask_labels(v_labels, masks)
        else:
            raise NotImplementedError

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        v_labels = torch.tensor(v_labels)
        
        return dict(
            idx1=self.indices1[idx], idx2=self.indices2[idx], 
            input_ids=input_ids, labels=labels, v_labels=v_labels, 
            qn_str=self.qns_str[idx], qn_tokens=self.qns_tokens[idx], sol_str=self.solutions_str[idx], sol_tokens=self.solutions_tokens[idx], v_class=self.v_classes[idx],
        )

    # input_ids, attention_mask, labels, v_labels
    # idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class
    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, v_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "v_labels"))
        idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class = tuple([instance[key] for instance in instances] for key in ("idx1", "idx2", "qn_str", "qn_tokens", "sol_str", "sol_tokens", "v_class"))

        #  (batch_size, max_length)
        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        # LLM_loss, Verifier_loss
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False) if self.loss_on_llm else None
        v_labels = right_pad_sequences(v_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            idx1=idx1, idx2=idx2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            v_labels=v_labels,
            qn_str=qn_str, qn_tokens=qn_tokens, sol_str=sol_str, sol_tokens=sol_tokens, v_class=v_class,
        )


# PRM
class ProcessVerifierDataset(torch.utils.data.Dataset):
    """Right Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k/model_generation', 
        target_set: str = None,
        generator_id: str = None, 
        per_problem_sampling_solution: str = None, 
        loss_level: str = 'token', 
        loss_on_llm: bool = False,
        dedup: bool = False
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ('token', 'step')

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # idx，input，question, answer, ground_truth, 
        # outputs(response, response_answer, label, step_labels, step_labels_progress)
        self.examples = get_model_solutions(data_dir, generator_id, target_set, process=True)
        # assert len(self.examples[0]['outputs']) >= per_problem_sampling_solution


        if per_problem_sampling_solution != -1:
            for example in self.examples:
                if len(example['outputs']) >= per_problem_sampling_solution:
                    example['outputs'] = example['outputs'][:per_problem_sampling_solution]
                else:
                    print(f"Warning: {len(example['outputs'])} solutions for problem {example['idx']} is less than per_problem_sampling_solution {per_problem_sampling_solution}")
                    continue
        else:
            per_problem_sampling_solution = len(self.examples[0]['outputs'])
        

        if dedup:
            for ex in self.examples:
                dedup_outputs = []
                responses = set()
                for output in ex['outputs']:
                    if output['response'] in responses:
                        continue
                    responses.add(output['response'])
                    dedup_outputs.append(output)
                ex['outputs'] = dedup_outputs


        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)]
        
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [[outputs["response"] for outputs in ex["outputs"]] for ex in self.examples]
        
        # vanilla PRM
        # [[True, False, False, ..., False], [True, False, False, ..., False], ..., [True, False, False, ..., False]]
        step_labels = [[outputs["step_labels"] for outputs in ex["outputs"]] for ex in self.examples]

        # PRM-math-shepherd
        # hard label
        # step_labels = [[outputs["step_h_label"] for outputs in ex["outputs"]] for ex in self.examples]
        # soft label
        # step_labels = [[outputs["step_s_label"] for outputs in ex["outputs"]] for ex in self.examples]

        # PRM-Entropy Regularized
        # step_labels = [[outputs["step_ER_label"] for outputs in ex["outputs"]] for ex in self.examples]

        v_classes = [[outputs["label"] for outputs in ex["outputs"]] for ex in self.examples]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        step_labels = self._flatten(step_labels)
        v_classes = self._flatten(v_classes)

        qns_tokens = tokenizer(qns_str, padding=False).input_ids

        print(f"All solutions_str: {len(solutions_str)}")

        # check
        for solution in solutions_str:
            assert solution.endswith(MODEL_NEWLINE), f"Solution does not end with newline token: {solution}"
            
        # xxxx kn, xxxx kn, xxxx kn <eos>
        steps_str = [
            list(map(lambda x: x+MODEL_NEWLINE, solution_str.split(MODEL_NEWLINE)[:-1]))
            for solution_str in solutions_str
        ]


        solutions_tokens = [
            [tokenizer.encode(step_str[0], add_special_tokens=False)]
              + [self._get_continued_input_ids(step) for step in step_str[1:]]
            for step_str in steps_str
        ]


        step_tokens_lens = [
            [len(step) for step in tokens]
            for tokens in solutions_tokens
        ]

        solutions_tokens = [self._flatten(tokens) for tokens in solutions_tokens]


        self.indices1 = indices1
        self.indices2 = indices2
        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.solutions_str = solutions_str
        self.solutions_tokens = solutions_tokens
        self.v_classes = v_classes

        self.step_tokens_lens = step_tokens_lens # [[4, 3, 2], [4, 3, 2], ...]
        self.step_labels = step_labels # [[True, False, False], [True, False, False], ...]
        self.n_question = len(self.examples)
        self.per_problem_sampling_solution = per_problem_sampling_solution

        print("Loading PRM dataset...")
        print(f'Number of examples = {len(qns_str)} with #deduplication = {self.n_question * self.per_problem_sampling_solution - len(qns_str)}')

        # question + solution + eos
        self.max_len = max([
                len(self.qns_tokens[i]) + len(self.solutions_tokens[i]) + 1
                for i in range(len(self.solutions_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")
    

    def _add_placeholder(self, text: str):
        if len(text) == 0 or text[0] != '#':
            return '####' + text
        return '$$' + text

    def _get_continued_input_ids(self, text: Union[str, List[str]], right_padding=False, return_tensors=False):
        if isinstance(text, str):
            text = self._add_placeholder(text)
        else:
            text = [self._add_placeholder(x) for x in text]

        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if isinstance(text, str):
            input_ids = input_ids[1:]
        else:
            input_ids = [ids[1:] for ids in input_ids]

        if right_padding and isinstance(text, list):
            max_length = max([len(ids) for ids in input_ids])
            input_ids = [
                ids 
                + [self.pad_token_id] * (max_length - len(ids))
                for ids in input_ids
            ]
        if return_tensors:
            input_ids = torch.tensor(input_ids)
        return input_ids
    
    def __len__(self):
        return len(self.solutions_tokens)

    def _flatten(self, ls):
        return [item for sublist in ls for item in sublist]

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        step_labels = self.step_labels[idx]
        step_tokens_lens = self.step_tokens_lens[idx]

        # input_ids, attention_mask
        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]

        masks = (
            ([0] * len(qn_tokens))
            + ([1] * len(sol_tokens))
            + ([1])
        )

        # create language modeling labels
        if self.loss_on_llm:
            labels = input_ids
            # -100
            labels = mask_labels(labels, masks)

        # create verifier labels
        if self.loss_level == 'token':
            v_labels = [0] * len(qn_tokens)

            for i, (tokens_len, step_label) in enumerate(zip(step_tokens_lens, step_labels)):
                if i < len(step_tokens_lens):
                    v_labels += [-100] * (tokens_len - 1)
                    if step_label:
                        v_labels += [1]
                    else:
                        v_labels += [0]

            # eos token
            v_labels += [-100]
            
            # # math-shepherd soft-label
            # v_labels = [0] * len(qn_tokens)

            # for i, (tokens_len, step_label) in enumerate(zip(step_tokens_lens, step_labels)):
            #     if i < len(step_tokens_lens) - 1:
            #         v_labels += [-100] * (tokens_len - 1)
            #         v_labels += [step_label]
            #     else:
            #         v_labels += [-100] * tokens_len

            # # eos token
            # v_labels += [step_labels[-1]]

            v_labels = mask_labels(v_labels, masks)

            assert len(v_labels) == len(input_ids)
        else:
            raise NotImplementedError

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        v_labels = torch.tensor(v_labels)
        return dict(
            idx1=self.indices1[idx], idx2=self.indices2[idx], 
            input_ids=input_ids, labels=labels, v_labels=v_labels, 
            qn_str=self.qns_str[idx], qn_tokens=self.qns_tokens[idx], sol_str=self.solutions_str[idx], sol_tokens=self.solutions_tokens[idx], v_class=self.v_classes[idx],
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, v_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "v_labels"))
        idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class = tuple([instance[key] for instance in instances] for key in ("idx1", "idx2", "qn_str", "qn_tokens", "sol_str", "sol_tokens", "v_class"))

        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False) if self.loss_on_llm else None
        v_labels = right_pad_sequences(v_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            idx1=idx1, idx2=idx2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            v_labels=v_labels,
            qn_str=qn_str, qn_tokens=qn_tokens, sol_str=sol_str, sol_tokens=sol_tokens, v_class=v_class,
        )


# BiRM
class BiVerifierDataset(torch.utils.data.Dataset):
    """Right Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k/model_generation', 
        target_set: str = None,
        generator_id: str = None, 
        per_problem_sampling_solution: str = None, 
        loss_level: str = 'token', 
        loss_on_llm: bool = False,
        dedup: bool = False
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ('token', 'step')

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # idx，input，question, answer, ground_truth, 
        # outputs(response, response_answer, label, step_labels, step_labels_progress)
        self.examples = get_model_solutions(data_dir, generator_id, target_set, process=True)

        # assert len(self.examples[0]['outputs']) >= per_problem_sampling_solution

        if per_problem_sampling_solution != -1:
            for example in self.examples:
                if len(example['outputs']) >= per_problem_sampling_solution:
                    example['outputs'] = example['outputs'][:per_problem_sampling_solution]
                else:
                    print(f"Warning: {len(example['outputs'])} solutions for problem {example['idx']} is less than per_problem_sampling_solution {per_problem_sampling_solution}")
                    continue
        else:
            per_problem_sampling_solution = len(self.examples[0]['outputs'])
        
        if dedup:
            for ex in self.examples:
                dedup_outputs = []
                responses = set()
                for output in ex['outputs']:
                    if output['response'] in responses:
                        continue
                    responses.add(output['response'])
                    dedup_outputs.append(output)
                ex['outputs'] = dedup_outputs

        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)]
        
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [[outputs["response"] for outputs in ex["outputs"]] for ex in self.examples]
        
        # reward label
        # [[True, False, False, ..., False], [True, False, False, ..., False], ..., [True, False, False, ..., False]]
        process_labels = [[outputs["step_labels"] for outputs in ex["outputs"]] for ex in self.examples]

        # outcome-supervised BiRM
        # outcome_labels = [[outputs["label"] for outputs in ex["outputs"]] for ex in self.examples]

        # value label
        # math-shepherd-plus BiRM
        outcome_labels = [[outputs["step_s_label"] for outputs in ex["outputs"]] for ex in self.examples]
        # outcome_labels = [[outputs["step_h_label"] for outputs in ex["outputs"]] for ex in self.examples]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        process_labels = self._flatten(process_labels)
        outcome_labels = self._flatten(outcome_labels)

        qns_tokens = tokenizer(qns_str, padding=False).input_ids

        print(f"All solutions_str: {len(solutions_str)}")

        # check
        for solution in solutions_str:
            assert solution.endswith(MODEL_NEWLINE), f"Solution does not end with newline token: {solution}"
            
        # xxxx kn, xxxx kn, xxxx kn <eos>
        steps_str = [
            list(map(lambda x: x+MODEL_NEWLINE, solution_str.split(MODEL_NEWLINE)[:-1]))
            for solution_str in solutions_str
        ]


        solutions_tokens = [
            [tokenizer.encode(step_str[0], add_special_tokens=False)]
              + [self._get_continued_input_ids(step) for step in step_str[1:]]
            for step_str in steps_str
        ]

        step_tokens_lens = [
            [len(step) for step in tokens]
            for tokens in solutions_tokens
        ]

        solutions_tokens = [self._flatten(tokens) for tokens in solutions_tokens]


        self.indices1 = indices1
        self.indices2 = indices2
        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.solutions_str = solutions_str
        self.solutions_tokens = solutions_tokens # [solution1, solution2, solution3, ...]

        
        self.step_tokens_lens = step_tokens_lens # [[4, 3, 2], [4, 3, 2], ...]
        self.process_labels = process_labels # [[True, False, False], [True, False, False], ...]
        self.outcome_labels = outcome_labels
        self.n_question = len(self.examples)
        self.per_problem_sampling_solution = per_problem_sampling_solution

        print("Loading BiRM dataset...")
        print(f'Number of examples = {len(qns_str)} with #deduplication = {self.n_question * self.per_problem_sampling_solution - len(qns_str)}')

        # question + solution + eos
        self.max_len = max([
                len(self.qns_tokens[i]) + len(self.solutions_tokens[i]) + 1
                for i in range(len(self.solutions_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")
    

    def _add_placeholder(self, text: str):
        if len(text) == 0 or text[0] != '#':
            return '####' + text
        return '$$' + text

    def _get_continued_input_ids(self, text: Union[str, List[str]], right_padding=False, return_tensors=False):
        if isinstance(text, str):
            text = self._add_placeholder(text)
        else:
            text = [self._add_placeholder(x) for x in text]

        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if isinstance(text, str):
            input_ids = input_ids[1:]
        else:
            input_ids = [ids[1:] for ids in input_ids]

        if right_padding and isinstance(text, list):
            max_length = max([len(ids) for ids in input_ids])
            input_ids = [
                ids 
                + [self.pad_token_id] * (max_length - len(ids))
                for ids in input_ids
            ]
        if return_tensors:
            input_ids = torch.tensor(input_ids)
        return input_ids
    
    
    def __len__(self):
        return len(self.solutions_tokens)

    def _flatten(self, ls):
        return [item for sublist in ls for item in sublist]

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        process_labels = self.process_labels[idx]
        outcome_labels = self.outcome_labels[idx]
        step_tokens_lens = self.step_tokens_lens[idx]

        # input_ids, attention_mask
        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]

        masks = (
            ([0] * len(qn_tokens))
            + ([1] * len(sol_tokens))
            + ([1])
        )

        # create language modeling labels
        if self.loss_on_llm:
            labels = input_ids
            # -100
            labels = mask_labels(labels, masks)

        # create BiRM labels
        if self.loss_level == 'token':

            # For reward labels
            p_labels = [0] * len(qn_tokens)

            for i, (tokens_len, step_label) in enumerate(zip(step_tokens_lens, process_labels)):
                if i < len(step_tokens_lens):
                    p_labels += [-100] * (tokens_len - 1)
                    if step_label:
                        p_labels += [1]
                    else:
                        p_labels += [0]

            # eos token
            p_labels += [-100]

            # -100
            p_labels = mask_labels(p_labels, masks)

            assert len(p_labels) == len(input_ids)

            # For value labels
            o_labels = [0] * len(qn_tokens)

            # outcome-supervised value
            # for i, (tokens_len, step_label) in enumerate(zip(step_tokens_lens, process_labels)):
            #     if i < len(step_tokens_lens):
            #         o_labels += [-100] * (tokens_len - 1)
            #         if outcome_labels:
            #             o_labels += [1]
            #         else:
            #             o_labels += [0]

            # math-shepherd value
            for i, (tokens_len, step_label) in enumerate(zip(step_tokens_lens, outcome_labels)):
                if i < len(step_tokens_lens):
                    o_labels += [-100] * (tokens_len - 1)
                    if step_label:
                        o_labels += [step_label]
                    else:
                        o_labels += [0]

            # eos token
            o_labels += [-100]

            # -100
            o_labels = mask_labels(o_labels, masks)

            assert len(o_labels) == len(input_ids)

            # print(f"p-label {p_labels}, o-label {o_labels}")
        else:
            raise NotImplementedError

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        p_labels = torch.tensor(p_labels)
        o_labels = torch.tensor(o_labels)

        return dict(
            idx1=self.indices1[idx], idx2=self.indices2[idx], 
            input_ids=input_ids, labels=labels,
            process_labels=p_labels, outcome_labels=o_labels,
            qn_str=self.qns_str[idx], qn_tokens=self.qns_tokens[idx], sol_str=self.solutions_str[idx], sol_tokens=self.solutions_tokens[idx],
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, p_labels, o_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "process_labels", "outcome_labels"))
        idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens = tuple([instance[key] for instance in instances] for key in ("idx1", "idx2", "qn_str", "qn_tokens", "sol_str", "sol_tokens"))

        # right padding for training
        #  (batch_size, max_length)
        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        # LLM_loss & Verifier_loss
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False) if self.loss_on_llm else None
        p_labels = right_pad_sequences(p_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        o_labels = right_pad_sequences(o_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            idx1=idx1, idx2=idx2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            process_labels=p_labels, outcome_labels=o_labels,
            qn_str=qn_str, qn_tokens=qn_tokens, sol_str=sol_str, sol_tokens=sol_tokens
        )


def left_pad_sequences(sequences: List[torch.LongTensor], padding_value: int, return_attention_mask: bool = False):
    max_length = max(len(x) for x in sequences)
    padded_sequences = torch.stack([F.pad(seq, (max_length - seq.shape[-1], 0), value=padding_value) for seq in sequences], dim=0)
    if return_attention_mask:
        attention_mask = padded_sequences.ne(padding_value)
        return padded_sequences, attention_mask
    return padded_sequences

def right_pad_sequences(sequences: List[torch.LongTensor], padding_value: int, return_attention_mask: bool = False):
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
    )
    if return_attention_mask:
        attention_mask = padded_sequences.ne(padding_value)
        return padded_sequences, attention_mask
    return padded_sequences

def mask_labels(labels: List[int], masks: List[bool]):
    """Mask the corresponding label into IGNORE_INDEX"""
    assert len(labels) == len(masks)
    return [
        token if mask
        else IGNORE_INDEX
        for token, mask in zip(labels, masks) 
    ]
