from utils.states import set_random_seed
from utils.verifier_models import load_two_verifiers
from utils.verifier_models_bi import load_bi_verifier
from utils.datasets import make_testing_dataloader
from utils.sampling_vllm import SamplingWithVLLM
import torch
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm
import torch.distributed as dist
from typing import Optional
from accelerate import Accelerator
import os
import json
import gc
from datetime import timedelta


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    verifier_model_name_or_path: Optional[str] = field(default=None)
    outcome_verifier_model_name_or_path: Optional[str] = field(default=None)
    process_verifier_model_name_or_path: Optional[str] = field(default=None)
    birm_verifier_model_name_or_path: Optional[str] = field(default=None)

    fp16: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    dataset: str = field(default="gsm8k")

    data_dir: str = field(
        default="data/gsm8k/model_generation",
        metadata={"help": "Path to the training data."},
    )
    target_set: str = field(
        default="test", metadata={"help": "specify which data set to generate"}
    )

    output_dir: str = field(
        default="eval_results/gsm8k/generator_with_verifier",
        metadata={"help": "Path to save the responses and metrics."},
    )


@dataclass
class GenerationArguments:
    do_sample: bool = field(default=False)
    num_beams: int = field(default=1)

    temperature: float = field(default=0.7)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)

    max_length: int = field(default=2048)
    max_new_tokens: int = field(default=-1)


@dataclass
class InferenceArguments:
    batch_size: int = field(default=26)
    vs_batch_size: int = field(default=64)

    # expand to n_sampling_steps
    # rerank and choose n_beam
    n_sampling_steps: int = field(default=10)
    n_beam: int = field(default=1)

    max_step_length: int = field(default=100)
    max_n_step: int = field(default=10)

    inference_mode: str = field(default="beam")
    dedup_mode: int = field(default=0)
    agg_mode: str = field(
        default="last"
    )  # for birm, agg_mode={min, max, last, mean}
    beta: float = field(default=1.0)

    seed: int = field(default=None)


def get_save_files(
    model_args: dataclass, data_args: dataclass, inference_args: dataclass
):
    output_dir = os.path.join(data_args.output_dir, data_args.target_set)

    if inference_args.inference_mode == "beam":
        if model_args.birm_verifier_model_name_or_path is not None:
            verifier_id = os.path.basename(
                os.path.normpath(model_args.birm_verifier_model_name_or_path)
            )
        elif (
            model_args.outcome_verifier_model_name_or_path is not None
            and model_args.process_verifier_model_name_or_path is not None
        ):
            verifier_id = os.path.basename(
                os.path.normpath(model_args.outcome_verifier_model_name_or_path)
            ) + os.path.basename(
                os.path.normpath(model_args.process_verifier_model_name_or_path)
            )
        elif model_args.outcome_verifier_model_name_or_path is not None:
            verifier_id = os.path.basename(
                os.path.normpath(model_args.outcome_verifier_model_name_or_path)
            )
        elif model_args.process_verifier_model_name_or_path is not None:
            verifier_id = os.path.basename(
                os.path.normpath(model_args.process_verifier_model_name_or_path)
            )
        else:
            raise ValueError("Either outcome or process verifier must be provided")
        output_dir = os.path.join(output_dir, verifier_id)
        os.makedirs(output_dir, exist_ok=True)

        generator_id = os.path.basename(os.path.normpath(model_args.model_name_or_path))

        # _total20_beam4_llama3_8b_ep2_last_42
        suffix = f"_total{inference_args.n_sampling_steps}_beam{inference_args.n_beam}_{generator_id}_{inference_args.agg_mode}_{inference_args.seed}".strip(
            "_"
        )


    responses_file = f"responses_{suffix}.jsonl"
    metrics_file = f"metrics_{suffix}.json"

    return os.path.join(output_dir, responses_file), os.path.join(
        output_dir, metrics_file
    )


def main():
    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # not to enforce timeout
    RANK = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        "nccl" if dist.is_nccl_available() else "gloo",
        timeout=timedelta(seconds=7200000),
        rank=RANK,
        world_size=world_size,
    )

    print(f"Rank {RANK} initialized.")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GenerationArguments, InferenceArguments)
    )
    model_args, data_args, generation_args, inference_args = (
        parser.parse_args_into_dataclasses()
    )

    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)

    if data_args.dataset == "gsm8k":
        from utils.gsm8k.datasets import make_test_generator_data_module
        from utils.gsm8k.decoding import extract_answer, get_answer_label
        from utils.gsm8k.metrics import GeneratorAnswerAcc
    elif data_args.dataset == "math500":
        from utils.math500.datasets import make_test_generator_data_module
        from utils.math500.decoding import extract_answer, get_answer_label
        from utils.math500.metrics import GeneratorAnswerAcc
    elif data_args.dataset == "gaokao2023":
        from utils.math500.datasets import make_test_generator_data_module
        from utils.math500.decoding import extract_answer, get_answer_label
        from utils.math500.metrics import GeneratorAnswerAcc
    else:
        raise NotImplementedError

    responses_file, metrics_file = get_save_files(model_args, data_args, inference_args)

    accelerator = Accelerator()

    if inference_args.inference_mode == "beam":
        if model_args.birm_verifier_model_name_or_path is not None:
            birm_verifier, tokenizer = load_bi_verifier(model_args)
            outcome_verifier = None
            process_verifier = None
        else:
            outcome_verifier, process_verifier, tokenizer = load_two_verifiers(
                model_args
            )
            birm_verifier = None
    else:
        raise NotImplementedError

    dataset = make_test_generator_data_module(tokenizer, data_args, inference_args)
    dataloader = make_testing_dataloader(dataset, batch_size=1)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)

    sampler = SamplingWithVLLM(
        accelerator=accelerator,
        model_name_or_path=model_args.model_name_or_path,
        outcome_verifier=outcome_verifier,
        process_verifier=process_verifier,
        birm_verifier=birm_verifier,
        tokenizer=tokenizer,
        generation_args=generation_args,
        beta=inference_args.beta,
    )
    acc_metric = GeneratorAnswerAcc(n_data=len(dataset))  # greedy metric

    if birm_verifier is not None:
        birm_verifier.eval().cuda()
        accelerator.unwrap_model(birm_verifier).gradient_checkpointing_enable()
    if outcome_verifier is not None:
        outcome_verifier.eval().cuda()
        accelerator.unwrap_model(outcome_verifier).gradient_checkpointing_enable()
    if process_verifier is not None:
        process_verifier.eval().cuda()
        accelerator.unwrap_model(process_verifier).gradient_checkpointing_enable()
    accelerator.wait_for_everyone()

    # idx, input, question, answer, ground_truth
    # => response, response_answer, label, intermediate_steps [sequences, choices]
    response_list = [
        {
            "idx": data["idx"],
            "input": data["input"],
            "question": data["question"],
            **data["record_data"],
        }
        for data in dataset
    ]
    print(f"len(dataloader): {len(dataloader)}")
    progress = (
        tqdm(
            enumerate(iterable=dataloader),
            total=len(dataloader),
            desc=f"Evaluation with beam {inference_args.n_beam} / sample {inference_args.n_sampling_steps}",
        )
        if accelerator.is_main_process
        else enumerate(dataloader)
    )
    # reference = standardize ground truth
    all_idxs_list, all_references_list, all_completions_list, all_intermediates_list = (
        tuple([] for _ in range(4))
    )

    for _, batch in progress:
        idx = batch["idx"][0]
        inp = batch["input"][0]
        reference = batch["reference"][0]

        # verify generation by step
        completion, intermediates = sampler.sample_by_steps(
            qn_str=inp,
            batch_size=inference_args.batch_size,
            vs_batch_size=inference_args.vs_batch_size,
            n_beam=inference_args.n_beam,
            n_sampling_steps=inference_args.n_sampling_steps,
            max_step_length=inference_args.max_step_length,
            max_n_step=inference_args.max_n_step,
            inference_mode=inference_args.inference_mode,
            dedup_mode=inference_args.dedup_mode,
            agg_mode=inference_args.agg_mode,
        )

        acc_metric([completion], [reference])

        for obj, container in [
            (idx, all_idxs_list),
            (reference, all_references_list),
            (completion, all_completions_list),
            (intermediates, all_intermediates_list),
        ]:
            container.append(obj)

        if accelerator.is_main_process:
            progress.update(1)

        gc.collect()
        torch.cuda.empty_cache()

    # gather all gpu results
    if accelerator.num_processes != 1:
        (
            all_idxs_gather,
            all_references_gather,
            all_completions_gather,
            all_intermediates_gather,
        ) = tuple([None] * dist.get_world_size() for _ in range(4))
        for obj, container in [
            (all_idxs_list, all_idxs_gather),
            (all_references_list, all_references_gather),
            (all_completions_list, all_completions_gather),
            (all_intermediates_list, all_intermediates_gather),
        ]:
            dist.all_gather_object(container, obj)

        (
            all_idxs_gather,
            all_references_gather,
            all_completions_gather,
            all_intermediates_gather,
        ) = tuple(
            [item for sublist in container for item in sublist]
            for container in [
                all_idxs_gather,
                all_references_gather,
                all_completions_gather,
                all_intermediates_gather,
            ]
        )
    else:
        (
            all_idxs_gather,
            all_references_gather,
            all_completions_gather,
            all_intermediates_gather,
        ) = (
            all_idxs_list,
            all_references_list,
            all_completions_list,
            all_intermediates_list,
        )

    # record
    for idx, reference, completion, intermediates in zip(
        all_idxs_gather,
        all_references_gather,
        all_completions_gather,
        all_intermediates_gather,
    ):
        if "response" in response_list[idx]:
            continue

        # save final results
        response_answer = extract_answer(completion)
        response_list[idx].update(
            {
                "response": completion,
                "response_answer": response_answer,
                "label": get_answer_label(response_answer, reference),
                "intermediate_steps": intermediates,  # List[Dict(sequences, choices)]
            }
        )

    # save outputs
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(responses_file), exist_ok=True)
        with open(responses_file, "w") as fp:
            fp.writelines([json.dumps(data) + "\n" for data in response_list])
        print(f"+ [Save] Save Responses to {responses_file}")

    # calculate metrics
    # greedy
    metrics = {"accuracy": acc_metric.get_metric()}
    accelerator.print(metrics)

    # save metrics
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        json.dump(metrics, open(metrics_file, "w"), indent=4, ensure_ascii=False)
        print(f"+ [Save] Save Metrics to {metrics_file}")


if __name__ == "__main__":
    main()
