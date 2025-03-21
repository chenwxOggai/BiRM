from utils.states import set_random_seed
from utils.verifier_models_bi import load_verifier
from utils.datasets import make_test_verifier_data_module, make_testing_dataloader
from utils.metrics import VerifierClassificationAcc, VerifierMPk, GenWithVerifierAcc


import torch
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm
import torch.distributed as dist
from typing import Optional, List
from accelerate import Accelerator
import os
import json
import pandas as pd
import gc


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    fp16: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/gsm8k/model_generation",
        metadata={"help": "Path to the training data."},
    )
    target_set: str = field(
        default="test", metadata={"help": "specify which data set to generate"}
    )
    generator_id: str = field(default="llama7b-2-ep2")

    # total_acc, ORM rerank 1st acc
    verifier_output_dir: str = field(
        default="eval_results/gsm8k/verifier",
        metadata={"help": "Path to save the responses and metrics."},
    )
    # test-time scaling, ORM rerank acc
    generator_metric_dir: str = field(
        default="eval_results/gsm8k/generator_with_verifier",
        metadata={"help": "Path to save the responses and metrics."},
    )


@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)

    seed: int = field(default=None)
    agg_mode: str = field(default="last")
    beta: float = field(default=1.0)


def get_save_files(
    model_args: dataclass, data_args: dataclass, inference_args: dataclass
):
    verifier_output_dir = os.path.join(
        data_args.verifier_output_dir, data_args.target_set
    )

    generator_metric_dir = os.path.join(
        data_args.generator_metric_dir, data_args.target_set
    )

    verifier_id = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    generator_metric_dir = os.path.join(generator_metric_dir, verifier_id)
    os.makedirs(generator_metric_dir, exist_ok=True)

    # "_g(llama7b-2-ep2)"
    generator_id_suffix = "_g(%s)" % data_args.generator_id
    # "_v(llama7b-2-ep2_n100_scahead_mse_lm_token)"
    verifier_id_suffix = "_v(%s)" % os.path.basename(
        os.path.normpath(model_args.model_name_or_path)
    )

    verifier_suffix = (verifier_id_suffix + generator_id_suffix).lstrip("_")
    generator_suffix = (generator_id_suffix + verifier_id_suffix).lstrip("_")

    # v_scores
    verifier_outputs_file = f"responses_{verifier_suffix}_{inference_args.agg_mode}_seed{inference_args.seed}.jsonl"
    verifier_metrics_file = f"metrics_{verifier_suffix}_{inference_args.agg_mode}_seed{inference_args.seed}.json"
    # test-time scaling metrics
    generator_metrics_file = f"metrics_{generator_suffix}_{inference_args.agg_mode}_seed{inference_args.seed}.csv"
    return (
        os.path.join(verifier_output_dir, verifier_outputs_file),
        os.path.join(verifier_output_dir, verifier_metrics_file),
        os.path.join(generator_metric_dir, generator_metrics_file),
    )


# solutions v_scores
def extract_sol_vscores(
    qns_tokens: List[List[int]],
    sols_tokens: List[List[int]],
    batch_vscores: torch.FloatTensor,
) -> List[list]:
    sol_vscores = []
    # qn_tokens, sol_tokens = (batch_size, n_seq), (batch_size, n_seq)
    # batch_vscores = (batch_size, n_total_seq, 1)
    for qn_tokens, sol_tokens, vscores in zip(qns_tokens, sols_tokens, batch_vscores):
        svs = vscores[len(qn_tokens) : len(qn_tokens) + len(sol_tokens) + 1][:, 0]
        sol_vscores.append(svs.tolist())  # (padded_response_len, )
    return sol_vscores


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferenceArguments)
    )
    model_args, data_args, inference_args = parser.parse_args_into_dataclasses()
    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)

    verifier_outputs_file, verifier_metrics_file, generator_metrics_file = (
        get_save_files(model_args, data_args, inference_args)
    )

    accelerator = Accelerator()

    verifier, tokenizer = load_verifier(model_args)
    dataset = make_test_verifier_data_module(tokenizer, data_args)
    dataloader = make_testing_dataloader(dataset, batch_size=inference_args.batch_size)

    n_question = dataset.n_question
    per_problem_sampling_solution = dataset.per_problem_sampling_solution

    # total_solutions accuracy
    verifier_acc_metric = VerifierClassificationAcc(n_data=len(dataset))
    # Mean Precision at K
    verifier_mpk_metric = VerifierMPk(
        n_data=len(dataset), n_solution_per_problem=per_problem_sampling_solution
    )
    # Take the top k predictions, sort, and calculate the accuracy of the correct answer with the highest predicted score
    generator_acc_metric = GenWithVerifierAcc(
        n_data=len(dataset), n_solution_per_problem=per_problem_sampling_solution
    )

    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)

    verifier_outputs = []
    for data in dataset:
        if len(verifier_outputs) == 0 or verifier_outputs[-1]["idx"] != data["idx1"]:
            verifier_outputs.append(
                {
                    "idx": data["idx1"],  # "idx": 0
                    "question": data["qn_str"],
                    "outputs": [],
                }
            )
        verifier_outputs[-1]["outputs"].append(
            {
                "response": data["sol_str"],
                "tokens": data["sol_tokens"],
                "label": data["v_class"],
            }
        )

    verifier.eval().cuda()
    accelerator.unwrap_model(verifier).gradient_checkpointing_enable()
    accelerator.wait_for_everyone()

    dataloader_iterator = (
        tqdm(
            enumerate(iterable=dataloader),
            total=len(dataloader),
            desc="Evaluation with Verifier",
        )
        if accelerator.is_main_process
        else enumerate(dataloader)
    )
    (
        all_idxs1_list,
        all_idxs2_list,
        all_vscores_list,
        all_outcome_list,
        all_process_list,
    ) = tuple([] for _ in range(5))
    for _, batch in dataloader_iterator:
        with torch.inference_mode(mode=True):
            output_all, output_o, output_p = verifier.scoring_sequences_by_step(
                batch["input_ids"],
                agg_mode=inference_args.agg_mode,
                beta=inference_args.beta,
            )  # (batch_size,)

            if inference_args.agg_mode == "full":
                # (batch_size, seq_len)
                # => (batch_size, n_total_seq, 1)
                assert output_all.shape[1] == batch["input_ids"].shape[1], (
                    f"seq_len ({output_all.shape[1]}) and n_total_seq ({batch['input_ids'].shape[1]}) must be consistent!"
                )
                v_scores = output_all.unsqueeze(2)  # (batch_size, seq_len, 1)
                outcome_v_scores = output_o.unsqueeze(2)  # (batch_size, seq_len, 1)
                process_v_scores = output_p.unsqueeze(2)  # (batch_size, seq_len, 1)
            else:
                # 'min', 'max', 'mean', 'last'
                n_total_seq = batch["input_ids"].shape[1]
                v_scores = (
                    output_all.unsqueeze(1).unsqueeze(2).expand(-1, n_total_seq, 1)
                )  # (batch_size, n_total_seq, 1)
                outcome_v_scores = (
                    output_o.unsqueeze(1).unsqueeze(2).expand(-1, n_total_seq, 1)
                )  # (batch_size, n_total_seq, 1)
                process_v_scores = (
                    output_p.unsqueeze(1).unsqueeze(2).expand(-1, n_total_seq, 1)
                )  # (batch_size, n_total_seq, 1)

        verifier_acc_metric(v_scores, batch["v_labels"])
        verifier_mpk_metric(v_scores, batch["v_labels"])
        generator_acc_metric(v_scores, batch["v_labels"])

        idx1, idx2, qn_tokens, sol_tokens = tuple(
            batch[key] for key in ("idx1", "idx2", "qn_tokens", "sol_tokens")
        )
        sol_vscores: List[torch.FloatTensor] = extract_sol_vscores(
            qn_tokens, sol_tokens, v_scores
        )
        sol_outcome_v_scores: List[torch.FloatTensor] = extract_sol_vscores(
            qn_tokens, sol_tokens, outcome_v_scores
        )
        sol_process_v_scores: List[torch.FloatTensor] = extract_sol_vscores(
            qn_tokens, sol_tokens, process_v_scores
        )

        for obj, container in [
            (idx1, all_idxs1_list),
            (idx2, all_idxs2_list),
            (sol_vscores, all_vscores_list),
            (sol_outcome_v_scores, all_outcome_list),
            (sol_process_v_scores, all_process_list),
        ]:
            container.extend(obj)

    gc.collect()
    torch.cuda.empty_cache()

    # gather
    if accelerator.num_processes != 1:
        (
            all_idxs1_gather,
            all_idxs2_gather,
            all_vscores_gather,
            all_outcome_gather,
            all_process_gather,
        ) = tuple([None] * dist.get_world_size() for _ in range(5))
        for obj, container in [
            (all_idxs1_list, all_idxs1_gather),
            (all_idxs2_list, all_idxs2_gather),
            (all_vscores_list, all_vscores_gather),
            (all_outcome_list, all_outcome_gather),
            (all_process_list, all_process_gather),
        ]:
            dist.all_gather_object(container, obj)

        (
            all_idxs1_gather,
            all_idxs2_gather,
            all_vscores_gather,
            all_outcome_gather,
            all_process_gather,
        ) = tuple(
            [item for sublist in container for item in sublist]
            for container in [
                all_idxs1_gather,
                all_idxs2_gather,
                all_vscores_gather,
                all_outcome_gather,
                all_process_gather,
            ]
        )
    else:
        (
            all_idxs1_gather,
            all_idxs2_gather,
            all_vscores_gather,
            all_outcome_gather,
            all_process_gather,
        ) = (
            all_idxs1_list,
            all_idxs2_list,
            all_vscores_list,
            all_outcome_list,
            all_process_list,
        )

    # record
    for idx1, idx2, sol_vscores, sol_outcome_v_scores, sol_process_v_scores in zip(
        all_idxs1_gather,
        all_idxs2_gather,
        all_vscores_gather,
        all_outcome_gather,
        all_process_gather,
    ):
        if "vscores" in verifier_outputs[idx1]["outputs"][idx2]:
            continue

        verifier_outputs[idx1]["outputs"][idx2]["vscores"] = sol_vscores
        verifier_outputs[idx1]["outputs"][idx2]["outcome_vscores"] = (
            sol_outcome_v_scores
        )
        verifier_outputs[idx1]["outputs"][idx2]["process_vscores"] = (
            sol_process_v_scores
        )

    # save outputs
    # outputs (response, tokens, label, vscores)
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(verifier_outputs_file), exist_ok=True)
        with open(verifier_outputs_file, "w") as fp:
            fp.writelines(
                [
                    json.dumps(verifier_outputs[i]) + "\n"
                    for i in range(len(verifier_outputs))
                ]
            )
        print(f"+ [Save] Save Outputs to {verifier_outputs_file}")

    # calculate verifier metrics
    test_acc = verifier_acc_metric.get_metric()
    mp1 = verifier_mpk_metric.get_metric(1)

    metrics = {
        "#question": n_question,
        "#solution_per_problem": per_problem_sampling_solution,
        "#total_solutions": len(dataset),
        "accuracy": test_acc,
        "mp1-ORM/PRM rerank": mp1,
    }
    accelerator.print(metrics)

    # calculate generator metrics
    n_list = list(range(1, per_problem_sampling_solution + 1, 1))
    df = pd.DataFrame(columns=["acc"], index=n_list)
    df.columns.name = "n_solution"

    for i in n_list:
        df.loc[i] = generator_acc_metric.get_metric(i, reset=False)

    accelerator.print(df)

    # save metrics
    # verifier: total_acc, ORM rerank 1st acc
    # generator: test-time scaling acc
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(verifier_metrics_file), exist_ok=True)
        json.dump(
            metrics, open(verifier_metrics_file, "w"), indent=4, ensure_ascii=False
        )
        print(f"+ [Save] Save Verifier Metrics to {verifier_metrics_file}")

        df.to_csv(generator_metrics_file, index_label=df.columns.name)
        print(f"+ [Save] Save Generator Metrics to {generator_metrics_file}")


if __name__ == "__main__":
    main()
