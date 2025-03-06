import torch
import transformers
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional
import gc
from accelerate import Accelerator
import json

from utils.states import set_deepspeed_config, set_training_states, set_random_seed
from utils.optim import get_optimizers
from utils.models import save_training_args_with_accelerator
from utils.verifier_models_bi import (
    save_verifier,
    save_verifier_checkpoint,
    load_verifier_for_training,
)
from utils.datasets import make_training_verifier_data_module, make_training_dataloaders

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/gsm8k/model_generation",
        metadata={"help": "Path to the training data."},
    )
    target_set: str = field(default="train")
    val_target_set: str = field(default=None)
    generator_id: str = field(default="llama7b-2-ep2")

    # n = 15
    per_problem_sampling_solution: int = field(default=-1)
    loss_level: str = field(default="token")
    # Add language modeling loss
    loss_on_llm: bool = field(default=False)

    # 是否去重，默认True
    dedup: bool = field(default=False)
    # PRM or ORM
    process: bool = field(default=False)
    # BiRM
    bi_process: bool = field(default=False)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "When it is specified, num_train_epoches is ignored"},
    )
    num_train_epoches: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)

    eval_steps: int = field(
        default=-1, metadata={"help": "When it is specified, eval_epoches is ignored"}
    )
    eval_epoches: int = field(default=1)
    per_device_eval_batch_size: int = field(default=4)

    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0)
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(
        default=-1, metadata={"help": "When it is specified, warmup_ratio is ignored"}
    )
    warmup_ratio: float = field(default=0)

    num_lr_epoches_fs: int = field(default=-1)
    num_lr_epoches_scatter: int = field(default=-1)

    # wandb log
    logging_steps: int = field(
        default=-1,
        metadata={"help": "When it is specified, logging_epoches is ignored"},
    )
    logging_epoches: int = field(default=1)

    # save_epoches = 1
    save_steps: int = field(
        default=-1, metadata={"help": "When it is specified, save_epoches is ignored"}
    )
    save_epoches: int = field(default=1)
    save_total_limit: int = field(default=0)
    save_best: bool = field(default=False)

    seed: int = field(default=42)
    resume: bool = field(default=False)


@dataclass
class OutputArguments:
    logging_dir: str = field(default="wandb/")
    save_dir: str = field(default="checkpoints/")


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, OutputArguments)
    )
    model_args, data_args, training_args, output_args = (
        parser.parse_args_into_dataclasses()
    )

    config_args_dict = model_args.__dict__.copy()
    combined_dict = {**data_args.__dict__, **training_args.__dict__}
    config_args_dict.update(combined_dict)
    print(json.dumps(config_args_dict, ensure_ascii=False, indent=4))

    set_random_seed(training_args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps
    )

    set_deepspeed_config(accelerator, training_args)

    model, tokenizer = load_verifier_for_training(model_args, training_args)

    data_module = make_training_verifier_data_module(tokenizer, data_args)

    # Return: input_ids, attention_mask, labels, v_labels
    # idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class
    train_dataloader, val_dataloader = make_training_dataloaders(
        data_module, training_args
    )

    # config optimizer and scheduler
    set_training_states(data_module, training_args)

    # config optimizer 和 lr_scheduler
    optimizer, lr_scheduler = get_optimizers(model, training_args)

    if val_dataloader is not None:
        model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer
        )
    else:
        model, train_dataloader, optimizer = accelerator.prepare(
            model, train_dataloader, optimizer
        )

    cur_epoch = local_step = global_step = 0

    # training
    global_step = 0
    model.train()
    while global_step < training_args.num_training_steps:
        train_dataloader_iterator = (
            tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Training - Epoch {cur_epoch + 1} / {training_args.num_train_epoches}",
            )
            if accelerator.is_main_process
            else enumerate(train_dataloader)
        )

        for local_step, batch in train_dataloader_iterator:
            batch_input = {
                k: v
                for k, v in batch.items()
                if k
                in (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "process_labels",
                    "outcome_labels",
                )
            }
            # backpropagation
            with accelerator.accumulate(model):
                output = model(**batch_input, output_all_losses=True)
                loss = output.loss
                all_losses = output.all_losses  # {'loss': loss, 'process_loss': process_loss, 'outcome_loss': outcome_loss}
                accelerator.backward(loss)

                optimizer.step()
                if (
                    not accelerator.optimizer_step_was_skipped
                    and global_step % training_args.gradient_accumulation_steps == 0
                ):
                    lr_scheduler.step()
                optimizer.zero_grad()

            # training logging
            if accelerator.is_main_process:
                # loss, v_loss, llm_loss
                train_dataloader_iterator.set_postfix(
                    epoch=cur_epoch,
                    step=local_step,
                    loss=loss.item(),
                    process_loss=all_losses.get("process_loss").item(),
                    outcome_loss=all_losses.get("outcome_loss").item(),
                )

            # save checkpoint
            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                save_verifier_checkpoint(
                    accelerator,
                    model,
                    tokenizer,
                    output_args.save_dir,
                    global_step,
                    training_args.save_total_limit,
                )

            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                pass

            global_step += 1

        cur_epoch += 1
        if cur_epoch == 1:
            accelerator.wait_for_everyone()
            save_verifier_checkpoint(
                accelerator,
                model,
                tokenizer,
                output_args.save_dir,
                global_step,
                training_args.save_total_limit,
            )
            print(f"Epoch {cur_epoch} over - Save checkpoint")

        del train_dataloader_iterator
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    save_verifier(accelerator, model, tokenizer, output_args.save_dir)
    save_training_args_with_accelerator(
        accelerator, training_args, output_args.save_dir
    )
    print("Training Over - Save verifier")


if __name__ == "__main__":
    main()
