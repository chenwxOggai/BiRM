from utils.models import (
    wrapper_safe_save_model_with_accelerator,
    wrapper_save_checkpoint,
    wrapper_save_best_checkpoint,
    build_model,
    load_model,
    load_model_for_training,
)
from utils.sampling import (
    shift_padding_to_right_2D,
    find_rightmost_notpadded_positions,
)
from utils.constants import IGNORE_INDEX, MODEL_NEWLINE_TOKEN
from typing import Optional, List, Dict, Mapping
import transformers
from transformers.generation.utils import ModelOutput
from torch import nn
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from accelerate import Accelerator
import os


@dataclass
class VerifierModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    v_scores: torch.FloatTensor = None
    all_losses: Optional[Dict[str, torch.FloatTensor]] = None


class Verifier(nn.Module):
    def __init__(self, backbone, checkpoint_dir=None):
        super(Verifier, self).__init__()
        self.backbone = backbone

        self.embed_dim = self.backbone.get_input_embeddings().embedding_dim

        self.gain = nn.Parameter(
            torch.randn(
                1,
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                1,
            )
        )
        self.dropout = nn.Dropout(p=0.2)
        # (out_features, in_features) = (1, embedding_dim)
        self.vscore_head = nn.Linear(
            self.backbone.get_input_embeddings().embedding_dim, 1, bias=False
        )

        if checkpoint_dir and os.path.exists(
            os.path.join(checkpoint_dir, "verifier.pth")
        ):
            verifier_params = torch.load(os.path.join(checkpoint_dir, "verifier.pth"))
            self.load_state_dict(verifier_params, strict=False)
        else:
            self.init_head_params()

        self.pad_token_id = backbone.config.pad_token_id

    def init_head_params(self):
        # output_embeddings, shape = (vocab_size, embed_dim)
        output_embeddings = self.backbone.get_output_embeddings().weight.data
        # shape = (1, embed_dim)
        output_embeddings_avg = output_embeddings.mean(dim=0, keepdim=True)

        self.vscore_head.weight = nn.Parameter(output_embeddings_avg)

    def loss_fct(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        # (batch_size, n_seq, 1)
        # v_scores: (batch_size, n_seq, 1)
        # v_labels: (batch_size, n_seq)
        return mse_loss_with_mask(v_scores.squeeze(), v_labels.type_as(v_scores))

    def transform(self, last_hidden_states):
        return self.gain * last_hidden_states + self.bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        v_labels: Optional[torch.LongTensor] = None,
        output_all_losses: Optional[bool] = None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        llm_loss = outputs.loss

        llm_hidden_states = outputs.hidden_states
        # gain*last_hidden_states+bias
        # (batch_size, n_seq, embed_dim)
        v_hidden_states = self.transform(llm_hidden_states[-1])

        # (batch_size, n_seq, 1)
        v_scores = self.vscore_head(self.dropout(v_hidden_states))

        v_loss, loss = None, None
        if v_labels is not None:
            # mse_loss_with_mask
            v_loss = self.loss_fct(v_scores, v_labels)
            loss = v_loss + (llm_loss if labels is not None else 0)

        all_losses = None
        if output_all_losses:
            all_losses = {"llm_loss": llm_loss, "v_loss": v_loss}

        return VerifierModelOutput(
            loss=loss,
            v_scores=v_scores,
            all_losses=all_losses,
        )

    @torch.inference_mode(mode=True)
    def scoring_sequences(self, input_ids: torch.LongTensor):
        input_ids = shift_padding_to_right_2D(input_ids, pad_value=self.pad_token_id)
        outputs = self(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
        )
        inds = find_rightmost_notpadded_positions(
            input_ids, pad_value=self.pad_token_id
        )

        return outputs.v_scores[:, :, -1].gather(1, inds.view(-1, 1)).squeeze(-1)

    @torch.inference_mode(mode=True)
    def scoring_sequences_by_step(
        self, input_ids: torch.LongTensor, agg_mode: str = "mean"
    ):
        """
        Find the end position of each step in each input sequence (position of '\n'),
        retrieve the v_scores at these positions, and aggregate them to compute the final score for each sequence based on the agg_mode.

        Parameters:
            input_ids (torch.LongTensor): The input token ID sequence, with shape (batch_size, seq_len).
            agg_mode (str): Aggregation mode, currently supports 'min', 'max', 'mean', 'last', default is 'mean'.

        Returns:
            torch.FloatTensor: The aggregated score for each sequence, with shape (batch_size,).
        """

        newline_token_id = MODEL_NEWLINE_TOKEN

        input_ids = shift_padding_to_right_2D(input_ids, pad_value=self.pad_token_id)
        outputs = self(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
        )

        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        v_scores = outputs.v_scores.squeeze(-1)

        # (batch_size, seq_len)
        newline_mask = input_ids == newline_token_id

        if agg_mode == "min":
            # Set the v_scores of non-target positions to positive infinity to select the minimum value
            masked_scores = torch.where(
                newline_mask, v_scores, torch.full_like(v_scores, float("inf"))
            )
            agg_scores, _ = masked_scores.min(dim=1)
        elif agg_mode == "max":
            # Set the v_scores of non-target positions to negative infinity to select the maximum value
            masked_scores = torch.where(
                newline_mask, v_scores, torch.full_like(v_scores, float("-inf"))
            )
            agg_scores, _ = masked_scores.max(dim=1)
        elif agg_mode == "mean":
            # Set the v_scores of non-target positions to 0
            masked_scores = v_scores * newline_mask.float()
            sum_scores = masked_scores.sum(dim=1)
            # Calculate the valid count for each sequence
            count = newline_mask.sum(dim=1).clamp(min=1)
            agg_scores = sum_scores / count
        elif agg_mode == "last":
            # Find the position of the last newline_token_id in each sequence
            last_indices = (
                newline_mask.long()
                .cumsum(dim=1)
                .eq(newline_mask.sum(dim=1, keepdim=True))
                .long()
                .argmax(dim=1)
            )
            agg_scores = v_scores[torch.arange(v_scores.size(0)), last_indices]
        elif agg_mode == "full":
            # Return the v_score for the full sequence length, with scores only at the newline_token positions
            agg_scores = torch.where(
                newline_mask, v_scores, torch.full_like(v_scores, float("nan"))
            )
        else:
            raise ValueError(
                f"Unsupported agg_mode: {agg_mode}. Supported modes are 'min', 'max', 'mean', 'last'."
            )

        return agg_scores  # (batch_size,)

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()


def mse_loss_with_mask(scores: torch.FloatTensor, labels: torch.FloatTensor):
    scores = torch.where(labels.ne(IGNORE_INDEX), scores, 0)
    labels = torch.where(labels.ne(IGNORE_INDEX), labels, 0)

    return F.mse_loss(scores, target=labels, reduction="sum") / scores.shape[0]


# save backbone and value_head，-> verifier.pth
@wrapper_safe_save_model_with_accelerator
def save_verifier(
    accelerator: Accelerator,
    model: transformers.AutoModelForCausalLM,
    cpu_state_dict: Mapping,
    output_dir: str,
):
    cpu_state_dict_backbone = {
        k.split("backbone.")[1]: v
        for k, v in cpu_state_dict.items()
        if k.startswith("backbone")
    }
    cpu_state_dict_verifier = {
        k: v for k, v in cpu_state_dict.items() if not k.startswith("backbone")
    }
    accelerator.unwrap_model(model).backbone.save_pretrained(
        output_dir,
        state_dict=cpu_state_dict_backbone,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    accelerator.save(cpu_state_dict_verifier, os.path.join(output_dir, "verifier.pth"))


@wrapper_save_checkpoint(save_func=save_verifier)
def save_verifier_checkpoint(
    accelerator: Accelerator,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    checkpoint_output_dir: str,
): ...


@wrapper_save_best_checkpoint(save_checkpoint_func=save_verifier_checkpoint)
def save_best_verifier_checkpoint(
    accelerator: Accelerator,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    output_dir: str,
    global_step: int,
    save_total_limit: int = None,
): ...


def build_verifier(model_args: dataclass, training_args: dataclass):
    # load LLM and tokenizer, add special token
    backbone, tokenizer = build_model(model_args, training_args)
    return Verifier(backbone), tokenizer


def load_verifier(model_args: dataclass):
    backbone, tokenizer = load_model(model_args)
    return Verifier(backbone, checkpoint_dir=model_args.model_name_or_path), tokenizer


def load_two_verifiers(model_args: dataclass):
    _, tokenizer = load_model(model_args)

    if model_args.outcome_verifier_model_name_or_path is not None:
        outcome_v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.outcome_verifier_model_name_or_path,
            torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
        )
        outcome_verifier = Verifier(
            outcome_v_backbone,
            checkpoint_dir=model_args.outcome_verifier_model_name_or_path,
        )
        print(
            f"+ [Verifiers] Load outcome verifier from {model_args.outcome_verifier_model_name_or_path}"
        )
    else:
        outcome_verifier = None

    if model_args.process_verifier_model_name_or_path is not None:
        process_v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.process_verifier_model_name_or_path,
            torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
        )
        process_verifier = Verifier(
            process_v_backbone,
            checkpoint_dir=model_args.process_verifier_model_name_or_path,
        )
        print(
            f"+ [Verifiers] Load process verifier from {model_args.process_verifier_model_name_or_path}"
        )
    else:
        process_verifier = None

    return outcome_verifier, process_verifier, tokenizer


def load_verifier_for_training(model_args: dataclass, training_args: dataclass):
    backbone, tokenizer = load_model_for_training(model_args, training_args)
    return Verifier(backbone, checkpoint_dir=model_args.model_name_or_path), tokenizer


def load_generator_and_verifier(model_args: dataclass):
    # padding left
    generator, tokenizer = load_model(model_args)

    v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.verifier_model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    verifier = Verifier(
        v_backbone, checkpoint_dir=model_args.verifier_model_name_or_path
    )
    return generator, verifier, tokenizer


def load_generator_and_two_verifiers(model_args: dataclass):
    # padding left
    generator, tokenizer = load_model(model_args)

    outcome_v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.outcome_verifier_model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    process_v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.process_verifier_model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    outcome_verifier = Verifier(
        outcome_v_backbone,
        checkpoint_dir=model_args.outcome_verifier_model_name_or_path,
    )
    process_verifier = Verifier(
        process_v_backbone,
        checkpoint_dir=model_args.process_verifier_model_name_or_path,
    )

    return generator, outcome_verifier, process_verifier, tokenizer
