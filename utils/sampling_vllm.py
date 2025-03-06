"""
Batch Generation using vLLM
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from accelerate.accelerator import Accelerator
from typing import Union, List, Optional
from dataclasses import dataclass
import math
from transformers.generation.utils import ModelOutput
from utils.cached_models import PreTrainedTokenizer
from utils.constants import (
    MODEL_EQUALS_TOKENS,
    MODEL_LEFTMARK_TOKENS,
    MODEL_RIGHTMARK_TOKEN,
    MODEL_NEWLINE,
    MODEL_NEWLINE_TOKEN,
    DEFAULT_EOS_TOKEN,
)
from utils.vllm_utils import send_questions_to_vllm


@dataclass
class SamplingOutput(ModelOutput):
    sequences: torch.LongTensor = None


@dataclass
class StepSamplingOutput(ModelOutput):
    sequences: torch.LongTensor = None
    verifier_scores: Optional[torch.FloatTensor] = None


def find_leftmost_tokens_positions(
    input_ids: torch.LongTensor, tokens: Union[int, torch.LongTensor], wnby: bool = True
) -> torch.LongTensor:
    """
    Get the indices where `tokens` first appear in the `input_ids` for each sample in the batch. When there aren't `tokens`, return seq_len-1 when `within_boundary`

    e.g.
    input_ids = torch.tensor([[1, 2, 3, 3], [7, 0, 4, 0], [3, 2, 1, 2]])
    tokens = torch.tensor([3, 0])
    find_leftmost_tokens_positions(input_ids, tokens)
    >> tensor([2, 1, 0])

    tokens = torch.tensor([3, 2])
    find_leftmost_tokens_positions(input_ids, tokens, wnby=True)
    >> tensor([1, 3, 0])

    find_leftmost_tokens_positions(input_ids, tokens, wnby=False)
    >> tensor([1, 4, 0])
    """
    assert input_ids.ndim == 2
    bsz, seq_len = input_ids.shape
    if isinstance(tokens, int):
        mask = input_ids.eq(tokens)
    elif isinstance(tokens, torch.Tensor):
        mask = input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2)
    positions = torch.where(
        mask.any(1), mask.float().argmax(dim=1), seq_len - 1 if wnby else seq_len
    )
    return positions


def find_rightmost_tokens_positions(
    input_ids: torch.LongTensor, tokens: Union[int, torch.LongTensor], wnby: bool = True
) -> torch.LongTensor:
    """
    Get the index where `tokens` last appear in the `input_ids` for each sample in the batch. When there aren't `tokens`, return 0 when `within_boundary`

    e.g.
    input_ids = torch.tensor([[1, 2, 3, 3], [7, 0, 4, 0], [3, 2, 1, 2]])
    tokens = torch.tensor([3, 0])
    find_rightmost_tokens_positions(input_ids, tokens)
    >> tensor([3, 3, 0])

    tokens = torch.tensor([3, 2])
    find_rightmost_tokens_positions(input_ids, tokens, wnby=True)
    >> tensor([3, 0, 3])

    find_rightmost_tokens_positions(input_ids, tokens, wnby=False)
    >> tensor([3, -1, 3])
    """
    assert input_ids.ndim == 2
    bsz, seq_len = input_ids.shape
    if isinstance(tokens, int):
        mask = input_ids.eq(tokens)
    elif isinstance(tokens, torch.Tensor):
        mask = input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2)
    positions = torch.where(
        mask.any(1),
        (seq_len - 1) - mask.flip(dims=[1]).float().argmax(dim=1),
        0 if wnby else -1,
    )
    return positions


def find_leftmost_notpadded_positions(
    tensor: torch.Tensor, pad_value: Union[int, float], wnby: bool = True
) -> torch.Tensor:
    """Get the index of the first not-pad token in the left for each sample in the batch `tensor`. When they are all pad_value, return seq_len-1 when within_boundary"""
    assert tensor.ndim == 2
    bsz, seq_len = tensor.shape
    mask = tensor.ne(pad_value)
    positions = torch.where(
        mask.any(1), mask.float().argmax(dim=1), seq_len - 1 if wnby else seq_len
    )
    # (batch_size,)
    return positions


def find_rightmost_notpadded_positions(
    tensor: torch.Tensor, pad_value: Union[int, float], wnby: bool = True
) -> torch.Tensor:
    """For right padding. Get the index of the last not-pad token for each sample in the batch `tensor`. When they are all pad_value, return 0 when within_boundary"""
    assert tensor.ndim == 2
    bsz, seq_len = tensor.shape
    mask = tensor.ne(pad_value)

    positions = torch.where(
        mask.any(1),
        (seq_len - 1) - mask.flip(dims=[1]).float().argmax(dim=1),
        0 if wnby else -1,
    )
    return positions


def count_right_padding(
    tensor: torch.Tensor, pad_value: Union[int, float]
) -> torch.Tensor:
    """For right padding. Count pad_value in the right of `tensor`"""
    seq_len = tensor.shape[-1]
    positions = find_rightmost_notpadded_positions(
        tensor, pad_value=pad_value, wnby=False
    )
    return (seq_len - 1) - positions


def count_left_padding(
    tensor: torch.Tensor, pad_value: Union[int, float]
) -> torch.Tensor:
    """For left padding. Count pad_value in the left of `tensor`"""

    positions = find_leftmost_notpadded_positions(
        tensor, pad_value=pad_value, wnby=False
    )
    return positions


def count_not_left_padding(
    tensor: torch.Tensor, pad_value: Union[int, float]
) -> torch.Tensor:
    """For left padding. Count not pad_value of `tensor`"""
    counts = count_left_padding(tensor, pad_value=pad_value)
    return tensor.shape[-1] - counts


def count_shared_left_padding(
    tensor: torch.Tensor, pad_value: Union[int, float]
) -> torch.Tensor:
    """For left padding. Return the minimal padding length in the batch `tensor`"""
    return count_left_padding(tensor, pad_value).min()


# < right_borders or > left_borders
def get_mask_for_seq_area(
    tensor: torch.Tensor,
    left_borders: Optional[torch.LongTensor] = None,
    right_borders: Optional[torch.LongTensor] = None,
    include_left: bool = False,
    include_right: bool = False,
):
    """Return a mask with True in the specified areas"""
    assert not (left_borders is None and right_borders is None)
    bsz, seq_len = tensor.shape

    if include_left and left_borders is not None:
        left_borders = left_borders - 1
    if include_right and right_borders is not None:
        right_borders = right_borders + 1

    if left_borders is not None and right_borders is not None:
        mask = torch.logical_and(
            torch.arange(seq_len).view(1, -1).to(tensor.device)
            > left_borders.view(-1, 1),
            torch.arange(seq_len).view(1, -1).to(tensor.device)
            < right_borders.view(-1, 1),
        )
    elif left_borders is not None:
        mask = torch.arange(seq_len).view(1, -1).to(tensor.device) > left_borders.view(
            -1, 1
        )
    elif right_borders is not None:
        mask = torch.arange(seq_len).view(1, -1).to(tensor.device) < right_borders.view(
            -1, 1
        )
    return mask


def mask_by_borders_2D(
    tensor: torch.Tensor,
    left_borders: Optional[torch.LongTensor] = None,
    right_borders: Optional[torch.LongTensor] = None,
    include_left: bool = False,
    include_right: bool = False,
    value: Union[int, float] = 0,
):
    """Fill before/after borders into value"""
    mask = get_mask_for_seq_area(
        tensor=tensor,
        left_borders=left_borders,
        right_borders=right_borders,
        include_left=include_left,
        include_right=include_right,
    )
    return tensor.masked_fill(mask, value=value)


def count_tokens_after_positions(
    input_ids: torch.LongTensor,
    positions: torch.LongTensor,
    tokens: Union[int, torch.LongTensor],
    include_pos: bool = False,
) -> torch.LongTensor:
    """Count `tokens` after `positions`"""
    mask = get_mask_for_seq_area(
        input_ids, right_borders=positions, include_right=not include_pos
    )
    input_ids = input_ids.masked_fill(mask, value=-1)
    if isinstance(tokens, int):
        return input_ids.eq(tokens).sum(1)
    elif isinstance(tokens, torch.Tensor):
        return input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2).sum(1)


def get_new_generated_tokens(
    input_ids: torch.LongTensor,
    past_token_lens: torch.LongTensor,
    pad_token_id: int = 0,
):
    """Mask past tokens and only reserve the newly generated tokens"""
    n_paddings = count_left_padding(input_ids, pad_value=pad_token_id)

    return mask_by_borders_2D(
        input_ids,
        right_borders=n_paddings + past_token_lens,
        include_right=False,
        value=pad_token_id,
    )


def batched_shift_along_seq_dim_2D(
    tensor: torch.Tensor, shifts: torch.LongTensor = None
):
    """Shift a tensor based on the shifts along seq_dim"""
    bsz, seq_len = tensor.shape
    assert shifts.numel() == bsz

    arange1 = torch.arange(seq_len).view((1, seq_len)).to(tensor.device)
    arange2 = (arange1 - shifts.view((bsz, 1))) % seq_len

    return torch.gather(tensor, 1, arange2)


def shift_padding_to_left_2D(tensor: torch.Tensor, pad_value: Union[int, float] = 0):
    """Shift right padding in `tensor` to the left"""
    bsz, seq_len = tensor.shape
    shifts = count_right_padding(tensor, pad_value=pad_value)

    return batched_shift_along_seq_dim_2D(tensor, shifts=shifts)


def shift_padding_to_right_2D(tensor: torch.Tensor, pad_value: Union[int, float] = 0):
    """Shift left padding in `tensor` to the right"""
    bsz, seq_len = tensor.shape
    shifts = count_left_padding(tensor, pad_value=pad_value)

    return batched_shift_along_seq_dim_2D(tensor, shifts=-shifts)


class SamplingWithVLLM:
    def __init__(
        self,
        accelerator: Accelerator = None,
        model_name_or_path: str = None,
        outcome_verifier: nn.Module = None,
        process_verifier: nn.Module = None,
        birm_verifier: nn.Module = None,
        tokenizer: PreTrainedTokenizer = None,
        generation_args: dataclass = None,
        beta: float = None,
    ):
        self.accelerator = accelerator
        self.model_name_or_path = model_name_or_path
        self.outcome_verifier = outcome_verifier  #
        self.process_verifier = process_verifier  #
        self.birm_verifier = birm_verifier  #
        self.beta = beta
        self.tokenizer = tokenizer
        self.generation_args = generation_args
        self.device = accelerator.device
        self.rank_id = accelerator.process_index
        self.num_gpus = torch.cuda.device_count()
        self.port_base = 36100  # vLLM port
        self.port = self.port_base + self.rank_id

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.equal_token_ids = torch.LongTensor(list(MODEL_EQUALS_TOKENS)).to(
            self.device
        )
        # self.left_mark_token_ids = torch.LongTensor(list(MODEL_LEFTMARK_TOKENS)).to(self.device) # LLama2
        self.left_mark_token_ids = torch.LongTensor([MODEL_LEFTMARK_TOKENS]).to(
            self.device
        )  # LLama3
        self.right_mark_token_ids = torch.LongTensor([MODEL_RIGHTMARK_TOKEN]).to(
            self.device
        )
        self.newline_token_ids = torch.LongTensor([MODEL_NEWLINE_TOKEN]).to(self.device)
        self.inter_step_end_token_ids = self.newline_token_ids
        # tensor([ки, <eos>])
        self.step_end_token_ids = torch.concat(
            [
                self.newline_token_ids,
                torch.tensor([self.eos_token_id], device=self.device),
            ]
        )
        self.step_end_token_ids_list = [MODEL_NEWLINE, DEFAULT_EOS_TOKEN]

        self.max_new_tokens = generation_args.max_new_tokens
        self.max_length = generation_args.max_length
        self.temperature = generation_args.temperature
        self.generation_config = {
            k: v
            for k, v in generation_args.__dict__.items()
            if k not in ("max_new_tokens", "max_length", "temperature")
        }

        # For vLLM service
        self.vllm_url = f"http://localhost:{self.port}/v1"
        print(f"[rank{self.rank_id}: {self.vllm_url}")

    def _add_placeholder(self, text: str):
        if len(text) == 0 or text[0] != "#":
            return "####" + text
        return "$$" + text

    def _get_continued_input_ids(
        self, text: Union[str, List[str]], right_padding=False, return_tensors=False
    ):
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
                ids + [self.pad_token_id] * (max_length - len(ids)) for ids in input_ids
            ]
        if return_tensors:
            input_ids = torch.tensor(input_ids)
        return input_ids

    def _shift_padding_to_left(self, token_ids: torch.LongTensor):
        """Shift right padding in `token_ids` to the left, and adjust `past_key_values` and `transition_scores` correspondingly"""

        shifts = count_right_padding(token_ids, pad_value=self.pad_token_id)
        token_ids = batched_shift_along_seq_dim_2D(token_ids, shifts=shifts)

        return token_ids

    def _truncate_left_padding(self, token_ids: torch.LongTensor):
        # cut redundant left padding
        n_truncate = count_shared_left_padding(token_ids, pad_value=self.pad_token_id)
        token_ids = token_ids[:, n_truncate:]

        return token_ids

    def _convert_into_tensors(self, qns: Union[str, List[str], torch.LongTensor]):
        if isinstance(qns, list) and isinstance(qns[0], str):
            token_ids = self.tokenizer(qns, padding=True, return_tensors="pt").input_ids
        elif isinstance(qns, str):
            token_ids = self.tokenizer([qns], return_tensors="pt").input_ids
        elif isinstance(qns, torch.Tensor):
            token_ids = qns
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
        else:
            raise ValueError

        return token_ids.to(self.device)

    # mask and left padding
    def _cut_after_eos_lp(
        self, input_ids: torch.LongTensor, past_token_lens: torch.LongTensor = None
    ):
        """Mask the tokens after eos and keep it left padding"""

        valid_borders_right = find_leftmost_tokens_positions(
            input_ids, self.eos_token_id, wnby=True
        )

        new_input_ids = mask_by_borders_2D(
            input_ids,
            left_borders=valid_borders_right,
            include_left=False,
            value=self.pad_token_id,
        )

        new_input_ids = self._shift_padding_to_left(new_input_ids)
        return new_input_ids

    def _cut_latter_steps(
        self, input_ids: torch.LongTensor, past_token_lens: torch.LongTensor = None
    ):
        """Mask the latter steps and keep it left padding"""

        new_tokens = get_new_generated_tokens(
            input_ids, past_token_lens=past_token_lens, pad_token_id=self.pad_token_id
        )
        # ки
        cur_step_borders_right = find_leftmost_tokens_positions(
            new_tokens, self.step_end_token_ids, wnby=True
        )

        new_input_ids = mask_by_borders_2D(
            input_ids,
            left_borders=cur_step_borders_right,
            include_left=False,
            value=self.pad_token_id,
        )

        new_input_ids = self._shift_padding_to_left(new_input_ids)

        return new_input_ids

    def _mask_former_steps(
        self, input_ids: torch.LongTensor, past_token_lens: torch.LongTensor = None
    ):
        """Mask the former steps"""
        n_paddings = count_left_padding(tensor=input_ids, pad_value=self.pad_token_id)
        cur_step_borders_left = n_paddings + past_token_lens

        input_ids = mask_by_borders_2D(
            input_ids,
            right_borders=cur_step_borders_left,
            include_right=False,
            value=self.pad_token_id,
        )
        return input_ids

    # sequences: [n_beam * n_sampling_steps_per_beam, seq_len]
    # verifier_scores: [n_beam * n_sampling_steps_per_beam,]
    def _highlight_unique_sequences(
        self,
        sequences: torch.LongTensor,
        verifier_scores: torch.FloatTensor,
        dedup_mode: int = 0,
    ) -> torch.FloatTensor:
        """
        Prioritize unique sequences: linguistics-level (mode=1)
        """
        if dedup_mode == 0:
            return verifier_scores

        seq_len = sequences.shape[-1]

        seqs = shift_padding_to_left_2D(sequences, pad_value=self.pad_token_id)
        multipliers = torch.pow(
            torch.full((seq_len,), 31, dtype=seqs.dtype, device=self.device),
            torch.arange(seq_len, device=self.device),
        )
        hashes = (seqs * multipliers).sum(dim=1)

        unique_hashes = torch.unique(hashes)
        hightlighted_indices = (
            (unique_hashes[:, None] == hashes[None, :]).float().argmax(dim=1)
        )

        highlighted_vscores = verifier_scores.clone()
        highlighted_vscores[hightlighted_indices] += 100
        return highlighted_vscores

    # [n_beam * n_sampling_steps_per_beam, seq_len]
    def _concat_group_tensors(
        self,
        tensor_list: List[torch.Tensor],
        left_padding=True,
        pad_value: int = 0,
        dim: int = 0,
    ):
        max_len = max(tensor.shape[-1] for tensor in tensor_list)
        if left_padding:
            tensor_list = [
                F.pad(tensor, (max_len - tensor.shape[-1], 0), value=pad_value)
                for tensor in tensor_list
            ]
        else:
            tensor_list = [
                F.pad(tensor, (0, max_len - tensor.shape[-1]), value=pad_value)
                for tensor in tensor_list
            ]

        tensors = torch.concat(tensor_list, dim=dim)
        return tensors

    def _concat_group_steps(self, instances: List[StepSamplingOutput], dim: int = 0):
        sequences, verifier_scores = tuple(
            [instance.get(key) for instance in instances]
            for key in ("sequences", "verifier_scores")
        )

        # sequences: List[(batch_size, seq_len)]
        # -> [n_beam * n_sampling_steps_per_beam, seq_len]
        sequences = self._concat_group_tensors(
            sequences, pad_value=self.pad_token_id, dim=dim
        )
        verifier_scores = (
            torch.cat(verifier_scores, dim=dim)
            if verifier_scores[0] is not None
            else None
        )

        return StepSamplingOutput(
            sequences=sequences,
            verifier_scores=verifier_scores,
        )

    def verifier_scoring(
        self, sequences: torch.LongTensor, batch_size: int = 1, agg_mode: str = None
    ):
        # n_beam * n_sampling_steps_per_beam
        nseq = sequences.shape[0]
        # vs_batch_size
        n_split = math.ceil(nseq / batch_size)

        outputs = []
        for i in range(n_split):
            batch = sequences[i * batch_size : min((i + 1) * batch_size, nseq)]
            batch = batch.to(self.device)

            assert agg_mode in ["min", "max", "last", "mean", "full"]

            if self.birm_verifier is not None:
                vscores, _, _ = self.accelerator.unwrap_model(
                    self.birm_verifier
                ).scoring_sequences_by_step(
                    batch, agg_mode=agg_mode, beta=self.beta
                )  # (batch_size,)

            elif (
                self.outcome_verifier is not None and self.process_verifier is not None
            ):
                outcome_vscores = self.accelerator.unwrap_model(
                    self.outcome_verifier
                ).scoring_sequences_by_step(batch, agg_mode="last")
                process_vscores = self.accelerator.unwrap_model(
                    self.process_verifier
                ).scoring_sequences_by_step(batch, agg_mode=agg_mode)
                # print(f'outcome_vscores: {outcome_vscores}')
                # print(f'process_vscores: {process_vscores}')

                vscores = outcome_vscores + self.beta * process_vscores
                # print(f'outcome_score: {outcome_vscores}, process_score: {process_vscores}, vscores: {vscores}')

            elif self.outcome_verifier is not None:
                vscores = self.accelerator.unwrap_model(
                    self.outcome_verifier
                ).scoring_sequences_by_step(batch, agg_mode=agg_mode)

            elif self.process_verifier is not None:
                vscores = self.accelerator.unwrap_model(
                    self.process_verifier
                ).scoring_sequences_by_step(batch, agg_mode=agg_mode)

            else:
                raise ValueError(
                    "Either outcome, process or birm verifier must be provided"
                )

            # vscores: [cur_batch_size,]
            outputs.append(vscores)
        return torch.cat(outputs, dim=0)  # [total_n_sampling_steps,]

    def vanilla_sample(
        self,
        qns: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """
        Batch sampling with vllm (string-level)

        Return:
            responses (`Union[str, List[str]]`)
        """

        input_ids = self._convert_into_tensors(qns)

        cur_length = input_ids.shape[-1]
        if self.max_new_tokens > 0:
            max_length = cur_length + self.max_new_tokens
        else:
            max_length = self.max_length

        outputs = self._sample_tokens_with_vllm(
            input_ids=input_ids,
            max_length=max_length,
            n_response=1,
        )

        # decode
        completions = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        responses = [
            completion[len(qn) :].strip() for qn, completion in zip(qns, completions)
        ]

        if isinstance(qns, str):
            return responses[0]

        return responses

    # verify generation by step
    def sample_by_steps(
        self,
        qn_str: str = None,
        batch_size: int = 1,
        vs_batch_size: int = 1,
        n_beam: int = 1,
        n_sampling_steps: int = 2,
        max_n_step: int = 10,
        max_step_length: int = 100,
        inference_mode: str = "beam",
        dedup_mode: int = 0,
        agg_mode: str = None,
    ) -> str:
        """
        Sampling with step-level techniques

        Only support one string by now

        Parameters:
            qn_str (`str`)
            batch_size (`int`):
                used for sampling at each time
            vs_batch_size (`int`):
                batch size of verifier scoring
            n_beam (`int`)
            n_sampling_steps (`int`):
                number of total sampling sequences as next step candidates
            max_n_step (`int`):
                maximum number of steps
            max_step_length (`int`):
                maximum length for a single step
            inference_mode (`str`):
                'verifier', 'majority', or 'beam'
            dedup_mode (`int`):
                0/1
        """
        assert inference_mode in ("beam")
        input_ids = self._convert_into_tensors(qn_str)  # [1, question_len]

        # question_len + max_new_tokens
        if self.max_new_tokens > 0:
            max_length = input_ids.shape[-1] + self.max_new_tokens
        else:
            max_length = self.max_length

        if inference_mode == "beam":
            sequence, all_sequences, all_choices, all_vscores = self._steps_beam_search(
                input_ids=input_ids,
                batch_size=batch_size,
                vs_batch_size=vs_batch_size,
                n_beam=n_beam,
                n_sampling_steps=n_sampling_steps,
                max_n_step=max_n_step,
                max_step_length=max_step_length,
                max_length=max_length,
                dedup_mode=dedup_mode,
                agg_mode=agg_mode,
            )
            all_scores = all_vscores
        else:
            raise ValueError(f"Invalid inference mode: {inference_mode}")

        # rollouts
        completion = self.tokenizer.batch_decode(
            sequences=sequence, skip_special_tokens=True
        )[0]
        response = completion[len(qn_str) :].strip()

        # save
        if inference_mode == "beam":
            intermediates = [
                {
                    # n_sampling_steps
                    "sequences": [
                        {
                            "sample_id": i,
                            "str": self.tokenizer.decode(seq, skip_special_tokens=True)[
                                len(qn_str) :
                            ],
                            "vscore": score.item(),
                        }
                        for i, (seq, score) in enumerate(zip(sequences, scores))
                    ],
                    "choices": choices.tolist(),
                }
                for sequences, choices, scores in zip(
                    all_sequences, all_choices, all_scores
                )
            ]
        else:
            raise ValueError(f"Invalid inference mode: {inference_mode}")

        return response, intermediates

    # bsz=1
    def _steps_beam_search(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1,
        vs_batch_size: int = 1,
        n_beam: int = 2,
        n_sampling_steps: int = 2,
        max_n_step: int = 10,
        max_step_length: int = 100,
        max_length: int = 2048,
        dedup_mode: int = 0,
        agg_mode: str = None,
    ):
        # candidates = n_sampling_steps / n_beam
        assert n_sampling_steps % n_beam == 0
        n_sampling_steps_per_beam = n_sampling_steps // n_beam

        input_ids = input_ids.repeat_interleave(n_beam, dim=0)  # [n_beam, question_len]

        cur_length = input_ids.shape[-1]

        all_sequences = []
        all_vscores = []
        all_choices = []
        cur_step = 0

        # termination condition
        while cur_length < max_length and cur_step < max_n_step:
            cur_step_max_length = cur_length + max_step_length

            batch_candidates = self._group_step_level_sample(
                input_ids=input_ids,
                batch_size=batch_size,
                vs_batch_size=vs_batch_size,
                num_sampling_sequences=n_sampling_steps_per_beam,  # 20 / 4 = 5
                max_length=min(cur_step_max_length, max_length),
                agg_mode=agg_mode,
            )
            batch_sequences = (
                batch_candidates.sequences
            )  # [n_beam * n_sampling_steps_per_beam, seq_len]
            batch_vscores = (
                batch_candidates.verifier_scores
            )  # [n_beam * n_sampling_steps_per_beam,]

            # select the best steps/sequences
            # todo fix
            hvscores = self._highlight_unique_sequences(
                batch_sequences, batch_vscores, dedup_mode=dedup_mode
            )

            # choose top K
            # n_beam
            _, indices = torch.topk(hvscores, k=n_beam, dim=0, largest=True)

            sequences = batch_sequences.index_select(0, indices)  # [n_beam, seq_len]
            vscores = batch_vscores.index_select(0, indices)

            all_sequences.append(batch_sequences)
            all_vscores.append(batch_vscores)
            all_choices.append(indices)

            # termination condition
            if sequences.eq(self.eos_token_id).any(1).all():
                break

            # cut redundant left padding
            input_ids = self._truncate_left_padding(sequences)

            cur_length = input_ids.shape[-1]
            cur_step += 1

        # final selection
        _, best_index = torch.topk(vscores, k=1, dim=0, largest=True)

        all_sequences.append(sequences)
        all_vscores.append(vscores)
        all_choices.append(best_index)
        # top1 as final response
        sequence = sequences.index_select(0, best_index)

        return sequence, all_sequences, all_choices, all_vscores

    def _group_step_level_sample(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1,
        vs_batch_size: int = 1,
        num_sampling_sequences: int = 1,
        max_length: int = 2048,
        agg_mode: str = None,
    ) -> StepSamplingOutput:
        if (
            self.pad_token_id is not None
            and len(input_ids.shape) == 2
            and torch.sum(input_ids[:, -1] == self.pad_token_id) > 0
        ):
            print(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

        cur_token_lens = count_not_left_padding(input_ids, pad_value=self.pad_token_id)
        cur_token_lens = cur_token_lens.repeat_interleave(num_sampling_sequences, dim=0)

        outputs = self._sample_tokens_with_vllm(
            input_ids=input_ids,
            max_length=max_length,
            stop_tokens=self.step_end_token_ids_list,
            n_response=num_sampling_sequences,
            inference_mode="beam",
        )
        sequences = outputs.sequences
        # For the convenience of subsequent operations/processing
        # [n_beam * n_sampling_steps_per_beam, seq_len]
        sequences = self._cut_latter_steps(sequences, past_token_lens=cur_token_lens)
        outputs = StepSamplingOutput(sequences=sequences)

        outputs.verifier_scores = self.verifier_scoring(
            outputs.sequences, batch_size=vs_batch_size, agg_mode=agg_mode
        )

        return outputs

    # vLLM
    def _sample_tokens_with_vllm(
        self,
        input_ids: torch.LongTensor = None,
        max_length: int = None,
        stop_tokens: List[str] = None,
        n_response: Optional[int] = None,
        inference_mode: Optional[str] = None,
    ) -> SamplingOutput:
        """
        Batch sampling with vllm - model generation (token-level)
        """
        input_token_lens = count_not_left_padding(
            input_ids, pad_value=self.pad_token_id
        )
        cur_length = input_ids.shape[-1]

        max_new_tokens = max_length - cur_length
        # print(f"max_new_tokens: {max_new_tokens}, cur_length: {cur_length}")
        if max_new_tokens <= 0:
            max_new_tokens = 1
        # print(f"max_new_tokens: {max_new_tokens}")

        # vllm fix
        batch_questions = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        # print("======batch_questions========")
        # print(batch_questions)

        answers = send_questions_to_vllm(
            batch_questions,
            self.temperature,
            max_new_tokens,
            stop_tokens,
            n_response,
            self.vllm_url,
            self.model_name_or_path,
        )
        completions = answers.choices

        generated_text = []
        batch_questions_n = [
            question for question in batch_questions for _ in range(n_response)
        ]

        for question, completion in zip(batch_questions_n, completions):
            if completion.stop_reason == MODEL_NEWLINE:
                generated_text.append(question + completion.text + MODEL_NEWLINE)
            elif completion.finish_reason in ("stop", "length"):
                generated_text.append(question + completion.text)
            else:
                print(
                    f"Not implemented: finish_reason={completion.finish_reason}, total_info={completion}"
                )
                print(f"generated_text: {question + completion.text}")
                raise ValueError

        # print("======generated_text========")
        # for text in generated_text:
        #     print(text)

        if all("The answer is" in text for text in generated_text):
            generated_token_ids = self.tokenizer(generated_text, padding=True).input_ids
            generated_token_ids = torch.tensor(
                [
                    token_ids + [self.tokenizer.eos_token_id]
                    for token_ids in generated_token_ids
                ],
                device=self.device,
            )

            return SamplingOutput(sequences=generated_token_ids)

        generated_token_ids = self.tokenizer(generated_text, padding=True).input_ids
        generated_token_ids = torch.tensor(generated_token_ids, device=self.device)

        cur_length = generated_token_ids.shape[-1]

        # For the convenience of subsequent operations/processing
        generated_token_ids = self._cut_after_eos_lp(
            generated_token_ids, past_token_lens=input_token_lens
        )

        return SamplingOutput(sequences=generated_token_ids)
