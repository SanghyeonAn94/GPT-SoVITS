"""Utility functions for the text-to-semantic model: masks, sampling and DPO loss.

modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/utils.py
reference: https://github.com/lifeiteng/vall-e
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def make_pad_mask_left(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).repeat(n, 1)
    expaned_lengths -= (max_len - lengths).unsqueeze(-1)

    return expaned_lengths < 0


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


from typing import Optional


def multinomial_sample_one_no_sync(
    probs_sort,
):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(logits=logits, previous_tokens=previous_tokens, **sampling_kwargs)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


def get_batch_logps(
    logits_target: torch.FloatTensor,
    logits_reject: torch.FloatTensor,
    labels_target: torch.LongTensor,
    labels_reject: torch.LongTensor,
    average_log_prob: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    per_token_logps_target = torch.gather(
        logits_target.log_softmax(-1), dim=2, index=labels_target.unsqueeze(2)
    ).squeeze(2)
    per_token_logps_reject = torch.gather(
        logits_reject.log_softmax(-1), dim=2, index=labels_reject.unsqueeze(2)
    ).squeeze(2)

    return per_token_logps_target.sum(-1), per_token_logps_reject.sum(-1)


def make_reject_y(y_o, y_lens):
    def repeat_P(y):
        range_idx, _ = torch.randint(0, len(y), size=(2,)).sort()
        pre = y[: range_idx[0]]
        shf = y[range_idx[1] :]
        range_text = y[range_idx[0] : range_idx[1]]
        new_y = torch.cat([pre, range_text, range_text, shf])
        return new_y

    def lost_P(y):
        range_idx, _ = torch.randint(0, len(y), size=(2,)).sort()
        pre = y[: range_idx[0]]
        shf = y[range_idx[1] :]
        range_text = y[range_idx[0] : range_idx[1]]
        new_y = torch.cat([pre, shf])
        return new_y

    bs = len(y_lens)
    reject_y = []
    reject_y_lens = []
    for b in range(bs):
        process_item_idx = torch.randint(0, 1, size=(1,))[0]
        if process_item_idx == 0:
            new_y = repeat_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
        elif process_item_idx == 1:
            new_y = lost_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
    max_length = max(reject_y_lens)
    for b in range(bs):
        pad_length = max_length - reject_y_lens[b]
        reject_y[b] = torch.cat([reject_y[b], torch.zeros(pad_length, dtype=y_o.dtype, device=y_o.device)], dim=0)

    reject_y = torch.stack(reject_y, dim=0)
    reject_y_lens = torch.tensor(reject_y_lens, device=y_lens.device)

    return reject_y, reject_y_lens
