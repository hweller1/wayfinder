"""Support-restricted reverse KL (SVA) for on-policy distillation. Pure torch, CPU-testable.

For every scored assistant position t the teacher supplies its top-k (token id, logprob) at that position (vLLM
`prompt_logprobs=20`). With S_t the set of those ids that both models share (see vocab.py):

    p_S(i) = exp(lp_i) / sum_{j in S} exp(lp_j)        teacher, renormalised on S
    q_S(i) = exp(z_i)  / sum_{j in S} exp(z_j)         student, renormalised on S (z = the student's logits)
    loss_t = sum_{i in S} q_S(i) * (log q_S(i) - log p_S(i))              reverse KL(q_S || p_S)

The student logits are only read at S_t, so the gradient is zero everywhere else; the student's mass on S_t under its
full softmax is returned as `coverage` (a diagnostic: low coverage means the student puts its mass where the teacher
has no opinion we can see, and the restricted loss cannot push it back).

Masks (a masked position contributes nothing and is not counted): a position whose teacher entries are missing, whose
target token is an added token the two models do not share, or whose teacher mass on non-shared ids exceeds
`max_dropped_mass`. Non-shared ids inside an otherwise kept position are removed from S_t before renormalising.
Domain normalisation: the batch loss is the mean over domains of each domain's token-mean loss (`domain_token_weights`).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

NEG = -1e30


class SVAOut(NamedTuple):
    loss: torch.Tensor  # [n] per-position reverse KL on S (0 where masked)
    keep: torch.Tensor  # [n] bool: position contributes
    coverage: torch.Tensor  # [n] student mass on S under the full softmax (0 where masked)
    fwd_kl: torch.Tensor  # [n] KL(p_S || q_S) diagnostic
    teacher_mass: torch.Tensor  # [n] teacher mass on the (shared) top-k set before renormalising
    agree: torch.Tensor  # [n] bool: student argmax over S == teacher argmax over S


def entry_validity(topk_ids: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """[n,k] bool: id is present (>= 0) and shared by both vocabularies."""
    present = topk_ids >= 0
    safe = topk_ids.clamp(min=0)
    return present & allowed[safe]


def keep_mask(
    topk_ids: torch.Tensor,
    topk_lp: torch.Tensor,
    allowed: torch.Tensor,
    target_ids: torch.Tensor | None = None,
    max_dropped_mass: float = 0.05,
    min_entries: int = 2,
) -> torch.Tensor:
    """[n] bool: which positions contribute. Depends only on the teacher data, so it can be computed on the CPU at data-load
    time (the trainer needs per-sequence kept-token counts to build the domain-normalised weights before any forward)."""
    valid = entry_validity(topk_ids, allowed)
    p_all = torch.where(topk_ids >= 0, topk_lp.float().exp(), torch.zeros_like(topk_lp, dtype=torch.float32))
    dropped = (p_all * (~valid)).sum(-1)
    kept_mass = (p_all * valid).sum(-1)
    keep = (valid.sum(-1) >= min_entries) & (dropped <= max_dropped_mass) & (kept_mass > 0)
    if target_ids is not None:
        keep = keep & (target_ids >= 0) & allowed[target_ids.clamp(min=0)]
    return keep


def sva_from_logits(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_lp: torch.Tensor,
    allowed: torch.Tensor,
    target_ids: torch.Tensor | None = None,
    max_dropped_mass: float = 0.05,
    min_entries: int = 2,
) -> SVAOut:
    """logits [n,V] float (student, already scaled/softcapped, temperature 1); topk_ids [n,k] long (-1 = absent);
    topk_lp [n,k] float teacher logprobs; allowed [V] bool; target_ids [n] long, the token actually at the position."""
    logits = logits.float()
    topk_lp = topk_lp.float()
    valid = entry_validity(topk_ids, allowed)
    p_all = torch.where(topk_ids >= 0, topk_lp.exp(), torch.zeros_like(topk_lp))
    kept_mass = (p_all * valid).sum(-1)
    keep = keep_mask(topk_ids, topk_lp, allowed, target_ids, max_dropped_mass, min_entries)
    ids = topk_ids.clamp(min=0)
    z = logits.gather(1, ids)
    z = torch.where(valid, z, torch.full_like(z, NEG))
    lt = torch.where(valid, topk_lp, torch.full_like(topk_lp, NEG))
    log_q = z - torch.logsumexp(z, dim=-1, keepdim=True)
    log_p = lt - torch.logsumexp(lt, dim=-1, keepdim=True)
    q = log_q.exp()
    p = log_p.exp()
    zero = torch.zeros_like(q)
    # where() on both branches so masked entries cannot inject NaN/inf into the gradient
    rev = torch.where(valid, q * (log_q - log_p), zero).sum(-1)
    fwd = torch.where(valid, p * (log_p - log_q), zero).sum(-1)
    full_lse = torch.logsumexp(logits, dim=-1)
    cov = (torch.logsumexp(z, dim=-1) - full_lse).exp()
    agree = z.argmax(-1) == lt.argmax(-1)
    f = keep.float()
    return SVAOut(rev * f, keep, cov.detach() * f, fwd.detach() * f, kept_mass.detach() * f, agree & keep)


def domain_token_weights(domains: list[str], n_tokens: list[int]) -> list[float]:
    """Per-token weight for each sequence so that sum_t w_t * loss_t = mean over present domains of the domain's token mean.
    `domains[i]` / `n_tokens[i]` describe sequence i of the global step; n_tokens counts the tokens that will carry loss."""
    tot: dict[str, int] = {}
    for d, n in zip(domains, n_tokens):
        tot[d] = tot.get(d, 0) + n
    present = [d for d, n in tot.items() if n > 0]
    nd = max(1, len(present))
    return [1.0 / (nd * tot[d]) if tot[d] > 0 else 0.0 for d in domains]
