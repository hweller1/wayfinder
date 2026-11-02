"""GRPO token loss: clipped importance ratio against the behaviour policy plus a KL penalty to the reference. Pure torch.

Per sampled assistant token t of a rollout with advantage A:
    rho   = exp(logp_new - logp_old)           logp_old = what vLLM reported when it sampled the token (the behaviour policy,
                                               the adapter of the previous step when rollouts overlap training); NaN -> rho = 1
    pg    = -min(rho * A, clip(rho, 1 - eps_lo, 1 + eps_hi) * A)
    kl    = exp(ref - new) - (ref - new) - 1   (k3 estimator of KL(new || ref), >= 0, unbiased, low variance)
    loss  = sum_t (pg + beta * kl) * scale_t   scale_t = 1 / (number of loss tokens in the optimizer step, all replicas)
The ratio uses the TRUNCATED sampling distribution (top-k 64, top-p 0.95, renormalised; vLLM `processed_logprobs`), the KL the full tempered softmax.
The reference is the same network with the LoRA disabled (no second model). Log-probs are of softmax(logits / T) with T the
sampling temperature, i.e. of the distribution the rollouts were sampled from (top-p truncation is ignored).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class GRPOOut(NamedTuple):
    loss: torch.Tensor  # [n] per-token loss (pg + beta*kl), unscaled
    pg: torch.Tensor
    kl: torch.Tensor
    ratio: torch.Tensor
    clipped: torch.Tensor  # [n] bool: the clip bound was the active term


def grpo_token_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, logp_kl: torch.Tensor | None = None,
                    logp_ref: torch.Tensor | None = None, clip_lo: float = 0.2, clip_hi: float = 0.2, beta: float = 0.02) -> GRPOOut:
    """logp_new / logp_old: log-prob of the sampled token under the current policy (grad) and under the behaviour policy (vLLM's
    processed logprob; NaN = unknown, treated as on-policy). adv [n]. KL term: logp_kl (current policy, same distribution family as
    logp_ref, e.g. the full tempered softmax) against logp_ref (LoRA disabled); omitted when logp_ref is None."""
    old = torch.where(torch.isnan(logp_old), logp_new.detach(), logp_old)
    log_ratio = (logp_new - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped_v = ratio.clamp(1.0 - clip_lo, 1.0 + clip_hi) * adv
    pg = -torch.minimum(unclipped, clipped_v)
    if logp_ref is None:
        kl = torch.zeros_like(pg)
    else:
        d = ((logp_ref - (logp_kl if logp_kl is not None else logp_new))).clamp(-20.0, 20.0)
        kl = d.exp() - d - 1.0
    return GRPOOut(pg + beta * kl, pg, kl, ratio.detach(), (clipped_v < unclipped).detach())


TOP_K = 64  # the served model's generation_config.json defaults (vLLM applies them to requests that do not set them)
TOP_P = 0.95


def tempered_logps(z: torch.Tensor, y: torch.Tensor, temperature: float, top_k: int = TOP_K, top_p: float = TOP_P) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [n, V] float logits, y [n] sampled ids. Returns (logp_full, logp_trunc, in_kept):
    logp_full  = log softmax(z / T)[y]                                  the policy's own distribution (used for the KL term)
    logp_trunc = log of the distribution vLLM actually sampled from: top-k, then top-p on the renormalised top-k probabilities,
                 then renormalised (vLLM `processed_logprobs`). Used for the ratio against the behaviour logprobs and for the PG score.
    in_kept    = y survived truncation (False only on a numerical tie/mismatch; logp_trunc then falls back to logp_full)."""
    zt = z / temperature
    lse = torch.logsumexp(zt, dim=-1)
    full = zt.gather(-1, y[:, None])[:, 0] - lse
    k = min(top_k, zt.shape[-1])
    tv, ti = zt.topk(k, dim=-1)  # sorted descending
    p = torch.softmax(tv, dim=-1)
    keep = (p.cumsum(-1) - p) < top_p  # keep a token while the mass before it is below top_p (always keeps the first)
    kept_lse = torch.logsumexp(tv.masked_fill(~keep, float("-inf")), dim=-1)
    hit = (ti == y[:, None]) & keep
    in_kept = hit.any(-1)
    trunc = torch.where(in_kept, zt.gather(-1, y[:, None])[:, 0] - kept_lse, full)
    return full, trunc, in_kept


def softmax_entropy(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """[n] entropy (nats) of softmax(z / T), no grad. One extra [n, V] transient per chunk."""
    with torch.no_grad():
        outs = []
        for i in range(0, z.shape[0], 256):  # sub-chunks keep the extra [n, V] transients small next to the trainer's peak memory
            zt = z[i : i + 256].detach() / temperature
            outs.append(torch.logsumexp(zt, dim=-1) - (torch.softmax(zt, dim=-1) * zt).sum(-1))
        return torch.cat(outs) if outs else z.new_zeros(0)


def token_logps(head: Any, hidden: torch.Tensor, label_pos: torch.Tensor, labels: torch.Tensor, temperature: float,
                chunk: int = 2048, use_ckpt: bool = False, top_k: int = TOP_K, top_p: float = TOP_P, return_entropy: bool = False) -> Any:
    """(logp_full, logp_trunc, in_kept) at the predicting positions (label_pos - 1), in chunks so [n, V] never materialises.
    `head(h)` returns float logits (the model's exact softcapped head); hidden is [T, H]."""
    from torch.utils.checkpoint import checkpoint

    def one(h: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = head(h)
        full, trunc, kept = tempered_logps(z, y, temperature, top_k, top_p)
        if return_entropy:
            return full, trunc, kept.float(), softmax_entropy(z, temperature)
        return full, trunc, kept.float()

    src = label_pos - 1
    fs, ts, ks, es = [], [], [], []
    for s in range(0, src.numel(), chunk):
        h = hidden[src[s : s + chunk]]
        y = labels[s : s + chunk]
        res = checkpoint(one, h, y, use_reentrant=False) if use_ckpt and hidden.requires_grad else one(h, y)
        f, t, k = res[:3]
        if return_entropy:
            es.append(res[3])
        fs.append(f)
        ts.append(t)
        ks.append(k)
    if not fs:
        z = hidden.new_zeros(0)
        return (z, z, z.bool(), z) if return_entropy else (z, z, z.bool())
    if return_entropy:
        return torch.cat(fs), torch.cat(ts), torch.cat(ks) > 0.5, torch.cat(es)
    return torch.cat(fs), torch.cat(ts), torch.cat(ks) > 0.5
