"""The per-rollout GRPO objective, model-agnostic (callables for the policy forward, the reference forward and the head), so the
same code runs the real trainer and the CPU toy-policy tests. Pure torch."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from wayfinder.grpo.data import label_positions
from wayfinder.grpo.loss import TOP_K, TOP_P, grpo_token_loss, token_logps

STAT_KEYS = ("loss", "pg", "kl", "clipped", "abs_logratio", "logratio", "dfull", "in_kept", "tokens", "adv_tok", "entropy")


def rollout_chain_loss(fwd: Callable[[torch.Tensor], torch.Tensor], ref_fwd: Callable[[torch.Tensor], torch.Tensor] | None, head: Callable[[torch.Tensor], torch.Tensor],
                       chain: dict[str, np.ndarray], adv: float, n_total: float, *, temperature: float = 0.85, top_k: int = TOP_K, top_p: float = TOP_P,
                       clip_lo: float = 0.2, clip_hi: float = 0.2, beta: float = 0.02, device: Any = "cpu", use_ckpt: bool = True,
                       chunk: int = 2048, span_adv: dict[int, float] | None = None, h_offset: int = 0) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Loss of one chain (already divided by `n_total`, the loss-token count of the whole optimizer step across replicas) and stat sums.
    fwd(ids[T]) -> hidden [T,H] (grad); ref_fwd the same with the adapter disabled (no grad), or None when beta == 0.
    span_adv (forked-report mode only): {span index: extra advantage} added to the chain advantage on the tokens of that span (the original report's intra term).
    h_offset (--fork-cached only): fwd/ref_fwd return hidden states for ids[h_offset:] instead of all of ids (the prefix ran in a no-grad KV cache); 0 = the old behaviour."""
    ids = torch.from_numpy(np.asarray(chain["ids"], np.int64)).to(device)
    pos = torch.from_numpy(label_positions(chain["spans"], len(ids))).to(device)
    if pos.numel() == 0:
        return None, {k: 0.0 for k in STAT_KEYS}
    labels = ids[pos]
    old = torch.from_numpy(np.asarray(chain["lp"], np.float32)).to(device)[pos]
    ref_full = None
    if beta > 0 and ref_fwd is not None:
        with torch.no_grad():
            h_ref = ref_fwd(ids)
            ref_full, _, _ = token_logps(head, h_ref, pos - h_offset, labels, temperature, chunk, False, top_k, top_p)
            del h_ref
    h = fwd(ids)
    full, trunc, kept, ent = token_logps(head, h, pos - h_offset, labels, temperature, chunk, use_ckpt, top_k, top_p, return_entropy=True)
    a = torch.full_like(trunc, float(adv))
    if span_adv:
        a = a + span_adv_vector(chain["spans"], len(ids), span_adv).to(a.device, a.dtype)
    out = grpo_token_loss(trunc, old, a, full, ref_full, clip_lo, clip_hi, beta)
    loss = out.loss.sum() / n_total
    with torch.no_grad():
        lr = torch.where(torch.isnan(old), torch.zeros_like(trunc), trunc - old)
        st = {"loss": float(out.loss.sum()), "pg": float(out.pg.sum()), "kl": float(out.kl.sum()), "clipped": float(out.clipped.float().sum()),
              "abs_logratio": float(lr.abs().sum()), "logratio": float(lr.sum()), "dfull": float(torch.where(torch.isnan(old), torch.zeros_like(full), full - old).sum()),
              "in_kept": float(kept.float().sum()), "tokens": float(pos.numel()), "adv_tok": float(adv) * float(pos.numel()), "entropy": float(ent.sum())}
    return loss, st


def span_adv_vector(spans: np.ndarray, limit: int, extra: dict[int, float]) -> torch.Tensor:
    """Per-loss-token extra advantage, aligned with label_positions(spans, limit) (spans in order, each clipped to [1, limit))."""
    parts = []
    for i, (a, b) in enumerate(np.asarray(spans).reshape(-1, 2)):
        n = max(min(int(b), limit) - max(int(a), 1), 0)
        parts.append(torch.full((n,), float(extra.get(i, 0.0))))
    return torch.cat(parts) if parts else torch.zeros(0)


def merge_stat_sums(parts: list[dict[str, float]]) -> dict[str, float]:
    tot = {k: sum(p.get(k, 0.0) for p in parts) for k in STAT_KEYS}
    n = max(tot["tokens"], 1.0)
    return {"loss_per_tok": tot["loss"] / n, "pg_per_tok": tot["pg"] / n, "kl": tot["kl"] / n, "clip_frac": tot["clipped"] / n,
            "abs_logratio": tot["abs_logratio"] / n, "logratio": tot["logratio"] / n, "dfull_old": tot["dfull"] / n, "in_kept_frac": tot["in_kept"] / n,
            "loss_tokens": tot["tokens"], "entropy": tot["entropy"] / n}
