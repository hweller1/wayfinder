"""Cached-prefix fork training (--fork-cached; docs/grpo_fork_report.md, "Cached-prefix training").

A forked completion is trained as prefix + completion. The full-sequence forward re-runs the ~40k-token prefix with grad for every trained fork. Here the
prefix is forwarded ONCE per prefix under torch.no_grad() into a KV cache, and each fork's completion tokens are forwarded WITH grad on top of that cache
(position_ids continuing from the prefix length). Gradients flow only through the completion tokens' own computation, not through the prefix K/V: the
deliberate approximation (it equals the full forward with the prefix K/V detached, tests/test_grpo_fork_cached.py).

Two HF behaviours the module works around (both pinned by tests):
  * GradientCheckpointingLayer drops `past_key_values` (silently!) while checkpointing is active in training mode unless the layer declares
    `_can_checkpoint_with_cache`, so the prefix forward runs with checkpointing off and the completion forward runs with it on only over a read-only cache;
  * a DynamicCache is appended to on every forward, so a second fork (or a checkpoint recompute) would see a longer cache. `freeze_cache` turns every layer of
    a filled cache into a read-only layer whose `update` returns cat(prefix, new) without storing, which makes forwards over it idempotent.

The split is one token early: the last prefix token (the `assistant` start token) goes through the grad forward, so the hidden state that predicts the first
completion token is a with-grad one. prefix = ids[:P], tail = ids[P:], P = first_loss_index - 1.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

_RO_CLASSES: dict[type, type] = {}


def _ro_class(cls: type) -> type:
    """Read-only variant of a cache layer class (same fields, so an existing instance can be re-classed in place)."""
    if cls not in _RO_CLASSES:
        def update(self: Any, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.is_initialized or self.keys.numel() == 0:
                return key_states, value_states
            pk, pv = self.keys, self.values
            if pk.shape[0] != key_states.shape[0]:  # K completions share one prefix: broadcast the batch-1 prefix
                pk, pv = pk.expand(key_states.shape[0], -1, -1, -1), pv.expand(key_states.shape[0], -1, -1, -1)
            return torch.cat([pk, key_states], dim=-2), torch.cat([pv, value_states], dim=-2)

        _RO_CLASSES[cls] = type("ReadOnly" + cls.__name__, (cls,), {"update": update})
    return _RO_CLASSES[cls]


def freeze_cache(cache: Any) -> Any:
    """Make a filled DynamicCache read-only in place (see module doc). Returns the same object."""
    for layer in cache.layers:
        if not getattr(layer, "_read_only", False):
            layer.__class__ = _ro_class(type(layer))
            layer._read_only = True
    return cache


def cache_nbytes(cache: Any) -> int:
    return sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size() for l in cache.layers if l.is_initialized and l.keys.numel())


def _ckpt_layers(lm: Any) -> list[Any]:
    from transformers.modeling_layers import GradientCheckpointingLayer

    return [m for m in lm.modules() if isinstance(m, GradientCheckpointingLayer)]


@contextmanager
def checkpointing_off(lm: Any) -> Iterator[None]:
    """The prefix forward: with checkpointing active HF would drop the cache."""
    layers = _ckpt_layers(lm)
    prev = [l.gradient_checkpointing for l in layers]
    for l in layers:
        l.gradient_checkpointing = False
    try:
        yield
    finally:
        for l, p in zip(layers, prev):
            l.gradient_checkpointing = p


@contextmanager
def checkpointing_over_readonly_cache(lm: Any) -> Iterator[None]:
    """The completion forward: keep checkpointing, and let the layers keep `past_key_values` (safe only because the cache is read-only, so the recompute
    on the backward pass reads exactly what the forward read)."""
    layers = _ckpt_layers(lm)
    for l in layers:
        l._can_checkpoint_with_cache = True
    try:
        yield
    finally:
        for l in layers:
            l.__dict__.pop("_can_checkpoint_with_cache", None)


def prefix_cache(lm: Any, prefix_ids: torch.Tensor) -> Any:
    """KV cache of prefix_ids [P] (one sequence, positions 0..P-1) under no_grad, frozen. `lm` = the text model (inner.model.language_model)."""
    from transformers import DynamicCache

    dev = lm.embed_tokens.weight.device
    x = prefix_ids[None].to(dev)
    cache = DynamicCache(config=lm.config)
    with torch.no_grad(), checkpointing_off(lm):
        lm(input_ids=x, position_ids=torch.arange(x.shape[1], device=dev)[None], past_key_values=cache, use_cache=True)
    return freeze_cache(cache)


def tail_hidden(lm: Any, cache: Any, tail_ids: torch.Tensor, start: int) -> torch.Tensor:
    """Hidden states of the completion tokens `tail_ids` (positions start..start+T-1) over the frozen prefix cache, with grad: [T] -> [T, H]. A right-padded
    batch [K, T] of completions sharing the prefix -> [K, T, H] (causal attention: padding after a sequence's end never reaches its real tokens)."""
    dev = lm.embed_tokens.weight.device
    x = tail_ids[None].to(dev) if tail_ids.dim() == 1 else tail_ids.to(dev)
    pos = torch.arange(start, start + x.shape[1], device=dev)[None]
    with checkpointing_over_readonly_cache(lm):
        h = lm(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=False).last_hidden_state
    return h[0] if tail_ids.dim() == 1 else h


def split_point(spans: np.ndarray) -> int:
    """P: the number of prefix tokens forwarded without grad = (index of the first loss token) - 1."""
    return int(np.asarray(spans).reshape(-1, 2)[0][0]) - 1


def shares_prefix(a: np.ndarray, b: np.ndarray, p: int) -> bool:
    return len(a) >= p and len(b) >= p and bool(np.array_equal(a[:p], b[:p]))


def train_prefix_group(lm: Any, head: Any, chains: list[dict[str, np.ndarray]], advs: list[float], n_total: float, *, ref_ctx: Any = None, autocast: Any = None,
                       backward: bool = True, device: Any = "cpu", **loss_kw: Any) -> tuple[list[dict[str, float]], list[int], int]:
    """Loss (+ backward) of the forked completions of ONE prefix: one no-grad prefix forward (plus one more with the adapter disabled when beta > 0), then each
    completion with grad over the frozen cache, freeing each fork's graph before the next. `chains` = [{ids, spans, lp}] (single loss span each), `advs` their
    advantages, `ref_ctx()` = context manager disabling the adapter (reference policy), `autocast()` = the trainer's autocast context. Returns (stat dicts, indices
    of the chains that do not share chains[0]'s prefix: not handled here, the caller runs them through the full-sequence path, and the prefix length P)."""
    from contextlib import nullcontext

    from wayfinder.grpo.step import rollout_chain_loss

    ac = autocast or nullcontext
    P = split_point(chains[0]["spans"])
    ids0 = np.asarray(chains[0]["ids"])
    ok = [i for i, c in enumerate(chains) if split_point(c["spans"]) == P and shares_prefix(np.asarray(c["ids"]), ids0, P)]
    rest = [i for i in range(len(chains)) if i not in ok]
    beta = float(loss_kw.get("beta", 0.02))
    pre = torch.from_numpy(ids0[:P].astype(np.int64))
    with ac():
        cache = prefix_cache(lm, pre)
        ref_cache = None
        if beta > 0 and ref_ctx is not None:
            with ref_ctx():
                ref_cache = prefix_cache(lm, pre)

    def fwd(ids: torch.Tensor) -> torch.Tensor:
        with ac():
            return tail_hidden(lm, cache, ids[P:], P)

    def ref_fwd(ids: torch.Tensor) -> torch.Tensor:
        with ac(), ref_ctx():
            return tail_hidden(lm, ref_cache, ids[P:], P)

    stats = []
    try:
        for i in ok:
            loss, st = rollout_chain_loss(fwd, ref_fwd if ref_cache is not None else None, head, chains[i], advs[i], n_total, device=device, h_offset=P, **loss_kw)
            if loss is not None and backward:
                loss.backward()
            stats.append(st)
            del loss
    finally:
        del cache, ref_cache
    return stats, rest, P
