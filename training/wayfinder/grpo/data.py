"""Rollout files for GRPO and the per-step batch plan. numpy only.

One rollout = one `.npz` (written by the rollout worker, read by the trainer):
    meta     0-d str  JSON: rollout_id, task_id, seed, step, version, n_chains, ... (reward parts are carried by the driver's manifest)
    ids<i>   int32 [T_i]   token ids of chain i (prompt + sampled completions + tool results ...), exactly what vLLM saw and sampled
    sp<i>    int32 [m_i,2] assistant [start,end) spans = the sampled tokens (loss tokens)
    lp<i>    float32 [T_i] behaviour-policy logprob of token j (NaN where the token was not sampled by the policy or logprob unknown)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_rollout(path: str | Path, meta: dict[str, Any], chains: list[dict[str, Any]]) -> int:
    arrs: dict[str, Any] = {}
    n_loss = 0
    for i, c in enumerate(chains):
        ids = np.asarray(c["input_ids"], np.int32)
        arrs[f"ids{i}"] = ids
        arrs[f"sp{i}"] = np.asarray(c["assistant_spans"], np.int32).reshape(-1, 2)
        lp = c.get("logprobs")
        arrs[f"lp{i}"] = np.asarray(lp, np.float32) if lp is not None else np.full(len(ids), np.nan, np.float32)
        n_loss += int(sum(b - a for a, b in c["assistant_spans"]))
    meta = {**meta, "n_chains": len(chains), "n_loss_tokens": n_loss, "n_tokens": int(sum(len(c["input_ids"]) for c in chains))}
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrs)
    return n_loss


def load_rollout(path: str | Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    chains = [{"ids": z[f"ids{i}"], "spans": z[f"sp{i}"], "lp": z[f"lp{i}"]} for i in range(meta["n_chains"])]
    return meta, chains


def label_positions(spans: np.ndarray, limit: int) -> np.ndarray:
    """Token indices of the sampled tokens (j >= 1, j < limit)."""
    out = [np.arange(max(int(a), 1), min(int(b), limit), dtype=np.int64) for a, b in np.asarray(spans).reshape(-1, 2)]
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def plan_ranks(items: list[tuple[str, int]], world: int) -> list[list[str]]:
    """Assign work items (key, cost in tokens) to `world` replicas, longest first onto the least-loaded replica (deterministic:
    ties by key). Every replica computes the same plan from the same manifest, so no broadcast is needed."""
    bins: list[list[str]] = [[] for _ in range(world)]
    load = [0] * world
    for key, cost in sorted(items, key=lambda kc: (-kc[1], kc[0])):
        r = min(range(world), key=lambda i: (load[i], i))
        bins[r].append(key)
        load[r] += cost
    return bins


def step_normaliser(items: list[dict[str, Any]]) -> float:
    """Loss-token count of the whole optimizer step (global, over all replicas). Every replica divides its chain losses by this one number and the
    gradients are SUM-all-reduced, so the update is the same for any world size (a mean-reduce would scale it by 1/world)."""
    return float(sum(it["n_loss"] for it in items)) or 1.0


def fork_parent(rid: str) -> str | None:
    """Rollout id of the prefix a forked-completion item belongs to (`<rid>-f<j>`, written by app._save_fork); None for an ordinary rollout item."""
    head, sep, tail = rid.rpartition("-f")
    return head if sep and head and tail.isdigit() else None


def plan_ranks_grouped(items: list[tuple[str, int]], world: int, n_loss: dict[str, int] | None = None) -> list[list[str]]:
    """plan_ranks for --fork-cached: every forked completion of one prefix goes to the same replica (they share one no-grad prefix forward there). A unit is a
    prefix with all its forks, or an ordinary rollout alone. `items` = (rid, tokens) as for plan_ranks. A prefix unit costs its longest sequence once
    (prefix + one completion) plus the completion tokens (`n_loss`, ~ the completion length) of its other forks; without `n_loss` the sum of the sequence lengths
    is used. Deterministic and the same on every rank."""
    units: dict[str, list[tuple[str, int]]] = {}
    for rid, c in items:
        par = fork_parent(rid)
        units.setdefault(f"g:{par}" if par is not None else rid, []).append((rid, c))
    cost: dict[str, int] = {}
    for key, mem in units.items():
        if not key.startswith("g:") or n_loss is None:
            cost[key] = sum(c for _, c in mem)
        else:
            top = max(mem, key=lambda rc: (rc[1], rc[0]))
            cost[key] = top[1] + sum(n_loss.get(r, 0) for r, _ in mem if r != top[0])
    plan = plan_ranks([(k, cost[k]) for k in units], world)
    return [[rid for k in b for rid, _ in sorted(units[k])] for b in plan]
