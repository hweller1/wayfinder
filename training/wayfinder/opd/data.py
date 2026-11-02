"""Scored-chain files for OPD and the CPU-side batch builder (no GPU, no tokenizer).

One scored chain = one `.npz`:

    input_ids   int32 [T]        student token ids (Glimmer ids; BOS first)
    spans       int32 [m,2]      assistant [start,end) token spans
    pos         int32 [n]        token index j of every scored assistant position (j >= 1)
    topk_ids    int32 [n,K]      teacher top-K ids at the position of token j (-1 pad)
    topk_lp     float32 [n,K]    their teacher logprobs
    meta        0-d str          JSON: id, task_id, seed, domain, repo, shift, teacher_tokens, ...

`pos[i]` is the index of the token the teacher distribution `topk_*[i]` is about; the student predicts that token from its
hidden state at `pos[i]-1` (the same convention as wayfinder/sft/train.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

K = 20


def spans_to_positions(spans: list[list[int]] | np.ndarray) -> np.ndarray:
    out: list[np.ndarray] = []
    for a, b in np.asarray(spans).reshape(-1, 2):
        out.append(np.arange(max(int(a), 1), int(b), dtype=np.int32))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


def parse_prompt_logprobs(pl: list[dict[str, Any] | None], positions: np.ndarray, shift: int, k: int = K) -> tuple[np.ndarray, np.ndarray, int]:
    """vLLM `prompt_logprobs` (list over teacher prompt positions of {token_id_str: {logprob, rank, decoded_token}} or None) ->
    (ids [n,k] int32, lp [n,k] float32, n_missing) for the student positions `positions` (teacher index = j + shift).
    Each position keeps its top-k entries by logprob; the entry for the actual prompt token is dropped when it is outside
    the top-k. Missing / None entries give an all -1 row."""
    n = len(positions)
    ids = np.full((n, k), -1, np.int32)
    lp = np.zeros((n, k), np.float32)
    missing = 0
    for r, j in enumerate(positions):
        ti = int(j) + shift
        d = pl[ti] if 0 <= ti < len(pl) else None
        if not d:
            missing += 1
            continue
        items = sorted(((int(t), float(v["logprob"] if isinstance(v, dict) else v)) for t, v in d.items()), key=lambda x: -x[1])[:k]
        for c, (t, v) in enumerate(items):
            ids[r, c] = t
            lp[r, c] = v
    return ids, lp, missing


def save_chain(path: str | Path, input_ids: list[int], spans: list[list[int]], pos: np.ndarray, topk_ids: np.ndarray, topk_lp: np.ndarray, meta: dict[str, Any]) -> None:
    np.savez(path, input_ids=np.asarray(input_ids, np.int32), spans=np.asarray(spans, np.int32).reshape(-1, 2), pos=pos.astype(np.int32),
             topk_ids=topk_ids.astype(np.int32), topk_lp=topk_lp.astype(np.float32), meta=np.asarray(json.dumps(meta)))


def load_chain(path: str | Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    return {"input_ids": z["input_ids"], "spans": z["spans"], "pos": z["pos"], "topk_ids": z["topk_ids"], "topk_lp": z["topk_lp"],
            "meta": json.loads(str(z["meta"]))}


def load_manifest(scored_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(scored_dir) / "manifest.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def kept_counts(chain: dict[str, Any], allowed) -> int:
    """Number of positions of this chain that contribute to the loss (teacher data only; no student forward needed)."""
    import torch

    from wayfinder.opd.loss import keep_mask

    ids = torch.from_numpy(chain["topk_ids"]).long()
    lp = torch.from_numpy(chain["topk_lp"])
    tgt = torch.from_numpy(chain["input_ids"][chain["pos"]]).long()
    return int(keep_mask(ids, lp, allowed, tgt).sum())


def plan_steps(n_chains: int, per_step: int, steps: int, seed: int = 0) -> list[list[int]]:
    """Deterministic chain order for `steps` global steps of `per_step` chains (cycles through a fresh shuffle each epoch)."""
    import random

    out: list[list[int]] = []
    order: list[int] = []
    rng = random.Random(seed)
    while len(out) < steps:
        if len(order) < per_step:
            fresh = list(range(n_chains))
            rng.shuffle(fresh)
            order += fresh
        out.append(order[:per_step])
        order = order[per_step:]
    return out
