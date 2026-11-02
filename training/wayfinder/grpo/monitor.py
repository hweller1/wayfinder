"""Padding monitor against v3's OWN rollouts on the same tasks (run f1, rebased 2026-10-28 after the first stop was traced to task mix).

Each step's training rollouts (16 tasks x 8) are compared with v3's pool-screen rollouts of the SAME task ids (`artifacts/rl_pool/screen/*.jsonl`: v3, semantic
auto, max_turns 40, max_seconds 1200, retain on, 8 seeds per task, covers every rl_pool_v1 task; NO f1 prompt line, so a small prompt effect is part of the delta).
The screen rows carry `reported_paths` (report order) but not precision or recall@5, so both are RECOMPUTED here from the reported files and the pool task's gold
(distinct reported files; precision = gold hits / distinct reported, undefined when nothing was reported; recall@5 over the first 5). n_reported counts 0 for a row
with no report. Task means first, then the mean over the step's tasks (the step has 8 rollouts per task), so the batch's task mix cancels.

Flag (a step looks padded): n_reported above same-task v3 by more than 0.5 AND precision below it by more than 0.03 AND recall@5 not above it. Two consecutive
flagged steps stop the run (driver.py). The earlier rule (any of the first two, against the pooled step 0-1 mean, OR instead of AND) fired on a low baseline.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

N_RISE, PREC_FALL, SUSTAIN = 0.5, 0.03, 2
METRICS = ("n_reported", "precision", "recall_at_5", "recall_reported")


def row_metrics(reported: list[str], gold: set[str]) -> dict[str, float | None]:
    rep = list(dict.fromkeys(reported))
    hit = len(gold & set(rep))
    return {"n_reported": float(len(rep)), "precision": (hit / len(rep)) if rep else None,
            "recall_at_5": (len(gold & set(rep[:5])) / len(gold)) if gold else 0.0, "recall_reported": (hit / len(gold)) if gold else 0.0}


def load_screen(screen_dir: str | Path, gold_by_task: dict[str, set[str]]) -> dict[str, list[dict[str, float | None]]]:
    """task_id -> metrics of each valid v3 screen row (no error). Tasks without a gold list are skipped."""
    out: dict[str, list[dict[str, float | None]]] = {}
    for f in sorted(glob.glob(str(Path(screen_dir) / "*.jsonl"))):
        for ln in open(f):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("error") or r.get("infra_error") or r.get("task_id") not in gold_by_task:
                continue
            out.setdefault(r["task_id"], []).append(row_metrics(r.get("reported_paths") or [], gold_by_task[r["task_id"]]))
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def per_task_mean(rows_by_task: dict[str, list[dict[str, float | None]]], tasks: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in METRICS:
        means = [x for x in (_mean([r[m] for r in rows_by_task[t] if r[m] is not None]) for t in tasks if rows_by_task.get(t)) if x is not None]
        out[m] = _mean(means)
    return out


def step_vs_screen(step_rows: list[dict[str, Any]], screen: dict[str, list[dict[str, float | None]]], gold_by_task: dict[str, set[str]]) -> dict[str, Any] | None:
    """step_rows: the step's valid rollout summaries (they carry task_id, n_reported, precision, recall_at_5, recall_reported). Returns current, same-task v3, deltas."""
    by: dict[str, list[dict[str, float | None]]] = {}
    for r in step_rows:
        if r.get("task_id") in gold_by_task and r.get("n_reported") is not None:
            by.setdefault(r["task_id"], []).append({m: r.get(m) for m in METRICS})
    tasks = sorted(t for t in by if t in screen)
    if not tasks:
        return None
    cur, ref = per_task_mean(by, tasks), per_task_mean(screen, tasks)
    if any(cur[m] is None or ref[m] is None for m in METRICS):
        return None
    d = {m: round(cur[m] - ref[m], 4) for m in METRICS}  # rounded: a delta of exactly the threshold is not past it
    padded = d["n_reported"] > N_RISE and d["precision"] < -PREC_FALL
    return {"n_tasks": len(tasks), "cur": {m: round(cur[m], 4) for m in METRICS}, "v3_same_tasks": {m: round(ref[m], 4) for m in METRICS}, "delta": d,
            "flagged_same_task": padded and d["recall_at_5"] <= 0.0}


def should_stop(flags: list[bool], sustain: int = SUSTAIN) -> bool:
    return len(flags) >= sustain and all(flags[-sustain:])
