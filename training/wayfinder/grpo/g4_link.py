"""Optional GKE G4 rollout worker for GRPO (docs/grpo_g4_worker.md): the transport and merge logic, modal-free (queues are injected, `from_modal` imports modal lazily).

Three Modal objects per run carry everything between the Modal driver and the G4 pod (the pod polls them with a Modal token; the cluster exposes nothing):
    wayfinder-grpo-g4q-<run>    Queue  jobs      (driver -> G4)   one job = one task with its seeds, the adapter version, the run's rollout cfg and an expiry
    wayfinder-grpo-g4r-<run>    Queue  results   (G4 -> driver)   rollout summaries in the schema of wayfinder-grpo-results-<run> (+ source="g4")
    wayfinder-grpo-g4ctl-<run>  Dict   control   hb (G4 heartbeat), cut (highest step the driver has closed: the G4 cancels work for steps <= cut)
Rollout chains do not travel through the queues: the G4 uploads each <rid>.npz to the volume wayfinder-grpo at /<run>/rollouts/ (the path the trainer reads) BEFORE it
queues the summary, the same ordering the Modal workers keep. The adapter (419M LoRA params, ~0.84 GB bf16) is pulled from the same volume (/<run>/adapters/v<j>); it is
far too large for a Dict value (100 MiB cap), so it is never pushed through the queue.

Staleness rule: the G4 rows of step j count only if they carry the step's sampling version (v_(j-1)) and arrive before the driver closes the step; a task counts only when
all its seeds are in (groups stay intact). Anything else is dropped and logged, never used off-policy.
"""

from __future__ import annotations

import time
from typing import Any

JOB_QUEUE = "wayfinder-grpo-g4q-{run}"
RESULT_QUEUE = "wayfinder-grpo-g4r-{run}"
CTL_DICT = "wayfinder-grpo-g4ctl-{run}"
SEED_STREAM_OFFSET = 1_000_003  # the G4 lane draws its tasks from a second seeded stream (same sample_batch, same epoch logic), disjoint from the Modal lane within a step
HB_MAX_AGE_S = 300.0  # a G4 heartbeat older than this means the pod is down: the driver submits no G4 tasks (nothing piles up in the queue)


def names(run: str) -> dict[str, str]:
    return {"jobs": JOB_QUEUE.format(run=run), "results": RESULT_QUEUE.format(run=run), "ctl": CTL_DICT.format(run=run)}


def sample_g4_tasks(pool: list[dict[str, Any]], step: int, n: int, seed: int, exclude_ids: set[str]) -> list[dict[str, Any]]:
    """n extra tasks for `step`, same seeded-shuffle epoch logic as the Modal lane (pool.sample_batch), never one the Modal lane already has this step.
    Deterministic in (pool, step, n, seed, exclude_ids)."""
    from wayfinder.grpo.pool import sample_batch

    n = min(n, max(len(pool) - len(exclude_ids), 0))
    if n <= 0:
        return []
    cand = sample_batch(pool, step, n + len(exclude_ids), seed + SEED_STREAM_OFFSET)
    out: list[dict[str, Any]] = []
    for t in cand:
        if t["task_id"] not in exclude_ids and all(t["task_id"] != o["task_id"] for o in out):
            out.append(t)
        if len(out) == n:
            break
    return out


def make_job(step: int, version: int, task: dict[str, Any], seeds: list[int], cfg: dict[str, Any], expires: float) -> dict[str, Any]:
    return {"step": step, "version": version, "task": task, "seeds": seeds, "cfg": cfg, "expires": expires}


def merge_g4(g4_rows: list[dict[str, Any]], submitted: dict[str, list[int]], version: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick the usable G4 rows of one step. `submitted` = {task_id: seeds}; rows = every G4 row received before the step was closed.
    A task is used only when (a) every seed has a row, (b) every row carries `version` (the step's sampling adapter). Returns (rows to append to the Modal rows, info)."""
    by_task: dict[str, dict[int, dict[str, Any]]] = {}
    n_unexpected = n_dup = 0
    for r in g4_rows:
        tid = r.get("task_id")
        if tid not in submitted or r.get("seed") not in submitted[tid]:
            n_unexpected += 1
            continue
        slot = by_task.setdefault(tid, {})
        if r["seed"] in slot:
            n_dup += 1
            continue
        slot[r["seed"]] = r
    used: list[dict[str, Any]] = []
    reasons = {"incomplete_or_late": 0, "stale_version": 0, "infra_only": 0}
    n_used_tasks = 0
    for tid, seeds in submitted.items():
        got = by_task.get(tid, {})
        if len(got) < len(seeds):
            reasons["incomplete_or_late"] += 1
            continue
        if any(r.get("version") != version for r in got.values()):
            reasons["stale_version"] += 1
            continue
        if all(r.get("infra_error") for r in got.values()):
            reasons["infra_only"] += 1
            continue
        n_used_tasks += 1
        used.extend(dict(got[s], source="g4") for s in seeds)
    ok = [r for r in used if not r.get("infra_error")]
    walls = [r["elapsed_s"] for r in g4_rows if isinstance(r.get("elapsed_s"), (int, float))]
    info = {"tasks_submitted": len(submitted), "g4_tasks_used": n_used_tasks, "g4_dropped": len(submitted) - n_used_tasks, "drop_reasons": {k: v for k, v in reasons.items() if v},
            "rollouts_received": len(g4_rows), "rollouts_used": len(ok), "rollouts_infra_error": len(used) - len(ok), "rows_unexpected": n_unexpected, "rows_duplicate": n_dup,
            "rollout_wall_s_sum": round(sum(walls), 1), "rollout_wall_s_max": round(max(walls), 1) if walls else None}
    return used, info


class G4Link:
    """Driver side of the transport. jobs/results are Queue-like (put, put_many, get_many, clear), ctl Dict-like (put, get, pop)."""

    def __init__(self, jobs: Any, results: Any, ctl: Any, clock: Any = time.time) -> None:
        self.jobs, self.results, self.ctl, self.clock = jobs, results, ctl, clock

    def reset(self) -> None:
        """Driver (re)start: stale jobs and results of an earlier attempt go; the G4 pod itself keeps running."""
        self.jobs.clear()
        self.results.clear()
        try:
            self.ctl.pop("cut")
        except KeyError:
            pass

    def alive(self, max_age_s: float = HB_MAX_AGE_S) -> tuple[bool, dict[str, Any] | None]:
        hb = self.ctl.get("hb")
        return (bool(hb) and self.clock() - float(hb.get("t", 0)) <= max_age_s), hb

    def submit(self, job: dict[str, Any]) -> None:
        self.jobs.put(job)

    def mark_cut(self, step: int) -> None:
        self.ctl.put("cut", step)


def from_modal(run: str) -> G4Link:
    import modal

    n = names(run)
    return G4Link(modal.Queue.from_name(n["jobs"], create_if_missing=True), modal.Queue.from_name(n["results"], create_if_missing=True),
                  modal.Dict.from_name(n["ctl"], create_if_missing=True))
