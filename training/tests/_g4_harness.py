"""Test harness for the GRPO Driver with fake queues (no Modal, no network): one real Driver.run over fabricated rank5b rollout rows."""

from __future__ import annotations

import json
import queue as pyqueue
import time
import types
from pathlib import Path
from typing import Any

from wayfinder.grpo import driver as drv
from wayfinder.grpo.reward import compute_reward_rank5b

REAL_SLEEP = time.sleep
VOLATILE = ("t_elapsed_s", "est_usd", "train_usd")


def pool_tasks(n: int = 40) -> list[dict[str, Any]]:
    return [{"task_id": f"t{i:03d}", "repo": "o/r", "base_commit": "c", "question": "q", "gold_files": ["g1", "g2"], "category": "c"} for i in range(n)]


def make_row(task_id: str, seed: int, step: int, version: int, rep: list[str], turns: int = 10, worker: Any = 0) -> dict[str, Any]:
    row = {"gold_files": ["g1", "g2"], "turns": turns, "error": None, "messages": [],
           "explore_result": {"results": [{"path": p, "start_line": 1, "end_line": 2, "why": "w"} for p in rep], "budget_exhausted": False}}
    p = compute_reward_rank5b(row)
    rid = f"{task_id}-{step}-{seed}"
    p.update({"task_id": task_id, "rid": rid, "seed": seed, "step": step, "version": version, "worker": worker, "path": f"/gdata/x/rollouts/{rid}.npz", "n_tokens": 1000 + seed,
              "n_loss": 100 + seed, "rubric": None, "report_chars": 10, "recall_touched": 0.5, "tokens_in": 5, "wall_s": 1.0, "elapsed_s": 2.0, "budget_exhausted": False})
    return p


REPORTS = [["g1", "g2"], ["g1", "x"], ["x"], ["g1", "g2", "y"], ["g2"], ["g1", "g2"], ["x", "y"], ["g1"]]


def group_rows(task_id: str, step: int, version: int, group: int, worker: Any = 0, seed_base: int = 0) -> list[dict[str, Any]]:
    return [make_row(task_id, seed_base + s, step, version, REPORTS[s % len(REPORTS)], turns=8 + s, worker=worker) for s in range(group)]


class FakeCtl:
    def __init__(self) -> None:
        self.d: dict[str, Any] = {}
        self.puts: list[tuple[str, Any]] = []

    def put(self, k: str, v: Any) -> None:
        self.d[k] = v
        if k != "heartbeat":
            self.puts.append((k, v))

    def get(self, k: str, default: Any = None) -> Any:
        if k.startswith("done_") and k not in self.d:  # the "trainer" answers as soon as the batch is published
            if f"batch_{k[5:]}" in self.d:
                return {"step": int(k[5:]), "kl": 0.0, "clip_frac": 0.0, "gnorm": 1.0, "train_s": 1.0}
        return self.d.get(k, default)


class FakeJobs:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def put(self, job: dict[str, Any]) -> None:
        self.puts.append(job)


class FakeResults:
    """Serves a fixed list of rows once, then Empty."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def get_many(self, n: int, block: bool = True, timeout: float = 5) -> list[dict[str, Any]]:
        if self.rows:
            out, self.rows = self.rows[:n], self.rows[n:]
            return out
        REAL_SLEEP(0.01)
        raise pyqueue.Empty


def run_driver(tmp: Path, monkeypatch: Any, *, g4: Any = None, extra_args: dict[str, Any] | None = None, tasks_per_step: int = 2, group: int = 4) -> dict[str, Any]:
    monkeypatch.setattr(drv.time, "sleep", lambda s: REAL_SLEEP(0.005))
    a = types.SimpleNamespace(run="t", steps=1, tasks_per_step=tasks_per_step, group=group, workers=4, train_gpus=8, precision_coef=0.0, start="/x", eval_every=0, eval_shards=1,
                              pool="pool.json", no_pool_switch=True, budget_usd=1e9, seed=0, seed_salt=0, straggle_s=1, step_timeout_s=60, start_step=0, papo_lambda=0.0, monitor=False,
                              nudge="teacher", eval_budget_usd=1e9, fixed_tasks="", merge_steps=())
    for k, v in (extra_args or {}).items():
        setattr(a, k, v)
    pool = pool_tasks()
    from wayfinder.grpo.pool import sample_batch

    modal_tasks = sample_batch(pool, 0, tasks_per_step, a.seed)
    rows = [r for t in modal_tasks for r in group_rows(t["task_id"], 0, 0, group)]
    out = tmp / "run"
    jobs, ctl = FakeJobs(), FakeCtl()
    d = drv.Driver(a, ctl, jobs, FakeResults(rows), out)
    d.pool, d.pool_src = pool, "test-pool"
    if g4 is not None:
        d.g4 = g4
    d.run(lambda *x: {})
    d.finish()
    files = {p.name: p.read_text() for p in sorted(out.iterdir()) if p.is_file()}
    line = json.loads(files["steps.jsonl"].splitlines()[0])
    for k in VOLATILE:
        line.pop(k, None)
    return {"line": line, "jobs": jobs.puts, "ctl": [(k, v) for k, v in ctl.puts], "files": sorted(files), "tasks_step0": files["tasks_step0.json"], "state": json.loads(files["state.json"]) | {"est_usd": 0, "train_usd": 0}}
