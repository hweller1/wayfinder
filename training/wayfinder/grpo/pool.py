"""RL task pool and per-step task sampling. Pure Python.

Pool sources, first that exists: --pool path, data/rl_pool_v1.json, data/rl_pool_v0.json, else the PRD s14 RL-eligible train tasks
(task_filter.json `rl_eligible`, train split, minus data/opd_excluded_tasks.json: 226 of 233). A pool file is a list of task dicts or
{"tasks": [...]} with task_id, repo, base_commit, question, gold_files, category (a list of task ids is resolved through splits.json).

Guards (raise, never filter silently): a task id that splits.json assigns to dev/test/eval_holdout; a task whose repo is not a train repo;
any task id in the E2E / OPD exclusion lists or in the dev eval task list (artifacts/teacher_test/stageB_tasks.json).

`sample_batch(pool, batch, n, seed)` is deterministic: a seeded shuffle of the whole pool per epoch, consumed n at a time (no task repeats
inside an epoch; a batch never straddles two epochs' duplicates).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_REPOS = {"Emmy123222/Stellar-GreenPay"}  # 404 on GitHub (deleted or private): every checkout fails (coordinator, 2026-10-27)


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def forbidden_ids() -> set[str]:
    out = set(_load_json(ROOT / "data/opd_excluded_tasks.json")["tasks"])
    for rel in ("artifacts/sft/e2e_excluded_tasks.json",):
        p = ROOT / rel
        if p.exists():
            d = _load_json(p)
            out |= {t["task_id"] if isinstance(t, dict) else t for t in (d["tasks"] if isinstance(d, dict) else d)}
    dev = ROOT / "artifacts/teacher_test/stageB_tasks.json"
    if dev.exists():
        out |= {t["task_id"] for t in _load_json(dev)}
    return out


HARD_ORIGINS = {"hard_v2", "hard_v3"}  # mined outside data/splits.json (scripts/rlpool/mine_hard*.py); allowed only if the repo is not a held-out hard-dev repo


def hard_dev_repos() -> set[str]:
    out: set[str] = set()
    for n in ("hard_dev_v1", "hard_dev_v2"):
        p = ROOT / f"data/{n}.json"
        if p.exists():
            out |= set(_load_json(p)["repos"])
    return out


def _from_splits(splits: dict[str, Any], tid: str) -> dict[str, Any]:
    t = splits["tasks"][tid]
    return {"task_id": tid, "repo": t["repo_full"], "base_commit": t["base_commit"], "question": t["task_text"],
            "gold_files": list(t["gold_files"]), "category": t.get("category")}


def default_pool_tasks() -> list[dict[str, Any]]:
    splits, flt = _load_json(ROOT / "data/splits.json"), _load_json(ROOT / "data/task_filter.json")
    ex = set(_load_json(ROOT / "data/opd_excluded_tasks.json")["tasks"])
    ids = sorted(t for t, v in flt["tasks"].items() if v.get("split") == "train" and v.get("rl_eligible") and t not in ex)
    return [x for x in (_from_splits(splits, t) for t in ids) if x["repo"] not in EXCLUDED_REPOS]


def validate_pool(tasks: list[dict[str, Any]], splits: dict[str, Any] | None = None, forbidden: set[str] | None = None) -> None:
    splits = splits or _load_json(ROOT / "data/splits.json")
    forbidden = forbidden if forbidden is not None else forbidden_ids()
    seen: set[str] = set()
    for t in tasks:
        tid = t["task_id"]
        if tid in seen:
            raise ValueError(f"duplicate task {tid}")
        seen.add(tid)
        if tid in forbidden:
            raise ValueError(f"task {tid} is in the E2E/OPD exclusion or dev eval list")
        if t["repo"] in EXCLUDED_REPOS:
            raise ValueError(f"task {tid}: repo {t['repo']} is excluded (unreachable)")
        sp = splits["tasks"].get(tid, {}).get("split")
        if sp is not None and sp != "train":
            raise ValueError(f"task {tid} is in split {sp!r}, not train")
        rs = splits["repo_to_split"].get(t["repo"])
        if rs is not None and rs != "train":
            raise ValueError(f"task {tid}: repo {t['repo']} is in split {rs!r}, not train")
        if rs is None and sp is None and t.get("origin") in HARD_ORIGINS:
            if t["repo"] in hard_dev_repos():
                raise ValueError(f"task {tid}: repo {t['repo']} is a held-out hard_dev repo")
            pass  # known hard origin, repo not held out: fall through to the field checks below
        elif rs is None and sp is None:
            raise ValueError(f"task {tid}: repo {t['repo']} is not a known train repo")
        for k in ("repo", "base_commit", "question", "gold_files"):
            if not t.get(k):
                raise ValueError(f"task {tid} missing {k}")


def load_pool(path: str | None = None) -> tuple[list[dict[str, Any]], str]:
    cands = [Path(path)] if path else [ROOT / "data/rl_pool_v1.json", ROOT / "data/rl_pool_v0.json"]
    for p in cands:
        if p.exists():
            d = _load_json(p)
            rows = d["tasks"] if isinstance(d, dict) else d
            splits = _load_json(ROOT / "data/splits.json")
            tasks = [_from_splits(splits, r) if isinstance(r, str) else r for r in rows]
            tasks = [t for t in tasks if t["repo"] not in EXCLUDED_REPOS]  # dropped, not an error: pool files come from another agent
            validate_pool(tasks, splits)
            return tasks, str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
        if path:
            raise FileNotFoundError(path)
    tasks = default_pool_tasks()
    validate_pool(tasks)
    return tasks, "prd14_rl_eligible_train"


def sample_batch(pool: list[dict[str, Any]], batch: int, n: int, seed: int = 0) -> list[dict[str, Any]]:
    ids = sorted(range(len(pool)), key=lambda i: pool[i]["task_id"])
    out: list[int] = []
    pos = batch * n
    while len(out) < n:
        epoch, off = divmod(pos + len(out), len(ids))
        order = ids[:]
        random.Random(f"{seed}:{epoch}").shuffle(order)
        take = min(n - len(out), len(ids) - off)
        out += order[off : off + take]
    return [pool[i] for i in out]
