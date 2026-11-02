"""Local debug driver: python -m wayfinder.rollout.cli --task-id ... [--root CHECKOUT]. Laptop only for debugging;
data runs go through wayfinder.rollout.modal_app."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from wayfinder.rollout.chat import SPARK_OPEN_BASETEN, CompatChat, Sampling
from wayfinder.rollout.run import checkout, run_rollout
from wayfinder.rollout.score import score_row
from wayfinder.rollout.tasks import load_split


def make_chat(seed: int) -> CompatChat:
    return CompatChat(
        SPARK_OPEN_BASETEN["base_url"],
        os.environ["BASETEN_API_KEY"],
        SPARK_OPEN_BASETEN["model"],
        auth_scheme=SPARK_OPEN_BASETEN["auth_scheme"],
        sampling=Sampling(0.85, 0.95),
        seed=seed,
        retain_reasoning=False,  # Spark teacher smoke runs: unchanged protocol
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--root")
    ap.add_argument("--cache", default="/tmp/wf_repos")
    ap.add_argument("--out")
    a = ap.parse_args()
    tasks = load_split(a.split)
    task = next(t for t in tasks if t["task_id"] == a.task_id) if a.task_id else tasks[a.index]
    root = a.root or checkout(task["repo"], task["base_commit"], a.cache)[0]
    row = await run_rollout(task, make_chat(0), root, seed=0)
    row.update(score_row(row))
    if a.out:
        Path(a.out).write_text(json.dumps(row) + "\n")
    print({k: row[k] for k in ("task_id", "turns", "wall_s", "cost_usd", "usage", "recall", "recall_reported", "all_gold_found", "error")}, file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
