"""Student rollouts for OPD: the existing scout harness (`wayfinder.rollout.run.run_rollout`, unchanged) with token capture.

Each model call is recorded as {prompt_token_ids, token_ids, finish, n_elided}: the ids vLLM tokenised for the prompt and the
ids the student sampled (`return_token_ids`). Earlier turns' reasoning is re-sent (`CompatChat.retain_reasoning`), which scout's loop
itself drops from the history, so a later prompt really extends the earlier prompt + completion and consecutive calls
fall into one scoreable chain (spans.build_chains). Rows carry the usual harness fields (recall, turns, ...) plus the chains.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from wayfinder.opd.spans import build_chains

DEFAULT_EXCLUDE = Path(__file__).resolve().parents[2] / "data" / "opd_excluded_tasks.json"


def select_tasks(splits: dict[str, Any], flt: dict[str, Any], exclude: set[str], n: int, salt: str, accepted_only: bool = True) -> list[dict[str, Any]]:
    """Train-split tasks (dev/test/eval_holdout never appear), minus the E2E eval tasks, hash-ordered and spread over repos."""
    from wayfinder.rollout.tasks import pick

    tasks = []
    for tid, t in sorted(splits["tasks"].items()):
        if t["split"] != "train" or tid in exclude:
            continue
        if accepted_only and not flt["tasks"].get(tid, {}).get("accepted"):
            continue
        tasks.append({"task_id": tid, "repo": t["repo_full"], "base_commit": t["base_commit"], "question": t["task_text"],
                      "gold_files": list(t["gold_files"]), "category": t.get("category")})
    return pick(tasks, n, salt) if n else tasks


def load_exclude(path: str | Path = DEFAULT_EXCLUDE) -> set[str]:
    return set(json.loads(Path(path).read_text())["tasks"])


def rollout_id(task_id: str, seed: int) -> str:
    return hashlib.sha1(f"{task_id}|{seed}".encode()).hexdigest()[:16]


async def opd_rollout(task: dict[str, Any], base_url: str, model: str, root: str, *, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    from wayfinder.rollout.chat import CompatChat, Prices, Sampling
    from wayfinder.rollout.run import TEACHER_NUDGE, run_rollout
    from wayfinder.rollout.score import score_row

    chat = CompatChat(base_url, "x", model, sampling=Sampling(cfg["temperature"], cfg["top_p"]), prices=Prices(*cfg.get("prices", [1.25, 4.25, 0.15])),
                      seed=seed, vllm_extensions=False, timeout=900.0, max_retries=4,
                      extra_body={"return_token_ids": True})
    calls: list[dict[str, Any]] = []
    chat.capture = calls.append
    chat.retain_reasoning = bool(cfg.get("retain_reasoning", True))
    row = await run_rollout(task, chat, root, seed=seed, max_turns=cfg["max_turns"], max_seconds=cfg["max_seconds"],
                            nudge=TEACHER_NUDGE if cfg.get("nudge", True) else None, semantic=cfg.get("semantic", "off"))
    row.update(score_row(row))
    tools = next((c["tools"] for c in calls if c.get("tools")), None)
    system = (row.get("messages") or [{}])[0].get("content", "")
    chains, diag = build_chains(calls, max_len=cfg.get("max_chain_tokens", 49152), strict=cfg.get("strict_chains", False))
    row["opd"] = {"chains": chains, "diag": diag, "system": system, "tools": tools, "n_calls": len(calls)}
    for k in ("messages", "turn_log"):  # keep the file small: the chain ids are the payload
        row.pop(k, None)
    row["domain"] = task.get("category")
    row["rollout_id"] = rollout_id(task["task_id"], seed)
    return row


def write_rollout(out_dir: str | Path, row: dict[str, Any]) -> Path:
    d = Path(out_dir) / "rollouts"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{row['rollout_id']}.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(row, f)
    return p


def read_rollout(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)
