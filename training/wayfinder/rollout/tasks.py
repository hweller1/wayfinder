"""Task selection from data/splits.json (track A's file; read-only here) and the benchmark comparison rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits.json"


def load_split(split: str = "dev", path: Path = SPLITS) -> list[dict[str, Any]]:
    d = json.loads(path.read_text())
    out = []
    for tid, t in d["tasks"].items():
        if t["split"] != split:
            continue
        out.append(
            {
                "task_id": tid,
                "repo": t["repo_full"],
                "base_commit": t["base_commit"],
                "question": t["task_text"],
                "gold_files": list(t["gold_files"]),
                "category": t.get("category"),
            }
        )
    return sorted(out, key=lambda t: t["task_id"])


def pick(tasks: list[dict[str, Any]], n: int, salt: str, exclude: set[str] = frozenset()) -> list[dict[str, Any]]:  # type: ignore[assignment]
    """Deterministic hash-ordered sample; spreads over repos (round-robin) so one repo cannot dominate."""
    pool = [t for t in tasks if t["task_id"] not in exclude]
    pool.sort(key=lambda t: hashlib.sha1(f"{salt}:{t['task_id']}".encode()).hexdigest())
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for t in pool:
        by_repo.setdefault(t["repo"], []).append(t)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(by_repo.values()):
        for repo in sorted(by_repo, key=lambda r: hashlib.sha1(f"{salt}:{r}".encode()).hexdigest()):
            if by_repo[repo] and len(out) < n:
                out.append(by_repo[repo].pop(0))
    return out
