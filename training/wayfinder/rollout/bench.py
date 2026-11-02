"""Benchmark comparison rows (read-only from ../benchmark-repo): A1-35B b2/g and Sonnet 5.5 high b2, seed 1.

Scoring replicates ../benchmark-repo/scripts/score_v4.py (the definitions voyage-scout/eval/build.py uses) without
its inspect_ai dependency: recall = |retrieved_docids & gold_docids| / |gold_docids| (file level, deduped, union of
files the agent was shown), harness errors dropped unless model-authored (MODEL_FAULT_KEYS), template-leaky tasks
dropped. all_gold = recall == 1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parents[3] / "benchmark-repo"
ARMS = {
    "a1_35b_b2": "artifacts/remote/modal_pub/v3_pub_b2_sophisticated_agents-a1-35b_merged.jsonl",
    "a1_35b_g": "artifacts/remote/modal_pub/v3_pub_g_sophisticated_agents-a1-35b_merged.jsonl",
    "sonnet55high_b2": "artifacts/remote/sonnet_pubnew/v3_pubnew_b2_sophisticated_sonnet-5.5-high_checkpoint.jsonl",
}
MODEL_FAULT_KEYS = (
    "regex parse error", "PathEscapeError", "ToolArgumentError", "FileNotFoundError", "ripgrep failed",
    "not a file under the corpus root",
)
_SLOT = re.compile(r"(?:Where in this codebase is|What behaviour does) the (.+?) functionality")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def leaky(row: dict) -> bool:
    q = (row.get("record") or {}).get("question") or row.get("task_text") or ""
    m = _SLOT.search(q)
    if not m:
        return False
    slot = _norm(m.group(1))
    return bool(slot) and any(slot in _norm(g) for g in (row.get("gold_files") or []))


def scored_recall(row: dict) -> float | None:
    rec = row.get("record") or {}
    if not rec:
        return None
    if rec.get("status") == "harness_error" and not any(k in (rec.get("harness_reason") or "") for k in MODEL_FAULT_KEYS):
        return None
    gold = list(dict.fromkeys(rec.get("gold_docids") or []))
    if not gold:
        return None
    got = set(rec.get("retrieved_docids") or [])
    return sum(1 for g in gold if g in got) / len(gold)


def load_bench(only: set[str] | None = None) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for arm, rel in ARMS.items():
        with (BENCH / rel).open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = str(row["sample_id"]).split(":", 1)[-1]
                if only is not None and tid not in only:
                    continue
                rec = None if leaky(row) else scored_recall(row)
                r = row.get("record") or {}
                out.setdefault(tid, {})[arm] = {
                    "recall": rec,
                    "all_gold": None if rec is None else rec >= 1.0,
                    "turns": int(r["turns"]) if r.get("turns") not in (None, "") else None,
                }
    return out
