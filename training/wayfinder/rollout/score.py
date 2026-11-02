"""Row scoring: file recall and all-gold-found, in the benchmark's convention plus scout's product view.

Benchmark convention (benchmark-repo `score_v4.scored_recall` -> `document_metrics`): recall = |seen & gold| / |gold|
over the union of files the agent was shown across the rollout, file level, deduped; all_gold_found = recall == 1.
Scout's grep tool shows up to 200 files per call whereas the benchmark showed k=5, so the union over everything
scout showed is generous. Every row therefore carries four views, all file level:

  recall_reported   files in the ExploreResult (<= 12): what a parent agent actually receives (product metric)
  recall_opened     files opened with read_file
  recall_seen_k5    benchmark-like: opened files + first 5 files of each grep/glob result (closest analog to arm b2)
  recall_seen_all   everything any tool result showed (upper bound)
"""

from __future__ import annotations

import json
import re
from typing import Any

_GREP_LINE = re.compile(r"^(?P<path>[^\n:]+?):\d+: ")
_SEM_LINE = re.compile(r"^(?P<path>[^\n]+?):\d+-\d+ \(score ")  # scout's semantic_text(): `path:start-end (score s)`


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for m in messages:
        for c in m.get("tool_calls") or ():
            try:
                args = json.loads(c["function"]["arguments"])
            except (TypeError, ValueError):
                args = {}
            out[c["id"]] = (c["function"]["name"], args if isinstance(args, dict) else {})
    return out


def files_seen(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    calls = _tool_names(messages)
    opened: list[str] = []
    k5: list[str] = []
    allseen: list[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        name, args = calls.get(m.get("tool_call_id"), ("", {}))
        text = str(m.get("content") or "")
        if name == "read_file":
            if text.startswith("error:"):
                continue
            p = text.split(" lines ", 1)[0].strip()
            opened.append(p)
            k5.append(p)
            allseen.append(p)
        elif name == "grep":
            paths: list[str] = []
            for line in text.splitlines():
                mm = _GREP_LINE.match(line)
                if mm and mm["path"] not in paths:
                    paths.append(mm["path"])
            allseen += paths
            k5 += paths[:5]
        elif name == "semantic_search":
            paths = []
            for line in text.splitlines():
                mm = _SEM_LINE.match(line)
                if mm and mm["path"] not in paths:
                    paths.append(mm["path"])
            allseen += paths
            k5 += paths[:5]
        elif name == "glob":
            paths = [ln for ln in text.splitlines() if ln and not ln.startswith(("no files", "..."))]
            allseen += paths
            k5 += paths[:5]
    dedupe = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return {"opened": dedupe(opened), "seen_k5": dedupe(k5), "seen_all": dedupe(allseen)}


def _recall(found: list[str], gold: set[str]) -> float | None:
    return len(gold & set(found)) / len(gold) if gold else None


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    gold = set(row["gold_files"])
    seen = files_seen(row["messages"])
    reported = [h["path"] for h in ((row.get("explore_result") or {}).get("results") or [])]
    reported = list(dict.fromkeys(reported))
    r = {
        "recall_reported": _recall(reported, gold),
        "recall_opened": _recall(seen["opened"], gold),
        "recall_seen_k5": _recall(seen["seen_k5"], gold),
        "recall_seen_all": _recall(seen["seen_all"], gold),
    }
    r["recall"] = r["recall_seen_k5"]  # headline for benchmark comparison
    r["all_gold_found"] = None if r["recall"] is None else r["recall"] >= 1.0
    r["all_gold_reported"] = None if r["recall_reported"] is None else r["recall_reported"] >= 1.0
    r["n_reported"] = len(reported)
    r["n_opened"] = len(seen["opened"])
    r["rubric"] = _rubric(row)
    return r


def _rubric(row: dict[str, Any]) -> dict[str, Any] | None:
    """Rubric terms from wayfinder/rubric.py (track A) when it exists; null otherwise."""
    try:
        import wayfinder.rubric as rb  # type: ignore[import-not-found]
    except ImportError:
        return None
    fn = getattr(rb, "score_rollout", None) or getattr(rb, "score", None)
    if fn is None:
        return None
    try:
        return fn(row)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
