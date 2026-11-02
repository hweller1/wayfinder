"""Rubric scorer (PRD section 7) over one trajectory record. Stdlib only; no network, no repo checkout needed.

Trajectory record (the row written by wayfinder.rollout.run.run_rollout, aligned to scout's ExploreResult):

    {"gold_files": [...],                       # required
     "gold_hunks": {path: [[start, end], ...]}, # optional, gold diff hunks in base-commit line numbers
     "repo_files": [...] | None,                # optional, tracked paths at base; enables existence checks
     "semantic": bool,                          # optional, was semantic_search offered (default: any call to it seen)
     "max_turns": int,                          # optional budget (default 40)
     "messages": [OpenAI-style messages],       # assistant.tool_calls[{id, function{name, arguments(JSON str)}}], tool{tool_call_id, content}
     "explore_result": {results: [Hit], summary, gaps, usage{turns, tool_calls, model_tokens{in,out}}, budget_exhausted} | None,
     "usage": {"in": int, "out": int} | None,   # optional
     "error": str | None}

`score_rollout(row)` returns {"score": 0..1, "terms": {leaf: value or None}, "weights": ..., "detail": ...}. Every leaf is in
[0, 1]; a leaf that cannot be computed (for example hunk overlap without gold hunks) is None and drops out of the weighted
mean (weights renormalised over the available leaves). Weights come from data/rubric_weights.json (fit on E2E round 3).
Chunking robustness is eval-only and not part of the score: see `chunking_robustness`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "rubric_weights.json"
FOCUS_FILES = 60  # files shown across the rollout at which focus reaches 0 (same constant as the fit)
TOKEN_BUDGET = 8000  # completion tokens at which token_eff reaches 0 (same constant as the fit)
WHY_MAX = 300
_GREP_LINE = re.compile(r"^(?P<path>[^\n:]+?):\d+: ")
_IDENT = re.compile(r"^[A-Za-z_][\w.:/\\\-$#<>*()\[\]|^+?{}]*$")  # one token, no spaces: symbol, path, literal
DEFAULT_WEIGHTS = {
    "all_gold_found": 0.2771, "recall": 0.0792, "focus": 0.2937, "hunk_overlap": 0.05, "first_gold": 0.03, "turn_eff": 0.03,
    "token_eff": 0.02, "no_rereads": 0.02, "stops_in_budget": 0.02, "no_repeated_queries": 0.015, "single_report": 0.015,
    "routing": 0.05, "output_valid": 0.05, "evidence": 0.05,
}


def load_weights(path: Path = WEIGHTS_PATH) -> dict[str, float]:
    try:
        w = json.loads(path.read_text())["weights"]
        return {k: float(v) for k, v in w.items()}
    except (OSError, KeyError, ValueError):
        return dict(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------------------------------------------
# trajectory parsing
# ---------------------------------------------------------------------------------------------------------------
def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        a = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return a if isinstance(a, dict) else {}


def parse_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One step per tool call in order: {turn, name, args, result, error}. `turn` counts assistant messages with calls."""
    results = {m.get("tool_call_id"): str(m.get("content") or "") for m in messages if m.get("role") == "tool"}
    steps: list[dict[str, Any]] = []
    turn = 0
    for m in messages:
        calls = m.get("tool_calls") or []
        if m.get("role") != "assistant" or not calls:
            continue
        turn += 1
        for c in calls:
            res = results.get(c.get("id"), "")
            steps.append({
                "turn": turn, "name": (c.get("function") or {}).get("name", ""), "args": _args(c), "result": res,
                "error": res.startswith("error:"),
            })
    return steps


def _paths_in_result(step: dict[str, Any]) -> list[str]:
    name, text = step["name"], step["result"]
    if step["error"]:
        return []
    if name == "read_file":
        return [text.split(" lines ", 1)[0].strip()]
    if name == "grep":
        out: list[str] = []
        for line in text.splitlines():
            m = _GREP_LINE.match(line)
            if m and m["path"] not in out:
                out.append(m["path"])
        return out
    if name == "glob":
        return [ln for ln in text.splitlines() if ln and not ln.startswith(("no files", "..."))]
    if name == "semantic_search":
        return list(dict.fromkeys(re.findall(r"^([^\s:]+):\d+-\d+ \(score", text, flags=re.M)))
    return []


def _norm_query(step: dict[str, Any]) -> str:
    a = step["args"]
    q = a.get("query") if step["name"] == "semantic_search" else a.get("pattern") if step["name"] == "grep" else None
    return " ".join(str(q).lower().split()) if q is not None else ""


def classify_query(q: str) -> str:
    """'identifier' (one token: symbol, path, literal, error string without spaces) or 'conceptual' (3+ words)."""
    q = q.strip()
    if not q:
        return "other"
    if " " not in q and _IDENT.match(q):
        return "identifier"
    return "conceptual" if len(q.split()) >= 3 else "other"


# ---------------------------------------------------------------------------------------------------------------
# term computation
# ---------------------------------------------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hunk_overlap(reported: list[dict[str, Any]], hunks: dict[str, list[list[int]]]) -> float | None:
    total = sum(max(0, e - s + 1) for hs in hunks.values() for s, e in hs)
    if not total:
        return None
    covered = 0
    for path, hs in hunks.items():
        spans = sorted((h["start_line"], h["end_line"]) for h in reported if h.get("path") == path)
        for s, e in hs:
            lines = set(range(s, e + 1))
            for a, b in spans:
                lines -= set(range(a, b + 1))
            covered += (e - s + 1) - len(lines)
    return covered / total


def score_terms(row: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, Any]]:
    gold = list(dict.fromkeys(row.get("gold_files") or []))
    steps = parse_steps(row.get("messages") or [])
    er = row.get("explore_result") or None
    hits = list((er or {}).get("results") or [])
    reported = list(dict.fromkeys(h.get("path") for h in hits))
    max_turns = int(row.get("max_turns") or 40)
    repo_files = set(row["repo_files"]) if row.get("repo_files") else None

    seen_order: list[tuple[int, str]] = []  # (turn, path) for first sighting
    opened: list[str] = []
    for st in steps:
        for p in _paths_in_result(st):
            seen_order.append((st["turn"], p))
        if st["name"] == "read_file" and not st["error"]:
            opened.append(_paths_in_result(st)[0])
    seen = list(dict.fromkeys(p for _, p in seen_order))
    gold_set = set(gold)
    terms: dict[str, float | None] = {}
    detail: dict[str, Any] = {"n_steps": len(steps), "n_reported": len(reported), "n_seen": len(seen), "n_opened": len(set(opened))}

    # --- gold coverage ---
    if gold:
        terms["recall"] = len(gold_set & set(reported)) / len(gold_set)  # recall@k over the reported list (k <= 12)
        seen_gold = len(gold_set & set(seen)) / len(gold_set)
        terms["all_gold_found"] = 1.0 if seen_gold >= 1.0 else 0.0  # benchmark convention: all gold shown during the rollout
        detail["recall_seen"] = seen_gold
        terms["hunk_overlap"] = _hunk_overlap(hits, row.get("gold_hunks") or {})
        first = next((t for t, p in seen_order if p in gold_set), None)
        terms["first_gold"] = None if first is None else _clip01(1 - (first - 1) / max_turns)
        if first is None:
            terms["first_gold"] = 0.0
        detail["first_gold_turn"] = first
    else:
        terms.update(recall=None, all_gold_found=None, hunk_overlap=None, first_gold=None)
    terms["focus"] = _clip01(1 - len(seen) / FOCUS_FILES)

    # --- efficiency ---
    turns = int((er or {}).get("usage", {}).get("turns") or (steps[-1]["turn"] if steps else 0))
    terms["turn_eff"] = _clip01(1 - turns / max_turns)
    out_tok = ((er or {}).get("usage", {}).get("model_tokens") or {}).get("out")
    if out_tok is None:
        out_tok = (row.get("usage") or {}).get("out")
    terms["token_eff"] = None if out_tok is None else _clip01(1 - out_tok / TOKEN_BUDGET)
    reads = [(st["args"].get("path"), st["args"].get("start_line", 1), st["args"].get("end_line")) for st in steps
             if st["name"] == "read_file" and not st["error"]]
    terms["no_rereads"] = 1.0 if not reads else len(set(reads)) / len(reads)

    # --- convergence ---
    reports = [st for st in steps if st["name"] == "report"]
    budget_exhausted = bool((er or {}).get("budget_exhausted")) or row.get("error") is not None or er is None
    terms["stops_in_budget"] = 0.0 if budget_exhausted else 1.0
    qs = [(st["name"], _norm_query(st)) for st in steps if st["name"] in ("grep", "semantic_search") and _norm_query(st)]
    terms["no_repeated_queries"] = 1.0 if not qs else len(set(qs)) / len(qs)
    terms["single_report"] = 1.0 if len(reports) == 1 else 0.0
    detail["n_report_calls"] = len(reports)

    # --- routing ---
    sem_on = row.get("semantic")
    if sem_on is None:
        sem_on = any(st["name"] == "semantic_search" for st in steps)
    routed, scored_q = 0, 0
    for st in steps:
        if st["name"] not in ("grep", "semantic_search"):
            continue
        cls = classify_query(_norm_query(st))
        if cls == "other":
            continue
        if cls == "conceptual" and not sem_on:
            continue  # nothing better was available: do not penalise
        scored_q += 1
        routed += int((cls == "conceptual") == (st["name"] == "semantic_search"))
    bad_path_calls = 0
    for st in steps:
        if st["name"] in ("read_file", "grep") and st["args"].get("path"):
            missing = st["error"] and ("NOT_FOUND" in st["result"] or "not found" in st["result"].lower())
            if missing or (repo_files is not None and st["name"] == "read_file" and st["args"]["path"] not in repo_files):
                bad_path_calls += 1
    path_calls = sum(1 for st in steps if st["name"] in ("read_file", "grep") and st["args"].get("path"))
    route_frac = routed / scored_q if scored_q else 1.0
    terms["routing"] = route_frac * (1 - bad_path_calls / path_calls) if path_calls else route_frac
    detail["routing"] = {"queries_scored": scored_q, "routed_ok": routed, "nonexistent_path_calls": bad_path_calls}

    # --- output quality ---
    if hits:
        ok = 0
        for h in hits:
            p, s, e = h.get("path"), h.get("start_line"), h.get("end_line")
            valid = isinstance(s, int) and isinstance(e, int) and s >= 1 and e >= s and bool(p)
            valid = valid and (p in repo_files if repo_files is not None else p in set(seen))
            valid = valid and len(h.get("why") or "") <= WHY_MAX and bool((h.get("why") or "").strip())
            ok += int(valid)
        validity = ok / len(hits)
    else:
        validity = 0.0
    # gaps stated: a named gap beats a silent guess. Full credit when gaps are listed or nothing is missing from the report.
    complete = terms["recall"] is None or terms["recall"] >= 1.0
    gaps_ok = 1.0 if ((er or {}).get("gaps") or complete) else 0.0
    terms["output_valid"] = 0.8 * validity + 0.2 * gaps_ok if hits else 0.0

    # --- evidence ---
    if reported:
        opened_set, seen_set = set(opened), set(seen)
        never_seen = [p for p in reported if p not in seen_set]
        frac_opened = sum(p in opened_set for p in reported) / len(reported)
        terms["evidence"] = _clip01(frac_opened - 0.5 * len(never_seen) / len(reported))  # citing an unseen file = memorisation signal
        detail["cited_never_seen"] = never_seen
    else:
        terms["evidence"] = 0.0
    return terms, detail


def score_rollout(row: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    w = weights or load_weights()
    terms, detail = score_terms(row)
    used = {k: v for k, v in terms.items() if v is not None and k in w}
    z = sum(w[k] for k in used) or 1.0
    score = sum(w[k] * v for k, v in used.items()) / z
    return {"score": round(score, 6), "terms": {k: (None if v is None else round(v, 6)) for k, v in terms.items()},
            "weights_used": {k: round(w[k] / z, 6) for k in used}, "detail": detail}


score = score_rollout  # wayfinder.rollout.score looks for score_rollout, then score


def chunking_robustness(recall_declaration_index: float, recall_treesitter_index: float) -> float:
    """Eval only (PRD section 7): worst-case over indexes relative to best, 1.0 = same recall under both chunkings."""
    hi = max(recall_declaration_index, recall_treesitter_index)
    return 1.0 if hi <= 0 else min(recall_declaration_index, recall_treesitter_index) / hi
