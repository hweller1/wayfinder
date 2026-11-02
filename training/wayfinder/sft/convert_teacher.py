"""Convert Spark Open TEACHER rollout rows (wayfinder/rollout/run.py) into SFT examples for train_v1.py.

Pure library (no Modal): used by wayfinder/sft/convert_teacher_modal.py inside a Modal CPU job and by the tests.
Output rows have exactly the contract of wayfinder/sft/convert.py (and train.py `load_examples`):
`id, input_ids, assistant_spans, n_tokens, tools_variant, messages, meta`.

A teacher row already lives in scout's product tool surface (the rollout ran scout's own loop), so unlike the A1
converter nothing is re-rendered. What this module does:

  system   the row's own system message (scout product prompt + the recorded convergence nudge), unchanged
  user     the row's user message ("<brief>\\n\\nReport at most 12 results."), unchanged
  tools    scout `tool_specs`: 5 tools when the row offered semantic_search (`row["semantic"]`), else the 4-tool grep-only set
  calls    JSON-string arguments -> dict (the ATEM template raises on strings, data/tokenizer_check.json)
  results  the row's tool messages, unchanged; the trailing "report accepted" tool message is dropped (example ends at
           the report call, as in the A1 examples)
  reasoning  KEPT: Glimmer's template renders `reasoning_content` as a `<|start|>assistant to=self` segment, which is the
           model's own reasoning channel, so the teacher's reasoning tokens are trained on. The short chatty `content` that
           the teacher emits next to a tool call (e.g. "I caught your request, now I'll dig...") is DROPPED: the template
           never renders `content` beside tool calls, so keeping it would only mean silently losing it at render time.
  loss     assistant tokens only (everything after `<|start|>assistant` up to and including `<|eom|>`/`<|eot|>`)

Filters (sequential, counted in the funnel): train split; converted (exactly one report call, budget not exhausted, clean
protocol: every assistant turn is a tool call, no mid-run user nudges, no model-emitted special tokens); all_gold_found
(the row's benchmark-convention flag); evidence covered (every reported path was returned by a successful read_file);
not over the task's p75 turn count (p75 over every teacher row of that task); <= 32,768 tokens; dedupe on (task_id, seed).
`max_tokens` (default 32,768 = train.py --pack-len) and `min_recall` (default None = strict all_gold_found) are the two knobs
for a looser tier; the defaults are the PRD filters.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

from wayfinder.sft.convert import MAX_TOKENS, file_recall, tokenize_example

SPECIALS = ("<|start|>", "<|eot|>", "<|eom|>", "<|message|>", "<|begin_of_text|>", "<|end_of_text|>")
FUNNEL_ORDER = [
    "rows_in", "train_split", "not_e2e_task", "converted", "all_gold_found", "evidence_covered", "turns_le_p75", "fits_32k", "dedup",
]


def p75(values: list[int]) -> int:
    """Nearest-rank 75th percentile (a task with one row keeps that row)."""
    s = sorted(values)
    return s[max(0, math.ceil(0.75 * len(s)) - 1)]


def report_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [tc for m in row["messages"] if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])
            if tc["function"]["name"] == "report"]


def protocol_problem(row: dict[str, Any]) -> str | None:
    """Return why the transcript cannot be an SFT example, or None. Does not look at gold or quality."""
    msgs = row.get("messages") or []
    if len(msgs) < 4 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        return "bad_shape"
    if row.get("error"):
        return "row_error"
    if (row.get("explore_result") or {}).get("budget_exhausted"):
        return "budget_exhausted"
    reps = report_calls(row)
    if len(reps) != 1:
        return "no_report" if not reps else "multi_report"
    for m in msgs[2:]:
        if m["role"] == "user":
            return "mid_run_user_nudge"
        if m["role"] == "assistant" and not m.get("tool_calls"):
            return "assistant_without_tool_call"
    last_asst = [m for m in msgs if m["role"] == "assistant"][-1]
    if [tc["function"]["name"] for tc in last_asst["tool_calls"]] != ["report"]:
        return "report_not_last_alone"
    for m in msgs:
        text = (m.get("content") or "") + (m.get("reasoning_content") or "") + json.dumps(m.get("tool_calls") or "")
        if any(s in text for s in SPECIALS):
            return "special_token_in_text"
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            try:
                if not isinstance(json.loads(tc["function"]["arguments"]), dict):
                    return "bad_tool_args"
            except (json.JSONDecodeError, TypeError):
                return "bad_tool_args"
    return None


def reported_paths(row: dict[str, Any]) -> list[str]:
    rep = json.loads(report_calls(row)[0]["function"]["arguments"])
    return [h["path"] for h in rep.get("results", [])]


def opened_paths(row: dict[str, Any]) -> set[str]:
    """Paths a successful read_file returned (tool message not starting with 'error:')."""
    calls: dict[str, dict[str, Any]] = {}
    out: set[str] = set()
    for m in row["messages"]:
        for tc in m.get("tool_calls") or []:
            calls[tc["id"]] = tc
        if m["role"] == "tool":
            tc = calls.get(m.get("tool_call_id", ""))
            if tc and tc["function"]["name"] == "read_file" and not (m.get("content") or "").startswith("error:"):
                out.add(json.loads(tc["function"]["arguments"])["path"])
    return out


def evidence_covered(row: dict[str, Any]) -> bool:
    paths = reported_paths(row)
    return bool(paths) and set(paths) <= opened_paths(row)


def to_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Template-ready messages: dict tool arguments, reasoning kept, content beside tool calls dropped, no trailing tool result."""
    out: list[dict[str, Any]] = []
    for m in row["messages"]:
        if m["role"] == "assistant":
            am: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}}
                for tc in m["tool_calls"]]}
            if (m.get("reasoning_content") or "").strip():
                am["reasoning_content"] = m["reasoning_content"].strip()
            out.append(am)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content") or ""})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    while out and out[-1]["role"] == "tool":
        out.pop()
    return out


def tools_for(row: dict[str, Any], tools_all: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if row.get("semantic"):
        return tools_all, "scout5"
    return [t for t in tools_all if t["function"]["name"] != "semantic_search"], "scout4_grep_only"


def build_example(row: dict[str, Any], tok: Any, tools_all: list[dict[str, Any]], task: dict[str, Any], turn_cap: int,
                  max_tokens: int = MAX_TOKENS) -> tuple[dict[str, Any] | None, str | None]:
    msgs = to_messages(row)
    tools, variant = tools_for(row, tools_all)
    tk = tokenize_example(tok, msgs, tools)
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    n_calls = sum(len(m["tool_calls"]) for m in msgs if m["role"] == "assistant")
    n_reason = sum(1 for m in msgs if m["role"] == "assistant" and m.get("reasoning_content"))
    if len(tk["assistant_spans"]) != n_calls + n_reason:  # one span per call segment plus one per reasoning segment
        return None, "span_count_mismatch"
    if tk["n_tokens"] > max_tokens:
        return None, "fits_32k"
    gold = row["gold_files"]
    paths = reported_paths(row)
    meta = {
        "source": "teacher", "teacher_model": row.get("model"), "teacher_endpoint": row.get("endpoint"),
        "task_id": row["task_id"], "seed": row["seed"], "repo": row["repo"], "split": "train",
        "category": task.get("category"), "semantic": bool(row.get("semantic")),
        "semantic_calls": sum(1 for m in msgs if m["role"] == "assistant" for tc in m["tool_calls"]
                              if tc["function"]["name"] == "semantic_search"),
        "turns": row["turns"], "turn_cap_p75": turn_cap, "n_assistant_msgs": n_asst,
        "recall": row.get("recall"), "recall_reported": row.get("recall_reported"),
        "recall_opened": row.get("recall_opened"),
        "recall_report_file_level": round(file_recall(set(paths), gold), 4),
        "all_gold_found": bool(row.get("all_gold_found")), "all_gold_reported": bool(row.get("all_gold_reported")),
        "n_gold": len(set(gold)), "n_report_hits": len(paths), "evidence_covered": True,
        "tokens": tk["n_tokens"], "assistant_tokens": sum(b - a for a, b in tk["assistant_spans"]),
        "reasoning_msgs": n_reason, "tools_variant": variant, "tier": "teacher",
        "nudge": row.get("nudge"),
    }
    ex = {"id": f"{row['task_id']}|teacher|s{row['seed']}", "input_ids": tk["input_ids"],
          "assistant_spans": tk["assistant_spans"], "n_tokens": tk["n_tokens"], "tools_variant": variant,
          "messages": msgs, "meta": meta}
    return ex, None


def convert_rows(rows: list[dict[str, Any]], splits: dict[str, Any], tok: Any, tools_all: list[dict[str, Any]],
                 max_tokens: int = MAX_TOKENS, min_recall: float | None = None,
                 exclude_tasks: frozenset[str] | set[str] = frozenset()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (examples, stats). stats.funnel is the sequential filter funnel; stats.drops the reason of every dropped row.
    `exclude_tasks`: task ids never used for training (the E2E coding-eval tasks), dropped right after the split filter."""
    funnel: Counter = Counter()
    drops: Counter = Counter()
    funnel["rows_in"] = len(rows)
    turns_by_task: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        turns_by_task[r["task_id"]].append(int(r.get("turns") or 0))
    cap = {t: p75(v) for t, v in turns_by_task.items()}

    cand: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for r in rows:
        t = splits["tasks"].get(r["task_id"])
        if t is None or t["split"] != "train":
            drops["split:" + ("unknown_task" if t is None else t["split"])] += 1
            continue
        funnel["train_split"] += 1
        if r["task_id"] in exclude_tasks:
            drops["e2e_eval_task"] += 1
            continue
        funnel["not_e2e_task"] += 1
        why = protocol_problem(r)
        if why:
            drops["not_converged:" + why] += 1
            continue
        funnel["converted"] += 1
        if not r.get("all_gold_found") and not (min_recall is not None and (r.get("recall") or 0) >= min_recall):
            drops["not_all_gold"] += 1
            continue
        funnel["all_gold_found"] += 1
        if not evidence_covered(r):
            drops["evidence_not_covered"] += 1
            continue
        funnel["evidence_covered"] += 1
        if int(r["turns"]) > cap[r["task_id"]]:
            drops["over_p75_turns"] += 1
            continue
        funnel["turns_le_p75"] += 1
        cand.append((r, t))

    built: list[dict[str, Any]] = []
    for r, t in cand:
        ex, why = build_example(r, tok, tools_all, t, cap[r["task_id"]], max_tokens)
        if ex is None:
            drops[why or "build_failed"] += 1
            continue
        built.append(ex)
    funnel["fits_32k"] = len(built)

    best: dict[tuple[str, int], dict[str, Any]] = {}
    for ex in built:  # keep the shortest run per (task, seed); ties keep the first seen
        k = (ex["meta"]["task_id"], ex["meta"]["seed"])
        if k not in best or ex["meta"]["turns"] < best[k]["meta"]["turns"]:
            best[k] = ex
    drops["duplicate_task_seed"] += len(built) - len(best)
    out = sorted(best.values(), key=lambda e: e["id"])
    funnel["dedup"] = len(out)
    stats = {
        "funnel": {k: funnel[k] for k in FUNNEL_ORDER}, "drops": dict(drops),
        "semantic_examples": sum(1 for e in out if e["meta"]["semantic"]),
        "tasks": len({e["meta"]["task_id"] for e in out}), "repos": len({e["meta"]["repo"] for e in out}),
        "tokens_total": sum(e["n_tokens"] for e in out), "assistant_tokens_total": sum(e["meta"]["assistant_tokens"] for e in out),
        "rows_semantic_in": sum(1 for r in rows if r.get("semantic")),
    }
    return out, stats


def a1_examples(example_rows: list[dict[str, Any]], tools_all: list[dict[str, Any]],
                exclude_tasks: frozenset[str] | set[str] = frozenset()) -> list[dict[str, Any]]:
    """Strict all-gold A1 examples (already in train.py format). Flags meta.source = 'a1'; nothing is re-tokenized."""
    out = []
    for e in example_rows:
        m = e.get("meta") or {}
        if m.get("tier") != "strict" or not m.get("all_gold_found") or not e.get("input_ids") or not e.get("assistant_spans"):
            continue
        if m.get("task_id") in exclude_tasks:
            continue
        e = dict(e)
        e["meta"] = {**m, "source": "a1", "semantic": e.get("tools_variant") == "scout5", "seed": None}
        out.append(e)
    return out
