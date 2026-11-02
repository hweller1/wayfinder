"""Score the Agents-A1-35B matched baseline (same harness, tasks, scorer as the v3/v4 retain-on arms) and pair it against
v3, v4 and base Glimmer (task-bootstrap 95% CIs, seeds averaged within task, 10,000 resamples; metric definitions from
scripts/score_sft_v2.py unchanged).

    python3 scripts/score_a1_baseline.py   -> artifacts/a1_baseline/summary.json, sft_a1_{grep,sem}.jsonl (merged shards), stdout tables

Rows: artifacts/a1_baseline/shard*/sft_a1_{grep,sem}.jsonl (wayfinder/rollout/a1_eval_app.py). References:
  v3 grep (retain on): artifacts/opd/round1/rows/eval_d__sft_v3_grep.jsonl.gz     v3 sem (retain on): artifacts/sft_v4/sft_v3r_sem.jsonl
  v4 grep/sem (retain on): artifacts/sft_v4/sft_v4_{grep,sem}.jsonl               base Glimmer (retain OFF, grep only): artifacts/baseline/glimmer_dev.jsonl
Base minus A1 is a mixed protocol (base is retain-off); it is reported with that label. PRD s15 is A1's grep arm against base.
Tool-call health is counted from the rows: tool results the scout loop returned as `error: ... not valid JSON`, `unknown tool`, or
`invalid arguments` (malformed calls), assistant turns that carried no tool call (the loop appends its 'Reply with a tool call' nudge), and
rollouts the loop stopped after 3 consecutive malformed calls.
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sft_v2 import METRICS, deltas, ok_rows, prd15, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AD = ROOT / "artifacts/a1_baseline"
V4 = ROOT / "artifacts/sft_v4"
NUDGE_TEXT = "Reply with a tool call. When you have enough evidence, call report."
DEV = ROOT / "artifacts/teacher_test/stageB_tasks.json"


def load_any(p: Path) -> list[dict]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def merge_shards(arm: str) -> list[dict]:
    """Shard files (local, untracked) are merged into sft_a1_<arm>.jsonl and rows/sft_a1_<arm>.jsonl.gz (committed). A clean clone has only the .gz."""
    rows: list[dict] = []
    for f in sorted(AD.glob(f"shard*/sft_a1_{arm}.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    gz = AD / "rows" / f"sft_a1_{arm}.jsonl.gz"
    if rows:
        (AD / f"sft_a1_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        gz.parent.mkdir(exist_ok=True)
        with gzip.open(gz, "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    elif gz.exists():
        rows = load_any(gz)
        (AD / f"sft_a1_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")  # score_grpo.py reads the plain file
    return rows


def ok_of(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("error") and not r.get("infra_error")]


def validate(rows: list[dict], model: str = "a1", n_rows: int = 100, n_tasks: int = 50) -> dict:
    ok = ok_of(rows)
    turns = sorted(r["turns"] for r in ok)
    walls = sorted(r["wall_s"] for r in ok)
    dev = {t["task_id"] for t in json.loads(DEV.read_text())}
    v = {"rows": len(rows), "ok_rows": len(ok), "errors": len(rows) - len(ok), "error_messages": sorted({str(r.get("error"))[:120] for r in rows if r.get("error")}),
         "tasks": len({r["task_id"] for r in ok}), "tasks_match_dev_file": {r["task_id"] for r in ok} == dev,
         "pairs_unique": len({(r["task_id"], r["seed"]) for r in ok}) == len(ok), "seeds": sorted({r["seed"] for r in ok}),
         "models": sorted({str(r.get("model")) for r in ok}), "retain_reasoning": sorted({str(r.get("retain_reasoning")) for r in ok}),
         "sampling": sorted({f"T{r.get('temperature')} top_p{r.get('top_p')} top_k{r.get('top_k')} pp{r.get('presence_penalty')} think{r.get('enable_thinking')}" for r in ok}),
         "max_turns": sorted({r.get("max_turns") for r in ok}), "max_seconds": sorted({r.get("max_seconds") for r in ok}),
         "weights": sorted({str(r.get("weights")) for r in ok}), "max_model_len": sorted({r.get("max_model_len") for r in ok}),
         "turns_min_p50_max": [turns[0], turns[len(turns) // 2], turns[-1]] if turns else None,
         "wall_s_min_p50_max": [walls[0], walls[len(walls) // 2], walls[-1]] if walls else None,
         "n429": sum(r.get("n429", 0) for r in ok), "n_retries": sum(r.get("n_retries", 0) for r in ok),
         "with_reasoning_rate": round(sum(1 for r in ok if any(m.get("reasoning_content") for m in r["messages"] if m["role"] == "assistant")) / max(len(ok), 1), 3)}
    v["healthy"] = (len(rows) == n_rows and len(ok) == n_rows and v["tasks"] == n_tasks and v["tasks_match_dev_file"] and v["pairs_unique"] and v["seeds"] == [0, 1]
                    and v["models"] == [model] and v["retain_reasoning"] == ["True"] and v["max_turns"] == [40] and v["max_seconds"] == [3600])
    return v


def parse_health(rows: list[dict]) -> dict:
    """Counts over ok rows. A turn is an assistant message; a call is an entry of its tool_calls."""
    out = dict(rollouts=len(rows), assistant_turns=0, tool_calls=0, no_tool_call_turns=0, bad_json_args=0, unknown_tool=0, invalid_args=0, rollouts_with_malformed=0,
               rollouts_stopped_3_malformed=0, rollouts_with_no_tool_call_turn=0, leaked_tool_markup_in_content=0, empty_turns=0)
    for r in rows:
        ms = r["messages"]
        bad = 0
        for i, m in enumerate(ms):
            if m["role"] == "assistant":
                out["assistant_turns"] += 1
                calls = m.get("tool_calls") or []
                out["tool_calls"] += len(calls)
                c = m.get("content") or ""
                if not calls:
                    out["no_tool_call_turns"] += 1
                    if not c and not m.get("reasoning_content"):
                        out["empty_turns"] += 1
                if "<tool_call>" in c or "<function=" in c:
                    out["leaked_tool_markup_in_content"] += 1
            elif m["role"] == "tool":
                t = str(m.get("content", ""))
                if t.startswith("error: the arguments of") and "not valid JSON" in t:
                    out["bad_json_args"] += 1; bad += 1
                elif t.startswith("error: unknown tool"):
                    out["unknown_tool"] += 1; bad += 1
                elif t.startswith("error: invalid arguments"):
                    out["invalid_args"] += 1; bad += 1
        out["rollouts_with_malformed"] += bad > 0
        out["rollouts_with_no_tool_call_turn"] += any(m["role"] == "user" and m.get("content") == NUDGE_TEXT for m in ms)
        gaps = (r.get("explore_result") or {}).get("gaps", [])
        out["rollouts_stopped_3_malformed"] += any("consecutive malformed" in g for g in gaps)
    out["malformed_calls_total"] = out["bad_json_args"] + out["unknown_tool"] + out["invalid_args"]
    out["malformed_rate_per_call"] = round(out["malformed_calls_total"] / max(out["tool_calls"], 1), 4)
    out["no_tool_call_rate_per_turn"] = round(out["no_tool_call_turns"] / max(out["assistant_turns"], 1), 4)
    return out


def n_sem_calls(r: dict) -> int:
    return sum(1 for m in r["messages"] if m["role"] == "assistant" for t in (m.get("tool_calls") or []) if t["function"]["name"] == "semantic_search")


def extra(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {"semantic_offered": sum(1 for r in rows if r.get("semantic")), "rollouts_calling_semantic_search": sum(1 for r in rows if n_sem_calls(r) > 0),
            "semantic_search_calls_per_rollout": round(sum(n_sem_calls(r) for r in rows) / n, 2),
            "reasoning_tokens_per_rollout": round(sum(r["usage"].get("reasoning", 0) for r in rows) / n),  # 0 where the server does not report it
            "out_tokens_per_rollout": round(sum(r["usage"]["out"] for r in rows) / n)}


def reasoning_render_evidence(rows: list[dict]) -> dict:
    """Does A1's template put earlier reasoning back into the next prompt? Compare, between consecutive turns of a rollout, the growth of the
    prompt (usage.in) with the previous turn's output tokens, using only turns whose tool results are under 120 chars. If earlier reasoning were
    NOT rendered, growth would be about (output - reasoning) + result, i.e. far below the output size on turns with long reasoning."""
    pairs = []
    for r in rows:
        ms, tl = r["messages"], [t for t in r["turn_log"] if t.get("ok")]
        if any(m["role"] == "user" and m.get("content") == NUDGE_TEXT for m in ms):
            continue  # after the loop's plain user nudge the template drops earlier reasoning (last user turn moves), so exclude
        ai = [i for i, m in enumerate(ms) if m["role"] == "assistant"]
        for k in range(min(len(ai), len(tl)) - 1):
            res = sum(len(str(m["content"])) for m in ms[ai[k] + 1: ai[k + 1]] if m["role"] == "tool")
            if res < 120:
                u0, u1 = tl[k]["usage"], tl[k + 1]["usage"]
                pairs.append((u1["in"] - u0["in"], u0["out"]))
    big = [(d, o) for d, o in pairs if o >= 300]
    import statistics as st
    return {"turn_pairs_short_results": len(pairs), "median_prompt_growth_over_output_tokens": round(st.median(d / max(o, 1) for d, o in pairs), 2) if pairs else None,
            "turn_pairs_output_ge_300_tokens": len(big),
            "median_growth_over_output_ge_300": round(st.median(d / o for d, o in big), 2) if big else None,
            "share_growth_ge_0.8x_output_ge_300": round(sum(d >= 0.8 * o for d, o in big) / len(big), 2) if big else None,
            "reading": "ratio near 1 means the previous turn's output, reasoning included, is re-rendered into the next prompt (reasoning IS retained by the template and by vLLM's reasoning_content -> reasoning mapping); ratio near 0 would mean it is dropped"}


def cell(s: dict, k: str, nd: int = 3) -> str:
    x = s[k]
    return f"{x['mean']:.{nd}f} [{x['ci95'][0]:.{nd}f}, {x['ci95'][1]:.{nd}f}]"


def dcell(d: dict, k: str) -> str:
    x = d[k]
    if k == "tokens":
        return f"{x['delta'] / 1000:+.0f}k [{x['ci95'][0] / 1000:+.0f}k, {x['ci95'][1] / 1000:+.0f}k]"
    nd = 1 if k in ("turns", "wall_s", "turn_latency_s") else 3
    return f"{x['delta']:+.{nd}f} [{x['ci95'][0]:+.{nd}f}, {x['ci95'][1]:+.{nd}f}]"


def main() -> None:
    raw = {a: merge_shards(a) for a in ("grep", "sem")}
    a1 = {a: ok_of(raw[a]) for a in raw}
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    ref = {"v3_grep": ok_of(load_any(ROOT / "artifacts/opd/round1/rows/eval_d__sft_v3_grep.jsonl.gz")), "v3_sem": ok_rows(V4 / "sft_v3r_sem.jsonl"),
           "v4_grep": ok_rows(V4 / "sft_v4_grep.jsonl"), "v4_sem": ok_rows(V4 / "sft_v4_sem.jsonl")}
    out: dict = {"protocol": {"tasks": "artifacts/teacher_test/stageB_tasks.json (50 dev tasks x seeds 0,1)", "sampling": "A1 recommended: T 0.85, top_p 0.95, top_k 20, presence_penalty 1.1, enable_thinking true (scout a1 profile = serve/a1_35b.py SAMPLING)",
                             "retain_reasoning": True, "max_turns": 40, "max_seconds": 3600, "nudge": "run.TEACHER_NUDGE", "weights": "/weights/Agents-A1-FP8 (FP8), vLLM 0.32.0, qwen3_coder tool parser, qwen3 reasoning parser, max_model_len 65536, 1x H100 per arm-container"},
                 "base": summarize(base), "validation": {}, "parse_health": {}, "extra": {}, "a1": {}, "refs": {}, "minus_a1": {}, "a1_minus_base": {}}
    out["reasoning_render_evidence"] = reasoning_render_evidence(a1["grep"] + a1["sem"])
    for a in ("grep", "sem"):
        if raw[a]:
            out["validation"][a] = validate(raw[a])
            out["parse_health"][a] = parse_health(a1[a])
            out["extra"][a] = extra(a1[a])
        if a1[a]:
            out["a1"][a] = summarize(a1[a])
    for k, rows in ref.items():
        if rows:
            out["refs"][k] = summarize(rows)
            out["extra"][k] = extra(rows)
            out["parse_health"][k] = parse_health(rows)
    for arm in ("grep", "sem"):
        if not a1[arm]:
            continue
        for v in ("v3", "v4"):
            out["minus_a1"][f"{v}_{arm}"] = deltas(ref[f"{v}_{arm}"], a1[arm])
    if a1["grep"]:
        out["minus_a1"]["base_grep_MIXED_protocol_base_is_retain_off"] = deltas(base, a1["grep"])
        out["a1_minus_base"]["grep"] = deltas(a1["grep"], base)
        out["prd15_a1_grep_vs_base"] = prd15(out["a1"]["grep"], out["a1_minus_base"]["grep"], out["base"])
        out["prd15_passes"] = sum(1 for c in out["prd15_a1_grep_vs_base"].values() if c["pass"])
        for v in ("v3", "v4"):
            out[f"{v}_grep_prd15_vs_base_ref"] = prd15(out["refs"][f"{v}_grep"], deltas(ref[f"{v}_grep"], base), out["base"])
    if a1["grep"] and a1["sem"]:
        out["a1_sem_minus_grep"] = deltas(a1["sem"], a1["grep"])
    (AD / "summary.json").write_text(json.dumps(out, indent=2))

    cols = [("base (retain off)", out["base"])] + [(f"A1 {a}", out["a1"][a]) for a in ("grep", "sem") if a in out["a1"]] + [(k.replace("_", " "), out["refs"][k]) for k in ("v3_grep", "v4_grep", "v3_sem", "v4_sem") if k in out["refs"]]
    print("| metric | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in ("recall_reported", "all_gold_reported", "recall_touched", "all_gold_touched", "reported_before_cap", "turns", "tokens", "wall_s", "calls_per_turn", "turn_latency_s"):
        print(f"| {k} | " + " | ".join(cell(s, k, 2 if k in ("all_gold_reported", "all_gold_touched", "reported_before_cap") else 3) if k not in ("turns", "tokens", "wall_s", "calls_per_turn", "turn_latency_s")
                                      else f"{s[k]['mean']:.1f} (p50 {s[k].get('p50', '-')})" for _, s in cols) + " |")
    for k, d in out["minus_a1"].items():
        print(f"\n{k} minus A1 ({k.split('_')[1] if 'MIXED' not in k else 'grep'}): " + "; ".join(f"{m} {dcell(d, m)}" for m in ("recall_reported", "all_gold_reported", "recall_touched", "reported_before_cap", "turns", "tokens", "wall_s")))
    if "prd15_a1_grep_vs_base" in out:
        print("\nPRD15 A1 grep vs base:", json.dumps(out["prd15_a1_grep_vs_base"], indent=1), "passes", out["prd15_passes"])
    print("\nvalidation", json.dumps(out["validation"], indent=1))
    print("parse_health", json.dumps(out["parse_health"], indent=1))
    print("extra", json.dumps(out["extra"], indent=1))


if __name__ == "__main__":
    main()
