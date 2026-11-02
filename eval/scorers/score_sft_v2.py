"""Score SFT v2 variants (v2a, v2b) against untrained Glimmer on the PRD section 15 criteria, with paired task-bootstrap CIs.
v1 and the teacher are shown as reference columns when their rows exist.

    python3 scripts/score_sft_v2.py   -> artifacts/sft_v2/summary.json (+ prints a markdown table)

Rows: artifacts/sft_v2/sft_<variant>_<arm>.jsonl (arm grep|sem). Baseline: artifacts/baseline/glimmer_dev.jsonl (grep only).
Criteria are evaluated on the grep arm, as pre-registered; the semantic arm is exploratory.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_glimmer_baseline import boot, load, paired, per_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ok_rows(p: Path) -> list[dict]:
    return [r for r in load(p) if not r.get("error") and not r.get("infra_error")] if p.exists() else []


def n_calls(r: dict) -> int:
    return sum(len(m.get("tool_calls") or []) for m in r["messages"] if m["role"] == "assistant")


def n_asst(r: dict) -> int:
    return max(1, sum(1 for m in r["messages"] if m["role"] == "assistant"))


def stop_reason(r: dict) -> str:
    e = r.get("explore_result") or {}
    if not e.get("budget_exhausted"):
        return "report"
    g = [x for x in e.get("gaps", []) if "budget exhausted" in x]
    return g[0].split("(")[1].split(")")[0] if g else "budget"


def turn_lat(r: dict) -> float | None:
    xs = [t["latency_s"] for t in r.get("turn_log", []) if t.get("ok") and t.get("latency_s") is not None]
    return sum(xs) / len(xs) if xs else None


METRICS = {
    "recall_reported": lambda r: r["recall_reported"],
    "all_gold_reported": lambda r: r["all_gold_reported"],
    "recall_touched": lambda r: r["recall_seen_k5"],
    "all_gold_touched": lambda r: r["all_gold_found"],
    "reported_before_cap": lambda r: stop_reason(r) == "report",
    "turns": lambda r: r["turns"],
    "tokens": lambda r: r["usage"]["in"] + r["usage"]["out"],
    "calls_per_turn": lambda r: n_calls(r) / n_asst(r),
    "turn_latency_s": turn_lat,
    "wall_s": lambda r: r["wall_s"],
}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out: dict = {"n_rollouts": len(rows), "n_tasks": len({r["task_id"] for r in rows})}
    for name, f in METRICS.items():
        m, lo, hi = boot(per_task(rows, f))
        out[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for name in ("turns", "tokens", "wall_s"):
        xs = sorted(METRICS[name](r) for r in rows)
        out[name]["p50"] = round(st.median(xs), 1)
    from collections import Counter
    out["stop_reasons"] = dict(Counter(stop_reason(r) for r in rows))
    out["semantic_calls_mean"] = round(st.mean(r.get("semantic_calls") or 0 for r in rows), 2) if any("semantic_calls" in r for r in rows) else None
    return out


def deltas(a: list[dict], b: list[dict]) -> dict:
    out = {}
    for name, f in METRICS.items():
        d = paired(per_task(a, f), per_task(b, f))
        out[name] = {"delta": round(d[0], 4), "ci95": [round(d[1], 4), round(d[2], 4)], "n_tasks": d[3]}
    return out


def prd15(s: dict, d: dict, base: dict) -> dict:
    tok_down = 1 - s["tokens"]["mean"] / base["tokens"]["mean"]
    return {
        "reported_recall_delta (primary)": {**d["recall_reported"], "pass": d["recall_reported"]["ci95"][0] > 0,
                                            "target_ge_0.10": d["recall_reported"]["delta"] >= 0.10},
        "reported_before_cap_ge_0.80": {"value": s["reported_before_cap"]["mean"], "base": base["reported_before_cap"]["mean"],
                                        "pass": s["reported_before_cap"]["mean"] >= 0.80},
        "turns_p50_le_20": {"value": s["turns"]["p50"], "base": base["turns"]["p50"], "pass": s["turns"]["p50"] <= 20},
        "touched_recall_guardrail": {**d["recall_touched"], "pass": d["recall_touched"]["ci95"][0] > -0.10},
        "tokens_down_ge_30pct": {"down": round(tok_down, 4), "pass": tok_down >= 0.30},
    }


def main() -> None:
    sd = ROOT / "artifacts/sft_v2"
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    teacher = ok_rows(ROOT / "artifacts/teacher_test/stageB_selfhost.jsonl")
    arms = {"v1_grep": ok_rows(ROOT / "artifacts/sft_v1/sft_v1_grep.jsonl"), "v1_sem": ok_rows(ROOT / "artifacts/sft_v1/sft_v1_sem.jsonl")}
    for v in ("v2a", "v2b"):
        for a in ("grep", "sem"):
            arms[f"{v}_{a}"] = ok_rows(sd / f"sft_{v}_{a}.jsonl")
    out: dict = {"base": summarize(base), "teacher_stageB": summarize(teacher)}
    for name, rows in arms.items():
        if not rows:
            continue
        out[name] = summarize(rows)
        out[f"{name}_minus_base"] = deltas(rows, base)
        if name.endswith("grep"):
            out[f"{name}_prd15"] = prd15(out[name], out[f"{name}_minus_base"], out["base"])
            out[f"{name}_minus_teacher"] = deltas(rows, teacher)
    for v in ("v2a", "v2b"):
        if arms[f"{v}_grep"] and arms[f"{v}_sem"]:
            out[f"{v}_sem_minus_grep"] = deltas(arms[f"{v}_sem"], arms[f"{v}_grep"])
        if arms[f"{v}_grep"] and arms["v1_grep"]:
            out[f"{v}_grep_minus_v1_grep"] = deltas(arms[f"{v}_grep"], arms["v1_grep"])
    if arms["v2a_grep"] and arms["v2b_grep"]:
        out["v2a_minus_v2b_grep"] = deltas(arms["v2a_grep"], arms["v2b_grep"])
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "summary.json").write_text(json.dumps(out, indent=2))
    cols = [("base", out["base"])] + [(k, out[k]) for k in ("v1_grep", "v2a_grep", "v2b_grep", "v1_sem", "v2a_sem", "v2b_sem") if k in out] + [("teacher", out["teacher_stageB"])]

    def cell(s: dict, k: str) -> str:
        x = s[k]
        return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    print("| metric | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in METRICS:
        print(f"| {k} | " + " | ".join(cell(s, k) for _, s in cols) + " |")
    for k in [x for x in out if x.endswith("_prd15")]:
        print(k, json.dumps(out[k], indent=1))


if __name__ == "__main__":
    main()
