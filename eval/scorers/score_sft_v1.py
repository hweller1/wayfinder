"""Score SFT v1 eval arms vs the prompted-Glimmer baseline (same 50 dev tasks x 2 seeds), with paired task-bootstrap CIs.

    python3 scripts/score_sft_v1.py   -> artifacts/sft_v1/summary.json (+ prints a markdown table)
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_glimmer_baseline import boot, load, paired, per_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H100_USD_PER_S = 0.001097


def called_report(r: dict) -> bool:
    return any((tc.get("function") or {}).get("name") == "report" for m in r["messages"] if m["role"] == "assistant" for tc in (m.get("tool_calls") or []))


def ok_rows(p: Path) -> list[dict]:
    return [r for r in load(p) if not r.get("error") and not r.get("infra_error")]


def metrics() -> dict:
    return {
        "recall": lambda r: r["recall_seen_k5"],
        "all_gold": lambda r: r["all_gold_found"],
        "recall_reported": lambda r: r["recall_reported"],
        "all_gold_reported": lambda r: r["all_gold_reported"],
        "turns": lambda r: r["turns"],
        "hit_cap": lambda r: (r.get("turns") or 0) >= r["max_turns"] or bool((r.get("explore_result") or {}).get("budget_exhausted")),
        "called_report": called_report,
        "reported_before_cap": lambda r: not bool((r.get("explore_result") or {}).get("budget_exhausted")) and (r.get("turns") or 0) < r["max_turns"],
        "tokens": lambda r: r["usage"]["in"] + r["usage"]["out"],
        "wall_s": lambda r: r["wall_s"],
        "tool_calls_per_turn": lambda r: sum(len(m.get("tool_calls") or []) for m in r["messages"] if m["role"] == "assistant") / max(r["turns"], 1),
        "turn_latency_s": lambda r: r["model_latency_s"] / max(r["model_calls"], 1),
    }


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_rollouts": len(rows), "n_tasks": len({r["task_id"] for r in rows})}
    for name, f in metrics().items():
        m, lo, hi = boot(per_task(rows, f))
        out[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if name in ("turns", "tokens", "wall_s"):
            xs = sorted(f(r) for r in rows)
            out[name]["p50"] = round(st.median(xs), 1)
            out[name]["p90"] = round(xs[int(0.9 * len(xs))], 1)
    return out


def deltas(a: list[dict], b: list[dict]) -> dict:
    out = {}
    for name, f in metrics().items():
        d = paired(per_task(a, f), per_task(b, f))
        out[name] = {"delta": round(d[0], 4), "ci95": [round(d[1], 4), round(d[2], 4)], "n_tasks": d[3]}
    return out


def wall_capped(r: dict) -> bool:
    return "seconds" in str((r.get("explore_result") or {}).get("gaps"))


def build_views(sd: Path, base: list[dict], teacher: list[dict]) -> dict:
    """Views of the grep arm vs base: primary (all completed rows), not-wall-capped rows, and turn-cap-only (wall-capped
    pairs replaced by their 3600 s reruns, from artifacts/sft_v1/rerun_*/sft_v1_grep.jsonl)."""
    g = ok_rows(sd / "sft_v1_grep.jsonl")
    out = {"wall_capped_rows": sum(wall_capped(r) for r in g), "wall_capped_turns_mean": round(st.mean(r["turns"] for r in g if wall_capped(r)), 1)}
    views = {"primary_all_completed": g, "sensitivity_not_wall_capped": [r for r in g if not wall_capped(r)]}
    reruns = [r for p in sorted(sd.glob("rerun_*/sft_v1_grep.jsonl")) for r in ok_rows(p)]
    if reruns:
        key = {(r["task_id"], r["seed"]) for r in reruns}
        views["turn_cap_only_rerun"] = [r for r in g if (r["task_id"], r["seed"]) not in key and not wall_capped(r)] + reruns
        out["rerun_rows"] = len(reruns)
        out["rerun_still_wall_capped"] = sum(wall_capped(r) for r in reruns)
    for name, rows in views.items():
        v = {"summary": summarize(rows), "vs_base": deltas(rows, base)}
        c, d = v["summary"], v["vs_base"]
        tok_down = 1 - c["tokens"]["mean"] / summarize(base)["tokens"]["mean"]
        v["prd15"] = {
            "reported_recall_delta": {**d["recall_reported"], "pass": d["recall_reported"]["ci95"][0] > 0, "target_met": d["recall_reported"]["delta"] >= 0.10},
            "reported_before_cap": {"v1": c["reported_before_cap"]["mean"], "called_report_tool": c["called_report"]["mean"], "pass": c["reported_before_cap"]["mean"] >= 0.80},
            "turns_p50": {"v1": c["turns"]["p50"], "pass": c["turns"]["p50"] <= 20},
            "touched_recall_delta": {**d["recall"], "pass": d["recall"]["ci95"][0] > -0.10},
            "tokens_down_frac": {"down": round(tok_down, 4), "pass": tok_down >= 0.30},
        }
        out[name] = v
    return out


def main() -> None:
    sd = ROOT / "artifacts/sft_v1"
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    teacher = ok_rows(ROOT / "artifacts/teacher_test/stageB_selfhost.jsonl")
    arms = {"v1_grep": ok_rows(sd / "sft_v1_grep.jsonl"), "v1_sem": ok_rows(sd / "sft_v1_sem.jsonl")}
    allrows = {a: load(sd / f"sft_v1_{a.split('_')[1]}.jsonl") for a in arms}
    out: dict = {"base": summarize(base), "teacher_stageB": summarize(teacher)}
    for a, rows in arms.items():
        s = summarize(rows)
        meta = json.loads((sd / f"sft_v1_{a.split('_')[1]}.meta.json").read_text())
        gs = meta.get("gpu_function_seconds") or 0
        s["error_rows"] = len(allrows[a]) - len(rows)
        s["gpu_seconds"] = gs
        s["gpu_usd_total"] = round(gs * H100_USD_PER_S, 2)
        s["gpu_usd_per_rollout"] = round(gs * H100_USD_PER_S / max(len(rows), 1), 4)
        s["semantic_offered_rows"] = sum(bool(r.get("semantic")) for r in rows)
        s["semantic_calls_mean"] = round(st.mean(r.get("semantic_calls") or 0 for r in rows), 2) if rows else None
        out[a] = s
        out[f"{a}_minus_base"] = deltas(rows, base)
    # PRD section 15 pre-registered criteria, grep arm vs base
    d, c = out["v1_grep_minus_base"], out["v1_grep"]
    tok_down = 1 - c["tokens"]["mean"] / out["base"]["tokens"]["mean"]
    out["prd15"] = {
        "reported_recall_delta": {**d["recall_reported"], "pass": d["recall_reported"]["ci95"][0] > 0, "target_met": d["recall_reported"]["delta"] >= 0.10},
        "reported_before_cap": {"v1": c["reported_before_cap"]["mean"], "base": out["base"]["reported_before_cap"]["mean"], "pass": c["reported_before_cap"]["mean"] >= 0.80},
        "turns_p50": {"v1": c["turns"]["p50"], "base": out["base"]["turns"]["p50"], "pass": c["turns"]["p50"] <= 20},
        "touched_recall_delta": {**d["recall"], "pass": d["recall"]["ci95"][0] > -0.10},
        "tokens_down_frac": {"v1_mean": c["tokens"]["mean"], "base_mean": out["base"]["tokens"]["mean"], "down": round(tok_down, 4), "pass": tok_down >= 0.30},
    }
    out["grep_views"] = build_views(sd, base, teacher)
    out["v1_sem_minus_v1_grep"] = deltas(arms["v1_sem"], arms["v1_grep"])
    out["v1_grep_minus_teacher"] = deltas(arms["v1_grep"], teacher)
    out["v1_sem_minus_teacher"] = deltas(arms["v1_sem"], teacher)
    out["base_gpu_usd_per_rollout"] = round(983.3 * H100_USD_PER_S / 100, 4)
    (sd / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
