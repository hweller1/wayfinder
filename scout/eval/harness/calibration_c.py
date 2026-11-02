# ruff: noqa: E501, E731, E701, E741
"""Addendum C calibration table: fresh rows (artifacts/eval/hybrid) + stored rows (artifacts/eval) on the same tasks."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "<repo>/scout")
from eval.harness.arms import valid_row
from eval.harness.report import gold_by_task, length_stats

R = Path("<repo>/scout/artifacts/eval")


def load(p):
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if valid_row(r) is None and not r.get("fallthrough"):
                out[r["task_id"]] = r
    return out


ARM = sys.argv[1] if len(sys.argv) > 1 else "hybrid_a1"
SUB = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
fresh = {a: load(R / SUB / f"rows_{a}.jsonl") for a in ("explore_default", ARM)}
tasks = sorted(set(fresh[ARM]) & set(fresh["explore_default"]))
stored = {f"{a} (stored)": load(R / f"rows_{a}.jsonl") for a in ("explore_default", "super_v3", "super_a1")}
arms = {"explore_default (fresh)": fresh["explore_default"], ARM: fresh[ARM], **stored, "hybrid_a1 (Addendum C)": load(R / "hybrid" / "rows_hybrid_a1.jsonl")}
gold = gold_by_task()
res = {"tasks": tasks, "arms": {}}
for name, rows in arms.items():
    rs = [rows[t] for t in tasks if t in rows]
    ls = [length_stats(r, gold[r["task_id"]]) for r in rs]
    m = lambda xs: round(statistics.mean(xs), 3) if xs else None
    res["arms"][name] = {
        "n": len(rs),
        "recall12": m([r["recall12"] for r in rs]),
        "precision12": m([x["precision12"] for x in ls]),
        "n_paths": m([x["n_paths"] for x in ls]),
        "recall5": m([x["recall5"] for x in ls]),
        "recall7": m([x["recall7"] for x in ls]),
        "wall_s_mean": m([r["wall_s"] for r in rs]),
        "wall_s_median": round(statistics.median([r["wall_s"] for r in rs]), 1) if rs else None,
        "cost_usd": m([r["cost_usd"] for r in rs]),
        "per_task_recall12": {t[-10:]: round(rows[t]["recall12"], 3) for t in tasks if t in rows},
    }
h = [fresh[ARM][t] for t in tasks]
res[f"{ARM}_usage"] = {
    "explore_first": sum(r["explore_first"] for r in h),
    "mean_reads_after_explore": round(statistics.mean(r["reads_after_explore"] for r in h), 2),
    "mean_explore_calls": statistics.mean((r["plugin_tool_counts"] or {}).get("explore", 0) for r in h),
    "mean_subagent_tool_calls": round(statistics.mean(r["subagent_tool_calls"] or 0 for r in h), 1),
    "unverified_share": round(statistics.mean(r["handback_unverified"] / max(1, len(r["paths"])) for r in h), 3) if all("handback_unverified" in r for r in h) else None,
    "tool_sequences": {r["task_id"][-10:]: r["sub_tool_seq"] for r in h},
}
inv = {a: [json.loads(l) for l in (R / SUB / f"rows_{a}.jsonl").read_text().splitlines()] for a in fresh}
res["invalid_rows"] = {
    a: [
        (r["task_id"][-10:], r.get("problem"), r.get("fallthrough"))
        for r in rs
        if r.get("problem") or r.get("fallthrough")
    ]
    for a, rs in inv.items()
}
ed, hy = res["arms"]["explore_default (fresh)"], res["arms"][ARM]
res["gate"] = {
    "recall_within_0.05": hy["recall12"] >= ed["recall12"] - 0.05,
    "explore_first_4_of_5": res[f"{ARM}_usage"]["explore_first"] >= 4,
    "no_invalid_rows": len(tasks) == 5 and not any(res["invalid_rows"].values()),
}
if ARM == "hybrid_v3":  # Addendum D gate: at least even on recall@12, no invalid row
    res["gate"] = {"recall_at_least_even": hy["recall12"] >= ed["recall12"], "no_invalid_rows": res["gate"]["no_invalid_rows"]}
res["gate"]["pass"] = all(res["gate"].values())
print(json.dumps(res, indent=1))
(R / SUB / "calibration.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
