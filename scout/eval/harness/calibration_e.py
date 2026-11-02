# ruff: noqa: E501, E731
"""Addendum E calibration table: fresh rows in artifacts/eval/backfill, stored rows beside them."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.harness.arms import valid_row
from eval.harness.report import gold_by_task, length_stats

R = Path(__file__).resolve().parents[2] / "artifacts" / "eval"
ARMS = ("super_v3_bf", "hybrid_v3_bf")


def load(p):
    out, bad = {}, []
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            (out.__setitem__(r["task_id"], r) if valid_row(r) is None and not r.get("fallthrough") else bad.append(r["task_id"][-8:]))
    return out, bad


fresh, invalid = {}, {}
for a in ("explore_default", *ARMS):
    fresh[a], invalid[a] = load(R / "backfill" / f"rows_{a}.jsonl")
tasks = sorted(set.intersection(*(set(v) for v in fresh.values())))
stored = {"hybrid_v3 (Addendum D)": load(R / "hybrid_v3" / "rows_hybrid_v3.jsonl")[0], "super_v3 (stored)": load(R / "rows_super_v3.jsonl")[0]}
gold = gold_by_task()
m = lambda xs: round(statistics.mean(xs), 3) if xs else None
res = {"tasks": tasks, "arms": {}, "invalid_rows": invalid}
for name, rows in {**{("explore_default (fresh)" if a == "explore_default" else a): fresh[a] for a in fresh}, **stored}.items():
    rs = [rows[t] for t in tasks if t in rows]
    ls = [length_stats(r, gold[r["task_id"]]) for r in rs]
    res["arms"][name] = {
        "n": len(rs), "recall12": m([r["recall12"] for r in rs]), "precision12": m([x["precision12"] for x in ls]),
        "n_paths": m([x["n_paths"] for x in ls]), "recall5": m([x["recall5"] for x in ls]),
        "wall_s_median": round(statistics.median([r["wall_s"] for r in rs]), 1) if rs else None,
        "cost_usd": m([r["cost_usd"] for r in rs]), "final_recall12": m([r["final_recall12"] for r in rs]),
        "unverified_share": m([r.get("handback_unverified", 0) / max(1, len(r["paths"])) for r in rs]),
        "per_task_recall12": {t[-10:]: round(rows[t]["recall12"], 3) for t in tasks if t in rows},
    }
ed = res["arms"]["explore_default (fresh)"]["recall12"]
res["gate"] = {a: {"recall_at_least_even": res["arms"][a]["recall12"] >= ed, "no_invalid_rows": len(tasks) == 5 and not invalid[a]} for a in ARMS}
for g in res["gate"].values():
    g["pass"] = all(g.values())
print(json.dumps(res, indent=1))
(R / "backfill" / "calibration.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
