"""Stage B summary: recall / all_gold_found vs benchmark arms on the same tasks, task-bootstrap CIs (10k, seeded)."""
from __future__ import annotations

import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from wayfinder.rollout.bench import load_bench

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("a1_35b_b2", "a1_35b_g", "sonnet55high_b2")
rng = np.random.default_rng(20261026)


def ci(d: np.ndarray) -> list[float]:
    idx = rng.integers(0, len(d), size=(10_000, len(d)))
    m = d[idx].mean(axis=1)
    return [round(float(x), 4) for x in np.percentile(m, [2.5, 97.5])]


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "artifacts/teacher_test/stageB.jsonl").read_text().splitlines() if l.strip()]
    keep = {t["task_id"] for t in json.loads((ROOT / "artifacts/teacher_test/stageB_tasks.json").read_text())}
    rows = [r for r in rows if not r.get("infra_error") and r["task_id"] in keep]  # stage A rows reused for overlap only
    bench = load_bench({r["task_id"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r)
    tasks = sorted(t for t in by if all(bench.get(t, {}).get(a, {}).get("recall") is not None for a in ARMS))
    out: dict = {"n_tasks": len(tasks), "n_rollouts": sum(len(by[t]) for t in tasks), "seeds": sorted({r["seed"] for r in rows}),
                 "semantic": False, "metric_note": "recall = benchmark-like view (files opened + first 5 files of each grep/glob result); reported = files in the ExploreResult (<=12)"}
    for view in ("recall", "recall_reported", "recall_opened", "recall_seen_all"):
        t_mean = np.array([st.mean(r[view] for r in by[t]) for t in tasks])
        out[f"spark_{view}"] = {"mean": round(float(t_mean.mean()), 4), "ci": ci(t_mean)}
    agf = np.array([st.mean(float(r["all_gold_found"]) for r in by[t]) for t in tasks])
    agr = np.array([st.mean(float(r["all_gold_reported"]) for r in by[t]) for t in tasks])
    out["spark_all_gold_found"] = {"mean": round(float(agf.mean()), 4), "ci": ci(agf)}
    out["spark_all_gold_reported"] = {"mean": round(float(agr.mean()), 4), "ci": ci(agr)}
    for a in ARMS:
        rec = np.array([bench[t][a]["recall"] for t in tasks]); ag = np.array([float(bench[t][a]["all_gold"]) for t in tasks])
        out[a] = {"recall": {"mean": round(float(rec.mean()), 4), "ci": ci(rec)}, "all_gold": {"mean": round(float(ag.mean()), 4), "ci": ci(ag)}}
        sr = np.array([st.mean(r["recall"] for r in by[t]) for t in tasks])
        srep = np.array([st.mean(r["recall_reported"] for r in by[t]) for t in tasks])
        out[a]["delta_recall_spark_minus_arm"] = {"mean": round(float((sr - rec).mean()), 4), "ci": ci(sr - rec)}
        out[a]["delta_recall_reported_spark_minus_arm"] = {"mean": round(float((srep - rec).mean()), 4), "ci": ci(srep - rec)}
        out[a]["delta_all_gold_spark_minus_arm"] = {"mean": round(float((agf - ag).mean()), 4), "ci": ci(agf - ag)}
    rr = [r for t in tasks for r in by[t]]
    tot_in = sum(r["usage"]["in"] for r in rr); tot_c = sum(r["usage"]["cached"] for r in rr)
    out["cost"] = {"per_rollout_usd": round(st.mean(r["cost_usd"] for r in rr), 4), "total_usd": round(sum(r["cost_usd"] for r in rr), 2),
                   "turns_mean": round(st.mean(r["turns"] for r in rr), 1), "turns_median": st.median(r["turns"] for r in rr),
                   "hit_turn_cap_share": round(sum(r["explore_result"]["budget_exhausted"] for r in rr) / len(rr), 3),
                   "cached_share_of_input": round(tot_c / tot_in, 3), "in_M": round(tot_in / 1e6, 2), "out_M": round(sum(r["usage"]["out"] for r in rr) / 1e6, 2),
                   "model_latency_s_median": st.median(r["model_latency_s"] for r in rr), "budget_wall_s_median": st.median(r["wall_budget_s"] for r in rr),
                   "benchmark_turns_mean": {a: round(st.mean(bench[t][a]["turns"] or 0 for t in tasks), 1) for a in ARMS}}
    # seed-to-seed consistency
    two = [t for t in tasks if len(by[t]) >= 2]
    if two:
        out["seed_pairs"] = {"n_tasks": len(two), "mean_abs_recall_diff": round(st.mean(abs(by[t][0]["recall"] - by[t][1]["recall"]) for t in two), 4)}
    (ROOT / "artifacts/teacher_test/summary_baseten.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
