"""Summarise artifacts/baseline/glimmer_dev.jsonl with task-bootstrap CIs and compare to the benchmark's per-task
A1-35B / Sonnet recall (data/task_filter.json, b2 arm). Writes artifacts/baseline/glimmer_dev_summary.json.

    python3 scripts/score_glimmer_baseline.py [--rows PATH] [--teacher PATH]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def boot(per_task: dict[str, float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Task-level bootstrap of the mean over tasks (seeds averaged within task first)."""
    vals = list(per_task.values())
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired(a: dict[str, float], b: dict[str, float], n: int = 10000, seed: int = 1) -> tuple[float, float, float, int]:
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1], len(ks)


def per_task(rows: list[dict], f) -> dict[str, float]:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = f(r)
        if v is not None:
            g[r["task_id"]].append(float(v))
    return {k: sum(v) / len(v) for k, v in g.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROOT / "artifacts/baseline/glimmer_dev.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/baseline/glimmer_dev_summary.json"))
    ap.add_argument("--teacher", default=str(ROOT / "artifacts/teacher_test/stageB_selfhost.jsonl"))
    a = ap.parse_args()
    rows = load(Path(a.rows))
    ok = [r for r in rows if not r.get("error") and not r.get("infra_error")]
    err = [r for r in rows if r.get("error") or r.get("infra_error")]
    flt = json.loads((ROOT / "data/task_filter.json").read_text())["tasks"]
    tasks = {t["task_id"] for t in json.loads((ROOT / "artifacts/teacher_test/stageB_tasks.json").read_text())}
    out: dict = {"rows": len(rows), "ok_rows": len(ok), "error_rows": len(err), "tasks": len({r["task_id"] for r in ok}),
                 "errors": [r.get("error") for r in err][:10]}
    metrics = {
        "recall_seen_k5": lambda r: r["recall_seen_k5"],
        "all_gold_seen_k5": lambda r: r["all_gold_found"],
        "recall_reported": lambda r: r["recall_reported"],
        "all_gold_reported": lambda r: r["all_gold_reported"],
        "recall_opened": lambda r: r["recall_opened"],
        "turns": lambda r: r["turns"],
        "tokens_in": lambda r: r["usage"]["in"],
        "tokens_out": lambda r: r["usage"]["out"],
        "tokens_total": lambda r: r["usage"]["in"] + r["usage"]["out"],
        "wall_s": lambda r: r["wall_s"],
        "cost_usd_list": lambda r: r["cost_usd"],
        "hit_turn_cap": lambda r: (r.get("turns") or 0) >= r["max_turns"] or bool((r.get("explore_result") or {}).get("budget_exhausted")),
        "reported": lambda r: bool((r.get("explore_result") or {}).get("results")),
    }
    summ = {}
    for name, f in metrics.items():
        m, lo, hi = boot(per_task(ok, f))
        summ[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for name in ("turns", "tokens_total", "wall_s", "cost_usd_list"):
        xs = [metrics[name](r) for r in ok]
        summ[name]["p50"] = round(st.median(xs), 3)
        summ[name]["p90"] = round(sorted(xs)[int(0.9 * len(xs))], 3)
    out["summary"] = summ
    # per-seed headline, to show seed variance
    out["by_seed"] = {str(s): {"recall_seen_k5": round(st.mean(r["recall_seen_k5"] for r in ok if r["seed"] == s), 4),
                               "all_gold": round(st.mean(bool(r["all_gold_found"]) for r in ok if r["seed"] == s), 4)}
                      for s in sorted({r["seed"] for r in ok})}
    # benchmark comparison, per task (b2 arm = grep + index; "c" arm = grep only for 35B)
    mine = per_task(ok, lambda r: r["recall_seen_k5"])
    mine_ag = per_task(ok, lambda r: r["all_gold_found"])
    cmp = {}
    for label, key in (("a1_35b_b2", "recall_35b_b2"), ("a1_35b_c", "recall_35b_c"), ("a1_35b_g", "recall_35b_g"), ("sonnet_b2", "recall_sonnet_b2"), ("sonnet_c", "recall_sonnet_c")):
        bench = {t: flt[t][key] for t in tasks if t in flt and flt[t].get(key) is not None}
        common = {t: v for t, v in bench.items() if t in mine}
        if not common:
            continue
        m, lo, hi = boot(common)
        d, dlo, dhi, n = paired({t: mine[t] for t in common}, common)
        bag = {t: float(v >= 1.0) for t, v in common.items()}
        out_ag = boot(bag)
        mine_c = {t: mine_ag[t] for t in common}
        dag = paired(mine_c, bag)
        cmp[label] = {"n_tasks": n, "bench_recall": [round(m, 4), round(lo, 4), round(hi, 4)],
                      "glimmer_minus_bench_recall": [round(d, 4), round(dlo, 4), round(dhi, 4)],
                      "bench_all_gold": round(out_ag[0], 4), "glimmer_minus_bench_all_gold": [round(dag[0], 4), round(dag[1], 4), round(dag[2], 4)],
                      "glimmer_recall_on_same_tasks": round(sum(mine[t] for t in common) / n, 4)}
    out["vs_benchmark"] = cmp
    tp = Path(a.teacher)
    if tp.exists():
        trows = [r for r in load(tp) if not r.get("error") and not r.get("infra_error")]
        if trows:
            tm = per_task(trows, lambda r: r["recall_seen_k5"])
            d = paired(mine, tm)
            out["vs_spark_teacher"] = {"n_tasks": d[3], "teacher_recall": round(sum(tm.values()) / len(tm), 4),
                                       "glimmer_minus_teacher_recall": [round(x, 4) for x in d[:3]]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
