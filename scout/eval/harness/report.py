"""Results: per-arm means with task-bootstrap CIs, paired deltas, repo-cluster sensitivity, per-language split.

    uv run python -m eval.harness.report --rows artifacts/eval --out artifacts/eval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.harness import stats
from eval.harness.arms import A20, ALL_ARMS, AUGMENT, HYBRID, HYBRID_A1, HYBRIDS, RELAY_BF, valid_row

METRICS = ("recall12", "all_gold", "final_recall12", "wall_s", "explore_s", "cost_usd", "parent_cost_usd", "subagent_cost_usd")


def present_arms(rows_dir: Path) -> tuple[str, ...]:
    """The three original arms, plus `augment_v3` when its rows file exists (the original report is unchanged otherwise)."""
    optional = (AUGMENT, A20, *HYBRIDS, RELAY_BF)
    return tuple(a for a in ALL_ARMS if a not in optional or (rows_dir / f"rows_{a}.jsonl").exists())


def load(rows_dir: Path) -> dict[str, dict[str, dict]]:
    """arm -> task_id -> the last valid, non-fallthrough row (fallthrough rows are reported separately)."""
    good: dict[str, dict[str, dict]] = {}
    bad: dict[str, list[dict]] = {}
    for arm in present_arms(rows_dir):
        p = rows_dir / f"rows_{arm}.jsonl"
        good[arm], bad[arm] = {}, []
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if valid_row(r) is None and not r.get("fallthrough"):
                good[arm][r["task_id"]] = r
            else:
                bad[arm].append(r)
    load.bad = bad  # type: ignore[attr-defined]
    return good


def series(rows: dict[str, dict], metric: str) -> dict[str, float]:
    return {t: float(r[metric]) for t, r in rows.items() if r.get(metric) is not None}


def usage(rows: dict[str, dict], tasks: list[str]) -> dict:
    """How often the augment arm's subagent called scout tools (rows are the scored ones)."""
    rs = [rows[t] for t in tasks]
    used = [r for r in rs if r.get("scout_calls", 0) > 0]
    n = len(rs)
    other = [r for r in rs if r.get("scout_calls", 0) == 0]

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n,
        "tasks_with_scout_call": len(used),
        "share_with_scout_call": len(used) / n if n else None,
        "mean_scout_calls": mean([float(r.get("scout_calls", 0)) for r in rs]),
        "mean_explore_calls": mean([float(r.get("scout_explore_calls", 0)) for r in rs]),
        "mean_semantic_search_calls": mean([float(r.get("scout_search_calls", 0)) for r in rs]),
        "mean_subagent_tool_calls": mean([float(r["subagent_tool_calls"]) for r in rs if r.get("subagent_tool_calls") is not None]),
        "recall12_when_used": mean([r["recall12"] for r in used]),
        "recall12_when_not_used": mean([r["recall12"] for r in other]),
    }


def hybrid_usage(rows: dict[str, dict], tasks: list[str]) -> dict:
    """How the hybrid subagent worked: scout explore first, reads after it, tool calls per session."""
    rs = [rows[t] for t in tasks]
    n = len(rs)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n,
        "share_explore_first": (sum(1 for r in rs if r.get("explore_first")) / n) if n else None,
        "mean_explore_calls": mean([float((r.get("plugin_tool_counts") or {}).get("explore", 0)) for r in rs]),
        "mean_reads_after_explore": mean([float(r.get("reads_after_explore", 0)) for r in rs]),
        "mean_subagent_tool_calls": mean([float(r["subagent_tool_calls"]) for r in rs if r.get("subagent_tool_calls") is not None]),
    }


def gold_by_task() -> dict[str, set[str]]:
    """Gold files per task id from `eval/eval_tasks.jsonl` (rows carry the recall but not the gold set)."""
    p = Path(__file__).resolve().parents[1] / "eval_tasks.jsonl"
    return {r["native_id"]: set(r["gold_files"]) for r in map(json.loads, p.read_text().splitlines()) if r}


def length_stats(row: dict, gold: set[str]) -> dict[str, float]:
    """Handback length and precision of one scored row from its ranked `paths` (repo files named in the handback,
    validated against git ls-files). `precision_k` is `|gold & first k| / |first k|` (0 for an empty list);
    `paths_top12` is the number of paths that count toward recall@12."""
    paths = row.get("paths") or []
    out: dict[str, float] = {"n_paths": float(len(paths)), "paths_top12": float(min(len(paths), 12))}
    for k in (5, 7, 12):
        top = paths[:k]
        out[f"precision{k}"] = len(gold & set(top)) / len(top) if top else 0.0
        out[f"recall{k}"] = len(gold & set(top)) / len(gold) if gold else 0.0
    return out


def summarize(good: dict[str, dict[str, dict]], bad: dict[str, list[dict]]) -> dict:
    ARMS = tuple(a for a in good if good[a])  # noqa: N806  arms with no valid row are not required
    tasks_all = sorted(set().union(*(set(g) for g in good.values())))
    complete = sorted(t for t in tasks_all if all(t in good[a] for a in ARMS))
    repo_of = {t: good[a][t]["repo"] for a in ARMS for t in good[a]}
    lang_of = {t: good[a][t]["lang"] for a in ARMS for t in good[a]}
    gold_all = gold_by_task()
    out: dict = {"n_tasks_complete": len(complete), "arms": {}, "deltas": {}, "per_language": {}, "fallthrough": {}, "excluded": {}}
    for arm in ARMS:
        s: dict = {"n": len(good[arm])}
        for m in METRICS:
            v = series({t: good[arm][t] for t in complete}, m)
            if v:
                mean, lo, hi = stats.bootstrap_ci(list(v.values()))
                s[m] = {"mean": mean, "ci95": [lo, hi], "n": len(v)}
        if complete and all(t in gold_all and "paths" in good[arm][t] for t in complete):
            gold = gold_all
            ls = {t: length_stats(good[arm][t], gold[t]) for t in complete}
            s["length"] = {}
            for m in ("n_paths", "paths_top12", "precision5", "precision7", "precision12", "recall5", "recall7"):
                mean, lo, hi = stats.bootstrap_ci([v[m] for v in ls.values()])
                s["length"][m] = {"mean": mean, "ci95": [lo, hi], "n": len(ls)}
        if complete:
            med, p90 = stats.median_p90([good[arm][t]["wall_s"] for t in complete])
            s["wall_median_p90"] = [med, p90]
        out["arms"][arm] = s
        out["fallthrough"][arm] = sum(1 for r in bad[arm] if r.get("fallthrough"))
        out["excluded"][arm] = [{"task_id": r["task_id"], "problem": r.get("problem"), "fallthrough": r.get("fallthrough")} for r in bad[arm]]
    pairs = [("super_v3", "explore_default"), ("super_a1", "explore_default"), ("super_v3", "super_a1")]
    for h, relay in ((HYBRID, "super_v3"), (HYBRID_A1, "super_a1")):
        if h in ARMS:
            pairs = [(h, "explore_default"), *([(h, relay)] if relay in ARMS else []), *pairs]
            out.setdefault("hybrid_usage", {})[h] = hybrid_usage(good[h], complete)
    if A20 in ARMS:
        pairs = [(A20, "super_v3"), (A20, "explore_default"), *pairs]
    if AUGMENT in ARMS:
        pairs = [(AUGMENT, "explore_default"), (AUGMENT, "super_v3"), *pairs]
        out["augment_usage"] = usage(good[AUGMENT], complete)
    for a, b in pairs:
        key = f"{a}_minus_{b}"
        out["deltas"][key] = {}
        gold = gold_all
        have_paths = all(t in gold and "paths" in good[x][t] for x in (a, b) for t in complete)
        for m in ("recall12", "all_gold", "wall_s", "cost_usd", "final_recall12", "n_paths", "precision12", "recall5", "recall7"):
            if m in ("n_paths", "precision12", "recall5", "recall7"):
                if not have_paths:
                    continue
                sa = {t: length_stats(good[a][t], gold[t])[m] for t in complete}
                sb = {t: length_stats(good[b][t], gold[t])[m] for t in complete}
            else:
                sa, sb = series({t: good[a][t] for t in complete}, m), series({t: good[b][t] for t in complete}, m)
            d = stats.paired_delta(sa, sb)
            if not d:
                continue
            mean, lo, hi = stats.bootstrap_ci(list(d.values()))
            cm, clo, chi = stats.cluster_bootstrap_ci(stats.by_cluster(d, repo_of))
            out["deltas"][key][m] = {"mean": mean, "ci95": [lo, hi], "cluster_ci95": [clo, chi], "n": len(d)}
        langs: dict[str, dict[str, float]] = defaultdict(dict)
        d = stats.paired_delta(series({t: good[a][t] for t in complete}, "recall12"), series({t: good[b][t] for t in complete}, "recall12"))
        for t, v in d.items():
            langs[lang_of[t]][t] = v
        out["per_language"][key] = {lg: {"n": len(v), "mean_recall12_delta": sum(v.values()) / len(v)} for lg, v in sorted(langs.items())}
    out["per_task"] = {t: {a: good[a][t]["recall12"] for a in ARMS} for t in complete}
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def markdown(res: dict) -> str:
    def f(x: dict | None, d: int = 3) -> str:
        return "n/a" if not x else f"{x['mean']:.{d}f} [{x['ci95'][0]:.{d}f}, {x['ci95'][1]:.{d}f}]"

    lines = [f"Tasks with a valid row in every arm: {res['n_tasks_complete']}", "", "| Arm | recall@12 | all-gold | wall s (mean) | wall median / p90 | cost USD | parent USD | subagent USD |", "|---|---|---|---|---|---|---|---|"]
    for arm, s in res["arms"].items():
        w = s.get("wall_median_p90", [float("nan")] * 2)
        lines.append(f"| {arm} | {f(s.get('recall12'))} | {f(s.get('all_gold'), 2)} | {f(s.get('wall_s'), 1)} | {w[0]:.0f} / {w[1]:.0f} | {f(s.get('cost_usd'), 4)} | {f(s.get('parent_cost_usd'), 4)} | {f(s.get('subagent_cost_usd'), 4)} |")
    if any("length" in s for s in res["arms"].values()):
        lines += ["", "Handback length and precision (mean [95% task CI]; paths = repo files named in the handback):", "", "| Arm | paths per handback | paths in top 12 | precision@5 | precision@7 | precision@12 | recall@5 | recall@7 |", "|---|---|---|---|---|---|---|---|"]
        for arm, s in res["arms"].items():
            ln = s.get("length", {})
            lines.append(f"| {arm} | {f(ln.get('n_paths'), 2)} | {f(ln.get('paths_top12'), 2)} | {f(ln.get('precision5'))} | {f(ln.get('precision7'))} | {f(ln.get('precision12'))} | {f(ln.get('recall5'))} | {f(ln.get('recall7'))} |")
    lines += ["", "| Paired delta (a minus b) | metric | mean [95% task CI] | repo-cluster CI |", "|---|---|---|---|"]
    for key, ms in res["deltas"].items():
        for m, v in ms.items():
            lines.append(f"| {key} | {m} | {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | [{v['cluster_ci95'][0]:+.3f}, {v['cluster_ci95'][1]:+.3f}] |")
    lines += ["", "Per-language recall@12 delta (mean, n tasks):", ""]
    for key, ls in res["per_language"].items():
        lines.append(f"- {key}: " + ", ".join(f"{lg} {v['mean_recall12_delta']:+.3f} (n={v['n']})" for lg, v in ls.items()))
    for h, u in res.get("hybrid_usage", {}).items():
        lines += ["", f"{h} subagent ({u['n']} scored tasks): scout explore was the first tool in {u['share_explore_first']:.0%}; mean explore calls {u['mean_explore_calls']:.2f}; mean reads after explore {u['mean_reads_after_explore']:.1f}; mean subagent tool calls {u['mean_subagent_tool_calls']:.1f}."]
    if "augment_usage" in res:
        u = res["augment_usage"]
        lines += ["", f"augment_v3 scout-tool use ({u['n']} scored tasks): {u['tasks_with_scout_call']} tasks called a scout tool ({u['share_with_scout_call']:.0%}); mean calls per task {u['mean_scout_calls']:.2f} (explore {u['mean_explore_calls']:.2f}, semantic_search {u['mean_semantic_search_calls']:.2f}); mean subagent tool calls {u['mean_subagent_tool_calls']:.1f}; recall@12 when used {_fmt(u['recall12_when_used'])}, when not used {_fmt(u['recall12_when_not_used'])}."]
    lines += ["", "Fallthrough or invalid rows per arm (reported, not scored): " + ", ".join(f"{a} {len(v)} (fallthrough {res['fallthrough'][a]})" for a, v in res["excluded"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/eval")
    ap.add_argument("--out", default="artifacts/eval")
    a = ap.parse_args()
    good = load(Path(a.rows))
    res = summarize(good, load.bad)  # type: ignore[attr-defined]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    (out / "REPORT.md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
