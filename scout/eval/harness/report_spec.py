# ruff: noqa: E501, E701, E741
"""cc_scout_ablation report: per-arm table, paired task-bootstrap contrasts, delegation and cost.

uv run python -m eval.harness.report_spec --dir artifacts/eval/cc50/full --out artifacts/eval/cc50/full/report.json
"""

import argparse, json, statistics as st
from pathlib import Path

from eval.harness.stats import bootstrap_ci, paired_delta

ARMS = ("cc_explore", "cc_explore_spec", "cc_scout_spec", "cc_scout_spec_noidx")
CONTRASTS = (
    ("cc_scout_spec", "cc_explore_spec"),
    ("cc_scout_spec", "cc_explore"),
    ("cc_explore_spec", "cc_explore"),
    ("cc_scout_spec", "cc_scout_spec_noidx"),
)
METRICS = ("recall5", "recall12", "precision12", "n_reported", "cost_usd", "turns")


def load(d: Path, arm: str, seed: int = 0) -> dict[str, dict]:
    p = d / f"rows_{arm}_s{seed}.jsonl"
    out: dict[str, dict] = {}
    if p.exists():
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if not r.get("problem"):
                out[r["task_id"]] = r
    return out


def summarize(rows: dict[str, dict]) -> dict:
    rs = list(rows.values())
    if not rs:
        return {"n": 0}
    m = lambda f: round(st.mean(f(r) for r in rs), 4)
    return {
        "n": len(rs),
        **{k: m(lambda r, k=k: r[k]) for k in METRICS},
        "delegated_explore": sum(r["explore_agent_calls"] > 0 for r in rs),
        "delegated_scout": sum(r["scout_agent_calls"] > 0 for r in rs),
        "delegated_any": sum(r["explore_agent_calls"] + r["scout_agent_calls"] > 0 for r in rs),
        "scout_tool_called": sum(r["scout_tool_calls_any"] > 0 for r in rs),
        "parent_cost": m(lambda r: r["parent_cost_usd"] or 0.0),
        "subagent_cost": m(lambda r: r["subagent_cost_usd"] or 0.0),
        "wall_s": m(lambda r: r["wall_s"]),
        "layers": {
            k: sum(bool((r.get("spec_layers") or {}).get(k)) for r in rs)
            for k in ("session_context", "index_description", "workflow_step")
        }
        if any("spec_layers" in r for r in rs)
        else None,
        "search_first_denials": sum((r.get("spec_layers") or {}).get("search_first_denials", 0) for r in rs),
    }


def contrast(a: dict, b: dict, metric: str) -> dict | None:
    d = paired_delta({t: r[metric] for t, r in a.items()}, {t: r[metric] for t, r in b.items()})
    if len(d) < 2:
        return None
    mean, lo, hi = bootstrap_ci(list(d.values()))
    return {"n": len(d), "delta": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = {arm: load(d, arm) for arm in ARMS}
    res = {"arms": {arm: summarize(r) for arm, r in rows.items() if r}, "contrasts": {}}
    for x, y in CONTRASTS:
        if rows[x] and rows[y]:
            res["contrasts"][f"{x} - {y}"] = {
                m: contrast(rows[x], rows[y], m) for m in ("recall5", "recall12", "precision12", "cost_usd")
            }
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
