"""Confirmation set for GRPO run f1 (pre-registered in the GRPO write-up (not in this release) before the run launched): the 83 held-out E2E v4 tasks, SEARCH ONLY (no coder),
grep arm, seeds 1 and 2, retain on, max_turns 40, max_seconds 3600. Candidate (f1 prompt line) vs v3 vs A1, plus the v3+line control.

    python3 scripts/score_confirm_f1.py --candidate s20   -> artifacts/grpo/confirm/summary.json + stdout tables

Rows:  candidate  artifacts/grpo/confirm/rows/cand_<step>/sft_grpo-f1-<step>_grep.jsonl      (sft_eval_app.py, WF_EVAL_NUDGE=f1)
       v3         artifacts/e2e/search_wayfinder_v3.jsonl                                       (eval/e2e/search_app.py, the existing v4 search rows, no line)
       a1         artifacts/grpo/confirm/rows/a1/sft_a1_grep.jsonl                              (a1_eval_app.py, no line)
       v3line     artifacts/grpo/confirm/rows/v3line/sft_v3_grep.jsonl                          (sft_eval_app.py v3 weights, WF_EVAL_NUDGE=f1)
Primary: recall@5 candidate minus v3, task-bootstrap and repo-cluster (12 repos) 95% CIs (10,000 resamples, seed 11, scripts/score_test_split.py); holds if both lower bounds > 0.
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_test_split as T  # noqa: E402
from score_grpo import F1_METRICS  # noqa: E402
from score_sft_v2 import stop_reason  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "artifacts/grpo/confirm"
TASKS = json.loads((C / "tasks_v4_83.json").read_text())
REPO_OF = {t["task_id"]: t["repo"] for t in TASKS}
SEEDS = [1, 2]
METRICS = {**F1_METRICS, "recall_touched": lambda r: r["recall_seen_k5"], "all_gold_reported": lambda r: r["all_gold_reported"]}


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def healthy(name: str, rows: list[dict], want_nudge: str | None) -> dict:
    ok = [r for r in rows if not r.get("error") and not r.get("infra_error")]
    v = {"rows": len(rows), "ok": len(ok), "tasks": len({r["task_id"] for r in ok}), "seeds": sorted({r["seed"] for r in ok}), "models": sorted({str(r.get("model")) for r in ok}),
         "nudges": sorted({str(r.get("nudge")) for r in ok}), "retain": sorted({str(r.get("retain_reasoning")) for r in ok}), "max_turns": sorted({r.get("max_turns") for r in ok}),
         "pairs_unique": len({(r["task_id"], r["seed"]) for r in ok}) == len(ok), "in_task_set": all(r["task_id"] in REPO_OF for r in ok)}
    v["healthy"] = (v["ok"] == len(rows) == 166 and v["tasks"] == 83 and v["seeds"] == SEEDS and v["pairs_unique"] and v["in_task_set"]
                    and (want_nudge is None or v["nudges"] == [want_nudge]))
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="checkpoint step label, e.g. s20")
    a = ap.parse_args()
    from wayfinder.rollout.nudges import NUDGES

    sets = {"candidate": (C / f"rows/cand_{a.candidate}/sft_grpo-f1-{a.candidate}_grep.jsonl", NUDGES["f1"]),
            "v3": (ROOT / "artifacts/e2e/search_wayfinder_v3.jsonl", NUDGES["teacher"]),
            "a1": (C / "rows/a1/sft_a1_grep.jsonl", NUDGES["teacher"]),
            "v3line": (C / "rows/v3line/sft_v3_grep.jsonl", NUDGES["f1"])}
    raw = {k: load(p) for k, (p, _) in sets.items()}
    out: dict = {"candidate": a.candidate, "validation": {k: healthy(k, raw[k], sets[k][1]) for k in raw}}
    ok = {k: T.ok_of(v) for k, v in raw.items()}
    out["summary"] = {k: {m: T.mean_ci(T.per_task(rs, f)) for m, f in METRICS.items()} for k, rs in ok.items() if rs}
    out["contrasts"] = {}
    for other in ("v3", "a1", "v3line"):
        if ok["candidate"] and ok[other]:
            out["contrasts"][f"candidate_minus_{other}"] = {m: T.contrast(ok["candidate"], ok[other], f, REPO_OF) for m, f in METRICS.items()}
    if ok["v3"] and ok["v3line"]:
        out["contrasts"]["v3line_minus_v3"] = {m: T.contrast(ok["v3line"], ok["v3"], f, REPO_OF) for m, f in METRICS.items()}
    prim = out["contrasts"].get("candidate_minus_v3", {}).get("recall_at_5")
    allv = all(v["healthy"] for v in out["validation"].values())
    out["primary"] = {"metric": "recall_at_5, candidate minus v3", **(prim or {}), "all_cells_healthy": allv,
                      "holds": bool(prim and allv and prim["ci95_task"][0] > 0 and prim["ci95_repo_cluster"][0] > 0)}
    (C / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["validation"], indent=1))
    names = list(METRICS)
    print("| metric | " + " | ".join(f"{k}" for k in out["summary"]) + " |\n|---|" + "---|" * len(out["summary"]))
    for m in names:
        print(f"| {m} | " + " | ".join(f"{s[m]['mean']:.3f} [{s[m]['ci95'][0]:.3f}, {s[m]['ci95'][1]:.3f}]" for s in out["summary"].values()) + " |")
    for k, c in out["contrasts"].items():
        print(f"\n{k} (task CI | repo-cluster CI)")
        for m in names:
            x = c[m]
            if x.get("delta") is None:
                continue
            print(f"  {m:18s} {x['delta']:+.3f} [{x['ci95_task'][0]:+.3f}, {x['ci95_task'][1]:+.3f}] | [{x['ci95_repo_cluster'][0]:+.3f}, {x['ci95_repo_cluster'][1]:+.3f}]")
    print("\nPRIMARY:", json.dumps(out["primary"]))


if __name__ == "__main__":
    main()
