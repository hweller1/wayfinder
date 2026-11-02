"""Score SFT v3 against untrained Glimmer on the PRD section 15 criteria (paired task-bootstrap CIs) and paired against v2b.
v1, v2b and the teacher are reference columns.

    python3 scripts/score_sft_v3.py   -> artifacts/sft_v3/summary.json (+ markdown table on stdout)

Rows: artifacts/sft_v3/sft_v3_<arm>.jsonl (arm grep|sem); v2b rows from artifacts/sft_v2. Criteria on the grep arm, as pre-registered.
Reuses the metric definitions of scripts/score_sft_v2.py unchanged.
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sft_v2 import METRICS, deltas, ok_rows, prd15, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def validate(path: Path, model: str, n_rows: int = 100, n_tasks: int = 50) -> dict:
    allr = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    ok = [r for r in allr if not r.get("error") and not r.get("infra_error")]
    return {"file": path.name, "rows": len(allr), "ok_rows": len(ok), "errors": len(allr) - len(ok),
            "tasks": len({r["task_id"] for r in ok}), "models": sorted({r.get("model", "?") for r in ok}),
            "turns_min_med_max": [min(r["turns"] for r in ok), sorted(r["turns"] for r in ok)[len(ok) // 2], max(r["turns"] for r in ok)],
            "wall_s_med": sorted(r["wall_s"] for r in ok)[len(ok) // 2],
            "healthy": len(allr) == n_rows and len(ok) == n_rows and len({r["task_id"] for r in ok}) == n_tasks and set(r.get("model") for r in ok) == {model}}


def main() -> None:
    sd = ROOT / "artifacts/sft_v3"
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    teacher = ok_rows(ROOT / "artifacts/teacher_test/stageB_selfhost.jsonl")
    arms = {"v1_grep": ok_rows(ROOT / "artifacts/sft_v1/sft_v1_grep.jsonl"),
            "v2b_grep": ok_rows(ROOT / "artifacts/sft_v2/sft_v2b_grep.jsonl"), "v2b_sem": ok_rows(ROOT / "artifacts/sft_v2/sft_v2b_sem.jsonl"),
            "v3_grep": ok_rows(sd / "sft_v3_grep.jsonl"), "v3_sem": ok_rows(sd / "sft_v3_sem.jsonl")}
    out: dict = {"base": summarize(base), "teacher_stageB": summarize(teacher),
                 "validation": {a: validate(sd / f"sft_v3_{a}.jsonl", "wayfinder-v3") for a in ("grep", "sem") if (sd / f"sft_v3_{a}.jsonl").exists()}}
    for name, rows in arms.items():
        if rows:
            out[name] = summarize(rows)
    for name in ("v3_grep", "v3_sem", "v2b_grep", "v2b_sem"):
        if arms[name]:
            out[f"{name}_minus_base"] = deltas(arms[name], base)
    out["v3_grep_prd15"] = prd15(out["v3_grep"], out["v3_grep_minus_base"], out["base"])
    out["v3_grep_minus_teacher"] = deltas(arms["v3_grep"], teacher)
    out["v3_grep_minus_v2b_grep"] = deltas(arms["v3_grep"], arms["v2b_grep"])
    if arms["v3_sem"]:
        out["v3_sem_minus_v2b_sem"] = deltas(arms["v3_sem"], arms["v2b_sem"])
        out["v3_sem_minus_grep"] = deltas(arms["v3_sem"], arms["v3_grep"])
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "summary.json").write_text(json.dumps(out, indent=2))
    cols = [("base", out["base"])] + [(k, out[k]) for k in ("v1_grep", "v2b_grep", "v3_grep", "v2b_sem", "v3_sem") if k in out] + [("teacher", out["teacher_stageB"])]

    def cell(s: dict, k: str) -> str:
        x = s[k]
        return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    print("| metric | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in METRICS:
        print(f"| {k} | " + " | ".join(cell(s, k) for _, s in cols) + " |")
    print("validation", json.dumps(out["validation"], indent=1))
    print("v3_grep_prd15", json.dumps(out["v3_grep_prd15"], indent=1))
    for k in ("v3_grep_minus_v2b_grep", "v3_sem_minus_v2b_sem", "v3_sem_minus_grep", "v3_grep_minus_teacher"):
        if k in out:
            print(k)
            for m, d in out[k].items():
                print(f"  {m}: {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
