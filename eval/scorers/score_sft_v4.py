"""Score SFT v4 against untrained Glimmer on the PRD section 15 criteria (paired task-bootstrap CIs) and paired against v3.

    python3 scripts/score_sft_v4.py   -> artifacts/sft_v4/summary.json (+ markdown tables on stdout)

Both v4 and v3 are the retain-reasoning protocol (WF_EVAL_RETAIN=1, the default since commit 22b2bcb). Rows:
  v4 grep/sem  : artifacts/sft_v4/sft_v4_{grep,sem}.jsonl (concatenation of the shard outputs, see `merge_shards`)
  v3 retain grep: artifacts/opd/round1/eval_d/sft_v3_grep.jsonl (OPD agent's v3 retain arm, identical config, checked in this script)
  v3 retain sem : artifacts/sft_v4/sft_v3r_sem.jsonl (re-run for this comparison, sharded)
  v3 (retain off, original eval) and base/teacher (retain off) are reference columns.
Criteria are on the grep arm, as pre-registered. Metric definitions come from scripts/score_sft_v2.py unchanged.
"""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sft_v2 import METRICS, deltas, ok_rows, prd15, summarize  # noqa: E402
from score_sft_v3 import validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts/sft_v4"


def merge_shards(prefix: str, arm: str, variant: str, dst: Path) -> None:
    rows: list[str] = []
    for d in sorted(SD.glob(f"{prefix}_shard*")):
        f = d / f"sft_{variant}_{arm}.jsonl"
        if f.exists():
            rows += [ln for ln in f.read_text().splitlines() if ln.strip()]
    if rows:
        dst.write_text("\n".join(rows) + "\n")


def main() -> None:
    for arm in ("grep", "sem"):
        merge_shards(f"v4_{arm}", arm, "v4", SD / f"sft_v4_{arm}.jsonl")
    merge_shards("v3_sem", "sem", "v3", SD / "sft_v3r_sem.jsonl")
    v3r_grep_path = ROOT / "artifacts/opd/round1/eval_d/sft_v3_grep.jsonl"
    meta = json.loads(v3r_grep_path.with_suffix(".meta.json").read_text())
    assert meta["retain_reasoning"] is True and meta["cfg"]["max_turns"] == 40 and meta["cfg"]["max_seconds"] == 3600 and meta["cfg"]["temperature"] == 0.85
    base = ok_rows(ROOT / "artifacts/baseline/glimmer_dev.jsonl")
    teacher = ok_rows(ROOT / "artifacts/teacher_test/stageB_selfhost.jsonl")
    arms = {"v4_grep": ok_rows(SD / "sft_v4_grep.jsonl"), "v4_sem": ok_rows(SD / "sft_v4_sem.jsonl"),
            "v3r_grep": ok_rows(v3r_grep_path), "v3r_sem": ok_rows(SD / "sft_v3r_sem.jsonl"),
            "v3_grep": ok_rows(ROOT / "artifacts/sft_v3/sft_v3_grep.jsonl"), "v3_sem": ok_rows(ROOT / "artifacts/sft_v3/sft_v3_sem.jsonl")}
    out: dict = {"base": summarize(base), "teacher_stageB": summarize(teacher), "validation": {}}
    for key, path, model in (("v4_grep", SD / "sft_v4_grep.jsonl", "wayfinder-v4"), ("v4_sem", SD / "sft_v4_sem.jsonl", "wayfinder-v4"),
                             ("v3r_grep", v3r_grep_path, "wayfinder-v3"), ("v3r_sem", SD / "sft_v3r_sem.jsonl", "wayfinder-v3")):
        if path.exists():
            out["validation"][key] = validate(path, model)
            rr = {r.get("retain_reasoning") for r in map(json.loads, path.read_text().splitlines()) if True}
            out["validation"][key]["retain_reasoning_values"] = sorted(map(str, rr))
    for name, rows in arms.items():
        if rows:
            out[name] = summarize(rows)
    for name in ("v4_grep", "v4_sem", "v3r_grep", "v3r_sem"):
        if arms[name]:
            out[f"{name}_minus_base"] = deltas(arms[name], base)
    if arms["v4_grep"]:
        out["v4_grep_prd15"] = prd15(out["v4_grep"], out["v4_grep_minus_base"], out["base"])
        out["v4_grep_minus_teacher"] = deltas(arms["v4_grep"], teacher)
        out["v4_grep_minus_v3r_grep"] = deltas(arms["v4_grep"], arms["v3r_grep"])
    if arms["v3r_grep"]:
        out["v3r_grep_prd15"] = prd15(out["v3r_grep"], out["v3r_grep_minus_base"], out["base"])
    if arms["v4_sem"] and arms["v3r_sem"]:
        out["v4_sem_minus_v3r_sem"] = deltas(arms["v4_sem"], arms["v3r_sem"])
    if arms["v4_sem"] and arms["v4_grep"]:
        out["v4_sem_minus_grep"] = deltas(arms["v4_sem"], arms["v4_grep"])
    SD.mkdir(parents=True, exist_ok=True)
    (SD / "summary.json").write_text(json.dumps(out, indent=2))
    cols = [("base", out["base"])] + [(k, out[k]) for k in ("v3_grep", "v3r_grep", "v4_grep", "v3_sem", "v3r_sem", "v4_sem") if k in out] + [("teacher", out["teacher_stageB"])]

    def cell(s: dict, k: str) -> str:
        x = s[k]
        return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    print("| metric | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in METRICS:
        print(f"| {k} | " + " | ".join(cell(s, k) for _, s in cols) + " |")
    print("validation", json.dumps(out["validation"], indent=1))
    for k in ("v4_grep_prd15", "v3r_grep_prd15"):
        if k in out:
            print(k, json.dumps(out[k], indent=1))
    for k in ("v4_grep_minus_v3r_grep", "v4_sem_minus_v3r_sem", "v4_sem_minus_grep", "v4_grep_minus_teacher", "v4_grep_minus_base", "v4_sem_minus_base"):
        if k in out:
            print(k)
            for m, d in out[k].items():
                print(f"  {m}: {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
