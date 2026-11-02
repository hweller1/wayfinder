"""Score the OOD40 monitor of a GRPO checkpoint against v3 (Run B rows, same 40 tasks, seed 0): recall@5, reported recall, n_reported, precision, paired deltas.
    python3 scripts/score_grpo_ood.py <run> <step>      Monitor only: never select on it (Run B already used the 120-task set)."""
from __future__ import annotations

from display_names import disp, print  # noqa: E402,A004 - display-only relabel (v3 -> wayfinder-glimmer)
import glob
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_glimmer_baseline import boot, paired, per_task  # noqa: E402
from score_grpo import F1_METRICS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("recall_at_5", "recall_reported", "n_reported", "precision")


def main() -> None:
    run, step = sys.argv[1], int(sys.argv[2])
    ids = set(json.load(open(ROOT / "artifacts/grpo/ood40_ids.json")))
    for arm in ("grep", "sem"):
        rows = []
        for f in glob.glob(str(ROOT / f"artifacts/grpo/{run}/ood_s{step}/shard*/sft_grpo-{run}-s{step}_{arm}.jsonl")):
            rows += [json.loads(x) for x in open(f) if x.strip()]
        ok = [r for r in rows if not r.get("error") and not r.get("infra_error")]
        ref = [json.loads(x) for x in gzip.open(ROOT / f"artifacts/ood_langs/rows/v3_{arm}.jsonl.gz", "rt") if x.strip()]
        ref = [r for r in ref if r["task_id"] in ids and r.get("seed") == 0 and not r.get("error")]
        print(f"\n### OOD40 {run} s{step} {arm}: rows {len(rows)} ok {len(ok)} tasks {len({r['task_id'] for r in ok})} (v3 ref rows {len(ref)})")
        if not ok:
            continue
        print("| metric | v3 | " + run + f" s{step} | minus v3 |\n|---|---|---|---|")
        for k in KEYS:
            f = F1_METRICS[k]
            m, lo, hi = boot(per_task(ok, f))
            rm = boot(per_task(ref, f))[0]
            d = paired(per_task(ok, f), per_task(ref, f))
            print(f"| {k} | {rm:.3f} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")


if __name__ == "__main__":
    main()
