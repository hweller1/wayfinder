"""Paired task-bootstrap comparison of G4 (adapter-served) vs Modal (merged-served) eval rows for one checkpoint, same metrics/scorer as score_grpo.py.
usage: python3 scripts/grpo/g4_compare.py <run> <step> <g4_root_with_shard_dirs> [dev|ood]"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from score_grpo import rec_k, reported  # noqa: E402
from score_glimmer_baseline import boot, paired, per_task  # noqa: E402

run, step, g4root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else "dev"
M = {"recall@5": lambda r: rec_k(r, 5), "recall_reported": lambda r: r["recall_reported"], "n_reported": lambda r: len(reported(r)), "turns": lambda r: r["turns"]}
def load(pat): return [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f) if l.strip()]
for arm in ("grep", "sem"):
    mod = load(f"artifacts/grpo/{run}/{'eval' if kind == 'dev' else 'ood'}_s{step}/shard*/sft_grpo-{run}-s{step}_{arm}.jsonl")
    g4 = load(f"{g4root}/shard*/sft_grpo-{run}-s{step}_{arm}.jsonl")
    mod = [r for r in mod if not r.get("error")]; g4 = [r for r in g4 if not r.get("error")]
    print(f"\n## {arm}: modal rows {len(mod)} ({len({r['task_id'] for r in mod})} tasks), g4 rows {len(g4)} ({len({r['task_id'] for r in g4})} tasks)")
    print(f"{'metric':16s} {'modal':>8s} {'g4':>8s} {'g4-modal':>9s} {'95% CI':>20s} {'n_tasks':>7s}")
    for name, f in M.items():
        a, b = per_task(mod, f), per_task(g4, f)
        ks = set(a) & set(b); a = {k: a[k] for k in ks}; b = {k: b[k] for k in ks}
        d = paired(b, a)
        print(f"{name:16s} {sum(a.values())/len(a):8.3f} {sum(b.values())/len(b):8.3f} {d[0]:+9.3f} [{d[1]:+.3f}, {d[2]:+.3f}] {d[3]:7d}  {'within CI' if d[1] <= 0 <= d[2] else 'OUTSIDE CI'}")
