"""Compact live status of GRPO arms: one line per logged step, apps alive, spend estimate.   python3 scripts/grpo/status.py [a b c d]"""
import json
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[2] / "artifacts" / "grpo"
for run in (sys.argv[1:] or ["a", "b", "c", "d"]):
    p = G / run / "steps.jsonl"
    if not p.exists():
        print(f"{run}: no steps yet")
        continue
    print(f"== arm {run}")
    for ln in p.read_text().splitlines():
        r = json.loads(ln)
        ro, tr = r["rollout"], r["train"]
        f = lambda x, n=3: "-" if x is None else round(x, n)  # noqa: E731
        print(f" s{r['step']:>2} v{r['version_sampled']} R {f(ro['reward'])} rec {f(ro['recall_reported'])} ag {f(ro['all_gold'], 2)} rep {f(ro['report_rate'], 2)} turns {f(ro['turns'], 1)} "
              f"prec {f(ro['precision'], 2)} nrep {f(ro['n_reported'], 1)} kept {ro['kept_groups']}/{ro['groups']} | kl {f(tr.get('kl'), 5)} clip {f(tr.get('clip_frac'), 4)} "
              f"gn {f(tr.get('gnorm'))} train {f(tr.get('train_s'), 0)}s | t {r['t_elapsed_s'] // 60}m ${r['est_usd']:.0f}")
