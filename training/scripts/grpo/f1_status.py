"""Compact f1 monitor view: per step train metrics, the padding monitor, guard events, spend.   python3 scripts/grpo/f1_status.py [run]"""
import json
import sys
from pathlib import Path

run = sys.argv[1] if len(sys.argv) > 1 else "f1"
p = Path(__file__).resolve().parents[2] / "artifacts/grpo" / run / "steps.jsonl"
f = lambda x, n=3: "-" if x is None else round(x, n)  # noqa: E731
for ln in p.read_text().splitlines():
    r = json.loads(ln)
    ro, tr, mo = r["rollout"], r["train"], r.get("monitor") or {}
    print(f"s{r['step']:>2} nrep {f(ro['n_reported'], 2)} prec {f(ro['precision'])} r@5 {f(ro['recall_at_5'])} r@7 {f(ro['recall_at_7'])} f1 {f(ro['f1'])} rec {f(ro['recall_reported'])} ag8 {f(ro['all_gold_window'], 2)} "
          f"turns {f(ro['turns'], 1)} R {f(ro['reward'])} | kl {f(tr.get('kl'), 5)} gn {f(tr.get('gnorm'))} lr {tr.get('lr')} skip {tr.get('guard_skipped')} | "
          f"mon d_n {mo.get('d_n_reported', '-')} d_p {mo.get('d_precision', '-')} d_r5 {mo.get('d_recall_at_5', '-')} flag {mo.get('flagged', '-')} | train ${r.get('train_usd')} eval ${r.get('eval_usd')}")
