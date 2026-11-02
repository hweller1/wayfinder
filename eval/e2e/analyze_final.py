"""Analysis per the end-to-end final study plan (not in this release) (+Addendum 1). python3 eval/e2e/analyze_final.py clean|v4 -> runs/final/results_<set>.json
Unit = task (seeds 2,3 averaged). Primary: each subagent minus none, exact paired sign-flip (two-sided, on task mean diffs), task bootstrap 95% CI, Holm over 4.
Secondary (descriptive, MDE at achieved n, alpha .05 power .8 paired): v3-a1, sonnet-v3, gpt-v3, oracle minus each. Gold recall of handed files, pass by all-gold handed."""
import json, sys, itertools, random, math, glob
from pathlib import Path
import statistics as st
HERE = Path(__file__).resolve().parent
which = sys.argv[1]
CONDS = ["none", "oracle", "v3", "a1", "sonnet", "gpt"]; SUB = ["v3", "a1", "sonnet", "gpt"]
head = json.load(open(HERE / f"final_ids_{which}_headroom.json"))
man = [t for p in ([HERE / "tasks_v4.json"] if which == "v4" else sorted((HERE / "data").glob("tasks_clean_v1*.json"))) for t in json.load(open(p))["tasks"]]
slug = {t["task_id"]: t["slug"] for t in man}; lang = {t["task_id"]: t["language"] for t in man}
R = {}
for tid in head:
    for c in CONDS:
        for s in (2, 3):
            p = HERE / f"runs/final/{which}/issue/{c}/{slug[tid]}_s{s}.json"
            if p.exists():
                R[(tid, c, s)] = json.load(open(p))
tasks = [t for t in head if all((t, c, s) in R for c in CONDS for s in (2, 3))]
def pr(t, c): return st.mean(1.0 if R[(t, c, s)]["pass"] else 0.0 for s in (2, 3))
M = {c: [pr(t, c) for t in tasks] for c in CONDS}
n = len(tasks); rng = random.Random(0)
def boot(d, B=10000):
    xs = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(B)); return xs[int(.025 * B)], xs[int(.975 * B)]
def signflip(d):
    nz = [x for x in d if x != 0]
    if not nz: return 1.0
    obs = abs(sum(d))
    if len(nz) <= 20:
        cnt = sum(abs(sum(s * x for s, x in zip(sg, nz))) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=len(nz))); return cnt / 2 ** len(nz)
    cnt = sum(abs(sum(rng.choice((1, -1)) * x for x in nz)) >= obs - 1e-12 for _ in range(100000)); return (cnt + 1) / 100001
def mde(d):
    if len(d) < 2: return None
    sd = st.stdev(d)
    return round((1.96 + 0.8416) * sd / math.sqrt(len(d)), 3)
def contrast(a, b):
    d = [x - y for x, y in zip(M[a], M[b])]
    if not d: return None
    lo, hi = boot(d); return dict(diff=round(st.mean(d), 3), ci=[round(lo, 3), round(hi, 3)], p=round(signflip(d), 4), mde=mde(d), n=len(d))
out = dict(set=which, n_tasks=n, languages={l: sum(lang[t] == l for t in tasks) for l in set(lang[t] for t in tasks)}, headroom_listed=len(head), complete=n, tasks=[slug[t] for t in tasks])
out["pass_rate"] = {c: dict(rate=round(st.mean(M[c]), 3) if n else None, runs=f"{int(sum(1 for t in tasks for s in (2,3) if R[(t,c,s)]['pass']))}/{2*n}",
                            ci=[round(x, 3) for x in boot(M[c])] if n else None) for c in CONDS}
prim = {c: contrast(c, "none") for c in SUB} if n else {}
ps = sorted(((v["p"], c) for c, v in prim.items())); m = len(ps); run_max = 0
for i, (p, c) in enumerate(ps):
    run_max = max(run_max, min(1.0, (m - i) * p)); prim[c]["p_holm"] = round(run_max, 4)
out["primary_vs_none"] = prim
out["secondary"] = {f"{a}-{b}": contrast(a, b) for a, b in [("v3", "a1"), ("sonnet", "v3"), ("gpt", "v3")] + [("oracle", c) for c in SUB]} if n else {}
rec = {}
for c in SUB:
    rs = [R[(t, c, s)]["retrieval"] for t in tasks for s in (2, 3)]
    rec[c] = dict(gold_recall_handed_mean=round(st.mean(r["recall"] for r in rs if r["recall"] is not None), 3),
                  all_gold_handed_frac=round(st.mean(1.0 if r["all_gold_found"] else 0.0 for r in rs if r["all_gold_found"] is not None), 3))
    for k, lab in ((True, "pass_when_all_gold"), (False, "pass_when_missing_gold")):
        sel = [R[(t, c, s)] for t in tasks for s in (2, 3) if R[(t, c, s)]["retrieval"]["all_gold_found"] is k]
        rec[c][lab] = f"{sum(1 for r in sel if r['pass'])}/{len(sel)}"
out["handed_files"] = rec
out["cost"] = {c: dict(coder_usd=round(sum(R[(t, c, s)].get("coder_usd") or 0 for t in tasks for s in (2, 3)), 2)) for c in CONDS}
out["by_language"] = {l: {c: round(st.mean(pr(t, c) for t in tasks if lang[t] == l), 3) for c in CONDS} for l in out["languages"]}
fn = HERE / f"final_funnel_{which}.json"
if fn.exists(): out["funnel_by_language"] = json.load(open(fn))
json.dump(out, open(HERE / f"runs/final/results_{which}.json", "w"), indent=1)
import re as _re
# display-only: condition key `v3` is shown as wayfinder-glimmer; row keys and files keep `v3`
print(_re.sub(r'(?<![\w/.])v3(?![\w/])', "wayfinder-glimmer", json.dumps({k: out[k] for k in ("n_tasks", "pass_rate", "primary_vs_none", "secondary")}, indent=1)))
