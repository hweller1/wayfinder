"""Power calculation for the v3 task set: n tasks needed to detect a +0.10 paired pass-rate difference between two search conditions
at alpha 0.05 (two-sided), 80% power, using the round-3 per-task discordance as the variance estimate.

Unit: the task (per-task pass rate = mean over seeds 1-2). Variance: sd of the per-task paired difference between two search conditions,
over the 17 round-3 headroom tasks, for each of the 10 pairs among glimmer_base, a1_35b_scout, spark_teacher, wayfinder_v2a, wayfinder_v2b
(runs/issue/<cond>/*_s[12].json). Normal approximation n = ((z_a + z_b) * sd / delta)^2, plus a t-correction (n -> n + z_a^2/2), checked by
a paired-bootstrap-free Monte Carlo (paired t-test on resampled round-3 differences, mean shifted to delta).
  python3 eval/e2e/power_v3.py  -> eval/e2e/power_v3.json
"""
import glob
import itertools
import json
import math
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = json.load(open(HERE / "experiments/e2e/screen_r3.json"))["headroom"]
C = ["glimmer_base", "a1_35b_scout", "spark_teacher", "wayfinder_v2a", "wayfinder_v2b"]
ZA, ZB, DELTA = 1.959964, 0.841621, 0.10
R: dict = {}
for c in C:
    for f in glob.glob(str(HERE / f"runs/issue/{c}/*_s[12].json")):
        r = json.load(open(f))
        if r["task_id"] in H:
            R.setdefault((r["task_id"], c), []).append(float(bool(r["pass"])))
rate = {k: st.mean(v) for k, v in R.items()}
pairs = {}
for a, b in itertools.combinations(C, 2):
    d = [rate[(t, a)] - rate[(t, b)] for t in H]
    pairs[f"{a}-{b}"] = dict(mean=round(st.mean(d), 3), sd=round(st.stdev(d), 3), n_z=math.ceil(((ZA + ZB) * st.stdev(d) / DELTA) ** 2))
pool_var = st.mean(st.stdev([rate[(t, a)] - rate[(t, b)] for t in H]) ** 2 for a, b in itertools.combinations(C, 2))
sd = pool_var ** 0.5
n_norm = ((ZA + ZB) * sd / DELTA) ** 2
n_t = math.ceil(n_norm + ZA ** 2 / 2)
# per-run (single seed) discordance, for reference: the same tasks, seed treated as the unit
disc = []
for a, b in itertools.combinations(C, 2):
    s = 0
    n = 0
    for t in H:
        for i in range(2):
            x, y = R[(t, a)][i], R[(t, b)][i]
            s += x != y
            n += 1
    disc.append(s / n)
# Monte Carlo power at a given n: draw n differences from the pooled round-3 diffs (centred, shifted to +delta), paired t-test.
pool = []
for a, b in itertools.combinations(C, 2):
    d = [rate[(t, a)] - rate[(t, b)] for t in H]
    m = st.mean(d)
    pool += [x - m for x in d]
T975 = {n: 0 for n in range(2, 400)}


def tcrit(df):
    # two-sided 5% t critical value via a short Newton-free approximation (Cornish-Fisher), good to <0.5% for df>=10
    z = ZA
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def power(n, sims=4000, seed=5):
    rng = random.Random(seed)
    hit = 0
    for _ in range(sims):
        x = [rng.choice(pool) + DELTA for _ in range(n)]
        t = st.mean(x) / (st.stdev(x) / math.sqrt(n))
        hit += abs(t) > tcrit(n - 1)
    return hit / sims


out = dict(delta=DELTA, alpha=0.05, power=0.80, unit="task (mean of seeds 1-2)", n_round3_tasks=len(H), pairs=pairs,
           pooled_sd=round(sd, 4), n_normal=round(n_norm, 1), n_required=n_t, mean_run_discordance=round(st.mean(disc), 3),
           mc_power={str(n): round(power(n), 3) for n in (17, 30, 40, 50, 60, n_t, 70, 80)})
(HERE / "power_v3.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
