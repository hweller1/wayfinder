"""Fit rubric weights (PRD section 7) by logistic regression of E2E coding pass on per-run retrieval features.

Runs in the Modal CPU sandbox (numpy). Inputs under --inputs: screen_r3.json and runs_r2/issue/{35b_grep,35b_index,
sonnet_grep,sonnet_index}/*.json (public E2E records; the no-search and oracle conditions have no retrieval features).

Rounds are NEVER pooled:
  r3  = the pre-registered round: the 17 headroom tasks (screen_r3.json), seeds 1 and 2 only, 4 search conditions
        (n = 136 runs). Seed 0 was the screening seed and is excluded, so the selection effect does not leak in.
  r2  = every other search-condition record in runs_r2 (the pilot: 13 tasks, seed 0 screening runs plus pilot seed 1/2;
        n = 70). Seed 0 picked headroom tasks (none fails, oracle passes), so r2 is selection-biased; it is a sign check only.
Records carry per-run retrieval metrics (recall, all_gold_found, precision, n_retrieved, tool counts, turns, tokens, stop
reason) but not the subagent message list, so routing / output-quality / evidence terms cannot be fit and keep prior weights.

Features (all in [0, 1], standardised before fitting): agf = all_gold_found; rec = recall; focus = 1 - min(1, n_retrieved / 60);
turn_eff = 1 - min(1, turns / 40); tok_eff = 1 - min(1, completion_tokens / 8000); clean = stop_reason == "answered".
Model: L2 logistic (ridge 1.0 on standardised features), Newton; AUC in sample, leave-one-task-out AUC, task-bootstrap 90% CI.
Weights: non-negative refit of {agf, rec, focus} on r3 (drop negative coefficients, refit), proportional to the standardised
coefficients, with a floor of 10% of the fitted block per leaf (dense shaping for RL; AUC cannot separate agf from rec).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

IN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
H = set(json.load(open(IN / "screen_r3.json"))["headroom"])
FITTED_BLOCK = 0.65
FLOOR = 0.10
PRIOR = {  # leaves the E2E records cannot identify; sums with FITTED_BLOCK to 1.0
    "hunk_overlap": 0.05, "first_gold": 0.03, "turn_eff": 0.03, "token_eff": 0.02, "no_rereads": 0.02,
    "stops_in_budget": 0.02, "no_repeated_queries": 0.015, "single_report": 0.015,
    "routing": 0.05, "output_valid": 0.05, "evidence": 0.05,
}
assert abs(FITTED_BLOCK + sum(PRIOR.values()) - 1.0) < 1e-9

rows = []
for f in glob.glob(str(IN / "runs_r2/issue/*/*.json")):
    if f.endswith(("transcript.json", "error.json")):
        continue
    d = json.load(open(f))
    if d["condition"] in ("none", "oracle"):
        continue
    s, r = d["subagent"], d["retrieval"]
    rows.append({
        "task": d["task_id"], "cond": d["condition"], "seed": d["seed"], "y": int(bool(d["pass"])),
        "rnd": "r3" if (d["task_id"] in H and d["seed"] in (1, 2)) else "r2",
        "agf": float(r["all_gold_found"]), "rec": float(r["recall"]),
        "focus": 1 - min(1, r["n_retrieved"] / 60), "turn_eff": 1 - min(1, s["turns"] / 40),
        "tok_eff": 1 - min(1, s["completion_tokens"] / 8000), "clean": float(s["stop_reason"] == "answered"),
        "n_retrieved": r["n_retrieved"], "turns": s["turns"],
    })


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def fit(X, y, l2=1.0):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1)
    P = np.eye(d + 1) * l2
    P[0, 0] = 0
    for _ in range(200):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + P @ w
        Hs = (Xb * (p * (1 - p) + 1e-9)[:, None]).T @ Xb + P
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def pred(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def study(rs, names, seed=0):
    y = np.array([r["y"] for r in rs], float)
    X = np.array([[r[n] for n in names] for r in rs], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = fit(Z, y)
    tasks = sorted({r["task"] for r in rs})
    tid = np.array([r["task"] for r in rs])
    cv = np.zeros(len(rs))
    for t in tasks:
        te = tid == t
        cv[te] = pred(fit(Z[~te], y[~te]), Z[te])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(400):
        idx = np.concatenate([np.where(tid == t)[0] for t in rng.choice(tasks, len(tasks))])
        bs.append(fit(Z[idx], y[idx])[1:])
    lo, hi = np.percentile(bs, [5, 95], axis=0)
    raw = w[1:] / sd
    return {
        "features": list(names), "n": len(rs), "n_tasks": len(tasks), "pass_rate": float(y.mean()),
        "intercept_z": float(w[0]),
        "coef_standardised": {n: round(float(c), 4) for n, c in zip(names, w[1:])},
        "coef_per_unit": {n: round(float(c), 4) for n, c in zip(names, raw)},
        "boot90_standardised": {n: [round(float(a), 3), round(float(b), 3)] for n, a, b in zip(names, lo, hi)},
        "auc_in_sample": round(auc(y, pred(w, Z)), 4), "auc_leave_one_task_out": round(auc(y, cv), 4),
        "pearson_r_with_pass": {n: round(float(np.corrcoef(X[:, i], y)[0, 1]), 3) for i, n in enumerate(names)},
    }


ALL = ["agf", "rec", "focus", "turn_eff", "tok_eff", "clean"]
out = {"schema": "wayfinder-rubric-weights-v1", "method": __doc__, "rounds": {}}
for rnd in ("r3", "r2"):
    rs = [r for r in rows if r["rnd"] == rnd]
    out["rounds"][rnd] = {
        "conditions": {c: sum(r["cond"] == c for r in rs) for c in sorted({r["cond"] for r in rs})},
        "agf_only": study(rs, ["agf"]), "coverage_focus": study(rs, ["agf", "rec", "focus"]), "full": study(rs, ALL),
    }

# ---- weights from r3: non-negative refit of the coverage block ----
rs3 = [r for r in rows if r["rnd"] == "r3"]
active = ["agf", "rec", "focus"]
while True:
    st = study(rs3, active)
    neg = [n for n in active if st["coef_standardised"][n] < 0]
    if not neg:
        break
    active = [n for n in active if n not in neg]
coef = {n: max(0.0, st["coef_standardised"].get(n, 0.0)) for n in ("agf", "rec", "focus")}
tot = sum(coef.values())
share = {n: coef[n] / tot for n in coef}
floored = {n: max(FLOOR, s) for n, s in share.items()}
z = sum(floored.values())
fitted = {n: round(FITTED_BLOCK * v / z, 4) for n, v in floored.items()}
weights = {"all_gold_found": fitted["agf"], "recall": fitted["rec"], "focus": fitted["focus"], **PRIOR}
weights = {k: round(v, 4) for k, v in weights.items()}
weights["all_gold_found"] = round(weights["all_gold_found"] + (1.0 - sum(weights.values())), 4)  # absorb rounding
out["weights"] = weights
out["weights_derivation"] = {
    "source_round": "r3", "nonneg_active_features": active, "nonneg_coef_standardised": st["coef_standardised"],
    "share_before_floor": {k: round(v, 4) for k, v in share.items()}, "floor_per_leaf": FLOOR, "fitted_block_mass": FITTED_BLOCK,
    "prior_leaves": PRIOR, "weights_sum": round(sum(weights.values()), 6),
}
out["pooled_r2_r3"] = "not computed by design"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out["rounds"][k]["coverage_focus"] for k in out["rounds"]}, indent=1))
print(json.dumps(out["weights"], indent=1))
print(json.dumps(out["weights_derivation"], indent=1))
