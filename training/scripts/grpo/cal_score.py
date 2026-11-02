"""Score the round-2 calibration arms against the pre-registered rule (docs/grpo_r2_plan.md). usage: python3 scripts/grpo/cal_score.py [cal_lo cal_hi]
Reads artifacts/grpo/<run>/steps.jsonl and rollouts.jsonl (pull first: modal volume get --force wayfinder-grpo /<run>/driver/<file> artifacts/grpo/<run>/)."""
from __future__ import annotations

import collections
import json
import random
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_BOOT, SEED = 2000, 0


def slope(xs: list[int], ys: list[float]) -> float:
    mx, my = S.mean(xs), S.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def score(run: str) -> dict:
    d = ROOT / "artifacts/grpo" / run
    steps = [json.loads(ln) for ln in (d / "steps.jsonl").read_text().splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in (d / "rollouts.jsonl").read_text().splitlines() if ln.strip()]
    # a restarted driver (cal_lo, 2026-10-29) appends its steps again: keep the LAST record of each step, and the last n_rollouts rollout rows of each step
    last = {s["step"]: s for s in steps}
    steps = [last[k] for k in sorted(last)]
    keep = []
    for k, s in last.items():
        rs = [r for r in rows if r["step"] == k]
        keep += rs[-s["rollout"]["n_rollouts"] - s["rollout"].get("n_infra_errors", 0):]
    rows = keep
    by = collections.defaultdict(lambda: collections.defaultdict(list))  # task -> step -> [(reward)]
    for r in rows:
        if r.get("infra_error") or r["step"] > 9:
            continue
        by[r["task_id"]][r["step"]].append(r)
    # the trainer saw arm_reward = reward + tiebreak; rollouts.jsonl stores the worker's base reward, so use steps.jsonl (rollout.reward) for the batch mean and the per-task
    # base reward (+ tie-break recomputed) for the bootstrap
    from wayfinder.grpo.driver import apply_tiebreak

    per = collections.defaultdict(dict)
    for k in sorted({r["step"] for t in by.values() for r in [x for v in t.values() for x in v]}):
        step_rows = [r for t in by.values() for r in t.get(k, [])]
        for r in step_rows:
            r["arm_reward"] = r["reward"]
        apply_tiebreak(step_rows)
        for tid, t in by.items():
            if t.get(k):
                per[tid][k] = S.mean(r["arm_reward"] for r in t[k])
    ks = sorted({k for t in per.values() for k in t})
    full = [k for k in ks if all(k in per[t] for t in per)]
    means = [S.mean(per[t][k] for t in per) for k in full]
    sl = slope(full, means)
    rng, tasks, boots = random.Random(SEED), list(per), []
    for _ in range(N_BOOT):
        smp = [rng.choice(tasks) for _ in tasks]
        boots.append(slope(full, [S.mean(per[t][k] for t in smp) for k in full]))
    boots.sort()
    ci = (boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)])
    tr = [s["train"] for s in steps]
    nrep = [s["rollout"]["n_reported"] for s in steps]
    guard = sum(1 for t in tr if t.get("guard_skipped"))
    state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
    out = {"run": run, "steps": len(steps), "reward_by_step": [round(m, 4) for m in means], "slope": round(sl, 5), "slope_ci95": [round(ci[0], 5), round(ci[1], 5)],
           "abs_logratio": [round(t["abs_logratio"], 4) for t in tr], "kl": [round(t["kl"], 6) for t in tr], "clip_frac": [round(t["clip_frac"], 4) for t in tr],
           "entropy": [round(t.get("entropy", float("nan")), 4) for t in tr], "gnorm": [round(t["gnorm"], 3) for t in tr], "n_reported": nrep, "guard_skipped": guard,
           "stop_reason": state.get("stop_reason"), "est_usd": steps[-1]["est_usd"] if steps else None,
           "kept_groups": [s["rollout"]["kept_groups"] for s in steps], "recall_at_5_train": [s["rollout"]["recall_at_5"] for s in steps]}
    dev = [abs(x - nrep[0]) for x in nrep[1:10]]
    out["n_reported_max_dev"] = round(max(dev), 3) if dev else None
    out["qualifies"] = bool(len(steps) >= 10 and ci[0] > 0 and dev and max(dev) <= 1.0 and guard == 0 and out["stop_reason"] in (None, "completed"))
    ab = out["abs_logratio"]
    if len(ab) >= 6:  # linear fit over steps 2..9 (after warm-up), extrapolated to step 100
        xs = list(range(2, len(ab)))
        s_ = slope(xs, ab[2:])
        out["abs_logratio_step100_linear"] = round(S.mean(ab[2:]) + s_ * (100 - S.mean(xs)), 4)
        out["abs_logratio_step100_flag_gt_0.1"] = out["abs_logratio_step100_linear"] > 0.1
    return out


if __name__ == "__main__":
    for r in (sys.argv[1:] or ["cal_lo", "cal_hi"]):
        print(json.dumps(score(r), indent=1))
