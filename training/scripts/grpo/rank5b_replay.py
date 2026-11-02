"""Offline replay: rank5 vs rank5b on (a) rk's stored rollouts (artifacts/grpo/rk/rollouts.jsonl, summary rows) and (b) the v3 screen rows (artifacts/rl_pool/screen).
usage: python3 scripts/grpo/rank5b_replay.py [--scales] ; writes docs/research/rank5b_replay.md (markdown to stdout with --stdout)."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics as S
from pathlib import Path

from wayfinder.grpo.advantage import group_advantages
from wayfinder.grpo.driver import reward_terms
from wayfinder.grpo.reward import RB_COEF, TURN_FREE, TURN_PENALTY, rank5b_tiebreak, recall_at_capped

ROOT = Path(__file__).resolve().parents[2]


def load_gold() -> dict[str, set[str]]:
    g = {}
    for f in ("artifacts/rl_pool/existing_tasks.json", "artifacts/rl_pool/new_tasks.json"):
        d = json.loads((ROOT / f).read_text())
        for t in (d["tasks"] if isinstance(d, dict) else d):
            g[t["task_id"]] = set(t["gold_files"])
    return g


def screen_rows(gold):
    out = []
    for f in sorted(glob.glob(str(ROOT / "artifacts/rl_pool/screen/*.jsonl"))):
        for ln in open(f):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("error") or r["task_id"] not in gold:
                continue
            rep = list(dict.fromkeys(r.get("reported_paths") or []))
            g = gold[r["task_id"]]
            out.append({"task_id": r["task_id"], "step": 0, "n_reported": len(rep), "turns": r["turns"], "repeat_penalty": 0.0, "no_report_penalty": 0.0 if rep else 0.5,
                        "valid_report": bool(rep) and not r.get("budget_exhausted"), "r5": recall_at_capped(rep, g, 5), "r8": recall_at_capped(rep, g, 8),
                        "all8": bool(g) and g <= set(rep[:8]), "n_gold5": min(len(g), 5), "n_gold8": min(len(g), 8), "recall_at_5": len(g & set(rep[:5])) / len(g)})
    return out


def rk_rows(gold):
    out = []
    for ln in open(ROOT / "artifacts/grpo/rk/rollouts.jsonl"):
        r = json.loads(ln)
        if r.get("error") or r["task_id"] not in gold or r.get("recall_at_5_capped") is None:
            continue
        out.append({"task_id": r["task_id"], "step": r["step"], "n_reported": r["n_reported"], "turns": r["turns"], "repeat_penalty": r["repeat_penalty"],
                    "no_report_penalty": r["no_report_penalty"], "valid_report": r["valid_report"], "r5": r["recall_at_5_capped"], "r8": r["recall_at_8"],
                    "all8": bool(r["all_gold_window"]), "n_gold5": min(len(gold[r["task_id"]]), 5), "n_gold8": min(len(gold[r["task_id"]]), 8), "recall_at_5": r["recall_at_5"], "rank5_reward": r["reward"]})
    return out


def rank5(r):
    return r["r5"] + 0.25 * r["r8"] + 0.5 * float(r["all8"]) - TURN_PENALTY * max(0, r["turns"] - TURN_FREE) - r["repeat_penalty"] - r["no_report_penalty"]


def rank5b(r):
    return RB_COEF["r5"] * r["r5"] + RB_COEF["r8"] * r["r8"] + RB_COEF["ag"] * float(r["all8"]) - r["no_report_penalty"]


def terms5(r):
    return {"r5": r["r5"], "r8": 0.25 * r["r8"], "ag": 0.5 * float(r["all8"]), "turn": -TURN_PENALTY * max(0, r["turns"] - TURN_FREE), "repeat": -r["repeat_penalty"], "no_report": -r["no_report_penalty"]}


def terms5b(r):
    return {"r5": RB_COEF["r5"] * r["r5"], "r8": RB_COEF["r8"] * r["r8"], "ag": RB_COEF["ag"] * float(r["all8"]), "no_report": -r["no_report_penalty"], "tiebreak": r.get("tb", 0.0)}


def share(groups, kind):
    acc, n = collections.defaultdict(float), 0
    for g in groups:
        T = [(terms5 if kind == 5 else terms5b)(r) for r in g]
        tot = [sum(t.values()) for t in T]
        m = sum(tot) / len(tot)
        sd = math.sqrt(sum((x - m) ** 2 for x in tot) / len(tot))
        if sd < 1e-3:
            continue
        n += 1
        for k in {k for t in T for k in t}:
            xs = [t.get(k, 0.0) for t in T]
            mk = sum(xs) / len(xs)
            acc[k] += sum(abs(x - mk) for x in xs) / len(xs) / max(sd, 0.1)
    z = sum(acc.values()) or 1.0
    return {k: v / z for k, v in sorted(acc.items())}, n


def analyse(rows, label):
    gs = collections.defaultdict(list)
    for r in rows:
        gs[(r["step"], r["task_id"])].append(r)
    groups = [g for g in gs.values() if len(g) >= 4]
    for g in groups:
        tb = rank5b_tiebreak([{"valid_report": r["valid_report"], "recall_at_5_capped": r["r5"], "recall_at_8": r["r8"], "n_reported": r["n_reported"], "turns": r["turns"],
                               "n_gold5": r["n_gold5"], "n_gold8": r["n_gold8"]} for r in g])
        for r, b in zip(g, tb):
            r["tb"] = b
            r["R5"], r["R5b0"], r["R5b"] = rank5(r), rank5b(r), rank5b(r) + b

    def sd(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    out = {"label": label, "n_groups": len(groups), "n_rollouts": sum(len(g) for g in groups)}
    for name, key in (("rank5", "R5"), ("rank5b", "R5b"), ("rank5b without tie-break", "R5b0")):
        s = [sd([r[key] for r in g]) for g in groups]
        out[name] = {"signal(std>=1e-3)": sum(x >= 1e-3 for x in s), "std>=0.1": sum(x >= 0.1 for x in s), "mean_std": S.mean(s)}
    # advantage-level view: groups kept, and mean |adv|
    for name, key in (("rank5", "R5"), ("rank5b", "R5b")):
        a = [abs(x) for g in groups for x in group_advantages([r[key] for r in g], [True] * len(g), [None] * len(g), lam=0.0)["adv"]]
        out[name]["mean_abs_adv"] = S.mean(a)
    out["share5"], out["n_share5"] = share(groups, 5)
    out["share5b"], out["n_share5b"] = share(groups, "b")
    tb_only = sum(1 for g in groups if sd([r["R5b0"] for r in g]) < 1e-3 and sd([r["R5b"] for r in g]) >= 1e-3)
    out["tiebreak_only_groups"] = tb_only
    # pairs with equal recall (r5, r8): does a padded / truncated report outrank a cleaner one?
    pad = collections.Counter()
    for g in groups:
        for a in g:
            for b in g:
                if a is b or (a["r5"], a["r8"]) != (b["r5"], b["r8"]) or not a["valid_report"] or not b["valid_report"]:
                    continue
                fa, fb = max(a["n_reported"], a["n_gold5"]), max(b["n_reported"], b["n_gold5"])
                if fb > fa:  # b is padded relative to a (file counts floored at min(|gold|,5))
                    pad["pairs_padded_vs_clean"] += 1
                    pad["rank5_padded_above_clean"] += b["R5"] > a["R5"] + 1e-9
                    pad["rank5b_padded_above_clean"] += b["R5b"] > a["R5b"] + 1e-9
                if a["n_reported"] < a["n_gold5"] <= b["n_reported"]:  # a is shorter than min(|gold|,5), b is not; same recall
                    pad["pairs_truncated_vs_full"] += 1
                    pad["rank5_truncated_above_full"] += a["R5"] > b["R5"] + 1e-9
                    pad["rank5b_truncated_above_full"] += a["R5b"] > b["R5b"] + 1e-9
                    if b["n_reported"] == b["n_gold5"]:  # b is exactly at the floor: truncation below it must buy nothing unless a also used fewer turns
                        pad["pairs_truncated_vs_floor"] += 1
                        pad["rank5b_truncated_above_floor_with_turns_not_lower"] += (a["turns"] >= b["turns"]) and a["R5b"] > b["R5b"] + 1e-9
                        pad["rank5_truncated_above_floor_with_turns_not_lower"] += (a["turns"] >= b["turns"]) and a["R5"] > b["R5"] + 1e-9
    out["pairs"] = dict(pad)
    # does rank5b invert a pure-recall ordering? (a has strictly higher r5 + r8, lower reward)
    inv = tot = 0
    for g in groups:
        for a in g:
            for b in g:
                if a["r5"] + a["r8"] > b["r5"] + b["r8"] + 1e-9:
                    tot += 1
                    inv += a["R5b"] <= b["R5b"] + 1e-9
    out["recall_order_inversions_rank5b"] = (inv, tot)
    inv5 = sum(1 for g in groups for a in g for b in g if a["r5"] + a["r8"] > b["r5"] + b["r8"] + 1e-9 and a["R5"] <= b["R5"] + 1e-9)
    out["recall_order_inversions_rank5"] = (inv5, tot)
    return out


def scales(gold):
    sd = [[], [], []]
    gs = collections.defaultdict(list)
    for r in screen_rows(gold):
        gs[r["task_id"]].append(r)
    for g in gs.values():
        if len(g) < 8:
            continue
        for k, key in enumerate(("r5", "r8", "all8")):
            sd[k].append(S.pstdev([float(r[key]) for r in g]))
    return {n: {"groups": len(x), "mean_sd": round(S.mean(x), 4), "rms_sd": round(math.sqrt(S.mean(v * v for v in x)), 4), "frac_varying": round(sum(v > 0 for v in x) / len(x), 3)}
            for n, x in zip(("r5", "r8", "ag"), sd)}


def fmt(o):
    L = [f"### {o['label']}", "", f"{o['n_groups']} groups ({o['n_rollouts']} rollouts, groups of at least 4).", "",
         "| reward | groups with signal (std >= 1e-3) | groups with std >= 0.1 | mean group std | mean abs advantage |", "|---|---|---|---|---|"]
    for k in ("rank5", "rank5b without tie-break", "rank5b"):
        v = o[k]
        L.append(f"| {k} | {v['signal(std>=1e-3)']} | {v['std>=0.1']} | {v['mean_std']:.3f} | {v.get('mean_abs_adv', float('nan')):.3f} |")
    L += ["", f"Groups whose only signal is the tie-break: {o['tiebreak_only_groups']}.", "", "Share of the advantage by term (mean |term advantage| over groups with signal):", "",
          "| term | rank5 | rank5b |", "|---|---|---|"]
    for k in sorted(set(o["share5"]) | set(o["share5b"])):
        L.append(f"| {k} | {o['share5'].get(k, 0):.1%} | {o['share5b'].get(k, 0):.1%} |")
    p = o["pairs"]
    L += ["", "Pairs inside a group with equal recall@5 and recall@8 (file counts floored at min(|gold|,5)):", "",
          f"- padded report vs cleaner report: {p.get('pairs_padded_vs_clean', 0)} pairs; ranked padded-above-clean by rank5 {p.get('rank5_padded_above_clean', 0)}, by rank5b {p.get('rank5b_padded_above_clean', 0)}",
          f"- report truncated below min(|gold|,5) vs a full-length report: {p.get('pairs_truncated_vs_full', 0)} pairs; truncated ranked above by rank5 {p.get('rank5_truncated_above_full', 0)}, by rank5b {p.get('rank5b_truncated_above_full', 0)} (a floor-length report is also ranked above longer ones: intended)",
          f"- truncated vs a report exactly at min(|gold|,5) files ({p.get('pairs_truncated_vs_floor', 0)} pairs), truncated ranked above with turns not lower: rank5 {p.get('rank5_truncated_above_floor_with_turns_not_lower', 0)}, rank5b {p.get('rank5b_truncated_above_floor_with_turns_not_lower', 0)}",
          f"- pairs where recall (r5 + r8) is strictly higher but the reward is not: rank5 {o['recall_order_inversions_rank5'][0]} / {o['recall_order_inversions_rank5'][1]}, rank5b {o['recall_order_inversions_rank5b'][0]} / {o['recall_order_inversions_rank5b'][1]}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    gold = load_gold()
    if a.scales:
        print(json.dumps(scales(gold), indent=1))
        return
    res = [analyse(rk_rows(gold), "rk rollouts, steps 0-22 (what the driver stored)"), analyse(screen_rows(gold), "v3 screen rows, 1,272 tasks x 8 (repeat penalty unknown, taken as 0)")]
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(fmt(o) for o in res))


if __name__ == "__main__":
    main()
