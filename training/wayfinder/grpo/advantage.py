"""Group advantages with the PAPO rubric term on failures (PRD s6, s14).

Per group (the 8 rollouts of one task):
    A_out[i]    = (r_i - mean(r)) / max(std(r), STD_FLOOR)            group-normalised outcome advantage (STD_FLOOR 0.1, run f1)
    A_rubric[i] = (s_i - mean_F(s)) / (std_F(s) + EPS)  for i in F    rubric score s, normalised over the failures F only
    A[i]        = A_out[i] + LAMBDA * 1[i in F] * A_rubric[i]         LAMBDA = 0.5 by default; run f1 passes 0 (no rubric term)
A failure is a rollout that did not report all gold files (all_gold = 0). Successes carry no rubric term, so the weak rubric
(leave-one-task-out AUC 0.60) can only order failures against each other, never trade against finding the code. Groups whose
rewards have (almost) no variance are dropped: they carry no outcome signal, and the rubric term alone is not allowed to
drive an update (PAPO is an add-on to the outcome advantage). With fewer than 2 failures, or a flat rubric among them,
A_rubric is 0.

The std floor (Dr. GRPO / DAPO motivation): dividing by a tiny group std turns a near-flat group (rewards differing by a few thousandths, i.e.
noise) into advantages of order 1, which is how a group that only differs in the length of its report can dominate a step. With
max(std, 0.1) a group whose rewards spread less than 0.1 gets advantages shrunk in proportion to its spread; groups with std >= 0.1 are
normalised exactly as before. Groups below MIN_STD (1e-3) are still dropped.
"""

from __future__ import annotations

import math
from typing import Any

EPS = 1e-4  # rubric normaliser only
STD_FLOOR = 0.1  # outcome normaliser: max(std, STD_FLOOR)
LAMBDA = 0.5
MIN_STD = 1e-3  # reward std below this = zero-variance group, dropped
A_CLIP = 5.0


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def consistency_clip(adv: list[float], all_gold8: list[bool], zero_recall: list[bool]) -> list[float]:
    """rank5b: advantages agree with correctness. A rollout with every gold file in its top 8 never gets a negative advantage; one with no gold file in its
    top 8 never gets a positive one (a tie-break or a group mean can otherwise push a correct report below the group)."""
    return [max(0.0, a) if g else min(0.0, a) if z else a for a, g, z in zip(adv, all_gold8, zero_recall)]


def group_advantages(rewards: list[float], failures: list[bool], rubric: list[float | None], lam: float = LAMBDA,
                     min_std: float = MIN_STD, clip: float = A_CLIP, std_floor: float = STD_FLOOR, all_gold8: list[bool] | None = None,
                     zero_recall: list[bool] | None = None) -> dict[str, Any]:
    """Returns {"keep": bool, "adv": [...], "a_out": [...], "a_rubric": [...], "std": float}. adv is all zeros when keep is False.
    With all_gold8 and zero_recall (rank5b) the final advantages go through consistency_clip."""
    n = len(rewards)
    m, s = _mean_std(rewards)
    if n < 2 or s < min_std:
        return {"keep": False, "adv": [0.0] * n, "a_out": [0.0] * n, "a_rubric": [0.0] * n, "std": s}
    a_out = [(r - m) / max(s, std_floor) for r in rewards]
    a_rub = [0.0] * n
    idx = [i for i in range(n) if failures[i] and rubric[i] is not None]
    if len(idx) >= 2:
        fm, fs = _mean_std([rubric[i] for i in idx])  # type: ignore[misc]
        if fs > 1e-6:
            for i in idx:
                a_rub[i] = (rubric[i] - fm) / (fs + EPS)  # type: ignore[operator]
    adv = [max(-clip, min(clip, a_out[i] + lam * a_rub[i])) for i in range(n)]
    if all_gold8 is not None and zero_recall is not None:
        adv = consistency_clip(adv, all_gold8, zero_recall)
    return {"keep": True, "adv": adv, "a_out": a_out, "a_rubric": a_rub, "std": s}


def batch_advantages(items: list[dict[str, Any]], group_key: str = "task_id", min_group: int = 4, consistent_clip: bool = False, **kw: Any) -> dict[str, Any]:
    """items: dicts with `reward`, `all_gold` (bool), `rubric` (float or None) and the group key. Fills item["adv"] (0 for dropped)
    and item["keep"]. consistent_clip (rank5b) needs items to carry all_gold_window, r5 (recall@5 capped) and r8. Groups with fewer than `min_group` rollouts are dropped (stragglers did not come back). Returns counts."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(it[group_key], []).append(it)
    stats = {"groups": len(groups), "kept_groups": 0, "zero_var_groups": 0, "short_groups": 0}
    for g in groups.values():
        if len(g) < min_group:
            for it in g:
                it["adv"], it["keep"] = 0.0, False
            stats["short_groups"] += 1
            continue
        if consistent_clip:
            kw2 = {**kw, "all_gold8": [bool(it["all_gold_window"]) for it in g], "zero_recall": [it["r5"] == 0 and it["r8"] == 0 for it in g]}
        else:
            kw2 = kw
        out = group_advantages([it["reward"] for it in g], [not it["all_gold"] for it in g], [it.get("rubric") for it in g], **kw2)
        for it, a, ao, ar in zip(g, out["adv"], out["a_out"], out["a_rubric"]):
            it["adv"], it["keep"], it["a_out"], it["a_rubric"], it["g_std"] = a, out["keep"], ao, ar, out["std"]
        if out["keep"]:
            stats["kept_groups"] += 1
        else:
            stats["zero_var_groups"] += 1
    return stats
