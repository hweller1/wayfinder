"""Per-rollout reward. Pure Python over a scored harness row (needs `messages`).

F1 reward (run f1, `compute_reward`): keep the reported ORDER and score only the first K=12 distinct reported files, L.
    P = hits / |L|,  R = hits / |G|,  r = F1(P, R) + 0.5 * [all gold within the first 8 reported]
        - 0.03 * max(0, n_reported - 8) - 0.02 * max(0, turns - 15) - 0.05 * repeated_queries (cap 0.2) - 0.5 * [no valid report]
K, the all-gold window (8), the free report length (8) and the per-file cost (0.03) are `RewardCfg` fields. Why: a recall-dominant reward
(the legacy reward below, arms a/b/c) taught the policy to report ~0.7 more files (dev reported recall +0.03, recall@5 and @7 flat, the
product's recall@12 handback test fell). R1-Searcher (arXiv 2503.05592) and ZeroSearch (arXiv 2505.04588) moved from containment-style
rewards to F1 for the same reason. Padding now lowers P; the all-gold bonus needs the gold files near the top; files past 8 cost.
Rank reward (run rk, `compute_reward_rank5`): r = recall@5 + 0.25 * recall@8 + 0.5 * [all gold within the first 8] - turn / repeat / no-report penalties.
recall@5 is gold hits among the first 5 distinct reported files over min(|gold|, 5) (so a task with more than 5 gold files can still reach 1); recall@8 uses
min(|gold|, 8) the same way. NOTHING beyond position 8 is credited or costed: padding cannot raise r, and the product backfills handbacks to 12 files in code, so
the model's only job is to rank the right files first. `compute_reward_legacy` is the old reward, kept so a/b/c rollouts can be re-scored offline.

Legacy reward (arms a, b, c):
    r = reported_recall + 0.5 * all_gold_reported - 0.02 * max(0, turns - 15) - 0.05 * repeated_identical_queries (cap 0.2)
        - 0.5 * [no valid report]

reported_recall      files in the ExploreResult (what the parent agent receives) / gold files. This is the dense primary signal:
                     all-gold alone is almost always 0 for this model (v3: 0.08).
all_gold_reported    1 if every gold file is in the reported list.
repeated queries     grep/semantic_search calls whose (tool, normalised query) already appeared earlier in the rollout; the
                     penalty is 0.05 per repeat, capped at 0.2.
valid report         the explorer called `report` (budget not exhausted, no harness error) and reported at least one location.
The rubric score (wayfinder/rubric.py) is NOT part of r: it only ranks failures inside the advantage (advantage.py, PAPO).

Sweep arm (c) adds a precision term, applied by the driver (`total_reward`): - PRECISION_COEF * (fraction of reported distinct files that are not
gold), a guard against gaming recall by reporting everything. Workers always record the ingredients (precision, n_reported), so every arm
can be re-scored offline under any coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TURN_FREE = 15
TURN_PENALTY = 0.02
REPEAT_PENALTY = 0.05
REPEAT_CAP = 0.2
NO_REPORT_PENALTY = 0.5
ALL_GOLD_BONUS = 0.5
RANK_W8 = 0.25  # weight of recall@8 in the rank5 reward
RANK_WINDOW = 8


@dataclass(frozen=True)
class RewardCfg:
    """F1 reward parameters (the run's config.json records them; the workers receive them in cfg["reward"])."""

    k: int = 12  # score only the first k distinct reported files
    gold_window: int = 8  # the all-gold bonus needs every gold file within the first `gold_window` reported files
    free_files: int = 8  # reported files beyond this cost `file_cost` each
    file_cost: float = 0.03
    all_gold_bonus: float = ALL_GOLD_BONUS


DEFAULT_CFG = RewardCfg()


def reward_cfg(d: dict[str, Any] | None) -> RewardCfg:
    return RewardCfg(**{k: v for k, v in (d or {}).items() if k in RewardCfg.__dataclass_fields__})


def _norm_query(call: dict[str, Any]) -> tuple[str, str] | None:
    import json

    fn = call.get("function") or {}
    name = fn.get("name", "")
    if name not in ("grep", "semantic_search"):
        return None
    raw = fn.get("arguments")
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict):
        return None
    q = args.get("pattern") if name == "grep" else args.get("query")
    if q is None:
        return None
    # path / glob / flags are part of the query identity: the same pattern on another path is a different query
    extra = "|".join(str(args.get(k, "")) for k in ("path", "path_glob", "regex", "context"))
    return name, " ".join(str(q).lower().split()) + "||" + extra


def repeated_queries(messages: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, str]] = set()
    n = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or ():
            k = _norm_query(c)
            if k is None:
                continue
            if k in seen:
                n += 1
            seen.add(k)
    return n


def has_valid_report(row: dict[str, Any]) -> bool:
    er = row.get("explore_result") or None
    if row.get("error") or not er or er.get("budget_exhausted"):
        return False
    return bool(er.get("results"))


def compute_reward_legacy(row: dict[str, Any]) -> dict[str, Any]:
    """The recall-dominant reward of arms a, b, c. Row must carry gold_files, explore_result, turns, error and messages (call before the messages are dropped)."""
    gold = list(dict.fromkeys(row.get("gold_files") or []))
    reported = list(dict.fromkeys(h["path"] for h in ((row.get("explore_result") or {}).get("results") or [])))
    recall = (len(set(gold) & set(reported)) / len(set(gold))) if gold else 0.0
    all_gold = bool(gold) and set(gold) <= set(reported)
    turns = int(row.get("turns") or 0)
    reps = repeated_queries(row.get("messages") or [])
    valid = has_valid_report(row)
    turn_pen = TURN_PENALTY * max(0, turns - TURN_FREE)
    rep_pen = min(REPEAT_CAP, REPEAT_PENALTY * reps)
    no_rep = 0.0 if valid else NO_REPORT_PENALTY
    r = recall + ALL_GOLD_BONUS * float(all_gold) - turn_pen - rep_pen - no_rep
    n_rep = len(reported)
    n_hit = len(set(gold) & set(reported))
    precision = (n_hit / n_rep) if n_rep else None  # None: nothing reported (the no-report penalty covers it)
    return {"reward": round(r, 6), "recall_reported": round(recall, 6), "all_gold": all_gold, "turns": turns, "repeats": reps,
            "valid_report": valid, "turn_penalty": round(turn_pen, 6), "repeat_penalty": round(rep_pen, 6), "no_report_penalty": no_rep,
            "n_reported": n_rep, "precision": None if precision is None else round(precision, 6)}


def total_reward(parts: dict[str, Any], precision_coef: float = 0.0) -> float:
    """Arm reward: the base reward minus precision_coef * (1 - precision) (zero when nothing was reported)."""
    pen = precision_coef * (1.0 - parts["precision"]) if (precision_coef and parts.get("precision") is not None) else 0.0
    return round(parts["reward"] - pen, 6)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def recall_at(reported: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(reported[:k])) / len(gold) if gold else 0.0


def compute_reward(row: dict[str, Any], cfg: RewardCfg = DEFAULT_CFG) -> dict[str, Any]:
    """F1 reward (module docstring). Row must carry gold_files, explore_result (results in REPORT ORDER), turns, error and messages.
    Returns the legacy keys (reward, recall_reported, all_gold, ..., precision, n_reported) plus the monitor fields: f1, recall_at_5,
    recall_at_7, recall_at_k, all_gold_window, file_cost."""
    gold = set(row.get("gold_files") or [])
    reported = list(dict.fromkeys(h["path"] for h in ((row.get("explore_result") or {}).get("results") or [])))  # distinct, report order
    top = reported[: cfg.k]
    hits = len(gold & set(top))
    p = hits / len(top) if top else 0.0
    r = hits / len(gold) if gold else 0.0
    f1 = _f1(p, r)
    all_gold_window = bool(gold) and gold <= set(reported[: cfg.gold_window])
    all_gold = bool(gold) and gold <= set(reported)
    turns = int(row.get("turns") or 0)
    reps = repeated_queries(row.get("messages") or [])
    valid = has_valid_report(row)
    turn_pen = TURN_PENALTY * max(0, turns - TURN_FREE)
    rep_pen = min(REPEAT_CAP, REPEAT_PENALTY * reps)
    no_rep = 0.0 if valid else NO_REPORT_PENALTY
    n_rep = len(reported)
    file_pen = cfg.file_cost * max(0, n_rep - cfg.free_files)
    reward = f1 + cfg.all_gold_bonus * float(all_gold_window) - file_pen - turn_pen - rep_pen - no_rep
    recall_all = (len(gold & set(reported)) / len(gold)) if gold else 0.0
    return {"reward": round(reward, 6), "f1": round(f1, 6), "recall_reported": round(recall_all, 6), "all_gold": all_gold, "all_gold_window": all_gold_window,
            "turns": turns, "repeats": reps, "valid_report": valid, "turn_penalty": round(turn_pen, 6), "repeat_penalty": round(rep_pen, 6),
            "no_report_penalty": no_rep, "file_cost": round(file_pen, 6), "n_reported": n_rep, "precision": None if not n_rep else round(len(gold & set(reported)) / n_rep, 6),
            "precision_at_k": None if not top else round(p, 6), "recall_at_5": round(recall_at(reported, gold, 5), 6), "recall_at_7": round(recall_at(reported, gold, 7), 6),
            "recall_at_k": round(r, 6)}


def recall_at_capped(reported: list[str], gold: set[str], k: int) -> float:
    """Gold hits among the first k distinct reported files over min(|gold|, k): a task with more than k gold files can still reach 1."""
    return len(gold & set(reported[:k])) / min(len(gold), k) if gold else 0.0


def compute_reward_rank5(row: dict[str, Any], cfg: RewardCfg = DEFAULT_CFG) -> dict[str, Any]:
    """Rank reward of run rk (module docstring). Same row requirements and output keys as compute_reward (so the monitor and curves work unchanged);
    `reward` ignores every file past position 8; `recall_at_5` stays the UNCAPPED monitor metric (comparable with v3's baselines), `recall_at_5_capped` is the reward term. file_cost is 0 (no per-file cost, nothing past 8 counts)."""
    parts = compute_reward(row, cfg)
    gold = set(row.get("gold_files") or [])
    reported = list(dict.fromkeys(h["path"] for h in ((row.get("explore_result") or {}).get("results") or [])))
    r5, r8 = recall_at_capped(reported, gold, 5), recall_at_capped(reported, gold, RANK_WINDOW)
    all8 = bool(gold) and gold <= set(reported[:RANK_WINDOW])
    reward = r5 + RANK_W8 * r8 + cfg.all_gold_bonus * float(all8) - parts["turn_penalty"] - parts["repeat_penalty"] - parts["no_report_penalty"]
    parts.update({"reward": round(reward, 6), "file_cost": 0.0, "all_gold_window": all8, "recall_at_5_capped": round(r5, 6), "recall_at_8": round(r8, 6)})
    return parts


# ---------------------------------------------------------------------------------------------------------------------------------------------
# rank5b (calibration + full run r2). rank5 stays untouched above.
#
#   reward = S5 * (W5 * r5/S5 + W8 * r8/S8 + WAG * all8/SAG) - no_report_penalty            (+ tie-break, see rank5b_tiebreak, added per group by the driver)
#
# r5 = recall@5 over min(|gold|,5), r8 = recall@8 over min(|gold|,8), all8 = every gold file within the first 8 reported. S* are FIXED per-term scales
# (MO-GRPO's Pre-Norm): the rms over the 1,272 complete v3 screen groups of the within-group std of that term (artifacts/rl_pool/screen/*.jsonl,
# scripts/grpo/rank5b_replay.py --scales). W* are the shares of the unit-std reward each term gets: recall@5 dominant. The final multiplication by S5 puts
# the reward back in recall@5 units (so the advantage std floor 0.1 keeps its meaning); the all-gold bonus is then 0.146 per all-gold report (rank5: 0.5).
# Turn and repeat penalties are MONITORS (logged, weight 0). The no-report penalty stays: it is a validity term, not a style term.
# ---------------------------------------------------------------------------------------------------------------------------------------------
RB_SCALE = {"r5": 0.1274, "r8": 0.1165, "ag": 0.1736}  # fixed Pre-Norm scales (screen rms within-group std)
RB_W = {"r5": 1.0, "r8": 0.25, "ag": 0.2}  # weights in unit-std space
RB_COEF = {k: RB_W[k] * RB_SCALE["r5"] / RB_SCALE[k] for k in RB_W}  # raw coefficients in recall@5 units: r5 1.0, r8 0.273, all-gold 0.147
TIE_MAX = 0.05  # total tie-break budget; the order inside a recall class is lexicographic: fewer reported files first, then fewer turns


def rank5b_terms(r5: float, r8: float, all8: bool, cfg_ag: float | None = None) -> dict[str, float]:
    """Weighted reward contributions of the three recall terms (raw recall@5 units)."""
    return {"r5": RB_COEF["r5"] * r5, "r8": RB_COEF["r8"] * r8, "ag": (RB_COEF["ag"] if cfg_ag is None else cfg_ag) * float(all8)}


def compute_reward_rank5b(row: dict[str, Any], cfg: RewardCfg = DEFAULT_CFG) -> dict[str, Any]:
    """rank5b base reward (no tie-break: that needs the group). Same keys as compute_reward_rank5 plus `reward_kind`, `n_gold5` (min(|gold|,5), the file-count floor
    the tie-break uses), `n_gold8`, and `terms` (the weighted contributions). turn_penalty / repeat_penalty are returned as monitors and NOT subtracted."""
    parts = compute_reward(row, cfg)
    gold = set(row.get("gold_files") or [])
    reported = list(dict.fromkeys(h["path"] for h in ((row.get("explore_result") or {}).get("results") or [])))
    r5, r8 = recall_at_capped(reported, gold, 5), recall_at_capped(reported, gold, RANK_WINDOW)
    all8 = bool(gold) and gold <= set(reported[:RANK_WINDOW])
    terms = rank5b_terms(r5, r8, all8)
    reward = sum(terms.values()) - parts["no_report_penalty"]
    parts.update({"reward": round(reward, 6), "reward_kind": "rank5b", "file_cost": 0.0, "all_gold_window": all8, "recall_at_5_capped": round(r5, 6),
                  "recall_at_8": round(r8, 6), "n_gold5": min(len(gold), 5), "n_gold8": min(len(gold), RANK_WINDOW), "terms": {k: round(v, 6) for k, v in terms.items()}})
    return parts


def rank5b_tiebreak(rows: list[dict[str, Any]]) -> list[float]:
    """Per-rollout tie-break bonus for ONE group, applied only among rollouts with the same recall@5 and the same recall@8 (so the same all-gold flag) and a valid
    report. Inside such a class the rollouts are ranked LEXICOGRAPHICALLY by (reported files, turns), fewer first, and the dense rank is mapped linearly onto
    [0, cap]: the best gets cap, the worst 0, equal (files, turns) share a value, a class with one distinct key gets 0. So turns can only separate rollouts with the
    same file count (a longer report never wins on turns). The file count is floored at min(|gold|,5): a report shorter than that gains nothing for being shorter.
    cap = min(TIE_MAX, 0.9 * the smallest recall quantum of the task (the recall@8 step, RB_COEF['r8'] / min(|gold|,8))), so the tie-break can never outrank a
    rollout that has strictly more gold files in the window."""
    out = [0.0] * len(rows)
    classes: dict[tuple[float, float], list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("valid_report"):
            classes.setdefault((round(float(r["recall_at_5_capped"]), 5), round(float(r["recall_at_8"]), 5)), []).append(i)
    for idx in classes.values():
        if len(idx) < 2:
            continue
        key = {i: (max(int(rows[i]["n_reported"]), int(rows[i].get("n_gold5") or 0)), int(rows[i]["turns"])) for i in idx}
        levels = sorted(set(key.values()))
        if len(levels) < 2:
            continue
        n8 = max(1, int(rows[idx[0]].get("n_gold8") or rows[idx[0]].get("n_gold5") or 1))
        cap = min(TIE_MAX, 0.9 * RB_COEF["r8"] / n8)
        pos = {k: n for n, k in enumerate(levels)}
        for i in idx:
            out[i] = cap * (len(levels) - 1 - pos[key[i]]) / (len(levels) - 1)
    return out
