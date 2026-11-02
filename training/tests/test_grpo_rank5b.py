"""CPU tests for the rank5b reward (calibration + r2): fixed Pre-Norm scales, monitor-only penalties, tie-break only among recall ties, consistency clip."""

import json

import pytest
import torch

from wayfinder.grpo.advantage import batch_advantages, consistency_clip, group_advantages
from wayfinder.grpo.driver import apply_tiebreak, build_batch, fixed_batch, step_diagnostics
from wayfinder.grpo.loss import softmax_entropy
from wayfinder.grpo.reward import RB_COEF, TIE_MAX, compute_reward_rank5, compute_reward_rank5b, rank5b_tiebreak


def row(reported, gold, turns=10, error=None):
    return {"gold_files": gold, "turns": turns, "error": error, "messages": [],
            "explore_result": {"results": [{"path": p, "start_line": 1, "end_line": 2, "why": "w"} for p in reported], "budget_exhausted": False}}


def parts(reported, gold, turns=10):
    return compute_reward_rank5b(row(reported, gold, turns))


def group(specs, gold):
    """specs: list of (reported, turns) -> rows with arm_reward = base + tiebreak (as build_batch does)."""
    rows = []
    for i, (rep, t) in enumerate(specs):
        p = parts(rep, gold, t)
        p.update({"arm_reward": p["reward"], "task_id": "t", "rid": str(i)})
        rows.append(p)
    apply_tiebreak(rows)
    return rows


def test_rank5_untouched():
    assert compute_reward_rank5(row(["a", "b"], ["a", "b"]))["reward"] == pytest.approx(1.75)


def test_recall_dominant_and_all_gold_bonus_small():
    assert RB_COEF["r5"] == pytest.approx(1.0)
    assert 0.1 <= RB_COEF["ag"] <= 0.2
    assert parts(["a", "b"], ["a", "b"])["reward"] == pytest.approx(RB_COEF["r5"] + RB_COEF["r8"] + RB_COEF["ag"])


def test_turn_and_repeat_penalties_are_monitors_only():
    p = parts(["a"], ["a"], turns=40)
    assert p["turn_penalty"] > 0  # still logged
    assert p["reward"] == pytest.approx(parts(["a"], ["a"], turns=5)["reward"])


def test_no_report_penalty_kept():
    r = compute_reward_rank5b(row([], ["a"]))
    assert r["reward"] == pytest.approx(-0.5)


def test_padding_never_gains():
    gold = ["g1", "g2", "g3"]
    clean = ["g1", "x1", "x2"]
    for padded in (clean + ["x3", "x4"], clean + ["x3", "x4", "x5", "x6", "g2"], clean + [f"p{i}" for i in range(9)]):
        rows = group([(clean, 10), (padded, 10)], gold)
        if rows[0]["recall_at_5_capped"] == rows[1]["recall_at_5_capped"] and rows[0]["recall_at_8"] == rows[1]["recall_at_8"]:
            assert rows[0]["arm_reward"] > rows[1]["arm_reward"]
        else:  # padding with a gold file inside the window is a different recall class: gains only through recall
            assert rows[1]["recall_at_5_capped"] > rows[0]["recall_at_5_capped"] or rows[1]["recall_at_8"] > rows[0]["recall_at_8"]
    eight = clean + [f"x{i}" for i in range(3, 8)]
    assert parts(eight + ["g2", "g3"], gold)["reward"] == pytest.approx(parts(eight, gold)["reward"])  # past 8: nothing


def test_truncating_below_min_gold_5_never_gains():
    gold = [f"g{i}" for i in range(6)]  # min(|gold|,5) = 5
    full = ["g0", "x1", "x2", "x3", "x4"]
    short = ["g0"]  # same recall@5 and recall@8, fewer files than 5
    rows = group([(full, 10), (short, 10)], gold)
    assert rows[1]["arm_reward"] <= rows[0]["arm_reward"] + 1e-12
    assert rows[1]["tiebreak"] == pytest.approx(rows[0]["tiebreak"])  # floored: equal file count
    # and with turns equal, no gain at all for the truncated one
    assert rank5b_tiebreak(rows)[1] <= rank5b_tiebreak(rows)[0] + 1e-12


def test_tiebreak_only_acts_on_recall_ties():
    gold = ["g1", "g2"]
    a = ["g1", "g2"]  # r5 = 1
    b = ["g1", "x1", "x2", "x3", "x4", "g2"]  # r5 = 0.5, r8 = 1: a different class even though it is longer
    c = ["g1"]  # r5 = 0.5, r8 = 0.5: shortest, but lower recall
    rows = group([(a, 10), (b, 5), (c, 3)], gold)
    assert all(r["tiebreak"] == 0.0 for r in rows)  # three classes of one
    assert rows[0]["arm_reward"] > rows[1]["arm_reward"] > rows[2]["arm_reward"]
    # tie-break never reverses a recall difference
    big = group([(["g1"], 40), (["g1", "g2"], 40)] + [(["g1"], 1)] * 3, gold)
    assert big[1]["arm_reward"] > max(r["arm_reward"] for r in big if r is not big[1])


def test_tiebreak_bounded_and_ordered():
    gold = ["g1"]
    rows = group([(["g1"], 4), (["g1", "x"], 4), (["g1"], 20), (["g1", "x", "y", "z"], 30)], gold)
    tb = [r["tiebreak"] for r in rows]
    assert max(tb) <= TIE_MAX + 1e-12 and min(tb) >= 0
    assert tb[0] == max(tb) and tb[3] == min(tb)
    assert tb[0] > tb[2] > tb[1] > tb[3]  # lexicographic: files first (1 file, 20 turns beats 2 files, 4 turns), then turns


def test_turns_cannot_outweigh_files():
    gold = ["g1"]
    rows = group([(["g1"], 40), (["g1", "x"], 1)], gold)  # padded but 39 turns faster
    assert rows[0]["arm_reward"] > rows[1]["arm_reward"]


def test_tiebreak_never_outranks_more_recall():
    gold = [f"g{i}" for i in range(8)]  # recall@8 quantum is smallest here
    best_tie = ["g0", "g1", "g2", "g3", "g4", "x5", "x6", "g5"]  # r5 = 1, r8 = 6/8
    fewer_gold_short_fast = ["g0", "g1", "g2", "g3", "g4"]  # r5 = 1, r8 = 5/8
    rows = group([(best_tie, 40), (fewer_gold_short_fast, 1), (fewer_gold_short_fast, 2)], gold)
    assert rows[0]["arm_reward"] > max(rows[1]["arm_reward"], rows[2]["arm_reward"])
    assert max(r["tiebreak"] for r in rows) <= 0.9 * RB_COEF["r8"] / 8 + 1e-12


def test_tiebreak_skips_invalid_report():
    gold = ["g1"]
    good = parts(["g1"], gold)
    bad = compute_reward_rank5b(row([], gold))
    bad["recall_at_5_capped"], bad["recall_at_8"] = good["recall_at_5_capped"], good["recall_at_8"]
    assert rank5b_tiebreak([good, bad]) == [0.0, 0.0]


def test_clipping_signs():
    assert consistency_clip([-1.0, 1.0, -1.0, 1.0, -0.5], [True, True, False, False, False], [False, False, True, True, False]) == [0.0, 1.0, -1.0, 0.0, -0.5]
    # through group_advantages: the all-gold rollout with the lowest reward (tie-break loser) is clipped up to 0, the zero-recall one down to <= 0
    rewards = [0.3, 0.2, 0.1, 0.0]
    out = group_advantages(rewards, [True] * 4, [None] * 4, lam=0.0, all_gold8=[False, False, False, False], zero_recall=[False, False, False, False])
    assert out["adv"][2] < 0
    out = group_advantages(rewards, [True] * 4, [None] * 4, lam=0.0, all_gold8=[False, False, True, False], zero_recall=[True, False, False, True])
    assert out["adv"][2] == 0.0 and out["adv"][0] <= 0.0 and out["adv"][3] <= 0.0 and out["adv"][1] > -1e-12


def test_clip_through_build_batch():
    gold = ["g1", "g2"]
    specs = [(["g1", "g2"], 40)] + [(["g1"], 1)] * 3 + [(["x"], 5)] + [(["g1", "g2"], 40)] * 2 + [(["g2"], 3)]
    rows = []
    for i, (rep, t) in enumerate(specs):
        p = parts(rep, gold, t)
        p.update({"task_id": "t", "rid": str(i), "path": f"/p{i}", "n_tokens": 10, "n_loss": 5, "rubric": None, "all_gold": p["all_gold"], "valid_report": True, "report_chars": 10})
        rows.append(p)
    man, stats, adv = build_batch(rows, 0.0, lam=0.0)
    by = {it["rid"]: it["adv"] for it in man["items"]}
    for i, (rep, t) in enumerate(specs):
        if set(gold) <= set(rep[:8]):
            assert by[str(i)] >= 0
        if rep == ["x"]:
            assert by[str(i)] <= 0
    assert stats["diag"]["n_groups_signal"] == 1 and "r5" in stats["diag"]["adv_share"]


def test_diagnostics_shapes():
    gold = ["g1", "g2"]
    rows = group([(["g1", "g2"], 5), (["g1"], 5), (["x"], 5), (["g1", "x", "y", "z", "w", "v"], 5)], gold)
    d = step_diagnostics(rows)
    assert d["group_std"]["n"] == 1 and sum(d["adv_share"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(b["n"] for b in d["recall_by_n_reported"].values()) == 4 and sum(d["n_reported_hist"].values()) == 4


def test_fixed_batch(tmp_path):
    pool = [{"task_id": f"t{i}"} for i in range(5)]
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"task_ids": ["t3", "t1"]}))
    assert [t["task_id"] for t in fixed_batch(pool, str(p))] == ["t3", "t1"]
    p.write_text(json.dumps({"task_ids": ["t3", "zz"]}))
    with pytest.raises(ValueError):
        fixed_batch(pool, str(p))


def test_entropy():
    z = torch.zeros(2, 8)
    assert softmax_entropy(z, 1.0).tolist() == pytest.approx([torch.log(torch.tensor(8.0)).item()] * 2, abs=1e-5)
    z = torch.tensor([[50.0, 0, 0, 0]])
    assert softmax_entropy(z, 1.0).item() < 1e-6
