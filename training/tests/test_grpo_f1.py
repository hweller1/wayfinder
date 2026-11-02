"""CPU tests for GRPO run f1: F1 reward over the first K reported files, std floor, optimizer-state save/restore, lr re-warm, update guard,
padding monitor and stop rule, scorer metrics, and the prompt line."""

import json
import math

import pytest
import torch

from wayfinder.grpo.advantage import batch_advantages, group_advantages
from wayfinder.grpo.driver import build_batch
from wayfinder.grpo.monitor import load_screen, row_metrics, should_stop, step_vs_screen
from wayfinder.grpo.optstate import LrSchedule, UpdateGuard, load_state, publish_resume_slot, save_state
from wayfinder.grpo.reward import RewardCfg, compute_reward, compute_reward_legacy, recall_at, reward_cfg


def call(name, **args):
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def row(reported, gold, turns=10, messages=None, budget_exhausted=False, error=None, no_results=False):
    res = [] if no_results else [{"path": p, "start_line": 1, "end_line": 5, "why": "w"} for p in reported]
    return {"gold_files": gold, "turns": turns, "error": error, "messages": messages or [],
            "explore_result": {"results": res, "budget_exhausted": budget_exhausted}}


# ------------------------------------------------------------------ reward
def test_f1_perfect_report():
    r = compute_reward(row(["a", "b"], ["a", "b"]))
    assert r["reward"] == pytest.approx(1.0 + 0.5) and r["f1"] == pytest.approx(1.0) and r["all_gold_window"]


def test_f1_formula_hand_computed():
    # reported a x y z (4 files), gold a b: hits 1, P 0.25, R 0.5, F1 = 2*0.25*0.5/0.75 = 1/3; gold not all found, no bonus
    r = compute_reward(row(["a", "x", "y", "z"], ["a", "b"]))
    assert r["f1"] == pytest.approx(1 / 3, abs=1e-5) and r["reward"] == pytest.approx(1 / 3, abs=1e-5)
    assert r["precision"] == pytest.approx(0.25) and r["recall_reported"] == pytest.approx(0.5)


def test_padding_lowers_the_reward_even_when_recall_rises():
    base = compute_reward(row(["a", "b", "c", "d", "e", "f"], ["a", "b", "x1"]))  # 2 of 3 gold, 6 files: P 1/3, R 2/3
    padded = compute_reward(row(["a", "b", "c", "d", "e", "f", "p1", "p2", "p3", "x1", "p4", "p5"], ["a", "b", "x1"]))  # finds the third gold file at position 10
    assert padded["recall_reported"] > base["recall_reported"]
    # padded: all gold within first 12 but not within first 8 -> no bonus; P 3/12 = .25, R 1, F1 = .4; 4 files past 8 cost 0.12
    assert padded["all_gold_window"] is False and padded["file_cost"] == pytest.approx(0.12)
    assert padded["reward"] == pytest.approx(0.4 - 0.12)
    assert base["reward"] == pytest.approx(2 * (1 / 3) * (2 / 3) / 1.0)  # F1 of P=1/3, R=2/3 = 4/9
    assert padded["reward"] < base["reward"]


def test_only_first_k_distinct_files_are_scored_and_order_is_kept():
    rep = [f"x{i}" for i in range(12)] + ["gold"]  # the only gold file is 13th: outside K=12
    r = compute_reward(row(rep, ["gold"]))
    assert r["f1"] == 0.0 and r["recall_at_k"] == 0.0 and not r["all_gold_window"]
    assert r["n_reported"] == 13 and r["file_cost"] == pytest.approx(0.03 * 5)  # n_reported counts every distinct file; the cost sees them all
    assert r["recall_reported"] == 1.0  # the legacy field still counts the whole list
    r2 = compute_reward(row(["gold"] + [f"x{i}" for i in range(12)], ["gold"]))
    assert r2["recall_at_k"] == 1.0 and r2["all_gold_window"] and r2["recall_at_5"] == 1.0
    assert r2["reward"] > r["reward"]
    dup = compute_reward(row(["a", "a", "b", "b"], ["a", "b"]))  # duplicates collapse, order of first appearance
    assert dup["n_reported"] == 2 and dup["f1"] == pytest.approx(1.0)


def test_all_gold_bonus_needs_gold_within_first_8():
    gold = ["g1", "g2"]
    early = compute_reward(row(["g1", "g2", "x", "y"], gold))
    late = compute_reward(row(["x1", "x2", "x3", "x4", "x5", "x6", "x7", "g1", "g2"], gold))  # g2 is the 9th file
    assert early["all_gold_window"] and not late["all_gold_window"]
    edge = compute_reward(row(["x1", "x2", "x3", "x4", "x5", "x6", "g1", "g2"], gold))  # g2 is the 8th: inside the window
    assert edge["all_gold_window"]
    assert early["reward"] - edge["reward"] == pytest.approx(2 / 3 - 0.4, abs=1e-5)  # same bonus, F1 differs: P .5 vs .25


def test_per_file_cost_starts_after_8_and_params_are_configurable():
    r8 = compute_reward(row([f"f{i}" for i in range(8)], ["f0"]))
    r9 = compute_reward(row([f"f{i}" for i in range(9)], ["f0"]))
    assert r8["file_cost"] == 0.0 and r9["file_cost"] == pytest.approx(0.03)
    cfg = RewardCfg(k=3, gold_window=2, free_files=2, file_cost=0.1)
    r = compute_reward(row(["a", "b", "c", "d"], ["a", "c"]), cfg)
    assert r["precision_at_k"] == pytest.approx(2 / 3) and r["recall_at_k"] == 1.0 and not r["all_gold_window"]  # c is 3rd, outside window 2
    assert r["file_cost"] == pytest.approx(0.2)  # 2 files past 2
    assert reward_cfg({"k": 5, "junk": 1}).k == 5 and reward_cfg(None) == RewardCfg()


def test_f1_penalties_carry_over_turns_repeats_no_report():
    assert compute_reward(row(["a"], ["a"], turns=20))["reward"] == pytest.approx(1.5 - 0.1)
    many = [{"role": "assistant", "tool_calls": [call("grep", pattern="q")]}] * 9
    assert compute_reward(row(["a"], ["a"], messages=many))["repeat_penalty"] == pytest.approx(0.2)
    nr = compute_reward(row(["a"], ["a"], budget_exhausted=True))
    assert not nr["valid_report"] and nr["reward"] == pytest.approx(1.5 - 0.5)
    assert compute_reward(row([], ["a"], no_results=True))["reward"] == pytest.approx(-0.5)  # F1 0, nothing reported, no valid report


def test_recall_at_k_and_legacy_unchanged():
    assert recall_at(["a", "b", "c"], {"b", "c"}, 2) == 0.5
    assert compute_reward_legacy(row(["a", "x", "y", "z"], ["a", "b"]))["reward"] == pytest.approx(0.5)  # arms a/b/c reward kept for offline re-scoring


# ------------------------------------------------------------------ advantage
def test_std_floor_shrinks_near_flat_groups_and_leaves_normal_groups_alone():
    flat = group_advantages([0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50, 0.50], [True] * 8, [None] * 8, lam=0.0)
    assert flat["keep"] and max(abs(x) for x in flat["adv"]) < 0.3  # std 0.0106 -> divided by 0.1, not by 0.0106
    old_scale = max(abs(x - 0.50) for x in [0.5, 0.52, 0.48, 0.5, 0.51, 0.49, 0.5, 0.5]) / 0.0106
    assert old_scale > 1.5  # the old normaliser would have given advantages of order 1-2
    wide = [1.5, 0.5, 0.0, 0.2, 0.9, 0.1, 0.4, 0.3]
    out = group_advantages(wide, [True] * 8, [None] * 8, lam=0.0)
    m = sum(wide) / 8
    s = math.sqrt(sum((x - m) ** 2 for x in wide) / 8)
    assert s > 0.1 and out["adv"][0] == pytest.approx((1.5 - m) / s)
    assert not group_advantages([0.5] * 8, [True] * 8, [None] * 8)["keep"]  # below MIN_STD still dropped


def test_papo_lambda_zero_removes_the_rubric_term():
    rewards, fail, rub = [1.5, 0.5, 0.5, 0.2], [False, True, True, True], [0.9, 0.2, 0.8, 0.5]
    on = group_advantages(rewards, fail, rub, lam=0.5)
    off = group_advantages(rewards, fail, rub, lam=0.0)
    assert off["adv"] == pytest.approx(off["a_out"]) and on["adv"] != pytest.approx(on["a_out"])
    items = [{"task_id": "t", "reward": r, "all_gold": not f, "rubric": x} for r, f, x in zip(rewards * 2, fail * 2, rub * 2)]
    batch_advantages(items, lam=0.0)
    assert all(it["adv"] == pytest.approx(it["a_out"]) for it in items)
    rows = [{"task_id": "t", "rid": f"r{i}", "reward": r, "arm_reward": r, "all_gold": not f, "rubric": x, "precision": 0.5, "path": "p", "n_tokens": 10, "n_loss": 5,
             "recall_reported": 0.5, "valid_report": True, "turns": 5, "repeats": 0, "n_reported": 5, "report_chars": 1}
            for i, (r, f, x) in enumerate(zip(rewards * 2, fail * 2, rub * 2))]
    man, _, _ = build_batch(rows, 0.0, lam=0.0)
    assert [it["adv"] for it in man["items"]] == pytest.approx([round(x, 6) for x in off["a_out"] * 0 + group_advantages(rewards * 2, fail * 2, rub * 2, lam=0.0)["adv"]])


# ------------------------------------------------------------------ optimizer state
def _toy(seed=0):
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(6, 4))
    return w, torch.optim.AdamW([w], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)


def _step(w, opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    w.grad = torch.sin(w.detach() * 3) + 0.1
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_adamw_state_roundtrip_continues_identically(tmp_path):
    w1, o1 = _toy()
    for _ in range(4):
        _step(w1, o1, 1e-2)
    save_state(o1, tmp_path / "ckpt" / "optimizer.pt", {"step": 4, "applied": 4, "guard": {}})
    snap = w1.detach().clone()
    # path A: keep going with the live optimizer
    for _ in range(3):
        _step(w1, o1, 1e-2)
    # path B: a new process: same weights, restored state
    w2, o2 = _toy(seed=5)
    w2.data = snap.clone()
    meta = load_state(o2, tmp_path / "ckpt" / "optimizer.pt")
    assert meta["applied"] == 4
    for _ in range(3):
        _step(w2, o2, 1e-2)
    assert torch.allclose(w1, w2, atol=0, rtol=0)
    # path C: no optimizer state: a different trajectory (this is what diverged arm b)
    w3, o3 = _toy(seed=5)
    w3.data = snap.clone()
    for _ in range(3):
        _step(w3, o3, 1e-2)
    assert not torch.allclose(w1, w3)
    assert load_state(o3, tmp_path / "nope.pt") is None


def test_resume_slot_is_atomic_and_replaceable(tmp_path):
    w, o = _toy()
    _step(w, o, 1e-2)
    writes = []
    for k in range(3):
        publish_resume_slot(tmp_path, lambda d, k=k: (__import__("pathlib").Path(d).mkdir(parents=True, exist_ok=True), (__import__("pathlib").Path(d) / "adapter_model.safetensors").write_text(str(k)), writes.append(k)),
                            o, {"step": k + 1, "applied": k + 1, "guard": {}})
    assert (tmp_path / "resume" / "adapter_model.safetensors").read_text() == "2"
    assert load_state(o, tmp_path / "resume" / "optimizer.pt")["step"] == 3
    assert not (tmp_path / "resume.tmp").exists() and not (tmp_path / "resume.old").exists()


def test_lr_schedule_fresh_resumed_and_rewarm():
    fresh = LrSchedule(3e-5, warmup=2)
    assert [round(fresh.lr_at(u) / 3e-5, 3) for u in range(4)] == [0.5, 1.0, 1.0, 1.0]
    cont = LrSchedule(3e-5, warmup=2, resumed=True, has_opt_state=True)
    assert cont.mode == "resumed" and cont.lr_at(0) == 3e-5  # state restored: full lr at once
    rew = LrSchedule(3e-5, warmup=2, rewarm=10, resumed=True, has_opt_state=False)
    assert rew.mode == "rewarm"
    ratios = [round(rew.lr_at(u) / 3e-5, 3) for u in range(12)]
    assert ratios[0] == 0.0 and ratios[5] == 0.5 and ratios[10] == 1.0 and ratios[11] == 1.0
    assert all(b >= a for a, b in zip(ratios, ratios[1:]))


def test_fresh_start_never_rewarms_and_first_update_lr_is_positive():
    """Review fix: the 10-update re-warm from 0 applies ONLY to a resume without optimizer state; a fresh run warms up as before and its first update has lr > 0."""
    fresh = LrSchedule(3e-5, warmup=2, rewarm=10)  # resumed=False, no state
    assert fresh.mode == "fresh" and fresh.lr_at(0) == pytest.approx(1.5e-5) and fresh.lr_at(1) == pytest.approx(3e-5) and fresh.lr_at(50) == pytest.approx(3e-5)
    # a resume WITH state: full lr from the first update, no warm-up
    assert LrSchedule(3e-5, warmup=2, resumed=True, has_opt_state=True).lr_at(0) == pytest.approx(3e-5)
    # a resume WITHOUT state: lr 0 for the first update (it only fills the moments), linear to full at update 10; only this mode ever returns 0
    rew = LrSchedule(3e-5, warmup=2, rewarm=10, resumed=True, has_opt_state=False)
    assert rew.mode == "rewarm" and rew.lr_at(0) == 0.0 and rew.lr_at(1) == pytest.approx(3e-6) and rew.lr_at(10) == pytest.approx(3e-5)
    for sched in (fresh, LrSchedule(3e-5, resumed=True, has_opt_state=True)):
        assert all(sched.lr_at(u) > 0 for u in range(20))


def test_guard_skip_leaves_weights_and_schedule_untouched_and_a_real_update_moves_them():
    w, o = _toy()
    sched = LrSchedule(1e-2, warmup=2)
    g = UpdateGuard(kl_max=1e-4)
    w0 = w.detach().clone()
    assert g.check(0.3, 3.8e-4) is not None  # the smoke's step-1 case: skipped, applied stays 0
    g.commit(0.3, True)
    assert sched.applied == 0 and torch.equal(w, w0)
    assert g.check(0.3, 0.0) is None  # an update that passes the guard moves the weights at the scheduled lr (> 0)
    _step(w, o, sched.lr_at())
    sched.applied += 1
    assert not torch.equal(w, w0) and sched.lr_at() == pytest.approx(1e-2)


# ------------------------------------------------------------------ update guard
def test_guard_skips_on_grad_norm_spike_and_high_kl_and_logs():
    g = UpdateGuard(gn_mult=5.0, kl_max=0.1)
    for gn in (0.3, 0.35, 0.28, 0.32):
        assert g.check(gn, 0.01) is None
        g.commit(gn, skipped=False)
    assert g.gn_avg == pytest.approx(0.3125)
    assert "grad_norm" in g.check(1.6, 0.01)  # > 5 x 0.3125 = 1.5625
    assert g.check(1.5, 0.01) is None  # just under 5x
    assert "kl" in g.check(0.3, 0.2)  # KL above 0.1
    assert g.check(0.3, 0.1) is None  # exactly at the bound passes
    assert g.check(float("nan"), 0.0) == "nonfinite_grad_norm" and g.check(0.3, float("inf")) == "nonfinite_kl"
    avg0 = g.gn_avg
    g.commit(1.6, skipped=True)  # a skipped update neither enters the average nor resets the streak
    assert g.gn_avg == avg0 and g.consecutive == 1 and g.skipped_total == 1 and not g.stuck
    g.commit(9.0, skipped=True)
    g.commit(9.0, skipped=True)
    assert g.stuck
    g.commit(0.3, skipped=False)
    assert g.consecutive == 0 and not g.stuck


def test_guard_needs_history_before_the_grad_norm_rule_and_state_roundtrips():
    g = UpdateGuard()
    assert g.check(50.0, 0.0) is None  # no moving average yet: only the KL and finiteness rules apply
    g.commit(0.3, False)
    g.commit(0.3, False)
    assert g.check(50.0, 0.0) is None  # still fewer than min_history
    g.commit(0.3, False)
    assert g.check(50.0, 0.0) is not None
    h = UpdateGuard()
    h.load(g.state())
    assert h.gn_avg == pytest.approx(g.gn_avg)


def test_guarded_toy_training_republishes_previous_weights_on_a_bad_update():
    w, o = _toy()
    g = UpdateGuard(min_history=2)
    pub = []
    for i, (gn, kl) in enumerate([(0.3, 0.0), (0.3, 0.0), (0.3, 0.0), (5.0, 0.0), (0.3, 0.5), (0.3, 0.01)]):
        reason = g.check(gn, kl)
        if reason is None:
            _step(w, o, 1e-2)
        g.commit(gn, reason is not None)
        pub.append(w.detach().clone())
    assert torch.equal(pub[3], pub[2]) and torch.equal(pub[4], pub[3])  # steps 3 and 4 skipped: the previous adapter is published again
    assert not torch.equal(pub[5], pub[4]) and g.skipped_total == 2


# ------------------------------------------------------------------ monitor and stop rule (same-task v3 baseline)
GOLD = {"t1": {"g1", "g2"}, "t2": {"h1"}, "t3": {"k1", "k2", "k3"}}


def _screen_rows(tmp_path, per_task_reports):
    d = tmp_path / "screen"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "s0.jsonl").open("w") as f:
        for t, reps in per_task_reports.items():
            for rep in reps:
                f.write(json.dumps({"task_id": t, "error": None, "reported_paths": rep}) + "\n")
        f.write(json.dumps({"task_id": "t1", "error": "boom", "reported_paths": ["junk"] * 20}) + "\n")  # errored rows are ignored
        f.write(json.dumps({"task_id": "unknown", "error": None, "reported_paths": ["a"]}) + "\n")  # not in the pool gold map: ignored
    return d


def _step_rows(reports_by_task):
    rows = []
    for t, reps in reports_by_task.items():
        for rep in reps:
            m = row_metrics(rep, GOLD[t])
            rows.append({"task_id": t, **m})
    return rows


def test_row_metrics_recomputed_from_reported_files_and_gold():
    m = row_metrics(["g1", "x", "g1", "y", "z", "g2"], {"g1", "g2"})  # duplicates collapse: g1 x y z g2
    assert m["n_reported"] == 5 and m["precision"] == pytest.approx(2 / 5) and m["recall_at_5"] == 1.0 and m["recall_reported"] == 1.0
    m2 = row_metrics(["x", "y", "z", "w", "v", "g1"], {"g1", "g2"})  # g1 is 6th: outside the first 5
    assert m2["recall_at_5"] == 0.0 and m2["recall_reported"] == 0.5
    none = row_metrics([], {"g1"})
    assert none["n_reported"] == 0 and none["precision"] is None and none["recall_at_5"] == 0.0


def test_same_task_baseline_cancels_task_mix(tmp_path):
    # task t1 is easy (v3 reports 3 files), t3 is hard (v3 reports 9). A batch of only hard tasks has a high n_reported, but not relative to v3 on those tasks.
    scr = load_screen(_screen_rows(tmp_path, {"t1": [["g1", "g2", "x"]] * 4, "t3": [["k1"] + [f"p{i}" for i in range(8)]] * 4}), GOLD)
    assert set(scr) == {"t1", "t3"} and len(scr["t1"]) == 4
    hard_only = _step_rows({"t3": [["k1"] + [f"p{i}" for i in range(8)]] * 8})
    c = step_vs_screen(hard_only, scr, GOLD)
    assert c["delta"]["n_reported"] == 0.0 and not c["flagged_same_task"]
    # the OLD rule (pooled baseline of an easy batch, n 3) would have seen +6 files here: the same-task delta is what removes it
    assert c["cur"]["n_reported"] == 9.0 and c["v3_same_tasks"]["n_reported"] == 9.0 and c["n_tasks"] == 1


def test_flag_needs_all_three_conditions(tmp_path):
    base = [["g1", "g2", "x", "y"]] * 4  # v3 on t1: 4 files, precision .5, recall@5 1.0
    scr = load_screen(_screen_rows(tmp_path, {"t1": base, "t2": [["h1", "a", "b", "c"]] * 4}), GOLD)
    padded = _step_rows({"t1": [["x1", "x2", "x3", "x4", "g1", "x5", "x6", "g2"]] * 8, "t2": [["a", "b", "c", "d", "e", "f", "g", "h1"]] * 8})  # more files, lower precision, recall@5 down
    c = step_vs_screen(padded, scr, GOLD)
    assert c["delta"]["n_reported"] > 0.5 and c["delta"]["precision"] < -0.03 and c["delta"]["recall_at_5"] <= 0 and c["flagged_same_task"]
    # more files, lower precision, but recall@5 RISING: not flagged (the rule needs recall@5 not above v3)
    scr_low = load_screen(_screen_rows(tmp_path / "low", {"t1": [["x", "g1", "y", "z"]] * 4}), GOLD)  # v3: recall@5 .5, precision .25, 4 files
    better = _step_rows({"t1": [["g1", "g2", "x", "y", "p", "q", "r", "s", "t"]] * 8})  # recall@5 1.0, precision 2/9, 9 files
    c2 = step_vs_screen(better, scr_low, GOLD)
    assert c2["delta"]["n_reported"] > 0.5 and c2["delta"]["precision"] < -0.02 and c2["delta"]["recall_at_5"] > 0
    assert not c2["flagged_same_task"]
    # more files but precision flat or higher: AND rule, not flagged (the earlier OR rule would have flagged on n_reported alone)
    scr_p = load_screen(_screen_rows(tmp_path / "p", {"t1": [["g1", "x", "y", "z", "w"]] * 4}), GOLD)  # v3: 5 files, precision .2
    longer_precise = _step_rows({"t1": [["g1", "g2", "x", "y", "z", "w"]] * 8})  # 6 files, precision .333
    c3 = step_vs_screen(longer_precise, scr_p, GOLD)
    assert c3["delta"]["n_reported"] > 0.5 and c3["delta"]["precision"] >= -0.03 and not c3["flagged_same_task"]
    # fewer files and lower precision: not flagged (not padding)
    shorter = _step_rows({"t1": [["g1"]] * 8, "t2": [["a"]] * 8})
    assert not step_vs_screen(shorter, scr, GOLD)["flagged_same_task"]
    # exactly on the thresholds is not past them
    exact = _step_rows({"t1": [["g1", "g2", "x", "y"]] * 8, "t2": [["h1", "a", "b", "c"]] * 8})
    ce = step_vs_screen(exact, scr, GOLD)
    assert ce["delta"] == {"n_reported": 0.0, "precision": 0.0, "recall_at_5": 0.0, "recall_reported": 0.0} and not ce["flagged_same_task"]


def test_stop_rule_two_consecutive_and_missing_tasks(tmp_path):
    assert not should_stop([True]) and should_stop([False, True, True]) and not should_stop([True, False, True])
    scr = load_screen(_screen_rows(tmp_path, {"t1": [["g1"]] * 2}), GOLD)
    assert step_vs_screen(_step_rows({"t2": [["h1"]] * 8}), scr, GOLD) is None  # no screen rows for the step's tasks: no verdict, no flag


def test_rollout_stats_carry_monitor_fields():
    rows = []
    for i in range(8):
        parts = compute_reward(row(["g1", "x1", "x2"] if i % 2 else ["g1", "g2", "x1", "x2", "x3", "x4", "x5", "x6", "x7"], ["g1", "g2"]))
        rows.append({**parts, "task_id": "t", "rid": f"r{i}", "path": "p", "n_tokens": 10, "n_loss": 5, "rubric": 0.1, "report_chars": 10, "arm_reward": parts["reward"]})
    _, stats, _ = build_batch(rows, 0.0, lam=0.0)
    for k in ("n_reported", "precision", "recall_at_5", "recall_at_7", "f1", "all_gold_window"):
        assert stats[k] is not None
    assert stats["n_reported"] == pytest.approx(6.0)


# ------------------------------------------------------------------ prompt line
def test_order_line_is_an_extra_paragraph_and_default_nudge_unchanged():
    from wayfinder.rollout.nudges import NUDGES, ORDER_LINE, TEACHER_NUDGE

    assert NUDGES["teacher"] == TEACHER_NUDGE == "Report as soon as the evidence covers the brief; do not exhaust the turn budget."
    assert NUDGES["f1"] == TEACHER_NUDGE + "\n\n" + ORDER_LINE
    assert "listed first" in ORDER_LINE and "count the most" in ORDER_LINE


# ------------------------------------------------------------------ cloud driver: retry wrapper and config plumbing
def test_retry_transient_recovers_with_backoff():
    from wayfinder.grpo.driver import retry_transient

    calls, sleeps, t = {"n": 0}, [], {"now": 0.0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise ConnectionError("dns")
        return "ok"

    def sleep(d):
        sleeps.append(d)
        t["now"] += d

    assert retry_transient(flaky, sleep=sleep, clock=lambda: t["now"], max_s=600) == "ok"
    assert calls["n"] == 4 and sleeps == [2.0, 4.0, 8.0]


def test_retry_transient_gives_up_after_max_s_and_never_retries_stopped():
    import threading

    from wayfinder.grpo.driver import Stopped, retry_transient

    t = {"now": 0.0}

    def sleep(d):
        t["now"] += d

    def bad():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        retry_transient(bad, sleep=sleep, clock=lambda: t["now"], max_s=100, cap_s=30)
    assert 100 <= t["now"] <= 160
    with pytest.raises(Stopped):
        retry_transient(lambda: (_ for _ in ()).throw(Stopped("x")), sleep=sleep)
    ev = threading.Event(); ev.set()
    with pytest.raises(Stopped):
        retry_transient(bad, sleep=sleep, clock=lambda: t["now"], stop=ev)


def test_wait_done_survives_a_network_blip():
    import types

    from wayfinder.grpo import driver as dmod

    class Ctl:
        def __init__(self):
            self.n = 0

        def get(self, k):
            if k == "done_3":
                self.n += 1
                if self.n <= 2:
                    raise ConnectionError("dns")
                return {"kl": 0.1}
            return None

    d = dmod.Driver(types.SimpleNamespace(start_step=0), Ctl(), None, None, __import__("pathlib").Path("/tmp/_unused_driver_out"))
    orig = dmod.time.sleep
    dmod.time.sleep = lambda s: None
    try:
        assert d.wait_done(3) == {"kl": 0.1}
    finally:
        dmod.time.sleep = orig


def test_remote_paths_plumbing(tmp_path):
    from pathlib import Path

    from wayfinder.grpo.driver import remote_paths

    repo = Path("/Users/x/wayfinder")
    a = {"pool": "/Users/x/wayfinder/data/rl_pool_v1.json", "eval_every": 5, "start_step": 4}
    r = remote_paths(a, repo, "/root")
    assert r["pool"] == "/root/data/rl_pool_v1.json" and r["eval_every"] == 0 and r["screen_dir"] == "/root/artifacts/rl_pool/screen" and a["eval_every"] == 5
    assert remote_paths(r, Path("/nonexistent"), "/root") == r  # idempotent (applied again inside the container)
    assert remote_paths({"pool": ""}, repo)["pool"] == ""
