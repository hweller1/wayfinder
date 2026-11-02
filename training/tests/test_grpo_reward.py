"""CPU tests: GRPO reward, group advantages (PAPO on failures only), pool guards, task sampling."""

import json

from pathlib import Path

import pytest

from wayfinder.grpo.advantage import batch_advantages, group_advantages
from wayfinder.grpo.pool import default_pool_tasks, load_pool, sample_batch, validate_pool
from wayfinder.grpo.reward import compute_reward_legacy as compute_reward  # arms a, b, c reward; the F1 reward is tested in test_grpo_f1.py
from wayfinder.grpo.reward import repeated_queries, total_reward


def call(name, **args):
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def row(reported, gold, turns=10, messages=None, budget_exhausted=False, error=None, no_results=False):
    res = [] if no_results else [{"path": p, "start_line": 1, "end_line": 5, "why": "w"} for p in reported]
    return {"gold_files": gold, "turns": turns, "error": error, "messages": messages or [],
            "explore_result": {"results": res, "budget_exhausted": budget_exhausted}}


def test_perfect_report_gets_recall_plus_bonus():
    r = compute_reward(row(["a", "b"], ["a", "b"]))
    assert r["reward"] == pytest.approx(1.5) and r["all_gold"] and r["valid_report"] and r["precision"] == 1.0


def test_partial_recall_no_bonus_and_precision():
    r = compute_reward(row(["a", "x", "y", "z"], ["a", "b"]))
    assert r["reward"] == pytest.approx(0.5) and not r["all_gold"] and r["precision"] == pytest.approx(0.25)


def test_turn_penalty_is_linear_past_15():
    assert compute_reward(row(["a"], ["a"], turns=15))["reward"] == pytest.approx(1.5)
    assert compute_reward(row(["a"], ["a"], turns=20))["reward"] == pytest.approx(1.5 - 0.1)
    assert compute_reward(row(["a"], ["a"], turns=40))["turn_penalty"] == pytest.approx(0.5)


def test_repeated_identical_queries_penalised_and_capped():
    same = [{"role": "assistant", "tool_calls": [call("grep", pattern="Foo  bar")]}, {"role": "assistant", "tool_calls": [call("grep", pattern="foo bar")]}]
    assert repeated_queries(same) == 1
    other_path = same + [{"role": "assistant", "tool_calls": [call("grep", pattern="foo bar", path="src")]}]
    assert repeated_queries(other_path) == 1  # same pattern on another path is a different query
    many = [{"role": "assistant", "tool_calls": [call("semantic_search", query="q")]}] * 9
    r = compute_reward(row(["a"], ["a"], messages=many))
    assert r["repeats"] == 8 and r["repeat_penalty"] == pytest.approx(0.2)  # cap
    read_twice = [{"role": "assistant", "tool_calls": [call("read_file", path="a")]}] * 3
    assert repeated_queries(read_twice) == 0  # only search queries count


def test_no_valid_report_penalty():
    r = compute_reward(row(["a"], ["a"], budget_exhausted=True))  # best-effort hits after a budget stop are not a report
    assert not r["valid_report"] and r["reward"] == pytest.approx(1.5 - 0.5)
    assert compute_reward(row([], ["a"], no_results=True))["reward"] == pytest.approx(0 - 0.5)
    assert compute_reward(row(["a"], ["a"], error="boom"))["valid_report"] is False


def test_total_reward_precision_arm():
    p = compute_reward(row(["a", "x", "y", "z"], ["a", "b"]))
    assert total_reward(p) == pytest.approx(0.5)
    assert total_reward(p, 0.3) == pytest.approx(0.5 - 0.3 * 0.75)
    nothing = compute_reward(row([], ["a"], no_results=True))
    assert total_reward(nothing, 0.3) == pytest.approx(nothing["reward"])  # precision undefined: no extra penalty


def test_advantages_normalise_within_group():
    out = group_advantages([1.0, 0.0, 0.5, 0.5], [True] * 4, [None] * 4)
    assert out["keep"]
    a = out["adv"]
    assert sum(a) == pytest.approx(0, abs=1e-6)
    assert (sum(x * x for x in a) / 4) == pytest.approx(1.0, abs=1e-3)
    assert a[0] > a[2] > a[1]


def test_zero_variance_group_dropped():
    out = group_advantages([0.5] * 8, [True] * 8, [0.1 * i for i in range(8)])
    assert not out["keep"] and all(x == 0 for x in out["adv"])  # rubric alone cannot drive an update


def test_papo_rubric_only_on_failures_and_normalised_over_failures():
    rewards = [1.5, 0.5, 0.5, 0.2]
    failures = [False, True, True, True]
    rubric = [0.9, 0.2, 0.8, 0.5]
    out = group_advantages(rewards, failures, rubric)
    assert out["a_rubric"][0] == 0.0  # the success carries no rubric term
    fr = [out["a_rubric"][i] for i in (1, 2, 3)]
    assert sum(fr) == pytest.approx(0, abs=1e-6) and fr[1] > fr[2] > fr[0]
    for i in range(4):
        assert out["adv"][i] == pytest.approx(out["a_out"][i] + 0.5 * out["a_rubric"][i])
    # same outcomes, different rubric on the success: its advantage must not change
    out2 = group_advantages(rewards, failures, [0.1, 0.2, 0.8, 0.5])
    assert out2["adv"][0] == pytest.approx(out["adv"][0])
    # rubric reorders failures only: tied-reward failures separate by rubric
    assert out["adv"][2] > out["adv"][1]


def test_papo_single_failure_or_flat_rubric_is_noop():
    out = group_advantages([1.5, 1.5, 0.5], [False, False, True], [0.5, 0.6, 0.9])
    assert out["a_rubric"] == [0.0, 0.0, 0.0]
    out = group_advantages([1.5, 0.5, 0.4], [False, True, True], [0.5, 0.7, 0.7])
    assert out["a_rubric"] == [0.0, 0.0, 0.0]


def test_batch_advantages_drops_short_and_flat_groups():
    items = []
    for t, rs in (("t1", [1.5, 0.5, 0.0, 0.2, 0.9, 0.1, 0.4, 0.3]), ("t2", [0.5] * 8), ("t3", [1.0, 0.0])):
        for i, r in enumerate(rs):
            items.append({"task_id": t, "reward": r, "all_gold": r >= 1.5, "rubric": 0.1 * i})
    st = batch_advantages(items)
    assert st == {"groups": 3, "kept_groups": 1, "zero_var_groups": 1, "short_groups": 1}
    assert all(not it["keep"] and it["adv"] == 0 for it in items if it["task_id"] in ("t2", "t3"))
    assert all(it["keep"] for it in items if it["task_id"] == "t1")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data" / "splits.json").exists(), reason="needs task manifests or cluster specs withheld from the public release; see RELEASE_CHECKLIST.md")
def test_default_pool_matches_prd14_and_is_train_only():
    tasks = default_pool_tasks()
    assert len(tasks) == 224  # 233 RL-eligible train tasks minus 7 OPD/E2E-excluded and 2 in the unreachable repo Emmy123222/Stellar-GreenPay
    assert not any(t['repo'] == 'Emmy123222/Stellar-GreenPay' for t in tasks)
    validate_pool(tasks)
    pool, src = load_pool()
    assert src in ("prd14_rl_eligible_train", "data/rl_pool_v0.json", "data/rl_pool_v1.json")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data" / "splits.json").exists(), reason="needs task manifests or cluster specs withheld from the public release; see RELEASE_CHECKLIST.md")
def test_pool_guards_reject_dev_and_unknown_repos():
    import wayfinder.grpo.pool as P

    splits = json.loads((P.ROOT / "data/splits.json").read_text())
    dev = next(t for t, v in splits["tasks"].items() if v["split"] == "dev")
    bad = P._from_splits(splits, dev)
    with pytest.raises(ValueError, match="exclusion"):  # the 50 dev eval tasks are in the dev-list guard
        validate_pool([bad])
    with pytest.raises(ValueError, match="split"):  # and the split guard fires on its own
        validate_pool([bad], forbidden=set())
    ok = default_pool_tasks()[0]
    with pytest.raises(ValueError, match="not a known train repo"):
        validate_pool([{**ok, "task_id": "new1", "repo": "nobody/nothing"}])
    with pytest.raises(ValueError, match="exclusion"):
        validate_pool([ok], forbidden={ok["task_id"]})


def test_sample_batch_no_repeats_within_epoch_and_deterministic():
    pool = [{"task_id": f"t{i}"} for i in range(10)]
    seen = []
    for b in range(5):
        batch = sample_batch(pool, b, 2, seed=1)
        assert batch == sample_batch(pool, b, 2, seed=1)
        seen += [t["task_id"] for t in batch]
    assert sorted(seen) == sorted(t["task_id"] for t in pool)  # 5 batches of 2 = one full epoch, each task once
    nxt = [t["task_id"] for t in sample_batch(pool, 5, 2, seed=1)]
    assert len(set(nxt)) == 2
