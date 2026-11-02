"""CPU tests for the forked-report mode (wayfinder/grpo/fork.py, docs/grpo_fork_report.md): K=0 identity, intra-prefix advantage assembly, zero-variance drop,
rank5b clipping, tie-break/padding, the loss-side span advantage, fork sequences and the padding guard."""

import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

from wayfinder.grpo.driver import build_batch
from wayfinder.grpo.fork import (apply_forks, compact_parts, fork_chain, fork_diagnostics, intra_advantages, pick_train, report_args, sample_forks,
                                 score_fork_report)
from wayfinder.grpo.reward import compute_reward_rank5b
from wayfinder.grpo.step import rollout_chain_loss, span_adv_vector
from wayfinder.opd.spans import ASSISTANT, EOM, EOT, START

GOLD = ["g1", "g2"]


def row(rep, gold=GOLD, turns=10):
    return {"gold_files": gold, "turns": turns, "error": None, "messages": [],
            "explore_result": {"results": [{"path": p, "start_line": 1, "end_line": 2, "why": "w"} for p in rep], "budget_exhausted": False}}


def parts(rep, gold=GOLD, turns=10):
    return compact_parts(compute_reward_rank5b(row(rep, gold, turns)))


def rollout_row(task, i, rep, gold=GOLD, turns=10):
    p = compute_reward_rank5b(row(rep, gold, turns))
    p.update({"task_id": task, "rid": f"{task}-{i}", "path": f"/p/{task}-{i}", "n_tokens": 1000 + i, "n_loss": 100 + i, "rubric": None, "report_chars": 10})
    return p


# ------------------------------------------------------------------------------------------------------------------------------ K = 0 identity
def golden_rows():
    specs = {"t1": [(["g1", "g2"], 10), (["g1", "x"], 12), (["x"], 9), (["g1", "g2", "x", "y", "z", "w"], 10), (["g2"], 10), (["g1", "g2"], 30), (["x", "y"], 5), (["g1"], 8)],
             "t2": [(["a"], 10)] * 8}
    rows = []
    for t, sp in specs.items():
        gold = ["g1", "g2"] if t == "t1" else ["a"]
        for i, (rep, tu) in enumerate(sp):
            p = compute_reward_rank5b(row(rep, gold, tu))
            p.update({"task_id": t, "rid": f"{t}-{i}", "path": f"/p/{t}-{i}", "n_tokens": 1000 + i, "n_loss": 100 + i, "rubric": None, "report_chars": 10, "recall_touched": 0.5,
                      "tokens_in": 5, "wall_s": 1.0})
            rows.append(p)
    return rows


def test_k0_manifest_and_advantages_identical_to_pre_fork_golden():
    """The golden file was produced by build_batch BEFORE the fork code existed (same inputs); K=0 (fork_lambda None) must reproduce it exactly."""
    man, stats, adv = build_batch(golden_rows(), 0.0, lam=0.0)
    golden = json.loads((Path(__file__).parent / "data" / "grpo_k0_golden.json").read_text())
    assert json.loads(json.dumps({"manifest": man, "adv": adv, "stats": stats}, sort_keys=True)) == golden
    assert "fork" not in adv and "fork" not in stats["diag"] and all("fork" not in it for it in man["items"])


def test_k0_rows_without_fork_records_are_untouched_even_when_the_mode_is_on():
    base = build_batch(golden_rows(), 0.0, lam=0.0)[0]
    on = build_batch(golden_rows(), 0.0, lam=0.0, fork_lambda=1.0)[0]
    assert on == base  # no row carries a fork record: nothing is added


def test_train_and_app_defaults_are_off():
    import inspect

    from wayfinder.grpo import driver

    assert inspect.signature(build_batch).parameters["fork_lambda"].default is None
    src = inspect.getsource(driver.Driver.run)
    assert 'getattr(a, "fork_report", 0) > 0 else None' in src


# ------------------------------------------------------------------------------------------------------------------------------ intra advantages
def test_intra_group_advantage_basic_ordering_and_normalisation():
    ps = [parts(["g1", "g2"]), parts(["g1", "x"]), parts(["x"]), parts(["g2", "g1"])]
    g = intra_advantages(ps)
    assert g["keep"] and len(g["adv"]) == 4
    assert g["adv"][0] > 0 and g["adv"][3] > 0 and g["adv"][2] <= 0
    m = sum(g["rewards"]) / 4
    sd = (sum((r - m) ** 2 for r in g["rewards"]) / 4) ** 0.5
    assert sd > 0.1  # above the floor: exact normalisation
    assert sum(g["adv"]) == pytest.approx(0.0, abs=1.0)  # clipping may shift the mean, never by more than the clipped mass
    assert g["adv"][1] == pytest.approx((g["rewards"][1] - m) / sd)


def test_zero_variance_prefix_is_dropped():
    ps = [parts(["g1", "g2"])] * 5
    g = intra_advantages(ps)
    assert not g["keep"] and g["adv"] == [0.0] * 5
    # identical file lists with identical turns (forks share the search): the tie-break has nothing to rank
    assert not intra_advantages([parts(["g1", "x"]), parts(["g1", "x"])])["keep"]


def test_below_floor_spread_is_shrunk_not_amplified():
    # same recall, only the report length differs: the tie-break spread is <= 0.05, far below the 0.1 floor
    gold = ["g1", "g2", "g3"]  # not all found, so the all-gold clip is out of play
    g = intra_advantages([parts(["g1", "g2"], gold), parts(["g1", "g2", "x", "y"], gold), parts(["g1", "g2", "x", "y", "z", "w"], gold)])
    assert g["keep"] and max(abs(a) for a in g["adv"]) < 0.5
    assert g["adv"][0] > g["adv"][1] > g["adv"][2]  # shorter report first: the padding guard inside the group


def test_correctness_consistent_clip_in_intra_group():
    # all-gold-in-top-8 never negative (even at the bottom of its group), zero recall never positive
    ps = [parts(["g1", "g2"]), parts(["g1", "g2", "x", "y", "z", "w", "v"]), parts(["x"]), parts(["y"])]
    g = intra_advantages(ps)
    assert g["adv"][1] >= 0.0
    assert g["adv"][2] <= 0.0 and g["adv"][3] <= 0.0
    # a single strong report among failures: nothing is clipped negative for it, failures never positive
    g = intra_advantages([parts(["x"]), parts(["y"]), parts(["z"]), parts(["g1", "g2"])])
    assert g["adv"][3] > 0 and all(a <= 0.0 for a in g["adv"][:3])


def test_advantage_clip_bound():
    ps = [parts(["g1", "g2"])] + [parts(["x"])] * 30
    g = intra_advantages(ps)
    assert g["keep"] and max(abs(a) for a in g["adv"]) <= 5.0 + 1e-9


def test_no_report_fork_is_penalised():
    bad = score_fork_report(None, {"gold_files": GOLD}, {"turns": 10})
    assert bad["valid_report"] is False and bad["reward"] == pytest.approx(-0.5)
    ok = score_fork_report(["g1", "g2"], {"gold_files": GOLD}, {"turns": 10})
    assert ok["valid_report"] and ok["all_gold_window"]
    g = intra_advantages([ok, bad, bad, ok])
    assert g["adv"][0] > 0 and g["adv"][1] < 0


# ------------------------------------------------------------------------------------------------------------------------------ assembly
def fork_rows():
    rows = [rollout_row("t", i, rep) for i, rep in enumerate([["g1", "g2"], ["g1", "x"], ["x"], ["g1", "g2", "x"], ["g2"], ["g1", "g2"], ["x", "y"], ["g1"]])]
    for r in rows:
        r.update({"all_gold": r["all_gold"], "valid_report": True})
    # original 0: intra group varies; original 1: forks identical to it (zero variance); original 2: fork record but inter group dropped is simulated below
    rows[0]["fork"] = {"parts": [parts(["g1", "g2"]), parts(["x"]), parts(["g1"])], "ref": [0, 3], "scale": 1.0,
                       "train": [{"j": 1, "path": "/f/0-f1", "n_loss": 50, "n_tokens": 900}, {"j": 2, "path": "/f/0-f2", "n_loss": 60, "n_tokens": 910}]}
    rows[1]["fork"] = {"parts": [parts(["g1", "x"])] * 3, "ref": [0, 1], "scale": 1.0, "train": [{"j": 0, "path": "/f/1-f0", "n_loss": 50, "n_tokens": 900}]}
    return rows


def test_assembly_original_gets_intra_term_and_forks_get_intra_only():
    rows = fork_rows()
    man, stats, adv = build_batch(rows, 0.0, lam=0.0, fork_lambda=1.0)
    by = {it["rid"]: it for it in man["items"]}
    g = intra_advantages([compact_parts(rows[0])] + rows[0]["fork"]["parts"])
    assert by["t-0"]["fork"] == {"chain": 0, "span": 3, "adv": pytest.approx(round(g["adv"][0], 6))}
    # the inter advantage is untouched (identical to the K=0 build) and the fork term is carried separately
    base = {it["rid"]: it["adv"] for it in build_batch(fork_rows(), 0.0, lam=0.0)[0]["items"]}
    assert by["t-0"]["adv"] == base["t-0"]
    assert by["t-0-f1"]["adv"] == pytest.approx(round(g["adv"][2], 6)) and by["t-0-f1"]["path"] == "/f/0-f1"
    assert by["t-0-f2"]["adv"] == pytest.approx(round(g["adv"][3], 6))
    assert "fork" not in by["t-0-f1"]
    # zero-variance prefix: no fork term on the original, no fork items
    assert "fork" not in by["t-1"] and "t-1-f0" not in by
    assert adv["fork"]["n_prefixes"] == 2 and adv["fork"]["kept"] == 1 and adv["fork"]["zero_var"] == 1 and adv["fork"]["fork_items"] == 2


def test_lambda_fork_scales_intra_term_only():
    rows = fork_rows()
    man1 = build_batch(rows, 0.0, lam=0.0, fork_lambda=1.0)[0]
    man2 = build_batch(fork_rows(), 0.0, lam=0.0, fork_lambda=0.5)[0]
    a1 = {it["rid"]: it for it in man1["items"]}
    a2 = {it["rid"]: it for it in man2["items"]}
    assert a2["t-0"]["adv"] == a1["t-0"]["adv"]
    assert a2["t-0"]["fork"]["adv"] == pytest.approx(0.5 * a1["t-0"]["fork"]["adv"], abs=2e-6)
    assert a2["t-0-f1"]["adv"] == pytest.approx(0.5 * a1["t-0-f1"]["adv"], abs=2e-6)


def test_dropped_inter_group_still_trains_forks_but_not_the_original():
    rows = [rollout_row("flat", i, ["g1"]) for i in range(8)]  # zero inter variance: the group is dropped
    rows[0]["fork"] = {"parts": [parts(["g1", "g2"]), parts(["x"]), parts(["g1"])], "ref": [0, 1], "scale": 1.0, "train": [{"j": 0, "path": "/f/a", "n_loss": 5, "n_tokens": 99}]}
    man, _, adv = build_batch(rows, 0.0, lam=0.0, fork_lambda=1.0)
    assert [it["rid"] for it in man["items"]] == ["flat-0-f0"] and adv["fork"]["orig_untrained"] == 1


def test_pick_train_is_deterministic_and_bounded():
    assert pick_train(8, 2, "r") == pick_train(8, 2, "r") and len(pick_train(8, 2, "r")) == 2 and len(pick_train(3, 5, "r")) == 3 and pick_train(4, 0, "r") == []


# ------------------------------------------------------------------------------------------------------------------------------ loss side
class Toy(torch.nn.Module):
    def __init__(self, V=12, H=16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.E = torch.nn.Parameter(torch.randn(V, H, generator=g) * 0.1)
        self.W = torch.nn.Parameter(torch.randn(H, V, generator=g) * 0.1)

    def fwd(self, ids):
        return self.E[ids]

    def head(self, h):
        return h @ self.W


def test_span_adv_vector_alignment():
    v = span_adv_vector(np.array([[2, 4], [6, 7]]), 10, {1: 0.5})
    assert v.tolist() == [0.0, 0.0, 0.5]
    assert span_adv_vector(np.array([[0, 2]]), 10, {0: 1.0}).tolist() == [1.0]  # position 0 has no label (j >= 1)


def test_report_span_loss_is_inter_plus_lambda_intra():
    m = Toy()
    chain = {"ids": np.array([0, 3, 7, 4, 9], np.int32), "spans": np.array([[2, 3], [4, 5]], np.int32), "lp": np.full(5, np.nan, np.float32)}
    # on-policy (old = NaN -> ratio 1): pg = -adv per token
    _, st0 = rollout_chain_loss(m.fwd, None, m.head, chain, 0.7, 2.0, beta=0.0, use_ckpt=False)
    _, st1 = rollout_chain_loss(m.fwd, None, m.head, chain, 0.7, 2.0, beta=0.0, use_ckpt=False, span_adv={1: 0.3})
    assert st0["adv_tok"] == pytest.approx(0.7 * 2) and st0["pg"] == pytest.approx(-0.7 * 2)
    assert st1["pg"] == pytest.approx(-(0.7 + 0.7 + 0.3))  # search span: inter only; report span: inter + intra
    # no span_adv -> identical to before
    l0, _ = rollout_chain_loss(m.fwd, None, m.head, chain, 0.7, 2.0, beta=0.0, use_ckpt=False)
    l1, _ = rollout_chain_loss(m.fwd, None, m.head, chain, 0.7, 2.0, beta=0.0, use_ckpt=False, span_adv=None)
    assert l0.item() == l1.item()


def test_fork_item_ratio_uses_vllm_logprobs():
    """A forked sequence is a normal chain whose behaviour logprobs are vLLM's: the ratio is exp(new - old), not forced to 1."""
    m = Toy()
    with torch.no_grad():
        h = m.fwd(torch.tensor([0, 3, 7]))
        lp_true = torch.log_softmax(m.head(h[1]) / 0.85, -1)[7].item()
    chain = {"ids": np.array([0, 3, 7], np.int32), "spans": np.array([[2, 3]], np.int32), "lp": np.array([np.nan, np.nan, lp_true - 0.5], np.float32)}
    _, st = rollout_chain_loss(m.fwd, None, m.head, chain, 1.0, 1.0, beta=0.0, use_ckpt=False, top_k=12, top_p=1.0)
    assert st["abs_logratio"] == pytest.approx(0.5, abs=1e-3)


# ------------------------------------------------------------------------------------------------------------------------------ sampling side
def test_fork_chain_layout_logprobs_and_terminator():
    P = [200000, 5, 6, START, ASSISTANT]
    c = fork_chain(P, {"token_ids": [11, 12, EOM], "logprobs": [-0.1, -0.2, -0.3], "finish": "tool_calls"})
    assert c["input_ids"] == P + [11, 12, EOM] and c["assistant_spans"] == [[5, 8]]
    assert c["logprobs"][:5] == [pytest.approx(math.nan, nan_ok=True)] * 5 and c["logprobs"][5:] == [-0.1, -0.2, -0.3]
    c = fork_chain(P, {"token_ids": [11, 12], "logprobs": [-0.1, -0.2], "finish": "stop"})  # server stopped on <|eot|>, left it out
    assert c["input_ids"][-1] == EOT and c["assistant_spans"] == [[5, 8]] and math.isnan(c["logprobs"][-1]) and c["logprobs"][5:7] == [-0.1, -0.2]
    assert fork_chain(P, {"token_ids": [11, 12], "logprobs": [-0.1, -0.2], "finish": "length"}) is None  # cut by max_tokens
    assert fork_chain([1, 2, 3], {"token_ids": [11, EOM], "finish": "stop"}) is None  # prompt does not end with the assistant header
    c = fork_chain(P, {"token_ids": [11, EOM], "logprobs": [-0.1], "finish": "stop"})  # misaligned logprobs -> all NaN (on-policy)
    assert all(math.isnan(x) for x in c["logprobs"])


def test_report_args_parsing():
    tc = [{"function": {"name": "grep", "arguments": "{}"}}, {"function": {"name": "report", "arguments": json.dumps({"results": [], "summary": "s"})}}]
    assert report_args(tc) == {"results": [], "summary": "s"}
    assert report_args([{"function": {"name": "report", "arguments": "{bad"}}]) is None and report_args([]) is None


def test_sample_forks_requests_n_and_parses_logprobs(monkeypatch):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        ch = [{"message": {"tool_calls": []}, "finish_reason": "stop", "token_ids": [1, 2], "logprobs": {"content": [{"logprob": -1.0}, {"logprob": -2.0}]}} for _ in range(3)]
        return httpx.Response(200, json={"choices": ch, "prompt_token_ids": [9, 9], "usage": {"completion_tokens": 6}})

    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *a, **kw):
            super().__init__(*a, **{**kw, "transport": httpx.MockTransport(handler)})

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    import asyncio

    got = asyncio.run(sample_forks("http://x/v1/chat/completions", {}, {"model": "m", "messages": [], "seed": 7, "temperature": 0.85, "top_p": 0.95, "logprobs": True}, 3))
    assert seen["n"] == 3 and "seed" not in seen and seen["temperature"] == 0.85 and seen["top_p"] == 0.95 and seen["logprobs"] is True
    assert len(got["samples"]) == 3 and got["samples"][0]["logprobs"] == [-1.0, -2.0] and got["prompt_token_ids"] == [9, 9]


# ------------------------------------------------------------------------------------------------------------------------------ padding guard
def test_fork_padding_guard_flags_longer_less_precise_forks():
    rows = []
    for i in range(4):
        r = rollout_row("t", i, ["g1", "x"])
        r.update({"precision": 0.5, "recall_at_5": 0.5})
        r["fork"] = {"parts": [dict(parts(["g1", "x", "a", "b", "c", "d", "e"]), precision=0.14, recall_at_5=0.5, n_reported=7)] * 2}
        rows.append(r)
    d = fork_diagnostics(rows)
    assert d["n_prefixes"] == 4 and d["n_forks"] == 8 and d["delta"]["n_reported"] == pytest.approx(5.0) and d["flagged_padding"] is True
    assert d["fork_n_reported_hist"] == {"7": 8}
    for r in rows:  # forks no longer than the originals: not flagged
        r["fork"]["parts"] = [dict(parts(["g1", "x"]), precision=0.5, recall_at_5=0.5, n_reported=2)] * 2
    assert fork_diagnostics(rows)["flagged_padding"] is False
    assert fork_diagnostics([rollout_row("t", 0, ["g1"])]) == {"n_prefixes": 0}


def test_validate_report_paths_uses_scouts_path_guard(tmp_path):
    import sys

    pytest.importorskip("pathspec")  # scout's ignore matcher; present in the worker image, optional on a laptop

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voyage-scout" / "src"))
    from wayfinder.grpo.rollout import validate_report_paths

    (tmp_path / "a.py").write_text("x\ny\n")
    ok = {"results": [{"path": "a.py", "start_line": 1, "end_line": 9, "why": "w"}, {"path": "missing.py", "start_line": 1, "end_line": 2, "why": "w"},
                      {"path": "../etc/passwd", "start_line": 1, "end_line": 2, "why": "w"}], "summary": "s"}
    assert validate_report_paths(ok, str(tmp_path)) == ["a.py"]
    assert validate_report_paths({"results": "nope"}, str(tmp_path)) is None
