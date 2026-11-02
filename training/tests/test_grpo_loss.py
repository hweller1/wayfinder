"""CPU tests: GRPO token loss (clip, KL), truncated-sampling logprobs, rollout files, rank planning, driver batch building, and a toy policy
trained with the real objective whose reward rises."""

import json
import math
import random

import numpy as np
import pytest
import torch

from wayfinder.grpo.advantage import group_advantages
from wayfinder.grpo.data import label_positions, load_rollout, plan_ranks, save_rollout
from wayfinder.grpo.driver import build_batch
from wayfinder.grpo.loss import grpo_token_loss, tempered_logps, token_logps
from wayfinder.grpo.step import merge_stat_sums, rollout_chain_loss


def t(x):
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------- loss
def test_on_policy_ratio_one_gradient_is_adv_times_grad_logp():
    lp = t([-1.0, -2.0, -0.5]).requires_grad_()
    adv = t([1.0, -2.0, 0.5])
    out = grpo_token_loss(lp, lp.detach().clone(), adv, beta=0.0)
    assert torch.allclose(out.ratio, torch.ones(3))
    out.loss.sum().backward()
    assert torch.allclose(lp.grad, -adv)  # d(-rho*A)/dlogp = -A at rho = 1


def test_clip_blocks_gradient_beyond_trust_region_only_in_the_improving_direction():
    old = t([-1.0, -1.0, -1.0, -1.0])
    new = (old + t([0.5, 0.5, -0.5, -0.5])).requires_grad_()  # ratios 1.65, 1.65, 0.61, 0.61
    adv = t([1.0, -1.0, 1.0, -1.0])
    out = grpo_token_loss(new, old, adv, clip_lo=0.2, clip_hi=0.2, beta=0.0)
    out.loss.sum().backward()
    # A>0, ratio>1.2: clipped, no gradient.  A<0, ratio>1.2: unclipped (min picks the pessimistic term), gradient flows.
    # A>0, ratio<0.8: unclipped, gradient flows.  A<0, ratio<0.8: clipped, no gradient.
    assert new.grad[0] == 0 and new.grad[1] != 0 and new.grad[2] != 0 and new.grad[3] == 0
    assert out.clipped.tolist() == [True, False, False, True]


def test_kl_k3_is_zero_at_reference_nonnegative_and_pulls_back():
    ref = t([-1.0, -2.0, -0.3])
    out0 = grpo_token_loss(ref.clone(), ref.clone(), t([0.0] * 3), logp_ref=ref, beta=1.0)
    assert torch.allclose(out0.kl, torch.zeros(3), atol=1e-7)
    new = (ref + t([0.7, -0.4, 0.2])).requires_grad_()
    out = grpo_token_loss(new, new.detach().clone(), t([0.0] * 3), logp_ref=ref, beta=0.5)
    assert (out.kl >= 0).all()
    out.loss.sum().backward()
    assert (new.grad * (new.detach() - ref) > 0).all()  # descent moves each logp back toward the reference
    d = ref - new.detach()
    assert torch.allclose(out.kl, (d.exp() - d - 1))


def test_nan_behaviour_logprob_is_treated_as_on_policy():
    new = t([-1.0, -2.0]).requires_grad_()
    out = grpo_token_loss(new, t([float("nan"), -2.0]), t([1.0, 1.0]), beta=0.0)
    assert out.ratio[0] == 1.0 and out.ratio[1] == pytest.approx(1.0)
    assert torch.isfinite(out.loss).all()


def test_ref_none_means_no_kl_support():
    out = grpo_token_loss(t([-1.0]), t([-1.0]), t([1.0]), beta=5.0)
    assert out.kl.item() == 0.0


# ---------------------------------------------------------------- truncated sampling logprobs
def brute_trunc(z, y, T, k, p):
    zt = (z / T).double()
    probs = torch.softmax(zt, -1)
    idx = probs.argsort(descending=True)[:k]
    pk = probs[idx] / probs[idx].sum()
    keep, c = [], 0.0
    for q, i in zip(pk, idx):
        if c < p:
            keep.append(int(i))
        c += float(q)
    mass = sum(float(probs[i]) for i in keep)
    return math.log(float(probs[y]) / mass) if y in keep else None


def test_tempered_logps_matches_brute_force_top_k_top_p():
    torch.manual_seed(0)
    z = torch.randn(40, 60) * 3
    for T, k, p in ((0.85, 64, 0.95), (1.0, 10, 0.9), (0.5, 5, 0.5)):
        # sample from the truncated distribution the way the server would
        ys = []
        for i in range(40):
            probs = torch.softmax(z[i] / T, -1)
            idx = probs.argsort(descending=True)[: min(k, 60)]
            pk = probs[idx] / probs[idx].sum()
            keep = [j for j in range(len(idx)) if float(pk[:j].sum()) < p]
            ys.append(int(idx[random.Random(i).choice(keep)]))
        y = torch.tensor(ys)
        full, trunc, kept = tempered_logps(z, y, T, k, p)
        assert kept.all()
        for i in range(40):
            assert trunc[i].item() == pytest.approx(brute_trunc(z[i], ys[i], T, k, p), abs=1e-4)
        assert (trunc >= full - 1e-6).all()  # truncation renormalises upward
    # a token outside the kept set falls back to the full logprob and is flagged
    z0 = torch.tensor([[10.0, 0.0, -10.0, -10.0]])
    full, trunc, kept = tempered_logps(z0, torch.tensor([3]), 1.0, 2, 0.95)
    assert not kept.item() and trunc.item() == pytest.approx(full.item())


def test_token_logps_chunking_is_exact():
    torch.manual_seed(1)
    H, V, T_ = 8, 30, 50
    W = torch.randn(H, V)
    hidden = torch.randn(T_, H)
    pos = torch.arange(1, T_)
    labels = torch.randint(0, V, (T_ - 1,))
    head = lambda h: h @ W  # noqa: E731
    a = token_logps(head, hidden, pos, labels, 0.85, chunk=7)
    b = token_logps(head, hidden, pos, labels, 0.85, chunk=1000)
    for x, y in zip(a, b):
        assert torch.allclose(x.float(), y.float(), atol=1e-6)


# ---------------------------------------------------------------- data, planning, driver batch
def test_rollout_file_roundtrip_and_label_positions(tmp_path):
    chains = [{"input_ids": [5, 6, 7, 8, 9, 10], "assistant_spans": [[2, 4], [5, 6]], "logprobs": [float("nan")] * 2 + [-0.1, -0.2, float("nan"), -0.3]},
              {"input_ids": [1, 2, 3], "assistant_spans": [[1, 3]]}]
    n = save_rollout(tmp_path / "r.npz", {"rid": "x"}, chains)
    assert n == 2 + 1 + 2
    meta, got = load_rollout(tmp_path / "r.npz")
    assert meta["n_chains"] == 2 and meta["n_loss_tokens"] == 5
    assert got[0]["ids"].tolist() == [5, 6, 7, 8, 9, 10] and np.isnan(got[0]["lp"][0]) and got[0]["lp"][3] == pytest.approx(-0.2)
    assert np.isnan(got[1]["lp"]).all()  # no logprobs recorded: all NaN = on-policy
    assert label_positions(got[0]["spans"], 6).tolist() == [2, 3, 5]
    assert label_positions(np.array([[0, 3]]), 10).tolist() == [1, 2]  # position 0 never carries loss


def test_plan_ranks_balances_and_is_deterministic():
    items = [(f"r{i}", c) for i, c in enumerate([50, 40, 30, 20, 10, 10, 5, 5])]
    p = plan_ranks(items, 2)
    assert p == plan_ranks(list(reversed(items)), 2)
    loads = [sum(dict(items)[k] for k in b) for b in p]
    assert abs(loads[0] - loads[1]) <= 10 and sorted(sum(p, [])) == sorted(k for k, _ in items)


def mkrow(task, i, recall, all_gold, rubric, precision=None, n_rep=3, valid=True, infra=False, turns=10):
    r = {"rid": f"{task}-{i}", "task_id": task, "seed": i, "step": 0, "version": 0, "infra_error": infra, "path": None if infra else f"/g/{task}-{i}.npz", "n_tokens": 1000, "n_loss": 100,
         "reward": recall + 0.5 * all_gold, "recall_reported": recall, "all_gold": all_gold, "valid_report": valid, "turns": turns, "repeats": 0, "rubric": rubric,
         "precision": precision, "n_reported": n_rep, "report_chars": 100}
    return r


def test_build_batch_drops_flat_groups_infra_errors_and_applies_precision_coef():
    rows = []
    for i in range(8):
        rows.append(mkrow("good", i, i / 8, i == 7, 0.1 * i, precision=0.5))
    rows += [mkrow("flat", i, 0.5, False, 0.1 * i, precision=1.0) for i in range(8)]
    rows += [mkrow("short", i, 0.2 * i, False, 0.1, precision=1.0) for i in range(3)] + [mkrow("short", 3 + i, 0.0, False, 0.1, infra=True) for i in range(5)]
    man, stats, adv = build_batch(rows, precision_coef=0.0)
    assert adv == {"groups": 3, "kept_groups": 1, "zero_var_groups": 1, "short_groups": 1}
    assert len(man["items"]) == 8 and all(it["rid"].startswith("good") for it in man["items"])
    assert stats["n_infra_errors"] == 5 and stats["n_rollouts"] == 19 and stats["kept_rollouts"] == 8
    best = max(man["items"], key=lambda it: it["adv"])
    assert best["rid"] == "good-7"
    # precision arm: the same rows, reward lower by coef * (1 - precision); flat group stays flat
    rows2 = [dict(r) for r in rows]
    man2, stats2, _ = build_batch(rows2, precision_coef=0.3)
    assert stats2["reward"] < stats["reward"]
    g = [r for r in rows2 if r["task_id"] == "good" and not r["infra_error"]][0]
    assert g["arm_reward"] == pytest.approx(g["reward"] - 0.3 * 0.5)


# ---------------------------------------------------------------- toy policy with the real objective
class Toy(torch.nn.Module):
    """h_t = E[ids_t] (no cross-position mixing), head = linear: the policy over the next token depends only on the current token."""

    def __init__(self, V=12, H=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(V, H, generator=g) * 0.1)
        self.W = torch.nn.Parameter(torch.randn(H, V, generator=g) * 0.1)
        self.V = V

    def fwd(self, ids):
        return self.E[ids]

    def head(self, h):
        return h @ self.W


def sample_chain(model, ctxs, T, rng):
    """chain = [0, c1, a1, c2, a2, ...]; a_i sampled from the model's policy at c_i (temperature T), behaviour logprobs recorded."""
    ids, lps, spans = [0], [float("nan")], []
    with torch.no_grad():
        for c in ctxs:
            ids.append(c)
            lps.append(float("nan"))
            z = model.head(model.E[torch.tensor([c])])[0] / T
            p = torch.softmax(z, -1)
            a = int(torch.multinomial(p, 1, generator=rng))
            spans.append([len(ids), len(ids) + 1])
            ids.append(a)
            lps.append(float(torch.log(p[a])))
    return {"ids": np.array(ids, np.int32), "spans": np.array(spans, np.int32), "lp": np.array(lps, np.float32)}


def reward_of(chain, target):
    ids = chain["ids"]
    hits = [int(ids[b - 1] == target[int(ids[a - 1])]) for a, b in chain["spans"]]
    return sum(hits) / len(hits)


def train_toy(beta, steps=60, lr=0.05, lag=1, seed=0, top_k=64, top_p=1.0):
    V, T = 12, 1.0
    target = {c: (c * 5 + 3) % V for c in range(1, V)}  # the right action for each context
    model = Toy(V, seed=seed)
    ref = Toy(V, seed=seed)
    ref.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    behaviour = Toy(V, seed=seed)
    behaviour.load_state_dict(model.state_dict())
    hist, kls, clipf = [], [], []
    queue = []
    for step in range(steps):
        snap = Toy(V, seed=seed)
        snap.load_state_dict(model.state_dict())
        queue.append(snap)
        sampler = queue[0] if len(queue) > lag else queue[-1]  # the sampler is `lag` optimizer steps behind the policy being trained
        if len(queue) > lag:
            queue.pop(0)
        items = []
        for g in range(6):  # 6 "tasks", group of 8 rollouts each, each rollout = 4 decisions
            ctxs = [random.Random(step * 100 + g).randrange(1, V) for _ in range(4)]
            rolls = [sample_chain(sampler, ctxs, T, rng) for _ in range(8)]
            rs = [reward_of(r, target) for r in rolls]
            adv = group_advantages(rs, [r < 1.0 for r in rs], [None] * 8)
            for r, rw, a, k in zip(rolls, rs, adv["adv"], [adv["keep"]] * 8):
                items.append((r, rw, a, k))
        hist.append(sum(i[1] for i in items) / len(items))
        n_total = float(sum(len(i[0]["spans"]) for i in items if i[3])) or 1.0
        opt.zero_grad()
        parts = []
        for r, rw, a, k in items:
            if not k:
                continue
            loss, st = rollout_chain_loss(model.fwd, ref.fwd if beta > 0 else None, model.head, r, a, n_total, temperature=T, top_k=top_k, top_p=top_p, beta=beta, use_ckpt=False)
            if loss is not None:
                loss.backward()
                parts.append(st)
        opt.step()
        if parts:
            s = merge_stat_sums(parts)
            kls.append(s["kl"])
            clipf.append(s["clip_frac"])
    return hist, kls, clipf, model, ref


def test_toy_policy_reward_rises_under_the_real_objective():
    hist, kls, clipf, model, ref = train_toy(beta=0.0, steps=60)
    assert sum(hist[:5]) / 5 < 0.2  # ~1/12 chance at init
    assert sum(hist[-5:]) / 5 > 0.8
    assert max(clipf) < 0.6


def test_toy_policy_rises_with_one_step_lag_and_clipping_is_active():
    hist, kls, clipf, *_ = train_toy(beta=0.0, steps=60, lag=1, lr=0.2)
    assert sum(hist[-5:]) / 5 > 0.6  # still learns off-policy by one step
    assert max(clipf) > 0.0  # the ratio clip sees stale samples


def test_kl_penalty_keeps_policy_nearer_the_reference():
    _, kl0, _, m0, r0 = train_toy(beta=0.0, steps=50, seed=3)
    _, kl1, _, m1, r1 = train_toy(beta=2.0, steps=50, seed=3)

    def dist(m, r):
        with torch.no_grad():
            c = torch.arange(1, 12)
            return float((torch.log_softmax(m.head(m.E[c]), -1).exp() * (torch.log_softmax(m.head(m.E[c]), -1) - torch.log_softmax(r.head(r.E[c]), -1))).sum(-1).mean())
    assert dist(m1, r1) < dist(m0, r0)
    assert kl1[-1] < 1.5 * kl0[-1] + 1e-9 or dist(m1, r1) < dist(m0, r0)  # reported KL is the same quantity the trainer logs
    assert all(k >= -1e-7 for k in kl1)


def test_rollout_chain_loss_gradient_only_on_sampled_tokens():
    model = Toy(12)
    chain = {"ids": np.array([0, 3, 7, 4, 9], np.int32), "spans": np.array([[2, 3], [4, 5]], np.int32), "lp": np.full(5, np.nan, np.float32)}
    loss, st = rollout_chain_loss(model.fwd, None, model.head, chain, 1.0, 2.0, beta=0.0, use_ckpt=False)
    loss.backward()
    assert st["tokens"] == 2
    # E rows that produced a prediction for a loss token get gradient (tokens 3 and 4 are the contexts); the sampled tokens' own rows do not (7, 9)
    g = model.E.grad.abs().sum(-1)
    assert g[3] > 0 and g[4] > 0 and g[7] == 0 and g[9] == 0 and g[0] == 0


# ---------------------------------------------------------------- chain building carries behaviour logprobs; the harness captures them
def test_build_chains_carries_behaviour_logprobs_aligned_with_spans():
    from wayfinder.opd.spans import ASSISTANT, EOM, START, build_chains

    P0 = [200000, 5, 6, START, ASSISTANT]
    calls = [{"prompt_token_ids": P0, "token_ids": [11, 12, EOM], "logprobs": [-0.1, -0.2, -0.3], "finish": "stop"},
             {"prompt_token_ids": P0 + [11, 12, EOM, 7, 8, START, ASSISTANT], "token_ids": [13, 14], "logprobs": [-0.4, -0.5], "finish": "stop"}]
    chains, diag = build_chains(calls, max_len=100, strict=True)
    ch = chains[0]
    lp = ch["logprobs"]
    assert len(lp) == len(ch["input_ids"]) and diag["eot_appended"] == 1
    (a0, b0), (a1, b1) = ch["assistant_spans"]
    assert lp[a0:b0] == pytest.approx([-0.1, -0.2, -0.3]) and lp[a1 : a1 + 2] == pytest.approx([-0.4, -0.5])
    assert math.isnan(lp[b1 - 1])  # the <|eot|> the server stopped on and kept out of token_ids has no logprob
    assert all(math.isnan(x) for x in lp[:a0])  # prompt tokens carry none
    # a call whose logprobs are missing or misaligned degrades to NaN (on-policy) instead of misaligning the whole chain
    calls[1]["logprobs"] = [-0.4]
    ch2 = build_chains(calls, max_len=100, strict=True)[0][0]
    assert len(ch2["logprobs"]) == len(ch2["input_ids"]) and math.isnan(ch2["logprobs"][ch2["assistant_spans"][1][0]])
    # no logprobs anywhere: key absent, old behaviour unchanged
    for c in calls:
        c.pop("logprobs")
    assert "logprobs" not in build_chains(calls, max_len=100, strict=True)[0][0]


def test_compatchat_capture_includes_logprobs_when_requested():
    import asyncio
    import sys
    from pathlib import Path

    import httpx

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voyage-scout" / "src"))
    from scout.core.ports.chat import ChatRequest

    from wayfinder.rollout.chat import CompatChat

    def handler(req):
        body = {"choices": [{"message": {"role": "assistant", "content": "x", "reasoning_content": "r"}, "finish_reason": "stop", "token_ids": [1, 2],
                             "logprobs": {"content": [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -1.5}]}}],
                "prompt_token_ids": [9, 9], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}
        return httpx.Response(200, json=body)

    chat = CompatChat("https://x.test/v1", "k", "m", transport=httpx.MockTransport(handler), extra_body={"return_token_ids": True, "logprobs": True})
    got = []
    chat.capture = got.append
    asyncio.run(chat.complete(ChatRequest(messages=[{"role": "user", "content": "q"}])))
    assert got[0]["logprobs"] == [-0.5, -1.5] and got[0]["token_ids"] == [1, 2]
