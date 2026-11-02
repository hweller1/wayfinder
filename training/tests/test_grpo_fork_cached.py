"""Cached-prefix fork training (wayfinder/grpo/cached_fork.py, --fork-cached): CPU tests on a tiny random MuseGlimmer (the real architecture: sliding + full layers,
NoPE layers, QK-norm, gated attention, softcapped head) wrapped with LoRA through the same peft config as sft.train.load_model. Needs transformers with
MuseGlimmer and peft (skipped otherwise)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("peft")
tfm = pytest.importorskip("transformers")
if not hasattr(tfm, "MuseGlimmerForConditionalGeneration"):
    pytest.skip("transformers without MuseGlimmer", allow_module_level=True)

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration  # noqa: E402

from wayfinder.grpo.cached_fork import (cache_nbytes, freeze_cache, prefix_cache, split_point, tail_hidden, train_prefix_group)  # noqa: E402
from wayfinder.grpo.data import fork_parent, plan_ranks_grouped, step_normaliser  # noqa: E402
from wayfinder.grpo.loss import token_logps  # noqa: E402
from wayfinder.grpo.step import rollout_chain_loss  # noqa: E402
from wayfinder.sft.train import LORA_TARGETS  # noqa: E402

V, WINDOW = 97, 8
ATOL = 1e-5
RESULTS: dict[str, object] = {}  # measured quantities, printed at the end of the module (pytest -s)


def tiny_cfg():
    return MuseGlimmerConfig(
        text_config=dict(vocab_size=V, hidden_size=48, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         sliding_window=WINDOW, bos_token_id=1, eos_token_id=2),
        vision_config=dict(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, intermediate_size=64), out_hidden_size=48, projector_hidden_size=32)


def build(attn: str, seed: int = 0, ckpt: bool = False):
    torch.manual_seed(seed)
    m = MuseGlimmerForConditionalGeneration._from_config(tiny_cfg(), attn_implementation=attn, dtype=torch.float32)
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # exactly load_model's setting
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none"))
    g = torch.Generator().manual_seed(seed + 1)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data = torch.randn(p.shape, generator=g) * 0.05  # B = 0 at init would make every LoRA-A gradient zero
    m.config.use_cache = False
    pm.train()
    return pm


def inner_of(pm):
    return pm.base_model.model


def head_of(pm):
    inner = inner_of(pm)
    tc = inner.config.text_config

    def f(h):
        return torch.tanh(inner.lm_head(h).float() * tc.output_multiplier / tc.final_logit_softcapping) * tc.final_logit_softcapping

    return f


def lm_of(pm):
    return inner_of(pm).model.language_model


def chain(n_pre: int, n_comp: int, seed: int, prefix: np.ndarray | None = None):
    """A fork sequence: n_pre prefix tokens (last = the assistant-start token) + n_comp completion tokens, one loss span over the completion."""
    rng = np.random.default_rng(seed)
    pre = prefix if prefix is not None else rng.integers(3, V, n_pre)
    comp = rng.integers(3, V, n_comp)
    return {"ids": np.concatenate([pre, comp]).astype(np.int32), "spans": np.array([[n_pre, n_pre + n_comp]], np.int32), "lp": np.full(n_pre + n_comp, np.nan, np.float32)}


def logps_full(pm, ids: np.ndarray, pos: np.ndarray):
    lm = lm_of(pm)
    x = torch.from_numpy(ids.astype(np.int64))[None]
    h = lm(input_ids=x, position_ids=torch.arange(x.shape[1])[None], use_cache=False).last_hidden_state[0]
    y = x[0][torch.from_numpy(pos)]
    return token_logps(head_of(pm), h, torch.from_numpy(pos), y, 1.0, 64, False, 1000, 1.0)[0]


def logps_cached(pm, cache, ids: np.ndarray, P: int, pos: np.ndarray, pos_start: int | None = None):
    lm = lm_of(pm)
    x = torch.from_numpy(ids.astype(np.int64))
    h = tail_hidden(lm, cache, x[P:], P if pos_start is None else pos_start)
    return token_logps(head_of(pm), h, torch.from_numpy(pos - P), x[torch.from_numpy(pos)], 1.0, 64, False, 1000, 1.0)[0]


def positions(ch):
    a, b = ch["spans"][0]
    return np.arange(a, b, dtype=np.int64)


@pytest.fixture(scope="module", params=["sdpa", "eager"])
def pm_nockpt(request):
    return build(request.param)


# ------------------------------------------------------------------------------------------------------------------------------ 1. forward equivalence
@pytest.mark.parametrize("n_pre,n_comp", [(3, 2), (7, 5), (8, 6), (9, 4), (21, 7), (40, 13), (64, 3)])  # prefixes shorter than, equal to and longer than the window (8)
def test_cached_logprobs_equal_full_forward(pm_nockpt, n_pre, n_comp):
    pm = pm_nockpt
    ch = chain(n_pre, n_comp, seed=n_pre * 100 + n_comp)
    P = split_point(ch["spans"])
    assert P == n_pre - 1
    pos = positions(ch)
    with torch.no_grad():
        ref = logps_full(pm, ch["ids"], pos)
        cache = prefix_cache(lm_of(pm), torch.from_numpy(ch["ids"][:P].astype(np.int64)))
        got = logps_cached(pm, cache, ch["ids"], P, pos)
    err = float((ref - got).abs().max())
    RESULTS.setdefault("t1_max_abs_logp_err", []).append(err)
    assert err < ATOL, err


@pytest.mark.parametrize("n_pre,n_comp_list", [(12, [3, 5, 4, 5]), (30, [6, 2, 6, 6, 1, 4])])
def test_batch_of_k_completions_sharing_one_prefix(pm_nockpt, n_pre, n_comp_list):
    pm = pm_nockpt
    rng = np.random.default_rng(n_pre)
    pre = rng.integers(3, V, n_pre)
    chains = [chain(n_pre, c, seed=n_pre + i, prefix=pre) for i, c in enumerate(n_comp_list)]
    P = n_pre - 1
    T = max(n_comp_list) + 1
    with torch.no_grad():
        cache = prefix_cache(lm_of(pm), torch.from_numpy(pre[:P].astype(np.int64)))
        batch = torch.zeros(len(chains), T, dtype=torch.long)  # right-padded tails
        for i, ch in enumerate(chains):
            t = ch["ids"][P:]
            batch[i, : len(t)] = torch.from_numpy(t.astype(np.int64))
        h = tail_hidden(lm_of(pm), cache, batch, P)  # [K, T, H] in ONE forward over the shared cache
        for i, ch in enumerate(chains):
            pos = positions(ch)
            ref = logps_full(pm, ch["ids"], pos)
            got = token_logps(head_of(pm), h[i], torch.from_numpy(pos - P), torch.from_numpy(ch["ids"].astype(np.int64))[torch.from_numpy(pos)], 1.0, 64, False, 1000, 1.0)[0]
            assert float((ref - got).abs().max()) < ATOL
            single = logps_cached(pm, cache, ch["ids"], P, pos)  # the trainer's per-fork loop gives the same numbers
            assert float((single - got).abs().max()) < ATOL


# ------------------------------------------------------------------------------------------------------------------------------ 2. positions
def test_position_offset_is_load_bearing(pm_nockpt):
    pm = pm_nockpt
    ch = chain(21, 6, seed=5)
    P, pos = split_point(ch["spans"]), positions(ch)
    with torch.no_grad():
        ref = logps_full(pm, ch["ids"], pos)
        cache = prefix_cache(lm_of(pm), torch.from_numpy(ch["ids"][:P].astype(np.int64)))
        good = logps_cached(pm, cache, ch["ids"], P, pos)
        bad = logps_cached(pm, cache, ch["ids"], P, pos, pos_start=0)  # position_ids restart at 0: no offset
    assert float((ref - good).abs().max()) < ATOL
    err_bad = float((ref - bad).abs().max())
    RESULTS["t2_err_without_offset"] = err_bad
    assert err_bad > 1e-3, err_bad  # the equivalence test of (1) is sensitive to it
    with torch.no_grad():  # omitting position_ids entirely is also right: HF derives them from the cache length
        h = lm_of(pm)(input_ids=torch.from_numpy(ch["ids"][P:].astype(np.int64))[None], past_key_values=cache, use_cache=False).last_hidden_state[0]
        dflt = token_logps(head_of(pm), h, torch.from_numpy(pos - P), torch.from_numpy(ch["ids"].astype(np.int64))[torch.from_numpy(pos)], 1.0, 64, False, 1000, 1.0)[0]
    assert float((ref - dflt).abs().max()) < ATOL


# ------------------------------------------------------------------------------------------------------------------------------ 3. cache immutability
def snapshot(cache):
    return [(l.keys.clone(), l.values.clone(), l.get_seq_length()) for l in cache.layers]


def test_plain_dynamic_cache_is_mutated_by_a_forward_negative_control(pm_nockpt):
    """Why freeze_cache exists: an unfrozen DynamicCache grows on every forward, so fork j+1 would see fork j's keys."""
    from transformers import DynamicCache

    pm = pm_nockpt
    ch = chain(20, 5, seed=2)
    P = split_point(ch["spans"])
    lm = lm_of(pm)
    cache = DynamicCache(config=lm.config)
    with torch.no_grad():
        lm(input_ids=torch.from_numpy(ch["ids"][:P].astype(np.int64))[None], past_key_values=cache, use_cache=True)
        before = cache.get_seq_length()
        lm(input_ids=torch.from_numpy(ch["ids"][P:].astype(np.int64))[None], position_ids=torch.arange(P, len(ch["ids"]))[None], past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == before + len(ch["ids"]) - P


def test_forks_do_not_mutate_the_shared_cache(pm_nockpt):
    pm = pm_nockpt
    pre = np.random.default_rng(3).integers(3, V, 25)
    chains = [chain(25, c, seed=40 + c, prefix=pre) for c in (4, 7, 2, 7)]
    P = 24
    with torch.no_grad():
        cache = prefix_cache(lm_of(pm), torch.from_numpy(pre[:P].astype(np.int64)))
        snap = snapshot(cache)
        first = [logps_cached(pm, cache, ch["ids"], P, positions(ch)) for ch in chains]
        for (k, v, n), l in zip(snap, cache.layers):
            assert torch.equal(k, l.keys) and torch.equal(v, l.values) and n == l.get_seq_length() == P  # bit-identical after the forwards
        second = [logps_cached(pm, cache, ch["ids"], P, positions(ch)) for ch in reversed(chains)][::-1]  # other order, same cache object
        for a, b, ch in zip(first, second, chains):
            assert torch.equal(a, b)
            assert float((a - logps_full(pm, ch["ids"], positions(ch))).abs().max()) < ATOL
    assert cache_nbytes(cache) > 0


def test_freeze_cache_is_idempotent_and_keeps_class_hierarchy(pm_nockpt):
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

    cache = prefix_cache(lm_of(pm_nockpt), torch.arange(3, 20))
    assert isinstance(cache, DynamicCache)
    kinds = {type(l).__mro__[1] for l in cache.layers}
    assert kinds == {DynamicLayer, DynamicSlidingWindowLayer}
    assert freeze_cache(cache) is cache  # a second call changes nothing
    assert all(l._read_only for l in cache.layers)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_gradient_checkpointing_matches_the_non_checkpointed_cached_path(attn):
    """Checkpoint recompute re-runs each layer's forward on the backward pass; over a plain DynamicCache that appends KV a second time, and HF's
    GradientCheckpointingLayer silently drops past_key_values for layers that write the cache. The read-only cache + `_can_checkpoint_with_cache` fixes both."""
    pm_c, pm_n = build(attn, ckpt=True), build(attn, ckpt=False)
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(pm_c.named_parameters(), pm_n.named_parameters()))  # same weights
    ch = chain(21, 9, seed=11)
    P, pos = split_point(ch["spans"]), positions(ch)
    ref_full = logps_full(pm_n, ch["ids"], pos).detach()
    outs = {}
    for name, pm in (("ckpt", pm_c), ("plain", pm_n)):
        pm.zero_grad(set_to_none=True)
        cache = prefix_cache(lm_of(pm), torch.from_numpy(ch["ids"][:P].astype(np.int64)))  # runs with HF checkpointing switched off, and restores it
        snap = snapshot(cache)
        lp = logps_cached(pm, cache, ch["ids"], P, pos)
        (lp * torch.linspace(0.5, 1.5, lp.numel())).sum().backward()
        for (k, v, n), l in zip(snap, cache.layers):
            assert torch.equal(k, l.keys) and torch.equal(v, l.values) and l.get_seq_length() == P  # backward recompute did not append
        outs[name] = (lp.detach(), {n: p.grad.clone() for n, p in pm.named_parameters() if p.grad is not None})
    assert any(l.gradient_checkpointing for l in lm_of(pm_c).modules() if hasattr(l, "gradient_checkpointing") and l.__class__.__name__.endswith("DecoderLayer"))  # restored
    assert float((outs["ckpt"][0] - ref_full).abs().max()) < ATOL
    assert float((outs["ckpt"][0] - outs["plain"][0]).abs().max()) < ATOL
    assert outs["ckpt"][1].keys() == outs["plain"][1].keys() and len(outs["plain"][1]) > 0
    for n, g in outs["plain"][1].items():
        assert torch.allclose(outs["ckpt"][1][n], g, atol=1e-6, rtol=1e-4), n


def test_hf_checkpointing_drops_the_cache_negative_control():
    """Without the context managers a checkpointed (training-mode) forward over a filled cache has `past_key_values` stripped by HF: it either raises (mask built
    for prefix+tail, attention over the tail only) or computes the wrong thing. Either way the managers in cached_fork are required."""
    pm = build("sdpa", ckpt=True)
    ch = chain(21, 6, seed=1)
    P, pos = split_point(ch["spans"]), positions(ch)
    cache = prefix_cache(lm_of(pm), torch.from_numpy(ch["ids"][:P].astype(np.int64)))
    x = torch.from_numpy(ch["ids"][P:].astype(np.int64))[None]
    try:
        h = lm_of(pm)(input_ids=x, position_ids=torch.arange(P, len(ch["ids"]))[None], past_key_values=cache, use_cache=False).last_hidden_state[0]  # no override
    except RuntimeError:
        RESULTS["t3_hf_ckpt_drop"] = "raises"
        return
    bad = token_logps(head_of(pm), h, torch.from_numpy(pos - P), torch.from_numpy(ch["ids"].astype(np.int64))[torch.from_numpy(pos)], 1.0, 64, False, 1000, 1.0)[0]
    assert float((bad - logps_full(pm, ch["ids"], pos)).abs().max()) > 1e-3
    RESULTS["t3_hf_ckpt_drop"] = "silently wrong"


# ------------------------------------------------------------------------------------------------------------------------------ 4. gradient semantics
def lora_grads(pm):
    return {n: p.grad.detach().clone() for n, p in pm.named_parameters() if p.grad is not None}


def flat(g):
    return torch.cat([g[k].flatten() for k in sorted(g)])


def weights_for(n):
    return torch.linspace(0.5, 1.5, n)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("n_pre,n_comp", [(21, 9), (40, 12)])
def test_cached_grad_equals_full_forward_grad_with_prefix_kv_detached(attn, n_pre, n_comp, monkeypatch):
    from transformers.modeling_utils import AttentionInterface

    pm = build(attn)
    ch = chain(n_pre, n_comp, seed=7)
    P, pos = split_point(ch["spans"]), positions(ch)
    ids = torch.from_numpy(ch["ids"].astype(np.int64))
    # cached path
    pm.zero_grad(set_to_none=True)
    cache = prefix_cache(lm_of(pm), ids[:P])
    lp_c = logps_cached(pm, cache, ch["ids"], P, pos)
    (lp_c * weights_for(lp_c.numel())).sum().backward()
    g_cached = lora_grads(pm)
    # reference A: full forward, prefix K/V detached inside every attention call (the prefix's queries/hidden states still carry grad, but nothing at or after P reads them)
    import transformers.models.muse_glimmer.modeling_muse_glimmer as mg

    orig = AttentionInterface._global_mapping[attn] if attn != "eager" else mg.eager_attention_forward

    def detach_prefix(module, q, k, v, mask, *a, **kw):
        k = torch.cat([k[:, :, :P].detach(), k[:, :, P:]], dim=2)
        v = torch.cat([v[:, :, :P].detach(), v[:, :, P:]], dim=2)
        return orig(module, q, k, v, mask, *a, **kw)

    if attn == "eager":
        monkeypatch.setattr(mg, "eager_attention_forward", detach_prefix)  # the model's own eager function (not in the interface registry)
    else:
        monkeypatch.setitem(AttentionInterface._global_mapping, attn, detach_prefix)
    pm.zero_grad(set_to_none=True)
    lp_a = logps_full(pm, ch["ids"], pos)
    (lp_a * weights_for(lp_a.numel())).sum().backward()
    g_detached = lora_grads(pm)
    monkeypatch.undo()
    # reference B: the plain full forward (prefix K/V attached)
    pm.zero_grad(set_to_none=True)
    lp_b = logps_full(pm, ch["ids"], pos)
    (lp_b * weights_for(lp_b.numel())).sum().backward()
    g_full = lora_grads(pm)

    assert float((lp_c.detach() - lp_a.detach()).abs().max()) < ATOL and float((lp_a - lp_b).abs().max()) < ATOL
    # the assertion: same gradient as the detached-prefix full forward
    assert g_cached.keys() == g_detached.keys()
    worst = max(float((g_cached[n] - g_detached[n]).abs().max() / (g_detached[n].abs().max() + 1e-12)) for n in g_cached)
    RESULTS.setdefault("t4_max_rel_diff_vs_detached", {})[f"{attn}/{n_pre}+{n_comp}"] = worst
    assert worst < 1e-4, worst
    fc, fd, ff = flat(g_cached), flat(g_detached), flat(g_full)
    assert torch.isfinite(fc).all() and float(fc.norm()) > 0
    # semantics: grads exist only on LoRA params, and (almost) every LoRA tensor of the language model is touched
    assert all("lora_" in n for n in g_cached)
    lora_names = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert set(g_cached) <= set(lora_names)
    assert not any(p.grad is not None for n, p in pm.named_parameters() if not p.requires_grad)
    assert sum(1 for n in g_cached if float(g_cached[n].abs().sum()) > 0) > 0.5 * len(lora_names)
    # measured, not asserted: how far the approximation is from the exact full-trajectory gradient
    cos = float(torch.dot(fc, ff) / (fc.norm() * ff.norm()))
    ratio = float(fc.norm() / ff.norm())
    RESULTS.setdefault("t4_cos_cached_vs_full", {})[f"{attn}/{n_pre}+{n_comp}"] = (round(cos, 4), round(ratio, 4))
    RESULTS.setdefault("t4_cos_cached_vs_detached", {})[f"{attn}/{n_pre}+{n_comp}"] = round(float(torch.dot(fc, fd) / (fc.norm() * fd.norm())), 8)
    assert -1.0 <= cos <= 1.0 and ratio > 0


# ------------------------------------------------------------------------------------------------------------------------------ 5. loss integration
def make_item(rid, ch, adv):
    n_loss = int(ch["spans"][0][1] - ch["spans"][0][0])
    return {"rid": rid, "path": f"/p/{rid}", "n_tokens": len(ch["ids"]), "n_loss": n_loss, "adv": adv}


@pytest.mark.parametrize("beta", [0.0, 0.05])
def test_fork_loss_and_token_count_match_the_full_path(beta):
    pm = build("sdpa", ckpt=True)
    pre = np.random.default_rng(9).integers(3, V, 30)
    chains = [chain(30, c, seed=70 + c, prefix=pre) for c in (5, 9, 3, 6)]
    advs = [1.3, -0.7, 0.4, -1.1]  # lambda_fork * intra advantage, as apply_forks writes it
    items = [make_item(f"r-f{j}", ch, adv) for j, (ch, adv) in enumerate(zip(chains, advs))] + [{"rid": "other", "path": "/p/o", "n_tokens": 500, "n_loss": 37, "adv": 0.2}]
    n_total = step_normaliser(items)
    assert n_total == sum(it["n_loss"] for it in items)
    # behaviour logprobs: the model's own, perturbed so that the ratio leaves [0.8, 1.2] for some tokens (clipping active) and a few NaNs (unknown -> on-policy)
    rng = np.random.default_rng(1)
    for ch in chains:
        pos = positions(ch)
        with torch.no_grad():
            lp = logps_full(pm, ch["ids"], pos).numpy() - rng.normal(0, 0.25, len(pos))
        lp[rng.random(len(pos)) < 0.1] = np.nan
        ch["lp"] = np.full(len(ch["ids"]), np.nan, np.float32)
        ch["lp"][pos] = lp
    inner = inner_of(pm)
    kw = dict(temperature=1.0, top_k=1000, top_p=1.0, clip_lo=0.2, clip_hi=0.2, beta=beta)

    def full_fwd(ids):
        return lm_of(pm)(input_ids=ids[None], position_ids=torch.arange(len(ids))[None], use_cache=False).last_hidden_state[0]

    def ref_fwd(ids):
        with pm.disable_adapter():
            return full_fwd(ids)

    st_full, loss_full = [], []
    for ch, adv in zip(chains, advs):
        loss, st = rollout_chain_loss(full_fwd, ref_fwd if beta > 0 else None, head_of(pm), ch, adv, n_total, **kw)
        loss_full.append(float(loss))
        st_full.append(st)
    st_c, rest, P = train_prefix_group(lm_of(pm), head_of(pm), chains, advs, n_total, ref_ctx=pm.disable_adapter if beta > 0 else None, backward=False, **kw)
    assert rest == [] and P == 29
    for k in ("tokens", "clipped", "in_kept"):
        assert sum(s[k] for s in st_c) == sum(s[k] for s in st_full), k  # same token count, same clip decisions
    assert sum(s["tokens"] for s in st_c) == sum(it["n_loss"] for it in items[:4])
    for k in ("loss", "pg", "kl", "abs_logratio", "logratio", "adv_tok", "dfull"):
        a, b = sum(s[k] for s in st_c), sum(s[k] for s in st_full)
        assert abs(a - b) <= 1e-4 * max(1.0, abs(b)), (k, a, b)
    assert sum(s["clipped"] for s in st_full) > 0  # the test exercises the clip
    if beta > 0:
        assert sum(s["kl"] for s in st_full) > 0
    assert abs(sum(s["loss"] for s in st_c) / n_total - sum(loss_full)) < 1e-5  # the contribution to the step loss, normalised by the global token sum


def test_group_with_a_foreign_prefix_is_returned_for_the_full_path():
    pm = build("sdpa")
    pre = np.random.default_rng(2).integers(3, V, 12)
    good = chain(12, 4, seed=1, prefix=pre)
    other = chain(12, 4, seed=3)  # different prefix tokens
    st, rest, P = train_prefix_group(lm_of(pm), head_of(pm), [good, other], [0.5, 0.5], 8.0, backward=False, temperature=1.0, top_k=1000, top_p=1.0, beta=0.0)
    assert rest == [1] and len(st) == 1 and P == 11


# ------------------------------------------------------------------------------------------------------------------------------ the multimodal wrapper
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_multimodal_wrapper_accepts_cache_and_positions_for_text_only(attn):
    """MuseGlimmerForConditionalGeneration.forward -> MuseGlimmerModel.forward -> language_model: past_key_values and position_ids pass straight through
    (text-only: no pixel inputs). The trainer calls the language_model directly (as forward_hidden does); this checks the top-level model agrees."""
    pm = build(attn)
    top = inner_of(pm)
    ids = torch.from_numpy(np.random.default_rng(0).integers(3, V, 26).astype(np.int64))[None]
    top.eval()
    with torch.no_grad():
        full = top(input_ids=ids, use_cache=False).logits
        out = top(input_ids=ids[:, :17], use_cache=True)
        cache = freeze_cache(out.past_key_values)
        part = top(input_ids=ids[:, 17:], past_key_values=cache, position_ids=torch.arange(17, 26)[None], use_cache=False).logits
    assert float((part - full[:, 17:]).abs().max()) < ATOL
    assert cache.get_seq_length() == 17


# ------------------------------------------------------------------------------------------------------------------------------ 6. data-parallel placement
def fork_workload(n_prefix=40, k=8, n_plain=128, seed=0):
    rng = np.random.default_rng(seed)
    items, n_loss = [], {}
    for i in range(n_plain):
        items.append((f"p{i:03d}", int(rng.integers(8_000, 60_000))))
    for g in range(n_prefix):
        pre = int(rng.integers(25_000, 45_000))
        for j in range(k):
            c = int(rng.integers(1_000, 3_000))
            items.append((f"g{g:03d}-f{j}", pre + c))
            n_loss[f"g{g:03d}-f{j}"] = c
    return items, n_loss


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_forks_of_a_prefix_share_a_replica_and_the_load_is_balanced(world):
    items, n_loss = fork_workload()
    plan = plan_ranks_grouped(items, world, n_loss)
    assert len(plan) == world
    flat_ids = [r for b in plan for r in b]
    assert sorted(flat_ids) == sorted(r for r, _ in items)  # every item exactly once
    where = {r: i for i, b in enumerate(plan) for r in b}
    for g in range(40):
        assert len({where[f"g{g:03d}-f{j}"] for j in range(8)}) == 1  # a prefix's forks never straddle replicas
    assert plan == plan_ranks_grouped(list(reversed(items)), world, n_loss)  # same plan on every rank whatever the manifest order
    # balance by the cost the trainer actually pays: no-grad prefix once + each completion
    cost = dict(items)
    by_par: dict[str, list[str]] = {}
    for r, _ in items:
        by_par.setdefault(fork_parent(r) or r, []).append(r)
    unit = {p: (cost[p] if len(m) == 1 and fork_parent(m[0]) is None else max(cost[r] for r in m) + sum(n_loss[r] for r in m) - max(n_loss[r] for r in m if cost[r] == max(cost[q] for q in m)))
            for p, m in by_par.items()}
    loads = [sum(unit[fork_parent(r) or r] for r in {fork_parent(r) or r for r in b}) for b in plan]
    RESULTS.setdefault("t6_max_over_mean", {})[world] = round(max(loads) / (sum(loads) / world), 4)
    assert max(loads) / (sum(loads) / world) < 1.10


def test_plain_items_are_unchanged_by_grouping():
    items = [(f"r{i}", 1000 + 37 * i) for i in range(30)]
    from wayfinder.grpo.data import plan_ranks

    assert [sorted(b) for b in plan_ranks_grouped(items, 4)] == [sorted(b) for b in plan_ranks(items, 4)]
    assert fork_parent("abc-f3") == "abc" and fork_parent("abc") is None and fork_parent("a-fx") is None


# ------------------------------------------------------------------------------------------------------------------------------ 7. memory / compute estimate
CFG = json.loads((Path(__file__).parent / "data" / "glimmer_text_config.json").read_text())  # Muse-Glimmer-30B config.json (text_config), from the weights volume
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}[CFG["dtype"]]


def kv_bytes(prefix_tokens: int, cfg=CFG) -> dict[str, float]:
    per_tok_layer = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * BYTES  # K and V
    full_layers = sum(t == "full_attention" for t in cfg["layer_types"])
    slide_layers = sum(t == "sliding_attention" for t in cfg["layer_types"])
    full = full_layers * prefix_tokens * per_tok_layer
    slide = slide_layers * min(prefix_tokens, cfg["sliding_window"] - 1) * per_tok_layer  # DynamicSlidingWindowLayer keeps the last window-1 tokens
    return {"full_layers": full_layers, "sliding_layers": slide_layers, "per_token_full_layer_B": per_tok_layer, "full": full, "sliding": slide, "total": full + slide}


def test_kv_cache_and_compute_estimate_fits_an_h200():
    kv = kv_bytes(40_000)
    assert (kv["full_layers"], kv["sliding_layers"]) == (13, 39)
    assert kv["per_token_full_layer_B"] == 2 * 2 * 128 * 2 == 1024
    assert kv["total"] == 13 * 40_000 * 1024 + 39 * 2047 * 1024
    gib = kv["total"] / 2**30
    H200 = 141 * 10**9 / 2**30  # ~131 GiB usable-nominal
    weights_gib = 59_553_253_376 / 2**30  # model.safetensors.index.json total_size (bf16, vision tower included)
    comp, pre = 2_000, 40_000
    # with-grad tokens per fork and the activation memory they pin (gradient checkpointing keeps one [T, H] hidden per layer)
    ckpt_full = CFG["num_hidden_layers"] * (pre + comp) * CFG["hidden_size"] * BYTES / 2**30
    ckpt_cached = CFG["num_hidden_layers"] * comp * CFG["hidden_size"] * BYTES / 2**30
    # no-grad transient of the prefix forward: the widest per-layer tensor is the MLP up/gate [P, intermediate] in bf16 (x2 for gate and up) + residual stream
    transient_gib = (2 * pre * CFG["intermediate_size"] * BYTES + 3 * pre * CFG["hidden_size"] * BYTES) / 2**30
    est = {"kv_cache_GiB_40k_prefix": round(gib, 3), "kv_full_attention_GiB": round(kv["full"] / 2**30, 3), "kv_sliding_GiB": round(kv["sliding"] / 2**30, 3),
           "weights_GiB": round(weights_gib, 1), "with_grad_tokens_per_fork": {"full": pre + comp, "cached": comp}, "ratio": round((pre + comp) / comp, 1),
           "checkpointed_hidden_GiB": {"full": round(ckpt_full, 1), "cached": round(ckpt_cached, 2)}, "prefix_forward_transient_GiB": round(transient_gib, 1),
           "cached_peak_extra_GiB": round(2 * gib + transient_gib + ckpt_cached, 1), "h200_GiB": round(H200, 1)}
    RESULTS["t7"] = est
    assert gib < 1.0  # GQA with 2 KV heads: the 40k-token cache is well under 1 GiB
    # cache (policy) + cache (reference when beta > 0) + prefix transient + the completion's checkpointed hidden states, next to the weights, fits with a wide margin
    assert weights_gib + 2 * gib + transient_gib + ckpt_cached + 20 < H200
    assert ckpt_cached < ckpt_full / 15  # the with-grad forward is ~21x shorter
    assert est["ratio"] == 21.0


def teardown_module(module):
    print("\nMEASURED:", json.dumps(RESULTS, default=str, sort_keys=True))


# ------------------------------------------------------------------------------------------------------------------------------ the flag
def test_flag_is_off_by_default_and_gates_every_new_path():
    root = Path(__file__).resolve().parents[1] / "wayfinder" / "grpo"
    tr, ap = (root / "train_grpo.py").read_text(), (root / "app.py").read_text()
    assert 'ap.add_argument("--fork-cached", action="store_true"' in tr  # default False
    assert "use_cached = a.fork_cached and" in tr and "if use_cached and fork_parent(rid) is not None:" in tr  # plan and training path only behind the flag
    assert "fork_cached: bool = False" in ap
    assert 'if cfg.get("fork_cached"):' in ap  # no flag, no extra trainer argv, no extra config keys


def test_trainer_script_has_no_undefined_names():
    """The trainer needs a GPU, so its fork branch cannot run on CPU; at least no NameError (the first GPU smoke died on one in verify_group)."""
    import ast
    import builtins

    src = (Path(__file__).resolve().parents[1] / "wayfinder" / "grpo" / "train_grpo.py").read_text()
    tree = ast.parse(src)
    module_names = set(dir(builtins))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            module_names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.Assign):
            module_names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        local = {a.arg for a in fn.args.args} | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local |= {(a.asname or a.name).split(".")[0] for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        local |= {n.name for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)} | {a.arg for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.Lambda)) for a in n.args.args}
        loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert loads <= module_names | local, (fn.name, sorted(loads - module_names - local))
