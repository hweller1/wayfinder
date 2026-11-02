"""The GRPO trainer must give the same update for any replica count (2, 4, 8): the chain partition covers every rollout exactly once, is the
same on every rank, is balanced, and the loss normaliser is the global token sum, independent of world size."""

import random

import pytest
import torch

from wayfinder.grpo.data import plan_ranks, step_normaliser


def make_items(n=128, seed=0):
    rng = random.Random(seed)
    return [{"rid": f"r{i:03d}", "n_tokens": rng.randint(2_000, 60_000), "n_loss": rng.randint(100, 3_000)} for i in range(n)]


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
def test_partition_covers_each_rollout_once(world):
    items = make_items()
    plan = plan_ranks([(i["rid"], i["n_tokens"]) for i in items], world)
    assert len(plan) == world
    flat = [r for b in plan for r in b]
    assert sorted(flat) == sorted(i["rid"] for i in items)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_partition_deterministic_and_balanced(world):
    items = make_items()
    keyed = [(i["rid"], i["n_tokens"]) for i in items]
    assert plan_ranks(keyed, world) == plan_ranks(list(reversed(keyed)), world)  # input order does not matter: every rank computes the same plan
    cost = dict(keyed)
    loads = [sum(cost[r] for r in b) for b in plan_ranks(keyed, world)]
    assert max(loads) / (sum(loads) / world) < 1.05


def test_partition_with_fewer_rollouts_than_ranks():
    plan = plan_ranks([("a", 10), ("b", 5)], 8)
    assert sum(len(b) for b in plan) == 2 and len(plan) == 8


def test_normaliser_is_global_and_world_independent():
    items = make_items()
    n = step_normaliser(items)
    assert n == float(sum(i["n_loss"] for i in items))
    assert step_normaliser([]) == 1.0
    for world in (2, 4, 8):  # what each rank would use: the same manifest, hence the same number
        assert step_normaliser(items) == n


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_summed_gradient_is_world_invariant(world):
    """loss_chain = sum_tokens / n_total on each replica, grads SUM-all-reduced == the single-replica gradient."""
    torch.manual_seed(0)
    items = make_items(32)
    w = torch.randn(5, dtype=torch.float64)
    feats = {i["rid"]: torch.randn(i["n_loss"] % 7 + 1, 5, dtype=torch.float64) for i in items}
    n_total = step_normaliser(items)

    def grad(rids):
        g = torch.zeros(5, dtype=torch.float64)
        for r in rids:
            wp = w.clone().requires_grad_(True)
            (feats[r] @ wp).sum().div(n_total).backward()
            g += wp.grad
        return g

    ref = grad([i["rid"] for i in items])
    plan = plan_ranks([(i["rid"], i["n_tokens"]) for i in items], world)
    assert torch.allclose(sum(grad(b) for b in plan), ref, atol=1e-12)


def test_new_run_seeded_from_adapter_publishes_v0():
    """A run with --resume-adapter and start_step 0 must still write adapters/v0 (the driver samples steps 0 and 1 with v0)."""
    import re
    src = open("wayfinder/grpo/train_grpo.py").read()
    m = re.search(r'if a\.start_step == 0[^\n]*\n\s+save_adapter\(pm, str\(gd / "adapters" / "v0"\)', src)
    assert m and "resume_adapter" not in m.group(0).split("\n")[0].split("#")[0]
