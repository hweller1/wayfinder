"""Live-test fixtures (task 7.10): real turbopuffer/Voyage, scratch `scout_dev_` namespaces, cleaned up.

AC-14.6, FR-P7. Keys come from the environment or the gitignored `.env` (never printed). Tests skip, not fail,
when keys are absent.
"""

import uuid
from collections.abc import AsyncIterator, Mapping

import pytest
from helpers_live import LiveStore

from scout.adapters.turbopuffer import TurbopufferStore
from scout.demo.env import load_env

TENANTS = ("dev", "eval")
KEEP = frozenset({"scout_dev_keys"})


@pytest.fixture(scope="session")
def live_env() -> Mapping[str, str]:
    return load_env()


@pytest.fixture(scope="session")
def voyage_optout_confirmed(live_env: Mapping[str, str]) -> None:
    """Private/eval-code tests need the owner's Voyage opt-out confirmation; skip loudly otherwise."""
    if live_env.get("SCOUT_VOYAGE_OPTOUT_CONFIRMED") != "1":
        pytest.skip("SCOUT_VOYAGE_OPTOUT_CONFIRMED != 1: skipping tests sending non-synthetic code to Voyage")


@pytest.fixture
async def live_store(live_env: Mapping[str, str]) -> AsyncIterator[LiveStore]:
    """A real `TurbopufferStore` plus a unique prefix. In `finally` every namespace under the prefix is
    deleted; a leftover (the case did not clean up after itself) is deleted and then fails the test."""
    if not live_env.get("TURBOPUFFER_API_KEY"):
        pytest.skip("TURBOPUFFER_API_KEY not set")
    store = TurbopufferStore.from_env(live_env)
    prefix = f"scout_dev_suite_{uuid.uuid4().hex[:8]}_"
    try:
        yield LiveStore(store, prefix)
    finally:
        leftovers = [n for n in await store.list_namespaces(prefix) if n not in KEEP]
        for ns in leftovers:
            await store.delete_namespace(ns)
    assert not leftovers, f"namespaces left behind by the test: {leftovers}"
