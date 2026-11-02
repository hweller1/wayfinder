"""Runs the backend-agnostic store suite on `TurbopufferStore` over a stub SDK client (task 7.1).

AC-5.3, FR-P3, FR-X4. Proves the adapter's translation and error mapping end to end without the network;
task 7.10 runs the same `store_suite.ALL_CASES` against the real service.
"""

import uuid

import pytest
import store_suite
from tpuf_stub import StubTurbopuffer, no_sleep

from scout.adapters.turbopuffer import TurbopufferStore


@pytest.mark.parametrize("case", store_suite.ALL_CASES, ids=lambda c: c.__name__)
async def test_store_suite_on_turbopuffer_adapter(case: store_suite.Case) -> None:
    sleep, _ = no_sleep()
    store = TurbopufferStore(client=StubTurbopuffer(), sleep=sleep)
    await case(store, f"scout_dev_{uuid.uuid4().hex[:8]}_")
