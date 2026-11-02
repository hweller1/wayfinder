"""Task 7.10: the backend-agnostic store suite on REAL turbopuffer through `TurbopufferStore`.

AC-5.3, FR-P7, AC-14.6. The same `store_suite.ALL_CASES` pass on FakeStore (test_fake_store_suite.py), the
SDK stub (test_turbopuffer_store_suite.py) and here, which is what proves FakeStore matches the service.
A failure here means fix the fake (or the adapter), never the suite. Namespaces: `scout_dev_suite_<8hex>_*`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
import store_suite  # noqa: E402  # pyright: ignore[reportMissingImports]
from helpers_live import LiveStore  # noqa: E402

pytestmark = [pytest.mark.live]


@pytest.mark.parametrize("case", store_suite.ALL_CASES, ids=lambda c: c.__name__)
async def test_store_suite_on_real_turbopuffer(case: store_suite.Case, live_store: LiveStore) -> None:
    await case(live_store.store, live_store.prefix)
