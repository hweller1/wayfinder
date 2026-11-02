"""Task 7.10: Voyage embedding dims smoke on 3 SYNTHETIC strings (no real code). FR-P7, AC-14.6."""

import math
from collections.abc import Mapping

import pytest

from scout.adapters.voyage.client import VoyageEmbedder

pytestmark = [pytest.mark.live]

SYNTHETIC = [
    "def frobnicate_widget(count): return count * 2",
    "SELECT name FROM gadgets WHERE colour = 'teal'",
    "struct Sprocket { int teeth; double pitch; };",
]


async def test_voyage_dims_and_norm(live_env: Mapping[str, str]) -> None:
    if not live_env.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")
    res = await VoyageEmbedder.from_env(live_env).embed(SYNTHETIC, input_type="document")
    assert len(res.vectors) == 3
    assert res.tokens > 0
    for v in res.vectors:
        assert len(v) == 1024
        assert 0.9 < math.sqrt(sum(x * x for x in v)) < 1.1
