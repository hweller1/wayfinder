"""Explore support (task 1.10): render cap, budget by clock, pinned prompts, profiles.

AC-1.2 (render drops whole lowest-ranked lines), FR-C2, FR-C18, AC-1.4 (budget), NFR-6.
"""

import hashlib
import re
from pathlib import Path

from scout.adapters.fake_clock import FakeClock
from scout.core.contracts.types import Budget as BudgetLimits
from scout.core.contracts.types import ExploreResult, Hit, IndexStatus, ModelTokens, Usage
from scout.core.explore.budget import Budget
from scout.core.explore.profiles import PROFILES, load_prompt
from scout.core.explore.render import approx_tokens, render_text

# sha256 of ../benchmark-repo/configs/prompts.yaml: sophisticated (the variant benchmark arm g ran).
NEUTRAL_SHA256 = "3e1a1023a11647095ac8617d5024131ea5305e5f60816b0cca64e26cd2b5ea75"
PROMPTS = Path(__file__).parents[2] / "src" / "scout" / "core" / "explore" / "prompts"


def _result(
    n: int, gaps: list[str] | None = None, summary: str = "Session lookup lives in auth."
) -> ExploreResult:
    hits = [
        Hit(
            path=f"src/pkg/module_{i:02d}/session_handlers.py",
            start_line=10 * i + 1,
            end_line=10 * i + 30,
            why=f"rank {i} handler that resolves the session, checks expiry and refreshes the token for it",
            source="grep",
        )
        for i in range(n)
    ]
    return ExploreResult(
        results=hits,
        summary=summary,
        gaps=gaps or [],
        index=IndexStatus(state="absent", coverage=0.0, semantic_enabled=False),
        usage=Usage(
            turns=3, tool_calls={"grep": 2}, model_tokens=ModelTokens(**{"in": 10, "out": 5}), wall_ms=1
        ),
    )


def test_render_40_results_into_1500_tokens_none_cut() -> None:
    """AC-1.2: <= 1,500 tokens, whole lines only, highest-ranked kept; structured results untouched."""
    res = _result(40, gaps=["could not find the refresh path"])
    full = render_text(res, 100_000)
    assert approx_tokens(full) > 1500  # the cap really binds
    out = render_text(res, 1500)
    assert approx_tokens(out) <= 1500
    full_lines = set(full.splitlines())
    assert all(ln in full_lines or ln.startswith("... ") for ln in out.splitlines())  # none cut mid-line
    kept = [ln for ln in out.splitlines() if re.match(r"src/pkg/module_\d\d/", ln)]
    assert 0 < len(kept) < 40
    assert kept == [ln for ln in full.splitlines() if re.match(r"src/pkg/module_\d\d/", ln)][: len(kept)]
    assert "module_00/" in out
    assert "module_39/" not in out  # lowest-ranked dropped first
    assert "omitted" in out
    assert len(res.results) == 40


def test_render_small_cap_and_injected_counter() -> None:
    res = _result(5)
    assert approx_tokens(render_text(res, 200)) <= 200
    # one token per character makes the cap bind on whole lines at a tiny size
    out = render_text(res, 150, count=len)
    assert len(out) <= 150
    assert all(ln in render_text(res, 10**6).splitlines() or ln.startswith("... ") for ln in out.splitlines())


def test_render_fits_untouched_and_marks_budget_exhaustion() -> None:
    res = _result(2).model_copy(update={"budget_exhausted": True})
    out = render_text(res, 1500)
    assert "omitted" not in out
    assert "budget exhausted" in out
    assert out.splitlines()[-1].startswith("src/pkg/module_01/")


def test_approx_tokens_is_ceil_bytes_over_3_5() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("a" * 7) == 2
    assert approx_tokens("a" * 8) == 3
    assert approx_tokens("é" * 7) == 4  # 14 bytes


def test_budget_exhaustion_by_clock_and_turns() -> None:
    clock = FakeClock()
    b = Budget(clock, max_turns=3, max_seconds=10)
    assert not b.exhausted
    assert b.reason is None
    clock.advance(9.9)
    assert not b.exhausted
    clock.advance(0.1)
    assert b.exhausted
    assert b.reason == "seconds"

    b2 = Budget.from_limits(FakeClock(), BudgetLimits(max_turns=2, max_seconds=60))
    b2.tick()
    assert not b2.exhausted
    b2.tick()
    assert b2.exhausted
    assert b2.reason == "turns"
    assert b2.turns == 2


def test_neutral_prompt_sha_pinned() -> None:
    """FR-C18: neutral prompt is the benchmark `sophisticated` text, byte for byte."""
    data = (PROMPTS / "neutral.md").read_bytes()
    assert hashlib.sha256(data).hexdigest() == NEUTRAL_SHA256
    assert load_prompt("neutral").encode("utf-8") == data
    assert data.startswith(
        b"You are answering a question about a corpus you have not seen before.\n\nWork in three phases."
    )


def test_product_prompt_has_routing_mechanics_and_report() -> None:
    p = load_prompt("product")
    for needle in ("ROUTING", "TOOL MECHANICS", "report", "read_file", "semantic_search", "grep"):
        assert needle in p
    assert p != load_prompt("neutral")


def test_profiles() -> None:
    assert set(PROFILES) == {"product", "parity"}
    parity = PROFILES["parity"]
    assert parity.prompt == load_prompt("neutral")
    assert parity.chunker == "declaration"
    assert parity.terminal == "text"
    assert parity.thinking
    assert parity.tools == ("semantic_search", "grep", "glob", "read_file")
    assert "report" not in parity.tools
    assert parity.caps["grep_files"] == 200
    assert parity.caps["grep_matches_per_file"] == 20
    product = PROFILES["product"]
    assert product.terminal == "report"
    assert "report" in product.tools
    assert product.prompt == load_prompt("product")
    assert product.chunker == "tree-sitter"
