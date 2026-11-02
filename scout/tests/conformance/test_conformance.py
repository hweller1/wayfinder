"""Tasks 6.6 and 6.7: the conformance suite (design section 9, FR-S6, AC-6.1, AC-6.2, AC-6.4, AC-6.5).

Every case runs through every transport: SDK, CLI, MCP stdio, MCP streamable HTTP and the daemon. For each
(case, transport) the outcome must equal the SDK's, which is the reference built in-process from the same case
document: the same typed error code, or a success body that is byte-identical JSON, valid against `schema/v1`
and satisfying the case's own `expect`. Each driver also asserts its transport's error mapping as it decodes.

The full matrix spawns a subprocess per (case, transport), so it is large: a smoke subset (one case of each
kind on every transport) runs in `make check`; the rest is marked `slow` and runs under `make check-all` and
`pytest -m conformance`, which every checkpoint does.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from cases import BRIEF, BY_ID, CASES, ERRORS, SUCCESS, Case
from drivers import (
    OPS,
    REQUEST_MODELS,
    RESULT_MODELS,
    CliDriver,
    DaemonDriver,
    McpHttpDriver,
    McpStdioDriver,
    SdkDriver,
    _McpDriver,  # pyright: ignore[reportPrivateUsage]
)

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.explore.render import render_text
from scout.surfaces import ops

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schema" / "v1"
SCHEMA_NAME = {
    "explore": "ExploreResult",
    "semantic_search": "SemanticSearchResult",
    "grep": "GrepResult",
    "read_range": "ReadRangeResult",
    "index_status": "IndexStatus",
}
TRANSPORTS: dict[str, Any] = {
    "cli": CliDriver,
    "mcp-stdio": McpStdioDriver,
    "mcp-http": McpHttpDriver,
    "daemon": DaemonDriver,
}
# one case of each kind runs in `make check` on every transport; the rest of the matrix is `slow`
SMOKE = {
    "explore_ready",
    "explore_stale_hit",
    "grep",
    "index_status_partial",
    "error_path_refused",
    "error_rate_limited",
}


async def run(driver: Any, case: Case, tmp: Path) -> tuple[str | None, str | None, ErrorCode | None]:
    """`(raw body, canonical JSON, code)`: the body as delivered, its canonical dump, or the typed error."""
    await driver.setup(case.doc(), tmp)
    try:
        req = REQUEST_MODELS[case.op].model_validate(case.request)
        try:
            result = await getattr(driver, case.op)(req)
        except ScoutError as e:
            return None, None, e.code
        return driver.raw, ops.result_json(result), None
    finally:
        await driver.close()


def assert_expect(case: Case, body: str | None, code: ErrorCode | None) -> None:
    assert code == case.expect_code, f"{case.id}: expected {case.expect_code}, got {code}"
    if case.expect_code is not None:
        assert body is None
        return
    assert body is not None
    parsed = json.loads(body)
    schema = json.loads((SCHEMA_DIR / f"{SCHEMA_NAME[case.op]}.json").read_text())
    jsonschema.validate(parsed, schema)
    if case.check:
        case.check(parsed)


async def reference(case: Case, tmp: Path) -> tuple[str | None, ErrorCode | None]:
    """The SDK's outcome: canonical JSON, or the typed error."""
    _, canonical, code = await run(SdkDriver(), case, tmp / "sdk")
    assert_expect(case, canonical, code)
    return canonical, code


def params() -> list[Any]:
    out: list[Any] = []
    for case in CASES:
        for name in TRANSPORTS:
            heavy = case.id not in SMOKE
            out.append(
                pytest.param(case, name, id=f"{case.id}-{name}", marks=[pytest.mark.slow] if heavy else [])
            )
    return out


@pytest.mark.parametrize(("case", "transport"), params())
async def test_transport_equals_sdk(case: Case, transport: str, tmp_path: Path) -> None:
    """AC-6.1, AC-6.2, AC-6.5: same JSON (byte for byte), same error codes, on every transport."""
    want_json, want_code = await reference(case, tmp_path)
    raw, canonical, code = await run(TRANSPORTS[transport](), case, tmp_path / transport)
    assert code == want_code
    assert_expect(case, raw, code)
    assert raw == want_json  # byte-identical, not merely equal once parsed
    assert canonical == want_json


# --- AC-6.4: the default (non-structured) MCP content is plain text ----------------------------------------
TEXT_CASES = [
    "explore_ready",
    "explore_partial",
    "explore_budget_exhausted",
    "grep",
    "read_range",
    "index_status_absent",
]
TEXT_TRANSPORTS = {"mcp-stdio": McpStdioDriver, "mcp-http": McpHttpDriver}


def text_params() -> list[Any]:
    """Stdio checks every kind; HTTP the two representative ones."""
    out: list[Any] = []
    for transport in sorted(TEXT_TRANSPORTS):
        for case_id in TEXT_CASES:
            if transport == "mcp-http" and case_id not in {"explore_ready", "grep"}:
                continue
            out.append(pytest.param(case_id, transport, id=f"{case_id}-{transport}"))
    return out


@pytest.mark.parametrize(("case_id", "transport"), text_params())
async def test_default_mcp_content_is_one_plain_text_block(
    case_id: str, transport: str, tmp_path: Path
) -> None:
    """AC-6.4: one text block equal to `render_text(result)`, never `{"result": ...}`."""
    case = BY_ID[case_id]
    _, canonical, code = await run(SdkDriver(), case, tmp_path / "sdk")
    assert code is None
    result = RESULT_MODELS[case.op].model_validate_json(canonical)  # type: ignore[arg-type]
    req = REQUEST_MODELS[case.op].model_validate(case.request)
    driver: _McpDriver = TEXT_TRANSPORTS[transport](structured=False)
    await driver.setup(case.doc(), tmp_path / transport)
    try:
        text = await driver.text(case.op, req)
        names = await driver.tool_names()
    finally:
        await driver.close()
    assert names == list(OPS)
    assert not text.lstrip().startswith("{")
    assert '{"result"' not in text
    if case.op == "explore":
        assert text == render_text(result, req.render_max_tokens)  # type: ignore[attr-defined]
        assert BRIEF not in text  # the brief is input, not output
    assert text == ops.render(case.op, req, result)


async def test_default_mcp_errors_are_plain_text_with_the_code(tmp_path: Path) -> None:
    case = BY_ID["error_path_refused"]
    driver: _McpDriver = McpStdioDriver(structured=False)
    await driver.setup(case.doc(), tmp_path)
    try:
        req = REQUEST_MODELS[case.op].model_validate(case.request)
        with pytest.raises(ScoutError) as ei:
            await driver.text(case.op, req)
    finally:
        await driver.close()
    assert ei.value.code is ErrorCode.PATH_REFUSED


# --- (4) all five operations everywhere; (3) typed errors map on every transport -----------------------
async def test_all_five_operations_are_callable_on_every_transport(tmp_path: Path) -> None:
    """Both contract levels everywhere: one case of each operation through each of the four transports."""
    smoke_ops = {"explore": "explore_ready", "grep": "grep", "read_range": "read_range"}
    smoke_ops |= {"index_status": "index_status_ready", "semantic_search": "semantic_search_ready"}
    assert set(smoke_ops) == set(OPS)
    for transport, make in TRANSPORTS.items():
        for op, case_id in smoke_ops.items():
            case = BY_ID[case_id]
            raw, _, code = await run(make(), case, tmp_path / f"{transport}-{op}")
            assert code is None, f"{transport} {op}"
            assert_expect(case, raw, code)


@pytest.mark.parametrize("transport", sorted(TRANSPORTS))
async def test_every_error_code_maps_on_the_transport(transport: str, tmp_path: Path) -> None:
    """Section 4: the typed error survives the transport (the drivers also assert the transport's own
    mapping while decoding: CLI exit code and stderr line, HTTP status and Retry-After, MCP isError text)."""
    for case in ERRORS:
        _, _, code = await run(TRANSPORTS[transport](), case, tmp_path / case.id)
        assert code == case.expect_code, f"{transport} {case.id}"


# --- the suite covers what it claims to -----------------------------------------------------------------
def test_every_operation_has_a_success_case() -> None:
    assert {c.op for c in SUCCESS} == set(OPS)


def test_every_error_code_has_a_case() -> None:
    assert {c.expect_code for c in ERRORS} == set(ErrorCode)


def test_every_case_states_what_it_expects() -> None:
    for case in SUCCESS:
        assert case.check is not None, f"{case.id} asserts nothing about its result"


def test_every_index_state_and_budget_exhaustion_is_exercised() -> None:
    explore = {c.id for c in SUCCESS if c.op == "explore"}
    assert {"explore_ready", "explore_partial", "explore_absent", "explore_budget_exhausted"} <= explore
    assert "explore_stale_hit" in explore
    assert "explore_path_refused_in_loop" in explore
