"""Task 6.2: the MCP stdio surface, driven by the official MCP client over a real subprocess (FR-S3, AC-6.4).

AC-6.4 content is plain text, never `{"result":`; AC-7.6 the render ceiling clamps; the tools filter; the
error mapping (`isError` + `scout error <CODE>`); `mcp.pid` is written at startup with no network call.
Also records Spike 0a item (2) in `artifacts/spike/0a_mcp.json` (checked here, not rewritten).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreResult
from scout.core.explore.render import render_text
from scout.surfaces.mcp.server import build_mcp, select_tools

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
BRIEF = "where is getUserSession defined?"
PRIMITIVES = ["explore", "semantic_search", "grep", "read_range", "index_status"]


def params(*args: str, env: dict[str, str] | None = None) -> StdioServerParameters:
    base = {**os.environ, "SCOUT_PROFILE": "fake", "SCOUT_FAKE_CASE": str(CASE), **(env or {})}
    return StdioServerParameters(command=sys.executable, args=["-m", "scout", "mcp", *args], env=base)


async def call(client: Client, name: str, args: dict[str, Any]) -> tuple[bool, list[str]]:
    r = await client.call_tool(name, args)
    return r.is_error, [c.text for c in r.content]  # type: ignore[union-attr]


async def test_both_contract_levels_are_listed() -> None:
    async with Client(params()) as c:
        names = [t.name for t in (await c.list_tools()).tools]
    assert names == PRIMITIVES


async def test_content_is_one_plain_text_block_equal_to_render_text() -> None:
    """AC-6.4: one text block, the rendering of the result, never the SDK's `{"result": ...}` wrapper."""
    async with Client(params("--structured")) as c:
        _, [structured] = await call(c, "explore", {"brief": BRIEF})
    result = ExploreResult.model_validate_json(structured)
    async with Client(params()) as c:
        r = await c.call_tool("explore", {"brief": BRIEF})
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert not r.is_error
    assert len(r.content) == 1
    text = r.content[0].text  # type: ignore[union-attr]
    assert text == render_text(result, 1500)
    assert not text.lstrip().startswith("{")
    assert '{"result"' not in text
    assert r.structured_content is None
    assert all(t.output_schema is None for t in tools.values())


async def test_primitives_render_as_plain_text() -> None:
    async with Client(params()) as c:
        err, [status] = await call(c, "index_status", {})
        _, [grep] = await call(c, "grep", {"pattern": "getUserSession", "context": 0})
        _, [read] = await call(
            c, "read_range", {"path": "src/auth/session.py", "start_line": 6, "end_line": 7}
        )
        _, [sem] = await call(c, "semantic_search", {"query": "session lookup"})
    assert not err
    assert status.startswith("index absent coverage=0.00")
    assert "src/auth/session.py:6:" in grep
    assert read.splitlines()[0] == "src/auth/session.py:6-7"
    assert "grep" in sem  # absent index: the note says to use grep
    assert all('{"result"' not in t for t in (status, grep, read, sem))


async def test_structured_option_returns_the_json_of_the_contract_type() -> None:
    async with Client(params("--structured")) as c:
        _, [grep] = await call(c, "grep", {"pattern": "getUserSession"})
    doc = json.loads(grep)
    assert doc["schema_version"] == "1.0"
    assert "src/auth/session.py" in {f["path"] for f in doc["files"]}


async def test_tools_filter_exposes_only_the_named_tools() -> None:
    async with Client(params("--tools", "explore")) as c:
        names = [t.name for t in (await c.list_tools()).tools]
        assert names == ["explore"]
        assert not (await c.call_tool("explore", {"brief": BRIEF})).is_error
    async with Client(params("--tools", "grep,read_range")) as c:
        assert [t.name for t in (await c.list_tools()).tools] == ["grep", "read_range"]


async def test_render_ceiling_clamps_the_requested_budget() -> None:
    async with Client(params("--structured")) as c:
        _, [js] = await call(c, "explore", {"brief": BRIEF})
    result = ExploreResult.model_validate_json(js)
    async with Client(params("--render-ceiling", "200")) as c:
        _, [clamped] = await call(c, "explore", {"brief": BRIEF, "render_max_tokens": 4000})
    assert clamped == render_text(result, 200)
    async with Client(params("--render-ceiling", "4000")) as c:
        _, [low] = await call(c, "explore", {"brief": BRIEF, "render_max_tokens": 200})
    assert low == render_text(result, 200)  # the ceiling never raises a smaller request


async def test_errors_are_is_error_results_with_the_code() -> None:
    async with Client(params()) as c:
        err, [text] = await call(c, "read_range", {"path": "../escape"})
        bad, [bad_text] = await call(c, "grep", {"pattern": "x", "context": 99})
    assert err
    assert text.startswith("scout error PATH_REFUSED: ")
    assert bad
    assert bad_text.startswith("scout error INVALID_ARGUMENT: ")
    assert "Traceback" not in text + bad_text


async def test_login_required_names_the_fix() -> None:
    env = {"SCOUT_PROFILE": "default", "SCOUT_KEYSTORE": "file", "HOME": str(ROOT / ".test-temp-home")}
    async with Client(params(env=env)) as c:
        err, [text] = await call(c, "explore", {"brief": BRIEF})
    assert err
    assert text.startswith("scout error ")
    assert "scout login" in text


def test_bad_tools_flag_fails_before_serving() -> None:
    p = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "mcp", "--tools", "nope"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SCOUT_PROFILE": "fake", "SCOUT_FAKE_CASE": str(CASE)},
    )
    assert p.returncode == 2
    assert json.loads(p.stdout)["error"]["code"] == "INVALID_ARGUMENT"


def test_select_tools() -> None:
    assert select_tools(None) == tuple(PRIMITIVES)
    assert select_tools(["grep, explore", "grep"]) == ("explore", "grep")
    for bad in (["nope"], [""], [","]):
        with pytest.raises(ScoutError) as ei:
            select_tools(bad)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def make_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607


async def test_mcp_pid_is_written_at_startup_with_no_network(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo(repo)
    env = {
        "SCOUT_PROFILE": "default",
        "SCOUT_KEYSTORE": "file",
        "HOME": str(tmp_path),
        "SCOUT_ROOT": str(repo),
    }
    pid_file = repo / ".git" / "scout" / "mcp.pid"
    async with Client(params(env=env)) as c:
        assert [t.name for t in (await c.list_tools()).tools] == PRIMITIVES
        body = json.loads(pid_file.read_text())
        assert set(body) == {"pid", "started_at"}
        os.kill(body["pid"], 0)  # a live process: what the host hook checks
        assert body["pid"] != os.getpid()


async def test_the_app_is_built_on_the_first_call_not_at_startup() -> None:
    built: list[int] = []

    def factory() -> Any:
        built.append(1)
        raise ScoutError(ErrorCode.LOGIN_REQUIRED, "run scout login")

    server = build_mcp(factory)
    async with Client(server) as c:
        assert built == []
        err, [text] = await call(c, "index_status", {})
    assert err
    assert text.startswith("scout error LOGIN_REQUIRED")
    assert built == [1]


async def test_unexpected_exceptions_leak_neither_text_nor_trace() -> None:
    class Boom:
        async def index_status(self, req: Any) -> Any:
            raise RuntimeError("secret /home/x/.env")

    async with Client(build_mcp(Boom)) as c:  # type: ignore[arg-type]
        err, [text] = await call(c, "index_status", {})
    assert err
    assert text == "scout error INTERNAL: internal error: RuntimeError"


@pytest.mark.skipif("not (ROOT / 'artifacts' / 'spike' / '0a_mcp.json').exists()", reason="spike artifact not included in the public release")
async def test_spike_0a_item_2_record_matches_live_behaviour() -> None:
    """The SDK default wraps a returned string as `{"result": ...}`; `structured_output=False` does not."""
    rec = json.loads((ROOT / "artifacts" / "spike" / "0a_mcp.json").read_text())
    assert rec["text_only"] is True
    probe = MCPServer("probe")

    @probe.tool()
    async def default_tool() -> str:
        return "hi"

    @probe.tool(structured_output=False)
    async def plain_tool() -> str:
        return "hi"

    async with Client(probe) as c:
        default = await c.call_tool("default_tool", {})
        plain = await c.call_tool("plain_tool", {})
    assert default.structured_content == {"result": "hi"}  # the hazard the surface avoids
    assert plain.structured_content is None
    assert rec["default_wraps_result"] is True
