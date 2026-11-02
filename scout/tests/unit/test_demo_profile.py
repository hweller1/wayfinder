"""`--profile demo`: the product surfaces over the demo lane's composition (owner priority).

`scout mcp --profile demo` must be a drop-in for `scout-demo mcp` (same three tools, same behaviour), and
`scout <op> --profile demo` must answer the primitives and `explore` for a repo indexed by the demo. All
offline: the demo runtime is built over `FakeStore`, `FakeEmbedder` and `ScriptedChat`, exactly as
`test_demo_lane.py` does, and injected through the wiring seams.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreRequest, GrepRequest, IndexStatusRequest, ReadRangeRequest
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.demo.runtime import DemoRuntime
from scout.surfaces.mcp.server import DEMO_TOOLS, select_demo_tools
from scout.wiring import PROFILES, Config, build_demo_app, build_demo_mcp, resolve_config

REPORT = {
    "results": [{"path": "auth.py", "start_line": 1, "end_line": 2, "why": "token check"}],
    "summary": "auth.py verifies session tokens",
    "gaps": [],
}


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def verify_session_token(token):\n    return token.startswith('s_')\n\n" * 12
    )
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run(["git", *cmd], cwd=root, env=env, check=True)  # noqa: S603, S607
    return root


def script() -> ScriptedChat:
    def call(id_: str, name: str, args: dict[str, object]) -> ChatTurn:
        return ChatTurn(None, (ToolCall(id_, name, args),), 10, 2, "tool_calls")

    return ScriptedChat(
        [call("c1", "semantic_search", {"query": "verify session token"}), call("c2", "report", REPORT)]
    )


# selects the a1 preset; the chat itself is the injected script
MODEL_ENV = {"A1_BASE_URL": "http://unused/v1"}


def runtime(tmp_path: Path, env: dict[str, str] | None = None) -> DemoRuntime:
    return DemoRuntime(
        env=MODEL_ENV if env is None else env,
        home=tmp_path / "home",
        store=FakeStore(),
        embedder=FakeEmbedder(),
        chat=script(),
        log=lambda _m: None,
    )


def test_demo_is_a_known_profile() -> None:
    assert "demo" in PROFILES
    assert resolve_config({"profile": "demo"}, {}, Path("/nonexistent")).profile == "demo"


async def test_primitives_and_explore_over_an_indexed_demo_repo(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    rt = runtime(tmp_path)
    assert (await rt.index(str(repo), "projx")).state == "ready"
    app = build_demo_app(Config(profile="demo", root=str(repo)), rt)

    status = await app.index_status(IndexStatusRequest())
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert (await app.grep(GrepRequest(pattern="verify_session_token"))).files[0].path == "auth.py"
    assert (
        (await app.read_range(ReadRangeRequest(path="auth.py", start_line=1, end_line=2)))
        .lines[0]
        .startswith("def verify")
    )
    res = await app.explore(ExploreRequest(brief="where are session tokens verified?"))
    assert [h.path for h in res.results] == ["auth.py"]
    assert res.index.state == "ready"
    assert res.usage.tool_calls["semantic_search"] == 1


async def test_an_unindexed_root_names_the_fix(tmp_path: Path) -> None:
    rt = runtime(tmp_path)
    await rt.index(str(make_repo(tmp_path)), "projx")
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ScoutError) as ei:
        build_demo_app(Config(profile="demo", root=str(other)), rt)
    assert ei.value.code is ErrorCode.NOT_FOUND
    assert "scout-demo index" in ei.value.message
    assert "projx" in ei.value.message  # lists what IS indexed


async def test_a_missing_chat_model_fails_only_explore(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    rt = runtime(tmp_path, env={})
    rt._chat = None  # pyright: ignore[reportPrivateUsage]  # no injected chat and no model env: nothing to talk to
    await rt.index(str(repo), "projx")
    app = build_demo_app(Config(profile="demo", root=str(repo)), rt)
    assert (await app.index_status(IndexStatusRequest())).state == "ready"  # primitives still work
    assert (await app.grep(GrepRequest(pattern="verify_session_token"))).files
    with pytest.raises(ScoutError) as ei:
        await app.explore(ExploreRequest(brief="x"))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert "no chat model configured" in ei.value.message  # the demo's own, complete message


def test_demo_tool_selection() -> None:
    assert select_demo_tools(None) == DEMO_TOOLS == ("explore", "index_repo", "list_repos")
    assert select_demo_tools(["list_repos,explore"]) == ("explore", "list_repos")
    for bad in (["grep"], ["nope"], [","]):
        with pytest.raises(ScoutError) as ei:
            select_demo_tools(bad)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT


async def test_the_demo_mcp_server_is_the_demos_own(tmp_path: Path) -> None:
    """The server `scout mcp --profile demo` serves is the demo lane's: same tools, same behaviour."""
    rt = runtime(tmp_path)
    async with Client(build_demo_mcp(rt)) as c:
        assert sorted(t.name for t in (await c.list_tools()).tools) == sorted(DEMO_TOOLS)
        await c.call_tool("index_repo", {"source": str(make_repo(tmp_path)), "name": "projx"})
        listed = await c.call_tool("list_repos", {})
        out = await c.call_tool("explore", {"brief": "where are session tokens verified?", "repo": "projx"})
    assert not out.is_error
    text = out.content[0].text  # type: ignore[union-attr]
    assert "auth.py:1-2" in text
    assert "token check" in text
    assert listed.content


def run_cli(*argv: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SCOUT_DEMO_HOME": str(home),
        "SCOUT_DEMO_ENV_FILES": "/nonexistent",
        "SCOUT_KEYSTORE": "file",
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", *argv], capture_output=True, text=True, env=env, check=False
    )


async def test_scout_mcp_profile_demo_lists_the_demo_tools_over_stdio(tmp_path: Path) -> None:
    """The real command: `scout mcp --profile demo`, spawned, driven by the official client; no network."""
    env = {**os.environ, "SCOUT_DEMO_HOME": str(tmp_path / "home"), "SCOUT_DEMO_ENV_FILES": "/nonexistent"}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "scout", "mcp", "--profile", "demo"], env=env
    )
    async with Client(params) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert sorted(tools) == sorted(DEMO_TOOLS)
        listed = await c.call_tool("list_repos", {})  # an empty registry: no network, no error
        assert not listed.is_error
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "scout", "mcp", "--profile", "demo", "--tools", "explore"],
        env=env,
    )
    async with Client(params) as c:
        assert [t.name for t in (await c.list_tools()).tools] == ["explore"]


def test_a_bad_demo_tool_name_is_invalid_argument(tmp_path: Path) -> None:
    p = run_cli("mcp", "--profile", "demo", "--tools", "grep", home=tmp_path / "home")
    assert p.returncode == 2
    assert "profile demo has no tool grep" in p.stdout


def test_a_demo_primitive_on_an_unindexed_checkout_exits_not_found(tmp_path: Path) -> None:
    p = run_cli("status", "--profile", "demo", "--root", str(tmp_path), home=tmp_path / "home")
    assert p.returncode == 4  # NOT_FOUND
    assert "scout-demo index" in p.stdout


def test_wiring_imports_scout_demo_only_inside_the_demo_builders() -> None:
    """The one upward edge: no module-level import of `scout.demo`, and the contract lists exactly these."""
    import ast

    tree = ast.parse((Path(__file__).resolve().parents[2] / "src/scout/wiring.py").read_text())
    top_level = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    assert not any("demo" in (getattr(n, "module", "") or "") for n in top_level)
    inside = {
        (f.name, n.module)
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        for n in ast.walk(f)
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("scout.demo")
    }
    assert inside == {
        ("build_demo_runtime", "scout.demo.runtime"),
        ("build_demo_app", "scout.demo.runtime"),
        ("build_demo_mcp", "scout.demo.mcp_server"),
    }


def test_the_demo_import_never_loads_at_startup() -> None:
    """`scout mcp` on any other profile must not pay for (or pull in) the demo lane."""
    code = (
        "import sys\nfrom scout import wiring\nwiring.resolve_config({}, {})\n"
        "print([m for m in sys.modules if m.startswith('scout.demo')])\n"
    )
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert p.stdout.strip() == "[]"
