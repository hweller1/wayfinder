"""Demo lane: env parsing, namespace confinement, model presets, and a full index+explore over fakes."""

import asyncio
from pathlib import Path

import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.core.ports.store import NamespaceSchema
from scout.demo.env import parse_env_text, sanitize_name
from scout.demo.runtime import DemoRuntime, default_name, looks_remote, resolve_model
from scout.demo.store import DemoNamespaceStore

HEX32, HEX16 = "a" * 32, "b" * 16


def test_parse_env_text_handles_export_quotes_and_comments() -> None:
    got = parse_env_text("# c\nexport A=1\nB='two'\nC = \"three\"\nEMPTY=\nnot a line\n")
    assert got == {"A": "1", "B": "two", "C": "three"}


def test_sanitize_and_default_name() -> None:
    assert sanitize_name("My-Repo.v2") == "my_repo_v2"
    assert default_name("https://github.com/gorilla/mux.git") == "mux"
    assert looks_remote("git@github.com:a/b.git")
    assert not looks_remote("a/relative/path")
    with pytest.raises(ValueError, match="empty"):
        sanitize_name("---")


async def test_store_maps_into_the_demo_family_and_refuses_everything_else() -> None:
    inner = FakeStore()
    st = DemoNamespaceStore(inner, "demo1")
    schema = NamespaceSchema(dims=None, attrs={})
    await st.ensure_namespace(f"scout_dev_{HEX32}_{HEX16}", schema)
    await st.ensure_namespace(f"scout_dev_{HEX32}_{HEX16}__tree", schema)
    await st.ensure_namespace("scout_dev_cache", schema)
    await st.ensure_namespace("scout_dev_keys", schema)
    assert await inner.list_namespaces("") == [
        "scout_dev_demo_demo1",
        "scout_dev_demo_demo1__cache",
        "scout_dev_demo_demo1__keys",
        "scout_dev_demo_demo1__tree",
    ]
    with pytest.raises(ScoutError) as e:
        await st.ensure_namespace("production", schema)
    assert e.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ValueError, match="must match"):
        DemoNamespaceStore(inner, "../x")


def test_resolve_model_presets() -> None:
    a1 = resolve_model({"A1_BASE_URL": "https://h/v1", "A1_API_KEY": "k", "BASETEN_API_KEY": "b"})
    assert a1.base_url == "https://h/v1"
    assert a1.usd_per_hour > 0
    bt = resolve_model({"BASETEN_API_KEY": "b"})
    assert bt.model == "meta-models/muse-spark-open"
    assert bt.usd_per_hour == 0
    v3 = resolve_model(
        {"V3_BASE_URL": "https://v/v1", "V3_API_KEY": "k", "A1_BASE_URL": "https://h/v1"}, preset="v3"
    )
    assert (v3.base_url, v3.model, v3.sampling, v3.api_key) == ("https://v/v1", "wayfinder-v3", "sft", "k")
    assert a1.sampling == "a1"
    env = {"V3_BASE_URL": "https://v/v1", "A1_BASE_URL": "https://h/v1", "SCOUT_DEMO_PRESET": "v3"}
    assert resolve_model(env).model == "wayfinder-v3"  # the env switch picks v3 over the A1 default
    with pytest.raises(ScoutError):
        resolve_model({"A1_BASE_URL": "https://h/v1"}, preset="v3")
    custom = resolve_model({}, base_url="http://x/v1", model="m", api_key="k")
    assert (custom.base_url, custom.model, custom.api_key) == ("http://x/v1", "m", "k")
    with pytest.raises(ScoutError):
        resolve_model({})


def _repo(tmp_path: Path) -> Path:
    import subprocess

    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def verify_session_token(token):\n    return token.startswith('s_')\n\n" * 12
    )
    (root / "util.py").write_text("def slugify(text):\n    return text.lower().replace(' ', '-')\n" * 8)
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


def test_index_then_explore_end_to_end_over_fakes(tmp_path: Path) -> None:
    report_args = {
        "results": [{"path": "auth.py", "start_line": 1, "end_line": 2, "why": "token check"}],
        "summary": "auth.py verifies session tokens",
        "gaps": [],
    }
    chat = ScriptedChat(
        [
            ChatTurn(
                content=None,
                tool_calls=(ToolCall("c1", "semantic_search", {"query": "verify session token"}),),
                tokens_in=10,
                tokens_out=2,
                finish="tool_calls",
            ),
            ChatTurn(
                content=None,
                tool_calls=(ToolCall("c2", "report", report_args),),
                tokens_in=10,
                tokens_out=2,
                finish="tool_calls",
            ),
        ]
    )
    store = FakeStore()
    rt = DemoRuntime(
        env={}, home=tmp_path / "home", store=store, embedder=FakeEmbedder(), chat=chat, log=lambda _m: None
    )
    rep = asyncio.run(rt.index(str(_repo(tmp_path)), "projx"))
    assert rep.state == "ready"
    assert rep.chunks > 0
    assert rep.namespace == "scout_dev_demo_projx"
    assert all(n.startswith("scout_dev_demo_projx") for n in asyncio.run(store.list_namespaces("")))
    cfg = resolve_model({}, base_url="http://unused/v1", model="m", api_key="k")
    out = asyncio.run(rt.explore("where are session tokens verified?", "projx", cfg, warmup=False))
    assert out.result.results[0].path == "auth.py"
    assert out.result.usage.tool_calls["semantic_search"] == 1
    assert out.result.index.state == "ready"
    assert [r.name for r in rt.list_repos()] == ["projx"]
    with pytest.raises(ScoutError) as e:
        asyncio.run(rt.explore("x", "nope", cfg, warmup=False))
    assert e.value.code is ErrorCode.NOT_FOUND
