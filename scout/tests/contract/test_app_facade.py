"""Task 5.5: the complete `ScoutApp` facade and the `Scout` SDK over the fake system.

FR-S1, FR-S7, AC-6.2, AC-3.8 (facade half). Every primitive is exercised on a ready, a partial and an absent
index; the SDK must return values equal to the facade's; the facade raises `ScoutError` only.
"""

import asyncio
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scout import Scout
from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.adapters.fake_keystore import FakeKeyStore
from scout.core.app import KEY_NAME, ScoutApp
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import (
    ExploreRequest,
    GrepRequest,
    IndexStatusRequest,
    ReadRangeRequest,
    SemanticSearchRequest,
)
from scout.core.contracts.wire import EmbedReq, EmbedRes
from scout.core.crypto import ids, keys
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.core.ports.gateway import Gateway
from scout.core.sync.engine import SyncConfig

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_spec = importlib.util.spec_from_file_location("gen_repo", FIXTURES / "gen_repo.py")
assert _spec
assert _spec.loader
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

TENANT = "dev"
IDENT = "getUserSession"


def repo_small() -> dict[str, str | bytes]:
    return dict(gen.load_dir(FIXTURES / "repo_small"))


class FlakyGateway:
    """Delegates to a gateway but fails `embed` for batches holding a doomed file (partial coverage)."""

    def __init__(self, gw: Gateway, doomed: set[str]) -> None:
        self._gw, self.doomed = gw, doomed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gw, name)

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        if any(r.file_id in self.doomed for r in req.rows):
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "embedding upstream 503")
        return await self._gw.embed(req, idempotency_key=idempotency_key)


@dataclass
class Built:
    app: ScoutApp
    fs: FakeFileSystem
    chat: ScriptedChat
    keystore: FakeKeyStore
    flaky: FlakyGateway | None


async def build(
    tmp_path: Path,
    files: dict[str, str | bytes] | None = None,
    *,
    not_a_repo: bool = False,
    script: list[ChatTurn] | None = None,
    flaky: bool = False,
) -> Built:
    chat, clock = ScriptedChat(script or []), FakeClock()
    gw = make_fake_gateway(TENANT, chat=chat, clock=clock)
    content = files if files is not None else repo_small()
    fs = (
        FakeFileSystem(content, git=None)
        if not_a_repo
        else FakeFileSystem(content, state_dir=str(tmp_path / "scout"))
    )
    ks = FakeKeyStore()
    wrapped: Any = gw
    flaky_gw: FlakyGateway | None = None
    if flaky:
        k = keys.derive(await gw.team_key())
        names = sorted(p for p in fs.files if p.endswith(".py"))[: max(1, int(len(fs.files) * 0.7))]
        flaky_gw = FlakyGateway(gw, {ids.file_id(k, p) for p in names})
        wrapped = flaky_gw
    cfg = SyncConfig(batch_tokens=700, workers=2, retries=1)
    return Built(ScoutApp(wrapped, fs, clock, ks, sync_config=cfg), fs, chat, ks, flaky_gw)


# --- ready ---------------------------------------------------------------------------------------------
async def test_every_primitive_on_a_ready_index(tmp_path: Path) -> None:
    b = await build(tmp_path)
    status = await b.app.sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert (await b.app.index_status()).state == "ready"
    assert (await b.app.index_status(IndexStatusRequest(root=b.fs.root))).semantic_enabled

    sem = await b.app.semantic_search(SemanticSearchRequest(query=f"{IDENT} session lookup", k=5))
    assert sem.note is None
    assert sem.hits
    top = sem.hits[0]
    disk = b.fs.files[top.path].decode().split("\n")
    first_line = disk[top.start_line - 1]
    assert top.snippet.splitlines()[0] == f"{top.start_line}\t{first_line}"
    assert all(h.source == "semantic" and 1 <= h.start_line <= h.end_line for h in sem.hits)
    assert "src/auth/session.py" in {h.path for h in sem.hits}
    assert ".env" not in {h.path for h in sem.hits}
    assert not any(h.path.startswith("generated/") for h in sem.hits)

    scoped = await b.app.semantic_search(SemanticSearchRequest(query=IDENT, path_glob="web/**", k=10))
    assert scoped.hits
    assert all(h.path.startswith("web/") for h in scoped.hits)

    g = await b.app.grep(GrepRequest(pattern=IDENT))
    assert "src/auth/session.py" in {f.path for f in g.files}
    rr = await b.app.read_range(ReadRangeRequest(path="src/auth/session.py", start_line=6, end_line=7))
    assert rr.lines[0].startswith(f"def {IDENT}")


async def test_explore_uses_semantic_when_ready_and_the_facade_wires_identity(tmp_path: Path) -> None:
    script = [
        ChatTurn(
            None, (ToolCall("c1", "semantic_search", {"query": f"{IDENT} lookup"}),), 10, 1, "tool_calls"
        ),
        ChatTurn(
            None,
            (
                ToolCall(
                    "r",
                    "report",
                    {
                        "results": [
                            {"path": "src/auth/session.py", "start_line": 6, "end_line": 12, "why": "def"}
                        ],
                        "summary": "found",
                        "gaps": [],
                    },
                ),
            ),
            10,
            1,
            "tool_calls",
        ),
    ]
    b = await build(tmp_path, script=script)
    await b.app.sync()
    res = await b.app.explore(ExploreRequest(brief="where is the session lookup?"))
    assert res.index.state == "ready"
    assert res.usage.tool_calls == {"semantic_search": 1, "report": 1}
    assert [h.path for h in res.results if h.verified] == ["src/auth/session.py"]
    assert "semantic_search" in [s["function"]["name"] for s in b.chat.requests[0].tools]


# --- partial -------------------------------------------------------------------------------------------
async def test_partial_index_degrades_semantic_to_grep_only_with_a_note(tmp_path: Path) -> None:
    files: dict[str, str | bytes] = dict(gen.make_repo(60))
    b = await build(tmp_path, files, flaky=True)
    status = await b.app.sync()
    assert status.state == "partial"
    sem = await b.app.semantic_search(SemanticSearchRequest(query="handler"))
    assert sem.hits == []
    assert sem.index.state == "partial"
    assert sem.note
    assert "grep" in sem.note
    assert (await b.app.index_status()).state == "partial"
    assert (await b.app.grep(GrepRequest(pattern="handler_1_0"))).files  # the primitives still work


# --- absent --------------------------------------------------------------------------------------------
async def test_absent_index_degrades_and_primitives_still_work(tmp_path: Path) -> None:
    b = await build(tmp_path)
    assert (await b.app.index_status()).state == "absent"
    sem = await b.app.semantic_search(SemanticSearchRequest(query="anything"))
    assert (sem.hits, sem.index.state) == ([], "absent")
    assert sem.note
    assert "grep" in sem.note
    assert (await b.app.grep(GrepRequest(pattern=IDENT))).files
    assert (await b.app.read_range(ReadRangeRequest(path="README.md"))).lines


# --- errors --------------------------------------------------------------------------------------------
async def test_not_a_repo_is_a_typed_error_everywhere_a_repo_is_needed(tmp_path: Path) -> None:
    b = await build(tmp_path, not_a_repo=True)
    for call in (
        b.app.index_status(),
        b.app.semantic_search(SemanticSearchRequest(query="x")),
        b.app.explore(ExploreRequest(brief="x")),
        b.app.sync(),
    ):
        with pytest.raises(ScoutError) as e:
            await call
        assert e.value.code is ErrorCode.NOT_A_REPO


async def test_foreign_root_and_unexpected_exceptions_are_scout_errors(tmp_path: Path) -> None:
    b = await build(tmp_path)
    with pytest.raises(ScoutError) as e:
        await b.app.semantic_search(SemanticSearchRequest(query="x", root="/elsewhere"))
    assert e.value.code is ErrorCode.INVALID_ARGUMENT

    await b.app.sync()
    with pytest.raises(ScoutError) as e2:
        await b.app.semantic_search(SemanticSearchRequest(query="x", path_glob="/etc/*"))
    assert e2.value.code is ErrorCode.INVALID_ARGUMENT

    class Boom:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("secret path /home/x/.env")

    broken = ScoutApp(Boom(), b.fs, FakeClock())  # type: ignore[arg-type]
    with pytest.raises(ScoutError) as e3:
        await broken.index_status()
    assert e3.value.code is ErrorCode.INTERNAL
    assert "secret" not in e3.value.message


# --- login / watch -------------------------------------------------------------------------------------
async def test_login_stores_the_key_in_the_keystore(tmp_path: Path) -> None:
    b = await build(tmp_path)
    await b.app.login("  sk_scout_abc123  ")
    assert b.keystore.get(KEY_NAME) == b"sk_scout_abc123"
    with pytest.raises(ScoutError) as e:
        await b.app.login("   ")
    assert e.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as e2:
        await ScoutApp(make_fake_gateway(), b.fs, FakeClock()).login("k")
    assert e2.value.code is ErrorCode.INTERNAL


async def test_watch_returns_when_stopped(tmp_path: Path) -> None:
    b = await build(tmp_path)
    stop = threading.Event()
    stop.set()
    res = await asyncio.wait_for(b.app.watch(stop), timeout=10)
    assert res.started


# --- SDK -----------------------------------------------------------------------------------------------
def test_sdk_results_equal_the_facades(tmp_path: Path) -> None:
    async def facade() -> tuple[Any, ...]:
        b = await build(tmp_path / "a")
        await b.app.sync()
        return (
            await b.app.index_status(),
            await b.app.semantic_search(SemanticSearchRequest(query=IDENT, k=4)),
            await b.app.grep(GrepRequest(pattern=IDENT, context=1)),
            await b.app.read_range(ReadRangeRequest(path="README.md")),
        )

    async def sdk_side() -> Scout:
        b = await build(tmp_path / "b")
        sdk = Scout(app=b.app)
        status = await sdk.async_sync()
        assert status.state == "ready"
        return sdk

    status, sem, grp, rr = asyncio.run(facade())
    sdk = asyncio.run(sdk_side())
    assert sdk.index_status() == status
    assert sdk.semantic_search(IDENT, k=4) == sem
    assert sdk.grep(IDENT, context=1) == grp
    assert sdk.read_range("README.md") == rr
    sdk.login("sk_scout_x")


def test_sdk_sync_wrapper_indexes_and_errors_stay_typed(tmp_path: Path) -> None:
    holder: dict[str, Built] = {}

    async def make() -> None:
        holder["b"] = await build(tmp_path)

    asyncio.run(make())
    sdk = Scout(app=holder["b"].app)
    assert sdk.sync().state == "ready"
    assert sdk.semantic_search(IDENT).hits
    with pytest.raises(ScoutError) as e:
        sdk.grep(IDENT, root="/elsewhere")
    assert e.value.code is ErrorCode.INVALID_ARGUMENT
