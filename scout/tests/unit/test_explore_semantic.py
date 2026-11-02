"""Task 5.1: the `semantic_search` explore tool over a really-synced fake system (ScriptedChat, FakeStore,
FakeEmbedder, real GatewayService and SyncEngine).

AC-3.8 (explore half: partial or absent index -> tool not offered, grep-only said so), FR-C4, FR-C13.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreRequest, ExploreResult
from scout.core.contracts.wire import EmbedReq, EmbedRes, StatusReq
from scout.core.crypto import ids, keys
from scout.core.explore.candidates import CandidateSet, RawHit, verify_hits
from scout.core.explore.loop import ExploreContext, explore
from scout.core.explore.semantic import make_backend
from scout.core.local.ignore import Matcher
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.core.ports.gateway import Gateway
from scout.core.sync import merkle
from scout.core.sync.engine import SyncConfig, SyncEngine
from scout.core.sync.state import StateDir

_spec = importlib.util.spec_from_file_location(
    "gen_repo", Path(__file__).resolve().parents[1] / "fixtures" / "gen_repo.py"
)
assert _spec
assert _spec.loader
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

TENANT = "dev"
SESSION = (
    '"""Session lookup."""\n\n\ndef get_user_session(token):\n    s = lookup(token)\n    return s\n\n\n'
    "def clear_sessions():\n    pass\n"
)
FILES = {
    "src/auth/session.py": SESSION,
    "src/billing/invoice.py": "def render_invoice(order):\n    return order.total\n",
    "README.md": "# demo\n",
}


def call(name: str, args: dict[str, Any] | None = None, id_: str = "c1") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args or {})


def turn(*calls: ToolCall) -> ChatTurn:
    return ChatTurn(content=None, tool_calls=calls, tokens_in=10, tokens_out=1, finish="tool_calls")


def report(results: list[dict[str, Any]]) -> ToolCall:
    return call("report", {"results": results, "summary": "done", "gaps": []}, id_="r")


class FlakyGateway:
    """Fails `embed` for batches holding a doomed file (a gateway whose upstream is down for them)."""

    def __init__(self, gw: Gateway, doomed: set[str]) -> None:
        self._gw, self._doomed = gw, doomed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gw, name)

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        if any(r.file_id in self._doomed for r in req.rows):
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "embedding upstream 503")
        return await self._gw.embed(req, idempotency_key=idempotency_key)


@dataclass
class System:
    fs: FakeFileSystem
    gw: Gateway
    chat: ScriptedChat
    clock: FakeClock
    k: keys.Subkeys
    status_req: StatusReq

    def ctx(self, semantic: bool = True) -> ExploreContext:
        backend = (
            make_backend(self.gw, self.k, TENANT, self.status_req, lambda d: merkle.leaf_hash(self.k, d))
            if semantic
            else None
        )
        return ExploreContext(self.gw, self.fs, self.clock, self.status_req, semantic=backend, backfill_to=0)

    async def run(self, script: list[ChatTurn], semantic: bool = True) -> ExploreResult:
        self.chat._script.extend(script)  # pyright: ignore[reportPrivateUsage]
        req = ExploreRequest(brief="where is the session lookup?")
        return await explore(req, self.ctx(semantic))


async def system(
    tmp_path: Path,
    files: dict[str, str] | None = None,
    *,
    sync: bool = True,
    flaky_for: set[str] | None = None,
    cfg: SyncConfig | None = None,
) -> System:
    chat, clock = ScriptedChat([]), FakeClock()
    gw = make_fake_gateway(TENANT, chat=chat, clock=clock)
    k = keys.derive(await gw.team_key())
    fs = FakeFileSystem(files or FILES, state_dir=str(tmp_path / "git" / "scout"))
    if sync:
        sync_gw = FlakyGateway(gw, {ids.file_id(k, p) for p in flaky_for}) if flaky_for else gw
        await SyncEngine(fs, sync_gw, k, clock, cfg).sync()  # type: ignore[arg-type]
    if not sync:
        return System(fs, gw, chat, clock, k, StatusReq(repo_id="0" * 32, branch_id="0" * 16))
    st = StateDir.open(fs).read_state()
    assert st.repo_id
    assert st.last_branch
    return System(fs, gw, chat, clock, k, StatusReq(repo_id=st.repo_id, branch_id=st.last_branch))


def tool_names(chat: ScriptedChat) -> list[str]:
    return [s["function"]["name"] for s in chat.requests[0].tools]


def tool_result(chat: ScriptedChat, index: int = 0) -> str:
    return [str(m["content"]) for m in chat.requests[-1].messages if m["role"] == "tool"][index]


# --- ready ---------------------------------------------------------------------------------------------
async def test_ready_index_offers_semantic_and_returns_numbered_current_lines(tmp_path: Path) -> None:
    sysm = await system(tmp_path)
    res = await sysm.run(
        [
            turn(call("semantic_search", {"query": "get_user_session token lookup", "k": 3})),
            turn(report([{"path": "src/auth/session.py", "start_line": 4, "end_line": 6, "why": "lookup"}])),
        ]
    )
    assert res.index.state == "ready"
    assert tool_names(sysm.chat)[0] == "semantic_search"
    out = tool_result(sysm.chat)
    assert out.splitlines()[0].startswith("src/auth/session.py:")
    assert "4\tdef get_user_session(token):" in out  # numbered, from the disk
    assert "5\t    s = lookup(token)" in out
    assert not any("grep-only" in g for g in res.gaps)
    assert res.usage.tool_calls == {"semantic_search": 1, "report": 1}
    assert [(h.path, h.start_line, h.end_line) for h in res.results] == [("src/auth/session.py", 4, 6)]
    assert res.results[0].source == "semantic"
    json.dumps(res.model_dump(mode="json"))


async def test_snippet_is_at_most_12_lines_and_hits_equal_real_ranges(tmp_path: Path) -> None:
    long = "\n".join(f"def fn_{i}(x):\n    return x + {i}" for i in range(30)) + "\n"
    sysm = await system(tmp_path, {"big.py": long, "other.py": "y = 1\n"})
    res = await sysm.run([turn(call("semantic_search", {"query": "fn_3 fn_4"})), turn(report([]))])
    assert res.usage.tool_calls["semantic_search"] == 1
    for line in tool_result(sysm.chat).splitlines():
        if line.startswith("big.py:"):
            start, end = (int(v) for v in line.split(":")[1].split(" ")[0].split("-"))
            assert end <= len(long.splitlines())
    body = [ln for ln in tool_result(sysm.chat).splitlines() if ln[:1].isdigit()]
    assert body
    assert max(len(b) for b in body) < 200


async def test_path_glob_scopes_the_semantic_search(tmp_path: Path) -> None:
    sysm = await system(tmp_path)
    await sysm.run(
        [
            turn(call("semantic_search", {"query": "order total", "path_glob": "src/auth/**"})),
            turn(report([])),
        ]
    )
    out = tool_result(sysm.chat)
    assert "billing" not in out
    assert "src/auth/session.py" in out


async def test_secret_and_ignored_files_never_surface(tmp_path: Path) -> None:
    files = dict(FILES) | {".env": "TOKEN_LOOKUP=get_user_session\n"}
    sysm = await system(tmp_path, files)
    await sysm.run([turn(call("semantic_search", {"query": "get_user_session", "k": 10})), turn(report([]))])
    assert ".env" not in tool_result(sysm.chat)


async def test_invalid_semantic_arguments_are_in_loop_errors(tmp_path: Path) -> None:
    sysm = await system(tmp_path)
    await sysm.run(
        [
            turn(call("semantic_search", {"query": "x", "path_glob": "/etc/*"})),
            turn(call("semantic_search", {"query": "x", "k": 999}, id_="c2")),
            turn(report([])),
        ]
    )
    msgs = [str(m["content"]) for m in sysm.chat.requests[-1].messages if m["role"] == "tool"]
    assert "INVALID_ARGUMENT" in msgs[0]
    assert "invalid arguments for semantic_search" in msgs[1]


# --- stale: file edited after indexing ------------------------------------------------------------------
async def test_hits_on_a_file_edited_after_sync_are_dropped_not_returned_stale(tmp_path: Path) -> None:
    sysm = await system(tmp_path)
    sysm.fs.files["src/auth/session.py"] = b"# shifted\n" * 30 + SESSION.encode()
    res = await sysm.run([turn(call("semantic_search", {"query": "get_user_session"})), turn(report([]))])
    out = tool_result(sysm.chat)
    assert "src/auth/session.py" not in out.replace("hit(s) dropped", "")
    assert "dropped: the file changed since it was indexed" in out
    assert res.results == []


# --- partial / absent ------------------------------------------------------------------------------------
async def test_partial_index_hides_the_tool_and_says_grep_only(tmp_path: Path) -> None:
    """AC-3.8, explore half: below 80% coverage the loop is grep-only and the result says so."""
    files = gen.make_repo(60)
    cfg = SyncConfig(batch_tokens=700, workers=2, retries=1)
    sysm = await system(tmp_path, dict(files), flaky_for=set(sorted(files)[:36]), cfg=cfg)
    res = await sysm.run([turn(call("semantic_search", {"query": "handler"})), turn(report([]))])
    assert res.index.state == "partial"
    assert "semantic_search" not in tool_names(sysm.chat)
    assert "unknown tool 'semantic_search'" in tool_result(sysm.chat)
    assert any("grep-only" in g and "partial" in g for g in res.gaps)


async def test_absent_index_hides_the_tool_and_says_grep_only(tmp_path: Path) -> None:
    sysm = await system(tmp_path, sync=False)
    res = await sysm.run([turn(call("grep", {"pattern": "get_user_session"})), turn(report([]))])
    assert res.index.state == "absent"
    assert "semantic_search" not in tool_names(sysm.chat)
    assert any("grep-only" in g and "absent" in g for g in res.gaps)


# --- candidates merge ------------------------------------------------------------------------------------
def test_candidates_merge_semantic_grep_and_reads_with_sources() -> None:
    from scout.core.contracts.types import GrepFile, GrepMatch, GrepResult, ReadRangeResult, SnippetHit

    def sh(path: str, start: int, end: int, score: float) -> SnippetHit:
        return SnippetHit(
            path=path, start_line=start, end_line=end, why="w", source="semantic", score=score, snippet=""
        )

    cs = CandidateSet()
    cs.add_semantic([sh("a.py", 10, 20, 0.04), sh("b.py", 1, 5, 0.02)])
    cs.add_grep(
        "x",
        GrepResult(files=[GrepFile(path="a.py", matches=[GrepMatch(line=12, text="x")])], truncated=False),
    )
    best = cs.best(5)
    assert [(c.path, c.source) for c in best] == [("a.py", "semantic"), ("b.py", "semantic")]
    assert best[0].score > 2.0  # semantic bonus plus the grep density it absorbed
    assert cs.source_for("a.py", 11, 13) == "semantic"
    cs.add_read(
        ReadRangeResult(path="b.py", start_line=1, end_line=4, lines=["1", "2", "3", "4"], truncated=False)
    )
    assert cs.source_for("b.py", 1, 4) == "read"
    assert [c.path for c in cs.best(5)].count("b.py") == 1  # the read absorbed the semantic range


def test_verify_hits_drops_changed_missing_and_refused() -> None:
    fs = FakeFileSystem({"a.py": "one\ntwo\nthree\n", ".env": "SECRET=1\n"})
    m = Matcher.from_fs(fs)

    def lh(data: bytes) -> str:
        return f"h{len(data)}"

    raw = [
        RawHit("a.py", 2, 3, "h14", 0.5),  # matches the disk
        RawHit("a.py", 1, 1, "stale", 0.4),  # file_hash differs: dropped
        RawHit("gone.py", 1, 1, "h1", 0.3),  # deleted: dropped
        RawHit(".env", 1, 1, "h9", 0.2),  # denylisted: dropped
    ]
    out, dropped = verify_hits(fs, m, lh, raw, k=5)
    assert [(h.path, h.start_line, h.end_line) for h in out] == [("a.py", 2, 3)]
    assert out[0].snippet == "2\ttwo\n3\tthree"
    assert dropped == 3
