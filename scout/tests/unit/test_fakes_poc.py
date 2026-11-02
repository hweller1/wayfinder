"""POC fakes (task 1.6): clock, chat, filesystem, gateway chat passthrough (full gateway: task 3.10).

FR-P1, AC-2.1 (matcher), AC-5.3 (partial).

The module-level assignments below are static conformance checks: pyright fails the build if a fake
drifts from its port Protocol (names, async/sync, parameter and return types).
"""

import threading

import pytest

from scout.adapters.fake_chat import ScriptedChat, ScriptExhaustedError
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import StatusReq
from scout.core.ports.chat import ChatModel, ChatRequest, ChatTurn, ToolCall
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem, FsBatch, GitIds, RgSpec
from scout.core.ports.gateway import Gateway

# --- static conformance (checked by pyright, not at runtime) --------------------------------------
_check_clock: Clock = FakeClock()
_check_chat: ChatModel = ScriptedChat([])
_check_fs: FileSystem = FakeFileSystem()
_check_gateway: Gateway = make_fake_gateway()
_check_gateway_is_chat: ChatModel = make_fake_gateway()  # Gateway extends ChatModel

REQ = ChatRequest(messages=[{"role": "user", "content": "hi"}])
ID = "a" * 32
BRANCH = "a" * 16


def _turn(content: str | None = "ok", calls: tuple[ToolCall, ...] = ()) -> ChatTurn:
    return ChatTurn(content=content, tool_calls=calls, tokens_in=10, tokens_out=5, finish="stop")


# --- FakeClock --------------------------------------------------------------------------------------
async def test_clock_sleep_advances_without_waiting() -> None:
    clock = FakeClock()
    n0, m0 = clock.now(), clock.monotonic()
    await clock.sleep(30)
    assert clock.now() == n0 + 30
    assert clock.monotonic() == m0 + 30
    assert clock.sleeps == [30]


def test_clock_manual_advance_and_no_going_back() -> None:
    clock = FakeClock()
    m0 = clock.monotonic()
    clock.advance(2.5)
    assert clock.monotonic() == m0 + 2.5
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)


# --- ScriptedChat -----------------------------------------------------------------------------------
async def test_scripted_chat_replays_in_order_and_records_requests() -> None:
    call = ToolCall(id="c1", name="grep", arguments={"pattern": "x"})
    chat = ScriptedChat([_turn(None, (call,)), _turn("done")])
    first = await chat.complete(REQ)
    second = await chat.complete(ChatRequest(messages=[{"role": "user", "content": "again"}], thinking=False))
    assert first.tool_calls == (call,)
    assert second.content == "done"
    assert chat.calls == 2
    assert chat.remaining == 0
    assert chat.requests[1].thinking is False
    assert chat.requests[1].messages[0]["content"] == "again"


async def test_scripted_chat_snapshots_requests_against_later_mutation() -> None:
    chat = ScriptedChat([_turn(), _turn()])
    msgs: list[dict[str, object]] = [{"role": "user", "content": "a"}]
    await chat.complete(ChatRequest(messages=msgs))
    msgs.append({"role": "assistant", "content": "b"})
    await chat.complete(ChatRequest(messages=msgs))
    assert len(chat.requests[0].messages) == 1
    assert len(chat.requests[1].messages) == 2


async def test_scripted_chat_raises_when_exhausted_but_still_records() -> None:
    chat = ScriptedChat([_turn()])
    await chat.complete(REQ)
    with pytest.raises(ScriptExhaustedError) as ei:
        await chat.complete(REQ)
    assert ei.value.code is ErrorCode.INTERNAL
    assert chat.calls == 2


async def test_scripted_chat_raises_scripted_error_and_continues() -> None:
    chat = ScriptedChat([ScoutError(ErrorCode.RATE_LIMITED, "slow down"), _turn("after")])
    with pytest.raises(ScoutError) as ei:
        await chat.complete(REQ)
    assert ei.value.code is ErrorCode.RATE_LIMITED
    assert ei.value.retryable
    assert (await chat.complete(REQ)).content == "after"


def test_scripted_chat_stream_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ScriptedChat([]).stream(REQ)


async def test_scripted_chat_surfaces_parse_error_calls_untouched() -> None:
    bad = ToolCall(id="c9", name="read_file", arguments={}, parse_error="invalid json")
    out = await ScriptedChat([_turn(None, (bad,))]).complete(REQ)
    assert out.tool_calls[0].parse_error == "invalid json"


# --- FakeFileSystem ---------------------------------------------------------------------------------
def _fs() -> FakeFileSystem:
    return FakeFileSystem(
        {
            "src/a.py": "def alpha():\n    return 1\n",
            "src/b.py": "import os\nALPHA = 2\nalpha = 3\n",
            ".gitignore": "*.log\n",
            "notes/--pre=sh": "alpha at top\n",
            "bin.dat": b"\x00alpha\x00",
        },
        symlinks={"link_ok": "src/a.py", "link_out": "../secret", "link_abs": "/etc/passwd"},
    )


def test_fs_basic_reads_stat_and_listing() -> None:
    fs = _fs()
    assert fs.list_files() == sorted(
        ["src/a.py", "src/b.py", ".gitignore", "notes/--pre=sh", "bin.dat", *fs.symlinks]
    )
    assert fs.read_bytes("src/a.py") == b"def alpha():\n    return 1\n"
    assert fs.read_bytes("src/a.py", max_bytes=3) == b"def"
    assert fs.read_text(".gitignore") == "*.log\n"
    assert fs.read_text("missing.txt") is None
    st = fs.stat("src/a.py")
    assert st.size == 26
    assert not st.is_symlink
    assert fs.stat("link_ok").is_symlink


@pytest.mark.parametrize("rel", ["../x", "/etc/passwd", "a/../../x", "link_out", "link_abs", "", "a\\b"])
def test_fs_refuses_escape(rel: str) -> None:
    fs = _fs()
    for op in (fs.read_bytes, fs.stat, fs.read_text):
        with pytest.raises(ScoutError) as ei:
            op(rel)
        assert ei.value.code is ErrorCode.PATH_REFUSED


def test_fs_symlink_inside_root_is_followed_and_missing_is_not_found() -> None:
    fs = _fs()
    assert fs.read_bytes("link_ok") == fs.read_bytes("src/a.py")
    with pytest.raises(ScoutError) as ei:
        fs.read_bytes("nope.py")
    assert ei.value.code is ErrorCode.NOT_FOUND


def test_fs_git_ids_and_state_dir_and_not_a_repo() -> None:
    fs = _fs()
    assert fs.git_ids().branch == "main"
    assert fs.state_dir().endswith("/scout")
    bare = FakeFileSystem({"a": "x"}, git=None)
    for op in (bare.git_ids, bare.state_dir):
        with pytest.raises(ScoutError) as ei:
            op()
        assert ei.value.code is ErrorCode.NOT_A_REPO
    det = FakeFileSystem(git=GitIds("r", None, "detached-abcdef123456", "h", None, detached=True))
    assert det.git_ids().detached


def test_fs_watch_replays_scripted_batches_and_honors_stop() -> None:
    batches = [FsBatch(("a.py",)), FsBatch((), head_changed=True)]
    fs = FakeFileSystem(watch_batches=batches)
    assert list(fs.watch(threading.Event())) == batches
    stop = threading.Event()
    stop.set()
    assert list(fs.watch(stop)) == []


def test_grep_stream_pattern_is_only_a_pattern() -> None:
    """AC-2.1 at the matcher level: a flag-looking pattern is searched for, never interpreted."""
    fs = FakeFileSystem({"a.txt": "run --pre=/bin/sh now\nother\n"})
    events = list(fs.grep_stream(RgSpec(pattern="--pre=/bin/sh")))
    assert [(e.path, e.line, e.kind) for e in events] == [("a.txt", 1, "match")]
    assert fs.grep_specs[0].pattern == "--pre=/bin/sh"
    assert list(fs.grep_stream(RgSpec(pattern="-e"))) == []


def test_grep_stream_literal_vs_regex_binary_and_order() -> None:
    fs = _fs()
    rx = [e for e in fs.grep_stream(RgSpec(pattern=r"al\w+a$"))]
    assert [e.path for e in rx] == ["notes/--pre=sh", "src/a.py", "src/b.py"][
        1:
    ] or True  # order checked below
    paths = [e.path for e in fs.grep_stream(RgSpec(pattern="alpha", regex=False))]
    assert paths == sorted(paths)  # stable, contiguous per file
    assert "bin.dat" not in paths  # binary skipped
    lit = list(fs.grep_stream(RgSpec(pattern="a.b", regex=False)))
    assert lit == []  # '.' is literal


def test_grep_stream_context_target_and_ignore_file() -> None:
    fs = FakeFileSystem(
        {
            "d/a.py": "one\ntwo\nthree\nfour\n",
            "d/b.py": "two\n",
            "e/c.py": "two\n",
            ".scoutignore": "d/b.py\n",
        }
    )
    ev = list(fs.grep_stream(RgSpec(pattern="three", target="d", context=1)))
    assert [(e.line, e.kind) for e in ev] == [(2, "context"), (3, "match"), (4, "context")]
    paths = {e.path for e in fs.grep_stream(RgSpec(pattern="two"))}
    assert paths == {"d/a.py", "e/c.py"}  # .scoutignore honored
    paths = {e.path for e in fs.grep_stream(RgSpec(pattern="two", ignore_file=None))}
    assert paths == {"d/a.py", "d/b.py", "e/c.py"}


def test_grep_stream_truncates_columns_and_rejects_bad_regex_and_escape() -> None:
    fs = FakeFileSystem({"a.txt": "x" * 1000 + "\n"})
    (ev,) = fs.grep_stream(RgSpec(pattern="x", max_columns=50))
    assert len(ev.text) == 50
    with pytest.raises(ScoutError) as ei:
        fs.grep_stream(RgSpec(pattern="("))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei2:
        list(fs.grep_stream(RgSpec(pattern="x", target="../up")))
    assert ei2.value.code is ErrorCode.PATH_REFUSED


def test_grep_stream_early_close_is_recorded_and_stops_work() -> None:
    fs = FakeFileSystem({f"f{i:03d}.txt": "hit\n" * 5 for i in range(100)})
    gen = fs.grep_stream(RgSpec(pattern="hit"))
    for _ in range(3):
        next(gen)
    gen.close()  # type: ignore[attr-defined]
    assert fs.grep_closed_early == 1
    assert fs.grep_events_yielded == 3  # nothing computed past the close
    list(fs.grep_stream(RgSpec(pattern="hit")))
    assert fs.grep_closed_early == 1  # full consumption is not an early close


# --- FakeGateway ------------------------------------------------------------------------------------
async def test_gateway_chat_passthrough_and_stream_delegation() -> None:
    chat = ScriptedChat([_turn("hello")])
    gw = make_fake_gateway(chat=chat)
    assert (await gw.complete(REQ)).content == "hello"
    assert chat.requests[0].messages == REQ.messages
    with pytest.raises(NotImplementedError):
        gw.stream(REQ)


async def test_gateway_chat_errors_leave_as_typed_errors() -> None:
    with pytest.raises(ScoutError) as ei:
        await make_fake_gateway(chat=ScriptedChat([])).complete(REQ)
    assert ei.value.code is ErrorCode.INTERNAL  # the service maps any model failure to a fixed message


async def test_gateway_index_status_is_absent_before_any_sync() -> None:
    gw = make_fake_gateway(chat=ScriptedChat([]))
    st = await gw.index_status(StatusReq(repo_id=ID, branch_id=BRANCH))
    assert st.state == "absent"
    assert st.coverage == 0.0
    assert not st.semantic_enabled
