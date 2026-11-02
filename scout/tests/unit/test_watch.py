"""Task 4.5: watch loop and single-instance launcher (`core/sync/watch.py`).

FR-C14, FR-C15, AC-3.9. Everything runs on ports: a queue-driven `FileSystem.watch`, the real `SyncEngine`
over `make_fake_gateway` (or a scripted `Syncer`), and a manual clock (`StepClock`: `sleep` waits until the
test advances time, so the settle period, backoff and 60 s ping are driven deterministically). Real-time
polls with a bounded timeout are used only for thread hand-offs (the batch pump, status writes), and every
wait is on a counter the loop owns (`batches`, `syncs` = started, `completed` = finished with the status
written), never on a wall-clock delay: a test that must prove "nothing else happens" yields to the loop
(`quiesce`) instead of sleeping, because without a clock advance no further sync can progress.
"""

import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.crypto import keys
from scout.core.ports.fs import FsBatch, GitIds
from scout.core.ports.gateway import Gateway
from scout.core.sync.engine import SyncEngine
from scout.core.sync.state import StateDir, iso_utc
from scout.core.sync.watch import WatchConfig, WatchLoop, WatchResult, ensure_watch, read_pid

STATUS_SCHEMA = json.loads((Path(__file__).parents[2] / "schema" / "v1" / "status_file.json").read_text())
WAIT_S = 10.0
NEVER = 1e9  # a ping interval that never elapses in a test
FILES: dict[str, str | bytes] = {
    "src/a.py": "def alpha():\n    return 1\n",
    "src/b.py": "def beta():\n    return 2\n",
    "README.md": "# demo\n",
}


class StepClock(FakeClock):
    """`sleep` blocks until `advance` reaches its deadline. `sleeps` records every requested duration."""

    def __init__(self) -> None:
        super().__init__()
        self._waiters: list[tuple[float, float, asyncio.Future[None]]] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (self._mono + max(seconds, 0.0), seconds, fut)
        self._waiters.append(entry)
        try:
            await fut
        finally:
            self._waiters.remove(entry)

    def advance(self, seconds: float) -> None:
        super().advance(seconds)
        for deadline, _, fut in list(self._waiters):
            if deadline <= self._mono and not fut.done():
                fut.set_result(None)

    def pending(self) -> list[float]:
        return sorted(d for _, d, f in self._waiters if not f.done())

    def fire_next(self, *, below: float = NEVER) -> bool:
        """Advance exactly to the earliest pending sleep shorter than `below`; False when there is none."""
        due = [dl for dl, d, f in self._waiters if d < below and not f.done()]
        if not due:
            return False
        self.advance(max(min(due) - self._mono, 0.0))
        return True


class QueueFS(FakeFileSystem):
    """`FakeFileSystem` whose `watch` yields what the test `push`es, until `stop` or `close`."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.q: queue.Queue[FsBatch | None] = queue.Queue()
        self.debounces: list[int] = []

    def push(self, *paths: str, head: bool = False) -> None:
        self.q.put(FsBatch(paths=tuple(paths), head_changed=head))

    def close(self) -> None:
        self.q.put(None)

    def set_branch(self, branch: str) -> None:
        old = cast(GitIds, self._git)
        self._git = GitIds(
            old.root_commit, old.remote_norm, branch, old.head, old.default_branch, old.detached, old.shallow
        )

    def watch(self, stop: threading.Event, *, debounce_ms: int = 300) -> Iterator[FsBatch]:
        self.debounces.append(debounce_ms)
        while not stop.is_set():
            try:
                item = self.q.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item


class ScriptedEngine:
    """A `Syncer` that counts calls, fails the first `fail`, reports contention for `contended` calls, and
    can hold its first call on `gate`."""

    def __init__(self, *, fail: int = 0, contended: int = 0) -> None:
        self.calls = 0
        self.fail = fail
        self.contended = contended
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def sync(self, wait: bool = True, *, full: bool = False) -> IndexStatus:
        self.calls += 1
        self.entered.set()
        if self.gate is not None and self.calls == 1:
            await self.gate.wait()
        if self.calls <= self.fail:
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "gateway down", True)
        busy = self.calls <= self.contended
        return IndexStatus(state="ready", coverage=1.0, syncing=busy, semantic_enabled=True)


class FlakyGateway:
    """Delegates to the real fake gateway; `whoami` (the first call of every sync) and `ping` can fail."""

    def __init__(self, gw: Gateway, *, whoami_fails: int = 0, ping_fails: int = 0) -> None:
        self._gw = gw
        self.whoami_fails = whoami_fails
        self.ping_fails = ping_fails

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gw, name)

    async def whoami(self) -> Any:
        if self.whoami_fails > 0:
            self.whoami_fails -= 1
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "gateway down", True)
        return await self._gw.whoami()

    async def ping(self) -> None:
        if self.ping_fails > 0:
            self.ping_fails -= 1
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "gateway down", True)
        await self._gw.ping()


async def wait_for(cond: Callable[[], bool], what: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_S
    while not cond():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


async def quiesce() -> None:
    """Let the loop run every ready callback and task before an assertion that something did NOT happen.
    Every batch was already delivered (`push` waits on `batches`) and time is manual, so a stray sync could
    only show up as a settle sleep registered within a few loop iterations (no wall-clock delay)."""
    for _ in range(50):
        await asyncio.sleep(0)
    await asyncio.sleep(
        0.01
    )  # one real hand-off for a worker thread; correctness never depends on its length


async def drive(clock: StepClock, cond: Callable[[], bool], what: str, *, below: float = NEVER) -> None:
    """Fire pending sleeps (shorter than `below`, earliest first) until `cond` holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_S
    while not cond():
        if loop.time() > deadline:
            raise AssertionError(f"timed out driving the clock to {what}; sleeps={clock.sleeps}")
        if clock.fire_next(below=below):
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.005)


async def tick(clock: StepClock, loop_: WatchLoop, seconds: float) -> None:
    """Let the pinger reach its sleep, then advance one ping interval and wait for the next ping."""
    n = loop_.pings
    await wait_for(lambda: seconds in clock.pending(), "the pinger to sleep")
    clock.advance(seconds)
    await wait_for(lambda: loop_.pings == n + 1 and seconds in clock.pending(), "the next ping to finish")


def backoffs(clock: StepClock) -> list[float]:
    """The retry delays so far (the idle ping sleep is `NEVER`, the settle sleeps are < 1 s)."""
    return [s for s in clock.sleeps if 1.0 <= s < NEVER]


def cfg(**over: Any) -> WatchConfig:
    return WatchConfig(**{"ping_interval_s": NEVER, "settle_s": 0.1, **over})


class Harness:
    """A running `WatchLoop` on a `QueueFS`; entering waits for the first ping to write `status.json`."""

    def __init__(
        self,
        tmp_path: Path,
        engine: Any,
        gw: Any,
        config: WatchConfig,
        *,
        fs: QueueFS | None = None,
        clock: StepClock | None = None,
    ) -> None:
        self.clock = clock or StepClock()
        self.fs = fs or QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout"))
        self.state = StateDir.open(self.fs)
        self.stop = threading.Event()
        self.loop = WatchLoop(self.fs, gw, engine, self.clock, self.state, config)
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Harness":
        self.task = asyncio.create_task(self.loop.run(self.stop))
        await wait_for(
            lambda: self.loop.pings >= 1 and self.has_ok_at(), "the first ping to write status.json"
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop.set()
        self.fs.close()
        assert self.task is not None
        await asyncio.wait_for(self.task, WAIT_S)

    def status(self) -> Any:
        return self.state.read_status()

    def has_ok_at(self) -> bool:
        st = self.status()
        return st is not None and st.gateway_ok_at is not None

    def disk_status(self) -> dict[str, Any]:
        doc = json.loads(Path(self.state.path, "status.json").read_text())
        jsonschema.validate(doc, STATUS_SCHEMA)
        return doc

    async def push(self, *paths: str, head: bool = False, n: int = 1) -> None:
        target = self.loop.batches + n
        for _ in range(n):
            self.fs.push(*paths, head=head)
        await wait_for(lambda: self.loop.batches >= target, f"{n} batch(es) to reach the loop")


async def real_stack(tmp_path: Path, **gw_over: int) -> tuple[Harness, SyncEngine, FakeEmbedder]:
    clock, emb = StepClock(), FakeEmbedder()
    gw = make_fake_gateway("dev", embedder=emb, clock=clock)
    fs = QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout"))
    wired = FlakyGateway(gw, **gw_over)
    engine = SyncEngine(fs, cast(Gateway, wired), keys.derive(await gw.team_key()), clock)
    return Harness(tmp_path, engine, wired, cfg(), fs=fs, clock=clock), engine, emb


# --- port ---------------------------------------------------------------------------------------------
async def test_ping_answers_for_a_bound_tenant() -> None:
    """FR-C15: the real service behind the fake answers; a tenant name that fails validation never binds."""
    assert await make_fake_gateway("dev").ping() is None
    with pytest.raises(ScoutError) as ei:
        make_fake_gateway("bad_tenant")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


# --- AC-3.9 with the real engine ----------------------------------------------------------------------
async def test_edit_syncs_once_and_head_change_resyncs_the_new_branch(tmp_path: Path) -> None:
    """AC-3.9: an edit reaches the index through ONE sync, a HEAD change re-syncs the new branch."""
    h, _, emb = await real_stack(tmp_path)
    async with h:
        await h.push("src/a.py")
        await drive(h.clock, lambda: h.loop.completed == 1, "first sync")
        first = h.disk_status()
        assert (first["state"], first["syncing"], first["last_error"]) == ("ready", False, None)
        assert first["gateway_ok_at"] == iso_utc(h.clock.now())
        branch = h.state.read_state().last_branch
        embedded = len(emb.embedded)

        h.fs.files["src/a.py"] = b"def alpha():\n    return 42\n"  # the edit
        await h.push("src/a.py")
        await drive(h.clock, lambda: h.loop.completed == 2, "edit sync")
        assert len(emb.embedded) == embedded + 1  # only the edited file's chunk body was embedded

        h.fs.set_branch("feature")
        await h.push(head=True)  # a HEAD-only batch carries no paths and still syncs
        await drive(h.clock, lambda: h.loop.completed == 3, "branch sync")
        assert h.state.read_state().last_branch != branch
        assert h.status().state == "ready"
        assert h.fs.debounces == [300]  # the configured debounce reaches the port
        await h.push()  # an empty batch is not a change
        await quiesce()
        assert not h.clock.fire_next()
        assert (h.loop.syncs, h.loop.completed) == (3, 3)


# --- coalescing ---------------------------------------------------------------------------------------
async def test_fifty_events_in_one_window_cause_one_sync(tmp_path: Path) -> None:
    eng = ScriptedEngine()
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg()) as h:
        await h.push("src/a.py", n=50)
        await drive(h.clock, lambda: h.loop.completed == 1, "the sync")
        await quiesce()  # a stray second sync would have registered a settle sleep by now
        assert (eng.calls, h.loop.syncs, h.clock.fire_next()) == (1, 1, False)


async def test_a_never_ending_burst_still_syncs_after_the_cap(tmp_path: Path) -> None:
    eng = ScriptedEngine()
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg(settle_s=1.0, max_settle_s=3.0)) as h:
        await h.push("a.py")
        for _ in range(10):  # each settle step sees a new event: only the cap ends the wait
            if eng.calls:
                break
            await h.push("a.py")
            await wait_for(lambda: 1.0 in h.clock.pending() or eng.calls > 0, "a settle sleep")
            h.clock.fire_next()
        await wait_for(lambda: eng.calls == 1, "the capped sync")
        assert h.clock.monotonic() - 1_000.0 <= 4.0 + 1e-9  # <= max_settle plus one settle step


async def test_events_during_a_running_sync_cause_exactly_one_follow_up(tmp_path: Path) -> None:
    eng = ScriptedEngine()
    eng.gate = asyncio.Event()
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg()) as h:
        await h.push("src/a.py")
        await drive(h.clock, lambda: eng.entered.is_set(), "the sync to start")
        await h.push("src/a.py", n=10)  # while the first sync is still running
        assert eng.calls == 1
        eng.gate.set()
        await drive(h.clock, lambda: h.loop.completed == 2, "the follow-up")
        await quiesce()
        assert (eng.calls, h.clock.fire_next()) == (2, False)  # exactly one follow-up, nothing queued after


# --- failure and backoff ------------------------------------------------------------------------------
async def test_failing_sync_backs_off_exponentially_then_recovers(tmp_path: Path) -> None:
    """A gateway failing N times then succeeding: loop alive, error recorded, delays 1,2,4, then cleared."""
    eng = ScriptedEngine(fail=3)
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg(settle_s=0.0)) as h:
        ok_at = h.status().gateway_ok_at
        assert ok_at is not None
        h.clock.advance(5)  # time passes; a failure must not move `gateway_ok_at`
        await h.push("src/a.py")
        await wait_for(lambda: eng.calls == 1 and h.status().last_error is not None, "first failure recorded")
        err = h.disk_status()
        assert err["last_error"] == "UPSTREAM_UNAVAILABLE: gateway down"
        assert err["gateway_ok_at"] == ok_at
        await wait_for(lambda: 1.0 in h.clock.pending(), "backoff of 1 s")
        for expected_calls, delay in ((2, 2.0), (3, 4.0)):
            h.clock.fire_next()
            await wait_for(lambda n=expected_calls: eng.calls == n, "retry")
            await wait_for(lambda d=delay: d in h.clock.pending(), f"backoff of {delay} s")
            assert h.disk_status()["gateway_ok_at"] == ok_at
        h.clock.fire_next()
        await wait_for(lambda: eng.calls == 4 and h.status().last_error is None, "recovery")
        ok = h.disk_status()
        assert ok["gateway_ok_at"] == iso_utc(h.clock.now()) != ok_at
        assert backoffs(h.clock) == [1.0, 2.0, 4.0]
        assert not h.task.done() if h.task else False  # the loop never died


async def test_backoff_is_bounded(tmp_path: Path) -> None:
    eng = ScriptedEngine(fail=8)
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg(settle_s=0.0, backoff_max_s=10.0)) as h:
        await h.push("src/a.py")
        await drive(h.clock, lambda: eng.calls == 9, "nine attempts")
        assert backoffs(h.clock)[:8] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]


async def test_real_engine_survives_a_failing_gateway_and_converges(tmp_path: Path) -> None:
    """The failure comes from the gateway the engine calls: first sync attempts fail, the index follows."""
    h, _, _ = await real_stack(tmp_path, whoami_fails=2)
    async with h:
        await h.push("src/a.py")
        await drive(h.clock, lambda: h.status().last_error is not None, "the failure in status.json")
        assert h.status().state == "absent"
        await drive(
            h.clock, lambda: h.status().state == "ready" and h.status().last_error is None, "recovery"
        )
        assert h.loop.syncs == 3
        assert h.status().gateway_ok_at == iso_utc(h.clock.now())


async def test_contention_retries_without_recording_an_error(tmp_path: Path) -> None:
    """Another process holds `sync.lock`: nothing was done, so the loop tries again later, quietly."""
    eng = ScriptedEngine(contended=2)
    async with Harness(tmp_path, eng, make_fake_gateway(), cfg(settle_s=0.0)) as h:
        await h.push("src/a.py")
        await drive(h.clock, lambda: eng.calls == 2, "two attempts")
        await wait_for(lambda: 1.0 in h.clock.pending(), "the second quiet retry delay")
        assert h.status().last_error is None  # contended twice, nothing recorded in between
        await drive(h.clock, lambda: eng.calls == 3, "three attempts")
        await wait_for(lambda: h.loop.syncs == 3 and h.clock.pending() == [NEVER], "idle")
        assert h.status().last_error is None
        assert backoffs(h.clock) == [1.0, 1.0]  # base delay, not exponential


# --- gateway_ok_at (the hook's readiness signal) ------------------------------------------------------
async def test_ping_refreshes_gateway_ok_at_and_a_failed_ping_never_does(tmp_path: Path) -> None:
    clock = StepClock()
    gw = FlakyGateway(make_fake_gateway(clock=clock), ping_fails=0)
    state = StateDir.open(QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout")))
    state.write_status("ready", 0.9, commit="c" * 40, last_sync_at="2026-01-01T00:00:00Z", gateway_ok_at=None)
    async with Harness(tmp_path, ScriptedEngine(), gw, cfg(ping_interval_s=60.0), clock=clock) as h:
        t0 = h.disk_status()
        assert t0["gateway_ok_at"] == iso_utc(clock.now())
        assert (t0["state"], t0["coverage"], t0["commit"], t0["last_sync_at"]) == (
            "ready",
            0.9,
            "c" * 40,
            "2026-01-01T00:00:00Z",
        )  # the ping only touches `gateway_ok_at`
        await tick(clock, h.loop, 60.0)
        t1 = h.disk_status()["gateway_ok_at"]
        assert t1 == iso_utc(clock.now()) != t0["gateway_ok_at"]

        gw.ping_fails = 2
        await tick(clock, h.loop, 60.0)
        await tick(clock, h.loop, 60.0)
        assert h.disk_status()["gateway_ok_at"] == t1  # two failed pings: untouched, no error recorded
        assert h.disk_status()["last_error"] is None
        await tick(clock, h.loop, 60.0)
        assert h.disk_status()["gateway_ok_at"] == iso_utc(clock.now()) != t1  # recovered


# --- single instance ----------------------------------------------------------------------------------
async def launch(fs: QueueFS, stop: threading.Event, clock: StepClock | None = None) -> WatchResult:
    return await ensure_watch(fs, make_fake_gateway(), ScriptedEngine(), clock or StepClock(), stop, cfg())


async def test_a_second_launcher_returns_the_running_pid_and_starts_nothing(tmp_path: Path) -> None:
    """AC-3.9: two launches, one watcher. SessionStart refires on resume/clear/compact."""
    fs = QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout"))
    stop = threading.Event()
    first = asyncio.create_task(launch(fs, stop))
    pid_file = os.path.join(StateDir.open(fs).path, "watch.pid")
    await wait_for(lambda: read_pid(pid_file) == os.getpid(), "the first watcher's pid file")
    watchers = len(fs.debounces)
    second = await asyncio.wait_for(launch(fs, threading.Event()), 5)
    assert second == WatchResult(started=False, pid=os.getpid())
    assert len(fs.debounces) == watchers  # the second launcher never opened a watch
    assert not first.done()
    stop.set()
    fs.close()
    assert await asyncio.wait_for(first, WAIT_S) == WatchResult(started=True, pid=os.getpid())
    again = asyncio.create_task(launch(fs, stop := threading.Event()))  # the lock is free after a clean exit
    await wait_for(lambda: len(fs.debounces) == watchers + 1, "a relaunch")
    stop.set()
    fs.close()
    assert (await asyncio.wait_for(again, WAIT_S)).started


_HOLDER = (
    "import fcntl, json, os, sys\n"
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
    "os.write(fd, json.dumps({'pid': os.getpid(), 'started_at': 'x'}).encode())\n"
    "print('held', flush=True)\n"
    "sys.stdin.read()\n"
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_a_lock_held_by_another_process_blocks_and_a_dead_holder_is_reclaimed(tmp_path: Path) -> None:
    """The kernel drops a flock with its process, so a killed watcher never blocks the next one."""
    fs = QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout"))
    pid_file = os.path.join(StateDir.open(fs).path, "watch.pid")
    child = subprocess.Popen(  # noqa: S603, ASYNC220
        [sys.executable, "-c", _HOLDER, pid_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "held"
        res = await asyncio.wait_for(launch(fs, threading.Event()), 5)
        assert res == WatchResult(started=False, pid=child.pid)
        assert fs.debounces == []
        child.send_signal(signal.SIGKILL)  # crash: no cleanup, the pid file keeps the dead pid
    finally:
        child.kill()
        child.wait()
        child.stdout.close()
        assert child.stdin is not None
        child.stdin.close()
    assert not _pid_alive(child.pid)
    assert read_pid(pid_file) == child.pid  # stale content, no lock
    stop = threading.Event()
    task = asyncio.create_task(launch(fs, stop))
    await wait_for(lambda: read_pid(pid_file) == os.getpid(), "the stale lock to be reclaimed")
    stop.set()
    fs.close()
    assert (await asyncio.wait_for(task, WAIT_S)).started


async def test_a_stale_pid_file_without_a_lock_is_overwritten(tmp_path: Path) -> None:
    fs = QueueFS(FILES, state_dir=str(tmp_path / "git" / "scout"))
    pid_file = os.path.join(StateDir.open(fs).path, "watch.pid")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603, ASYNC220
    dead.wait()
    Path(pid_file).write_text(json.dumps({"pid": dead.pid, "started_at": "x", "pad": "p" * 300}))  # noqa: ASYNC240
    stop = threading.Event()
    task = asyncio.create_task(launch(fs, stop))
    await wait_for(lambda: read_pid(pid_file) == os.getpid(), "the stale pid to be replaced")
    leftover = json.loads(Path(pid_file).read_text())  # noqa: ASYNC240
    assert set(leftover) == {"pid", "started_at"}  # no stale bytes after the new, shorter document
    stop.set()
    fs.close()
    await asyncio.wait_for(task, WAIT_S)


async def test_stop_ends_the_loop_even_while_backing_off(tmp_path: Path) -> None:
    eng = ScriptedEngine(fail=100)
    h = Harness(tmp_path, eng, make_fake_gateway(), cfg(settle_s=0.0))
    async with h:
        await h.push("src/a.py")
        await wait_for(lambda: 1.0 in h.clock.pending(), "a backoff sleep")
    assert h.task is not None
    assert h.task.done()
    assert h.task.exception() is None
    assert h.stop.is_set()
