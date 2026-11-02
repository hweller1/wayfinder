"""Watch loop and single-instance launcher (task 4.5, design 5.2, 11; PRD 4.3 item 7, 8.3; FR-C14, FR-C15).

`ensure_watch` is what `scout watch` runs. It takes the exclusive `flock` on `<state_dir>/watch.pid`; a second
launcher (SessionStart refires on resume, clear and compact) fails to get it, reports the pid in the file and
returns at once. The kernel drops a flock when its process dies, so a crashed watcher never leaves a lock
that blocks the next launch (a stale pid in the file is simply overwritten).

`WatchLoop` depends only on ports (`FileSystem.watch`, `Gateway.ping`, `Clock`) and a `Syncer`; the OS
watcher lives behind `FileSystem.watch` (a blocking iterator, pumped on a worker thread):
- Coalescing: `watch` already debounces; the loop adds a quiet period (`settle_s`, clock-driven) and clears
  its dirty flag when a sync STARTS, so any number of batches that land during a running sync cause exactly
  one follow-up sync. A HEAD-only batch also syncs: `SyncEngine.sync` re-reads git ids, so a branch switch
  indexes the new branch.
- Failure: a sync that raises never ends the loop. The error is recorded in `status.json` (`last_error`, the
  previous fields kept) and the sync is retried after `min(backoff_base_s * 2**(n-1), backoff_max_s)` seconds
  of `Clock.sleep`, `n` consecutive failures. A sync that finds another process holding `sync.lock` did no
  work; it is retried after the base delay without recording an error.
- Readiness signal: `gateway_ok_at` in `status.json` is what the Claude Code hook reads. A successful sync
  sets it (the engine does, and the loop again after a clean return); a successful `Gateway.ping()` every
  `ping_interval_s` refreshes it. A failed ping or sync never moves it. The ping write skips while
  `sync.lock` is held (the sync writes its own).

`FakeClock.sleep` returns at once, so a loop on a `FakeClock` pings in a tight cycle until the source ends.
"""

import asyncio
import contextlib
import fcntl  # POSIX only; Windows is a v1 non-goal (PRD section 10). Move behind FileSystem for a port.
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Protocol

from scout.core.contracts.errors import ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem, FsBatch
from scout.core.ports.gateway import Gateway
from scout.core.sync.state import StateDir, iso_utc

LOGGER = logging.getLogger("scout.watch")
PID_FILE = "watch.pid"
PING_INTERVAL_S = 60.0
_PID_READ_TRIES = 10  # the holder writes its pid just after taking the lock; a fast second launcher waits
_PID_READ_WAIT_S = 0.05
_ERROR_MAX = 200


class Syncer(Protocol):
    async def sync(self, wait: bool = True, *, full: bool = False) -> IndexStatus: ...


@dataclass(frozen=True, slots=True)
class WatchConfig:
    debounce_ms: int = 300  # passed to `FileSystem.watch`
    settle_s: float = 0.1  # quiet period after the first event; 0 disables
    max_settle_s: float = 2.0  # a never-ending burst still syncs after this long
    ping_interval_s: float = PING_INTERVAL_S
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0

    def __post_init__(self) -> None:
        if min(self.ping_interval_s, self.backoff_base_s, self.backoff_max_s) <= 0:
            raise ValueError("intervals must be positive")
        if self.settle_s < 0 or self.max_settle_s < 0 or self.debounce_ms < 0:
            raise ValueError("settle and debounce must not be negative")


@dataclass(frozen=True, slots=True)
class WatchResult:
    started: bool  # False: another watcher already runs and this call did nothing
    pid: int | None  # pid of the running watcher (this process when `started`); None if unreadable


class WatchLock:
    """Exclusive `flock` on `watch.pid`, held for the life of the watcher; the file carries its pid."""

    def __init__(self, state: StateDir) -> None:
        self._path = os.path.join(state.path, PID_FILE)
        self._fd: int | None = None

    @property
    def path(self) -> str:
        return self._path

    def acquire(self, now: float) -> bool:
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        doc = json.dumps({"pid": os.getpid(), "started_at": iso_utc(now)}).encode("utf-8")
        os.pwrite(fd, doc, 0)
        os.ftruncate(fd, len(doc))  # overwrite a stale pid in place, never an empty window
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def read_pid(path: str) -> int | None:
    try:
        with open(path, "rb") as f:
            pid = json.loads(f.read()).get("pid")
    except (OSError, ValueError, AttributeError):
        return None
    return pid if isinstance(pid, int) and pid > 0 else None


async def _holder_pid(path: str) -> int | None:
    """The holder writes its pid just after taking the lock: wait briefly (real time) for it."""
    for i in range(_PID_READ_TRIES):
        pid = await asyncio.to_thread(read_pid, path)
        if pid is not None:
            return pid
        if i < _PID_READ_TRIES - 1:
            await asyncio.sleep(_PID_READ_WAIT_S)
    return None


async def ensure_watch(
    fs: FileSystem,
    gw: Gateway,
    engine: Syncer,
    clock: Clock,
    stop: threading.Event,
    cfg: WatchConfig | None = None,
) -> WatchResult:
    """Run the watcher until `stop` is set, unless one already runs for this clone (then return at once)."""
    state = await asyncio.to_thread(StateDir.open, fs)
    lock = WatchLock(state)
    if not await asyncio.to_thread(lock.acquire, clock.now()):
        return WatchResult(False, await _holder_pid(lock.path))
    try:
        await WatchLoop(fs, gw, engine, clock, state, cfg).run(stop)
    finally:
        lock.release()
    return WatchResult(True, os.getpid())


def _describe(exc: BaseException) -> str:
    """One line for `status.json`: a ScoutError's code and message (both are safe by contract); for any
    other exception only the type, since its text may carry a path."""
    text = f"{exc.code}: {exc.message}" if isinstance(exc, ScoutError) else type(exc).__name__
    return text[:_ERROR_MAX]


class WatchLoop:
    def __init__(
        self,
        fs: FileSystem,
        gw: Gateway,
        engine: Syncer,
        clock: Clock,
        state: StateDir,
        cfg: WatchConfig | None = None,
    ) -> None:
        self._fs, self._gw, self._engine, self._clock, self._state = fs, gw, engine, clock, state
        self._cfg = cfg or WatchConfig()
        self.syncs = 0  # sync attempts started (observability for tests and `scout watch` logs)
        self.completed = 0  # sync attempts finished, outcome and `status.json` already written
        self.batches = 0  # batches received from the file system
        self.pings = 0  # ping attempts, failed ones included
        self._dirty = False
        self._events = 0
        self._ended = False  # the batch source is exhausted (a real watcher only ends on stop)
        self._stopping = False  # `stop` was set: leave without finishing pending work
        self._wake = asyncio.Event()
        self._stop_evt = asyncio.Event()

    # --- batch source (worker thread -> loop thread) --------------------------------------------
    def _pump(self, stop: threading.Event, loop: asyncio.AbstractEventLoop) -> None:
        try:
            for batch in self._fs.watch(stop, debounce_ms=self._cfg.debounce_ms):
                loop.call_soon_threadsafe(self._on_batch, batch)
        finally:
            loop.call_soon_threadsafe(self._on_end, stop.is_set())

    def _on_batch(self, batch: FsBatch) -> None:
        self.batches += 1
        if batch.paths or batch.head_changed:
            self._dirty = True
            self._events += 1
            self._wake.set()

    def _on_end(self, stopped: bool) -> None:
        self._ended = True
        if stopped:
            self._stopping = True
            self._stop_evt.set()
        self._wake.set()

    # --- run ------------------------------------------------------------------------------------
    async def run(self, stop: threading.Event) -> None:
        """Ping and sync until `stop` is set (or the batch source ends). Sets `stop` on the way out so the
        watcher thread ends; a failure of the watcher itself is raised."""
        loop = asyncio.get_running_loop()
        pump = asyncio.create_task(asyncio.to_thread(self._pump, stop, loop))
        pinger = asyncio.create_task(self._ping_loop())
        try:
            await self._sync_loop()
        finally:
            stop.set()
            pinger.cancel()
            results = await asyncio.gather(pinger, pump, return_exceptions=True)
        failure = results[1]
        if isinstance(failure, BaseException):
            raise failure

    async def _sleep(self, seconds: float) -> bool:
        """`Clock.sleep` that a stop interrupts; True when the full time elapsed."""
        timer = asyncio.create_task(self._clock.sleep(seconds))
        stopped = asyncio.create_task(self._stop_evt.wait())
        try:
            await asyncio.wait({timer, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (timer, stopped):
                t.cancel()
            await asyncio.gather(timer, stopped, return_exceptions=True)
        return not self._stopping

    async def _settle(self) -> None:
        """Wait until no new batch arrived for `settle_s` (at most `max_settle_s` in total)."""
        started = self._clock.monotonic()
        while self._cfg.settle_s > 0 and not self._stopping:
            seen = self._events
            if not await self._sleep(self._cfg.settle_s):
                return
            if self._events == seen or self._clock.monotonic() - started >= self._cfg.max_settle_s:
                return

    async def _sync_loop(self) -> None:
        failures = 0
        retry = False
        while not self._stopping:
            if not self._dirty:
                if self._ended:
                    return
                self._wake.clear()
                if not (self._dirty or self._ended):
                    await self._wake.wait()
                continue
            if not retry:
                await self._settle()
                if self._stopping:
                    return
            self._dirty = False  # events from here on belong to the NEXT sync
            outcome = await self._sync_once()
            self.completed += 1
            if outcome is None:
                failures, retry = 0, False
                continue
            failures = failures + 1 if outcome else failures
            self._dirty, retry = True, True
            delay = min(self._cfg.backoff_base_s * 2.0 ** max(failures - 1, 0), self._cfg.backoff_max_s)
            if not await self._sleep(delay):
                return

    async def _sync_once(self) -> bool | None:
        """None: synced. True: failed (error recorded). False: another process was syncing (nothing done)."""
        self.syncs += 1
        try:
            status = await self._engine.sync()
        except Exception as exc:  # noqa: BLE001  the loop must outlive any sync failure
            msg = _describe(exc)
            LOGGER.warning("sync failed: %s", msg)
            await self._update_status(error=msg)
            return True
        if status.syncing:
            return False
        await self._update_status(ok_at=iso_utc(self._clock.now()), clear_error=True)
        return None

    # --- gateway readiness ----------------------------------------------------------------------
    async def _ping_loop(self) -> None:
        while not self._stopping:
            self.pings += 1
            try:
                await self._gw.ping()
            except Exception as exc:  # noqa: BLE001  an unreachable gateway only leaves `gateway_ok_at` old
                LOGGER.warning("ping failed: %s", _describe(exc))
            else:
                await self._update_status(ok_at=iso_utc(self._clock.now()))
            if not await self._sleep(self._cfg.ping_interval_s):
                return

    async def _update_status(
        self, *, ok_at: str | None = None, error: str | None = None, clear_error: bool = False
    ) -> None:
        await asyncio.to_thread(self._write_status, ok_at, error, clear_error)

    def _write_status(self, ok_at: str | None, error: str | None, clear_error: bool) -> None:
        """Read-modify-write `status.json`; skipped while a sync holds `sync.lock` (it writes its own)."""
        with self._state.lock() as held:
            if not held:
                return
            cur = self._state.read_status()
            self._state.write_status(
                cur.state if cur else "absent",
                cur.coverage if cur else 0.0,
                commit=cur.commit if cur else None,
                syncing=False,
                last_sync_at=cur.last_sync_at if cur else None,
                gateway_ok_at=ok_at if ok_at is not None else (cur.gateway_ok_at if cur else None),
                last_error=error
                if error is not None
                else (None if clear_error or not cur else cur.last_error),
            )
