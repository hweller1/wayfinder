"""fs_local watcher (task 2.16): debounced batches, HEAD tracking, ignore filtering, stop.

AC-3.9, FR-C14. Real temporary repos; every wait polls with a bounded timeout (FSEvents latency varies),
never a fixed sleep that assumes timing.
"""

import os
import queue
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from scout.adapters.fs_local import LocalFileSystem
from scout.core.ports.fs import FsBatch

WAIT_S = 5.0
_HERMETIC = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def sh(cwd: Path, *args: str) -> str:
    """Test-side git, hermetic (no user/global config)."""
    p = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        env={**os.environ, **_HERMETIC},
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.strip()


def init_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    sh(path, "init", "-q", "-b", "main")
    for rel, data in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(data)
    sh(path, "add", "-A")
    sh(path, "commit", "-q", "-m", "init")
    return path


class Watcher:
    """Runs `fs.watch` on a thread and exposes the batches through a queue."""

    def __init__(self, fs: LocalFileSystem) -> None:
        self.stop = threading.Event()
        self.batches: queue.Queue[FsBatch] = queue.Queue()
        self.error: BaseException | None = None
        self.done = threading.Event()
        self._it = fs.watch(self.stop, debounce_ms=100)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            for b in self._it:
                self.batches.put(b)
        except BaseException as e:  # noqa: BLE001 - surfaced to the test thread
            self.error = e
        finally:
            self.done.set()

    def wait_for(self, pred, timeout: float = WAIT_S) -> list[FsBatch]:  # type: ignore[no-untyped-def]
        """Collect batches until `pred(all_batches_so_far)` holds; fail on timeout."""
        got: list[FsBatch] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                got.append(self.batches.get(timeout=0.1))
            except queue.Empty:
                continue
            if pred(got):
                return got
        raise AssertionError(f"timed out after {timeout}s, batches so far: {got}")

    def drain(self, quiet: float = 0.2) -> list[FsBatch]:
        """Everything that arrives until `quiet` seconds pass with nothing new."""
        got: list[FsBatch] = []
        while True:
            try:
                got.append(self.batches.get(timeout=quiet))
            except queue.Empty:
                return got

    def arm(self, root: Path) -> None:
        """Return once the OS watch is live: touch a probe file until its batch arrives (no fixed sleep)."""
        probe = root / "probe.tmp"
        deadline = time.monotonic() + WAIT_S
        n = 0
        while time.monotonic() < deadline:
            n += 1
            probe.write_text(str(n))
            try:
                got = self.batches.get(timeout=0.2)
            except queue.Empty:
                continue
            assert "probe.tmp" in got.paths
            probe.unlink()
            self.drain(0.15)  # settle: later waits must not see probe leftovers
            return
        raise AssertionError("watcher never saw the probe file")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {"src/a.py": "x = 1\n", ".gitignore": "build/\n", ".scoutignore": "secret_notes.txt\n"},
    )


@pytest.fixture
def watcher(repo: Path) -> Iterator[Watcher]:
    w = Watcher(LocalFileSystem(str(repo)))
    w.arm(repo)
    try:
        yield w
    finally:
        w.close()


def _paths(batches: list[FsBatch]) -> set[str]:
    return {p for b in batches for p in b.paths}


def test_one_edit_one_batch_within_2s(repo: Path, watcher: Watcher) -> None:
    t0 = time.monotonic()
    (repo / "src" / "a.py").write_text("x = 2\n")
    got = watcher.wait_for(lambda bs: "src/a.py" in _paths(bs), timeout=2.0)
    assert time.monotonic() - t0 < 2.0
    assert [b.paths for b in got] == [("src/a.py",)]
    assert not any(b.head_changed for b in got)


def test_burst_is_debounced_into_one_batch_and_rename_delete_report_paths(
    repo: Path, watcher: Watcher
) -> None:
    for i in range(5):
        (repo / "src" / "a.py").write_text(f"x = {i}\n")
    (repo / "new.py").write_text("y\n")
    got = watcher.wait_for(lambda bs: {"src/a.py", "new.py"} <= _paths(bs))
    assert len(got) == 1
    assert got[0].paths == ("new.py", "src/a.py")
    (repo / "src" / "a.py").rename(repo / "src" / "b.py")  # rename = deleted + added
    watcher.wait_for(lambda bs: {"src/a.py", "src/b.py"} <= _paths(bs))
    (repo / "src" / "b.py").unlink()
    watcher.wait_for(lambda bs: "src/b.py" in _paths(bs))


def test_ignored_and_denylisted_paths_never_appear(repo: Path, watcher: Watcher) -> None:
    (repo / ".env").write_text("KEY=1\n")
    (repo / "secret_notes.txt").write_text("x\n")  # .scoutignore
    (repo / "build").mkdir()
    (repo / "build" / "out.js").write_text("x\n")  # .gitignore
    (repo / "visible.py").write_text("ok\n")
    got = watcher.wait_for(lambda bs: "visible.py" in _paths(bs))
    got += watcher.drain()
    assert _paths(got) == {"visible.py"}


def test_ignore_file_edit_applies_to_later_events(repo: Path, watcher: Watcher) -> None:
    (repo / ".scoutignore").write_text("secret_notes.txt\nlater.txt\n")
    watcher.wait_for(lambda bs: ".scoutignore" in _paths(bs))
    watcher.drain()
    (repo / "later.txt").write_text("x\n")
    (repo / "marker.py").write_text("x\n")
    got = watcher.wait_for(lambda bs: "marker.py" in _paths(bs)) + watcher.drain()
    assert "later.txt" not in _paths(got)


def test_checkout_and_commit_flag_head_change_and_hide_git_internals(repo: Path, watcher: Watcher) -> None:
    sh(repo, "checkout", "-q", "-b", "feature")
    got = watcher.wait_for(lambda bs: any(b.head_changed for b in bs))
    got += watcher.drain()
    (repo / "c.py").write_text("c\n")
    sh(repo, "add", "c.py")
    watcher.wait_for(lambda bs: "c.py" in _paths(bs))
    watcher.drain()
    sh(repo, "commit", "-q", "-m", "c")  # moves refs/heads/feature only, HEAD file untouched
    got += watcher.wait_for(lambda bs: any(b.head_changed for b in bs))
    got += watcher.drain()
    assert not any(p == ".git" or p.startswith(".git/") for p in _paths(got))


def test_head_change_in_linked_worktree(repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    sh(repo, "worktree", "add", "-q", "-b", "wtbranch", str(wt))
    w = Watcher(LocalFileSystem(str(wt)))
    try:
        w.arm(wt)
        sh(wt, "checkout", "-q", "-b", "other")
        w.wait_for(lambda bs: any(b.head_changed for b in bs))
        (wt / "e.py").write_text("e\n")
        sh(wt, "add", "e.py")
        w.drain()
        sh(wt, "commit", "-q", "-m", "e")
        w.wait_for(lambda bs: any(b.head_changed for b in bs))
    finally:
        w.close()


def test_stop_ends_iterator_within_1s_and_leaks_no_threads(repo: Path) -> None:
    baseline = threading.active_count()
    w = Watcher(LocalFileSystem(str(repo)))
    w.arm(repo)
    t0 = time.monotonic()
    w.stop.set()
    assert w.done.wait(timeout=1.0), "iterator did not end within 1 s of stop.set()"
    assert time.monotonic() - t0 < 1.0
    w.thread.join(timeout=1.0)
    assert w.error is None
    deadline = time.monotonic() + WAIT_S
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() == baseline


def test_stop_set_before_start_yields_nothing(repo: Path) -> None:
    stop = threading.Event()
    stop.set()
    assert list(LocalFileSystem(str(repo)).watch(stop)) == []
