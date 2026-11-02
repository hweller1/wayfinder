"""Sync file set, state dir, status file and readiness (task 2.23).

FR-C8, FR-C9, FR-C13, AC-2.5: one matcher for the file set, state under the git dir (worktree-safe), atomic
writes, flock exclusivity, schema-valid `status.json`, stat-cache skip rule, 80% readiness.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fs_local import LocalFileSystem
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.schema import StatusFile
from scout.core.local.ignore import MAX_FILE_BYTES, Matcher
from scout.core.ports.fs import FileStat
from scout.core.sync import files as sync_files
from scout.core.sync import state as sync_state
from scout.core.sync.files import list_indexable
from scout.core.sync.state import FileEntry, StateDir, TreeState

SCHEMA = json.loads((Path(__file__).parents[2] / "schema" / "v1" / "status_file.json").read_text())
BRANCH = "0123456789abcdef"
NOW = 1_700_000_000.0


def iso(epoch: float) -> str:
    return sync_state.iso_utc(epoch)


def sd_for(tmp_path: Path) -> StateDir:
    return StateDir.open(FakeFileSystem(state_dir=str(tmp_path / "git" / "scout")))


def ok_status(**over: Any) -> StatusFile:
    base: dict[str, Any] = {"state": "ready", "coverage": 1.0, "gateway_ok_at": iso(NOW - 5)}
    return StatusFile.model_validate({**base, **over})


# --- file set (FR-C8, AC-2.5) ---------------------------------------------------------------------
def test_list_indexable_uses_the_shared_matcher() -> None:
    """AC-2.5: sync lists exactly what the Matcher admits (gitignore, scoutignore, denylist, size, binary)."""
    fs = FakeFileSystem(
        {
            "a.py": "x = 1\n",
            "src/b.py": "y = 2\n",
            "build/out.js": "1\n",
            "vendor/v.py": "2\n",
            ".env": "K=1\n",
            "keys/id_rsa": "k\n",
            "big.txt": "a" * (MAX_FILE_BYTES + 1),
            "bin.dat": b"ab\x00cd",
            ".gitignore": "build/\n",
            ".scoutignore": "vendor/\n",
        }
    )
    got = list_indexable(fs)
    m = Matcher.from_fs(fs)
    assert got.paths == [p for p in sorted(fs.list_files()) if not m.is_excluded(p)]
    assert got.paths == [".gitignore", ".scoutignore", "a.py", "src/b.py"]
    assert got.warnings == []


def test_list_indexable_skips_escaping_symlinks_and_vanished_files() -> None:
    fs = FakeFileSystem({"ok.py": "1\n"}, symlinks={"esc": "../../etc/passwd"})
    assert list_indexable(fs).paths == ["ok.py"]


def test_list_indexable_caps_files_with_a_warning() -> None:
    fs = FakeFileSystem({f"f{i:03d}.py": "x\n" for i in range(12)})
    got = list_indexable(fs, max_files=5)
    assert got.paths == [f"f{i:03d}.py" for i in range(5)]
    assert len(got.warnings) == 1
    assert "5" in got.warnings[0]
    assert sync_files.MAX_FILES == 100_000
    assert list_indexable(fs, max_files=12).warnings == []


# --- state dir location ------------------------------------------------------------------------------
def _sh(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    p = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True)  # noqa: S603, S607
    return p.stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _sh(path, "init", "-q", "-b", "main")
    (path / "a.py").write_text("x = 1\n")
    _sh(path, "add", "-A")
    _sh(path, "commit", "-q", "-m", "init")
    return path


def test_state_dir_is_inside_git_dir_and_never_indexed(tmp_path: Path) -> None:
    root = _repo(tmp_path / "r")
    fs = LocalFileSystem(str(root))
    sd = StateDir.open(fs)
    sd.read_state()
    sd.save_tree(BRANCH, TreeState("r" * 64, {"a.py": FileEntry("l" * 64, 6, 1, 2)}))
    sd.write_status("ready", 1.0)
    real = os.path.realpath
    assert real(sd.path) == real(root / ".git" / "scout")
    assert not (root / "scout").exists()
    assert _sh(root, "status", "--porcelain") == ""  # nothing in the working tree
    assert all(".git/scout" not in p and not p.startswith("scout") for p in fs.list_files())
    assert list_indexable(fs).paths == ["a.py"]


def test_two_worktrees_of_one_repo_get_separate_state(tmp_path: Path) -> None:
    main = _repo(tmp_path / "main")
    wt = tmp_path / "wt"
    _sh(main, "worktree", "add", "-q", "-b", "wtb", str(wt))
    sd_main = StateDir.open(LocalFileSystem(str(main)))
    sd_wt = StateDir.open(LocalFileSystem(str(wt)))
    real = os.path.realpath
    assert real(sd_main.path) == real(main / ".git" / "scout")
    assert real(sd_wt.path) == real(main / ".git" / "worktrees" / "wt" / "scout")
    assert real(sd_main.path) != real(sd_wt.path)
    assert not (wt / ".git").is_dir()
    assert sd_main.clone_id() != sd_wt.clone_id()
    sd_wt.write_status("partial", 0.5)
    sd_main.write_status("ready", 1.0)
    rs = sd_wt.read_status()
    assert rs is not None
    assert rs.state == "partial"
    assert list_indexable(LocalFileSystem(str(wt))).paths == ["a.py"]


def test_state_dir_not_a_repo(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(ScoutError) as ei:
        StateDir.open(LocalFileSystem(str(tmp_path / "plain")))
    assert ei.value.code is ErrorCode.NOT_A_REPO


# --- clone_id ---------------------------------------------------------------------------------------
def test_clone_id_is_12_hex_written_once_and_stable(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    cid = sd.clone_id()
    assert len(cid) == 12
    assert int(cid, 16) >= 0
    assert cid == cid.lower()
    assert StateDir(sd.path).clone_id() == cid  # a new process-level handle reads the same value
    sd.update_state(tenant="acme", repo_id="a" * 32, last_branch="main")
    st = StateDir(sd.path).read_state()
    assert (st.clone_id, st.tenant, st.repo_id, st.last_branch) == (cid, "acme", "a" * 32, "main")
    doc = json.loads((Path(sd.path) / "state.json").read_text())
    assert doc["clone_id"] == cid
    assert doc["schema"] == "1.0"


def test_fresh_clone_gets_a_different_clone_id(tmp_path: Path) -> None:
    """Parent ruling Q1: clone_id is per clone, so a second clone of the same repo differs."""
    ids = {sd_for(tmp_path / str(i)).clone_id() for i in range(20)}
    assert len(ids) == 20


def test_concurrent_first_runs_agree_on_one_clone_id(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    seen: list[str] = []
    barrier = threading.Barrier(8)

    def work() -> None:
        barrier.wait()
        seen.append(StateDir(sd.path).clone_id())

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(seen) == 8
    assert len(set(seen)) == 1
    assert StateDir(sd.path).clone_id() == seen[0]


def test_corrupt_state_json_is_replaced(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    (Path(sd.path) / "state.json").write_text("{not json")
    cid = sd.clone_id()
    assert len(cid) == 12
    assert sd.clone_id() == cid


# --- trees: atomic, round trip, stat cache ------------------------------------------------------------
def test_tree_roundtrip_and_missing(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    assert sd.load_tree(BRANCH) is None
    t = TreeState(
        "ab" * 32, {"src/a.py": FileEntry("c" * 64, 10, 123, 7), "b.py": FileEntry("d" * 64, 1, 2, 3)}
    )
    sd.save_tree(BRANCH, t)
    assert sd.load_tree(BRANCH) == t
    assert sorted(os.listdir(Path(sd.path) / "trees")) == [f"{BRANCH}.json"]  # no stray tmp files
    with pytest.raises(ValueError, match="branch_id"):
        sd.save_tree("../../evil", t)


def test_unreadable_tree_loads_as_none(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    p = Path(sd.path) / "trees" / f"{BRANCH}.json"
    for bad in ("", "[]", '{"root": 1}', '{"root": null, "files": {"a": {"leaf": "x"}}}', "\x00\x01"):
        p.write_text(bad)
        assert sd.load_tree(BRANCH) is None, bad


def test_tmp_failure_leaves_previous_file_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("ready", 1.0)
    before = (Path(sd.path) / "status.json").read_bytes()

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        sd.write_status("partial", 0.1)
    monkeypatch.undo()
    assert (Path(sd.path) / "status.json").read_bytes() == before
    assert [n for n in os.listdir(sd.path) if n.startswith(".tmp-")] == []


def _st(size: int = 5, mtime_ns: int = 10, ino: int = 3) -> FileStat:
    return FileStat(size=size, mtime_ns=mtime_ns, ino=ino, is_symlink=False)


def test_stat_cache_skip_rule() -> None:
    saved = FileEntry("l" * 64, 5, 10, 3)
    assert sync_state.stat_unchanged(saved, _st())
    assert not sync_state.stat_unchanged(saved, _st(size=6))
    assert not sync_state.stat_unchanged(saved, _st(mtime_ns=11))
    assert not sync_state.stat_unchanged(saved, _st(ino=4))
    assert not sync_state.stat_unchanged(None, _st())
    assert not sync_state.stat_unchanged(saved, _st(), full=True)  # --full re-hashes everything


# --- status.json ------------------------------------------------------------------------------------
def test_written_status_validates_against_published_schema(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    cases: list[dict[str, Any]] = [
        {"state": "absent", "coverage": 0.0},
        {"state": "partial", "coverage": 0.42, "commit": "a" * 40, "syncing": True},
        {
            "state": "ready",
            "coverage": 1.0,
            "commit": None,
            "last_sync_at": iso(NOW),
            "gateway_ok_at": iso(NOW),
        },
    ]
    for kw in cases:
        sd.write_status(kw.pop("state"), kw.pop("coverage"), **kw)
        raw = json.loads((Path(sd.path) / "status.json").read_text())
        jsonschema.validate(raw, SCHEMA)
        assert raw["schema"] == "1.0"
        assert set(raw) == {
            "schema",
            "state",
            "coverage",
            "commit",
            "syncing",
            "last_sync_at",
            "gateway_ok_at",
            "last_error",  # task 4.5: the watcher's last failed sync, additive and optional for readers
        }
        assert sd.read_status() == StatusFile.model_validate(raw)


def test_write_status_rejects_invalid_values_without_touching_the_file(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("ready", 1.0)
    before = (Path(sd.path) / "status.json").read_bytes()
    for state, cov in (("bogus", 0.5), ("ready", 1.5), ("ready", -0.1)):
        with pytest.raises(ValueError, match="validation|Input"):
            sd.write_status(state, cov)
    assert (Path(sd.path) / "status.json").read_bytes() == before


def test_read_status_absent_or_invalid_is_none(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    assert sd.read_status() is None
    (Path(sd.path) / "status.json").write_text('{"state": "wat"}')
    assert sd.read_status() is None
    (Path(sd.path) / "status.json").write_text("[1]")
    assert sd.read_status() is None


_WRITER = """
import sys
from scout.core.sync.state import StateDir
sd = StateDir(sys.argv[1])
n = int(sys.argv[2])
for i in range(400):
    sd.write_status("partial" if i % 2 else "ready", (i % 10) / 10, commit="%040x" % (n * 1000 + i),
                    syncing=bool(i % 3))
"""


def _read_loop(path: Path, stop: threading.Event, bad: list[str], reads: list[int]) -> None:
    while not stop.is_set():
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        reads[0] += 1
        try:
            jsonschema.validate(json.loads(raw), SCHEMA)  # stdlib json read, as the hook does
        except Exception as e:  # noqa: BLE001
            bad.append(f"{type(e).__name__}: {raw[:80]!r}")


def test_concurrent_thread_writers_never_expose_a_partial_file(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("absent", 0.0)
    stop, bad, reads = threading.Event(), [], [0]
    reader = threading.Thread(target=_read_loop, args=(Path(sd.path) / "status.json", stop, bad, reads))
    reader.start()

    def work(n: int) -> None:
        for i in range(150):
            sd.write_status("ready" if i % 2 else "partial", (i % 10) / 10, commit=f"{n}-{i}")

    ts = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    stop.set()
    reader.join()
    assert reads[0] > 0
    assert bad == []
    assert [n for n in os.listdir(sd.path) if n.startswith(".tmp-")] == []


def test_concurrent_process_writers_never_expose_a_partial_file(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("absent", 0.0)
    stop, bad, reads = threading.Event(), [], [0]
    reader = threading.Thread(target=_read_loop, args=(Path(sd.path) / "status.json", stop, bad, reads))
    reader.start()
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, sd.path, str(n)])  # noqa: S603
        for n in range(4)
    ]
    codes = [p.wait(timeout=120) for p in procs]
    stop.set()
    reader.join()
    assert codes == [0, 0, 0, 0]
    assert reads[0] > 0
    assert bad == []
    assert sd.read_status() is not None
    assert [n for n in os.listdir(sd.path) if n.startswith(".tmp-")] == []


# --- sync.lock ---------------------------------------------------------------------------------------
def test_lock_is_exclusive_and_contention_returns_current_status(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("partial", 0.5, syncing=True)
    other = StateDir(sd.path)
    with sd.lock() as held:
        assert held is True
        with other.lock() as second:  # a second sync attempt: not held, nothing blocks
            assert second is False
            cur = other.read_status()  # "current status" path
            assert cur is not None
            assert (cur.state, cur.syncing) == ("partial", True)
        sd.write_status("ready", 1.0, syncing=False)  # status stays writable and readable under the lock
        done = other.read_status()
        assert done is not None
        assert done.state == "ready"
    with other.lock() as again:  # released on exit
        assert again is True


def test_lock_released_after_exception(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    with pytest.raises(RuntimeError), sd.lock():
        raise RuntimeError("sync failed")
    with sd.lock() as held:
        assert held is True


def test_lock_excludes_another_process_and_status_stays_readable(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    sd.write_status("partial", 0.6, syncing=True)
    code = (
        "import sys\nfrom scout.core.sync.state import StateDir\n"
        "sd = StateDir(sys.argv[1])\n"
        "with sd.lock() as h:\n"
        "    s = sd.read_status()\n"
        "    print(h, s.state if s else None)\n"
    )

    def child() -> str:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", code, sd.path], capture_output=True, text=True, check=True, timeout=60
        ).stdout.strip()

    with sd.lock() as held:
        assert held is True
        assert child() == "False partial"
    assert child() == "True partial"


def test_lock_wait_blocks_until_release(tmp_path: Path) -> None:
    sd = sd_for(tmp_path)
    got: list[bool] = []
    entered = threading.Event()

    def waiter() -> None:
        with StateDir(sd.path).lock(wait=True) as h:
            got.append(h)
            entered.set()

    with sd.lock() as held:
        assert held
        t = threading.Thread(target=waiter)
        t.start()
        assert not entered.wait(0.3)
    t.join(timeout=10)
    assert got == [True]


# --- no secrets in state files ---------------------------------------------------------------------
def test_state_files_hold_no_secret_or_plaintext(tmp_path: Path) -> None:
    k_team = bytes(range(32))
    scout_key = "sk_scout_FIXTURE_KEY_0123456789"
    secret_text = "SECRET_PLAINTEXT_CONTENT_xyz"  # noqa: S105
    fs = FakeFileSystem({"a.py": f"x = '{secret_text}'\n"}, state_dir=str(tmp_path / "git" / "scout"))
    sd = StateDir.open(fs)
    st = fs.stat("a.py")
    sd.update_state(tenant="acme", repo_id="b" * 32, last_branch="main")
    sd.save_tree(BRANCH, TreeState("e" * 64, {"a.py": FileEntry("f" * 64, st.size, st.mtime_ns, st.ino)}))
    sd.write_status("ready", 1.0, commit="c" * 40, last_sync_at=iso(NOW), gateway_ok_at=iso(NOW))
    with sd.lock():
        pass
    blob = b"".join(p.read_bytes() for p in Path(sd.path).rglob("*") if p.is_file())
    assert blob
    for needle in (k_team, k_team.hex().encode(), scout_key.encode(), secret_text.encode(), b"FIXTURE_KEY"):
        assert needle not in blob


# --- readiness (pure; hook parity) ---------------------------------------------------------------------
def test_classify_threshold_is_80_percent() -> None:
    assert sync_state.classify(0, 0, has_root=False) == ("absent", 0.0)
    assert sync_state.classify(80, 100, has_root=True) == ("ready", 0.8)
    assert sync_state.classify(79, 100, has_root=True)[0] == "partial"
    assert sync_state.classify(0, 100, has_root=True) == ("partial", 0.0)
    assert sync_state.classify(0, 0, has_root=True) == ("ready", 1.0)  # empty expected set: nothing missing
    assert sync_state.classify(150, 100, has_root=True) == ("ready", 1.0)  # clamped


def _ready(over: dict[str, Any] | None, creds: bool = True, mcp: bool = True) -> bool:
    status = None if over is None else ok_status(**over)
    return sync_state.readiness(status, NOW, creds, mcp)


def test_readiness_rule() -> None:
    assert _ready({}) is True
    assert _ready({"state": "partial", "coverage": 0.5}) is True
    assert _ready({"state": "absent", "coverage": 0.0}) is False
    assert _ready(None) is False
    assert _ready({}, creds=False) is False
    assert _ready({}, mcp=False) is False
    assert _ready({"gateway_ok_at": None}) is False
    assert _ready({"gateway_ok_at": "nope"}) is False
    assert _ready({"gateway_ok_at": iso(NOW - 599)}) is True
    assert _ready({"gateway_ok_at": iso(NOW - 600)}) is False
