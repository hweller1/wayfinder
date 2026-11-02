"""fs_local adapter (task 2.14): filesystem + git behaviour on real temporary repos.

AC-3.2 (worktree state dir), FR-P1, FR-C8. Covers: the same core local tools produce identical results over
`FakeFileSystem` and `LocalFileSystem` (so the fake cannot drift), root validation (the guard's in-memory skip
branch is never taken), symlink confinement, git ids (detached, shallow, remote, empty repo), NOT_A_REPO,
worktrees, and hostile `.git/config` (fsmonitor/pager) never executing.
"""

import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fs_local import LocalFileSystem, git
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ReadRangeRequest
from scout.core.local.glob import glob
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import MAX_FILE_BYTES, Matcher
from scout.core.local.readfile import read_range
from scout.core.ports.fs import FileSystem

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
    env = {**os.environ, **_HERMETIC}
    p = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.strip()


def init_repo(path: Path, files: dict[str, str | bytes] | None = None, *, commit: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    sh(path, "init", "-q", "-b", "main")
    for rel, data in (files or {"README.md": "hi\n"}).items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data.encode() if isinstance(data, str) else data)
    if commit:
        sh(path, "add", "-A")
        sh(path, "commit", "-q", "-m", "init")
    return path


def _norm(p: str | Path) -> str:
    return os.path.realpath(p)


# --- same behaviour as the fake ---------------------------------------------------------------------
FILES: dict[str, str | bytes] = {
    "src/a.py": "one\ntwo\nthree\n",
    "src/sub/c.py": "x = 1\n",
    "README.md": "hi\n",
    "crlf.txt": "a\r\nb\r\n",
    "long.txt": "".join(f"line{i}\n" for i in range(1, 1001)),
    ".env": "KEY=1\n",
    "certs/server.pem": "pem\n",
    ".gitignore": "build/\n*.log\n",
    ".scoutignore": "vendor/\ngenerated_*.py\n",
    "build/out.js": "x",
    "app.log": "x",
    "vendor/lib.py": "x",
    "generated_api.py": "x",
    "big.txt": b"a" * (MAX_FILE_BYTES + 1),
    "edge.txt": b"a" * MAX_FILE_BYTES,
    "bin.dat": b"abc\x00def",
}
SYMLINKS = {
    "link_in": "src/a.py",
    "link_secret": ".env",
    "link_out": "../outside/secret.txt",
    "dir_out": "../outside",
}


def make_pair(tmp_path: Path) -> tuple[FakeFileSystem, LocalFileSystem]:
    """The same tree as an in-memory fake and as a real git repo (links and ignored files included)."""
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("top secret\n")
    root = init_repo(tmp_path / "root", FILES, commit=False)
    for name, target in SYMLINKS.items():
        os.symlink(target, root / name)
    # The fake gets the real root path: with a non-directory root the guard skips realpath containment
    # (in-memory mode), which would make link handling differ for reasons unrelated to the adapters.
    fake = FakeFileSystem(FILES, root=_norm(root), symlinks=SYMLINKS)
    return fake, LocalFileSystem(str(root))


@pytest.fixture
def pair(tmp_path: Path) -> tuple[FakeFileSystem, LocalFileSystem]:
    return make_pair(tmp_path)


def outcome(fn: Callable[[FileSystem], Any], fs: FileSystem) -> Any:
    try:
        return ("ok", fn(fs))
    except ScoutError as e:
        return ("err", e.code)


def same(pair: tuple[FakeFileSystem, LocalFileSystem], fn: Callable[[FileSystem], Any]) -> Any:
    a, b = outcome(fn, pair[0]), outcome(fn, pair[1])
    assert a == b
    return a


@pytest.mark.parametrize(
    "path",
    [
        "src/a.py",
        "src/./a.py",
        "README.md",
        "crlf.txt",
        "nope.py",
        "src",
        ".",
        "../x",
        "/etc/passwd",
        "a\x00b",
        ".env",
        "certs/server.pem",
        ".gitignore",
        "build/out.js",
        "app.log",
        "vendor/lib.py",
        "generated_api.py",
        "big.txt",
        "edge.txt",
        "bin.dat",
        "link_in",
        "link_secret",
        "link_out",
        "dir_out",
        "dir_out/secret.txt",
        ".git/config",
    ],
)
def test_read_range_same_as_fake(pair: tuple[FakeFileSystem, LocalFileSystem], path: str) -> None:
    """AC-2.3, AC-2.4, AC-13.2: read_file refuses/serves identically on the fake and the real adapter."""
    kind, val = same(pair, lambda fs: read_range(fs, ReadRangeRequest(path=path)))
    if path in ("link_out", "dir_out", "dir_out/secret.txt", "../x", ".env", "link_secret", "bin.dat"):
        assert (kind, val) == ("err", ErrorCode.PATH_REFUSED)


def test_read_range_ranges_and_caps_same_as_fake(pair: tuple[FakeFileSystem, LocalFileSystem]) -> None:
    for req in (
        ReadRangeRequest(path="src/a.py", start_line=2, end_line=3),
        ReadRangeRequest(path="long.txt", start_line=10, end_line=900),
        ReadRangeRequest(path="long.txt"),
        ReadRangeRequest(path="src/a.py", start_line=50),
        ReadRangeRequest(path="src/a.py", start_line=5, end_line=2),
    ):
        same(pair, lambda fs, r=req: read_range(fs, r))
    assert same(pair, lambda fs: read_range(fs, ReadRangeRequest(path="crlf.txt")))[1].lines == ["a\r", "b\r"]


@pytest.mark.parametrize(
    "pat", ["**/*", "**/*.py", "*.py", "src/*.py", "src/**/*.py", "*.txt", "*", "*.md", "../*", "/etc/*"]
)
def test_glob_same_as_fake(pair: tuple[FakeFileSystem, LocalFileSystem], pat: str) -> None:
    """AC-2.4, AC-2.5: glob lists the same files (ignored, denylisted, oversize, binary, escaping hidden)."""
    kind, val = same(pair, lambda fs: glob(fs, pat))
    if pat == "**/*":
        assert kind == "ok"
        listed = set(val.paths)
        assert {"src/a.py", "src/sub/c.py", "link_in"} <= listed
        assert not listed & {
            ".env",
            "link_secret",
            "link_out",
            "dir_out",
            "build/out.js",
            "big.txt",
            "bin.dat",
        }


def test_matcher_exclusions_same_as_fake(pair: tuple[FakeFileSystem, LocalFileSystem]) -> None:
    for p in FILES:
        same(pair, lambda fs, p=p: Matcher.from_fs(fs).is_excluded(p))


def test_stat_and_read_text_same_as_fake(pair: tuple[FakeFileSystem, LocalFileSystem]) -> None:
    fake, local = pair
    for p in ("src/a.py", "README.md", "big.txt", "link_in", "nope.py", "link_out", "../x", "src"):
        a, b = outcome(lambda fs, p=p: fs.stat(p), fake), outcome(lambda fs, p=p: fs.stat(p), local)
        assert a[0] == b[0], p
        if a[0] == "err":
            assert a == b, p
        else:
            assert (a[1].size, a[1].is_symlink) == (b[1].size, b[1].is_symlink), p
            assert b[1].mtime_ns > 0
            assert b[1].ino > 0
    for p in (".gitignore", ".scoutignore", "nope.py", "src/a.py"):
        same(pair, lambda fs, p=p: fs.read_text(p))


def test_read_bytes_max_bytes_and_missing(pair: tuple[FakeFileSystem, LocalFileSystem]) -> None:
    for fs in pair:
        assert fs.read_bytes("src/a.py", max_bytes=3) == b"one"
        assert fs.read_bytes("src/a.py") == b"one\ntwo\nthree\n"
        with pytest.raises(ScoutError) as ei:
            fs.read_bytes("nope.py")
        assert ei.value.code is ErrorCode.NOT_FOUND
    assert len(pair[1].read_bytes("big.txt", max_bytes=10)) == 10


def test_list_files_is_git_ls_files_co(tmp_path: Path) -> None:
    """FR-C8: tracked + untracked-not-ignored; ignored and deleted-but-tracked files are absent."""
    root = init_repo(tmp_path / "r", {"a.py": "1", "b.py": "2", ".gitignore": "ig.txt\n"})
    (root / "untracked.py").write_text("3")
    (root / "ig.txt").write_text("4")
    (root / "b.py").unlink()
    (root / "sp ace.py").write_text("5")
    (root / "ünï.py").write_text("6")
    assert LocalFileSystem(str(root)).list_files() == sorted(
        [".gitignore", "a.py", "sp ace.py", "untracked.py", "ünï.py"]
    )


def test_list_files_skips_nested_repo_entries(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r", {"a.py": "1"})
    init_repo(root / "nested", {"n.py": "1"})
    names = LocalFileSystem(str(root)).list_files()
    assert "a.py" in names
    assert "nested" not in names


# --- root validation: the guard's in-memory skip branch is never reachable ---------------------------
def test_refuses_nonexistent_or_non_directory_root(tmp_path: Path) -> None:
    """Parent note on 1.7: core.local.guard skips realpath containment for a non-dir root."""
    with pytest.raises(ScoutError) as ei:
        LocalFileSystem(str(tmp_path / "missing"))
    assert ei.value.code is ErrorCode.NOT_FOUND
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ScoutError) as ei:
        LocalFileSystem(str(f))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError):
        LocalFileSystem(str(tmp_path / "nolink" / "x"))


def test_resolve_in_root_never_takes_skip_branch_on_local_fs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on the guard's `os.path.isdir(fs.root)`: every call made on behalf of fs_local sees a real dir,
    and a symlink escape is refused by realpath containment (not by lexical checks)."""
    root = init_repo(tmp_path / "r", {"a.py": "1\n"})
    (tmp_path / "out.txt").write_text("secret")
    os.symlink("../out.txt", root / "esc")
    fs = LocalFileSystem(str(root))
    seen: list[tuple[str, bool]] = []
    real_isdir = os.path.isdir

    def spy(p: Any) -> bool:
        r = real_isdir(p)
        if str(p) == fs.root:
            seen.append((str(p), r))
        return r

    monkeypatch.setattr(os.path, "isdir", spy)
    assert resolve_in_root(fs, "a.py") == "a.py"
    assert fs.read_bytes("a.py") == b"1\n"
    assert fs.stat("a.py").size == 2
    with pytest.raises(ScoutError) as ei:
        fs.read_bytes("esc")
    assert ei.value.code is ErrorCode.PATH_REFUSED
    assert seen
    assert all(ok for _, ok in seen)


def test_root_removed_after_construction_is_not_served_via_skip_branch(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r", {"a.py": "1\n"})
    fs = LocalFileSystem(str(root))
    shutil.rmtree(root)
    for call in (lambda: fs.read_bytes("a.py"), lambda: fs.stat("a.py")):
        with pytest.raises(ScoutError) as ei:
            call()
        assert ei.value.code is ErrorCode.NOT_FOUND
    assert fs.read_text("a.py") is None


def test_symlinked_root_is_resolved(tmp_path: Path) -> None:
    real = init_repo(tmp_path / "real", {"a.py": "1\n"})
    os.symlink(real, tmp_path / "alias")
    fs = LocalFileSystem(str(tmp_path / "alias"))
    assert fs.root == _norm(real)
    assert fs.read_bytes("a.py") == b"1\n"


# --- symlinks and special files ---------------------------------------------------------------------
def test_symlink_escape_refused_for_read_stat_and_text(tmp_path: Path) -> None:
    """AC-2.3: a link leaving the root is refused by the adapter itself (no tool layer involved)."""
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("top secret\n")
    root = init_repo(tmp_path / "root", {"ok.py": "ok\n"})
    os.symlink(tmp_path / "outside" / "secret.txt", root / "abs_link")
    os.symlink("../outside/secret.txt", root / "rel_link")
    os.symlink(tmp_path / "outside", root / "dir_link")
    os.symlink("rel_link", root / "chain")
    os.symlink("loop", root / "loop")
    os.symlink("ok.py", root / "inside")
    fs = LocalFileSystem(str(root))
    for bad in ("abs_link", "rel_link", "dir_link", "dir_link/secret.txt", "chain"):
        for call in (fs.read_bytes, fs.stat, fs.read_text):
            with pytest.raises(ScoutError) as ei:
                call(bad)
            assert ei.value.code is ErrorCode.PATH_REFUSED, (bad, call)
    assert fs.read_bytes("inside") == b"ok\n"
    assert fs.stat("inside").is_symlink
    assert not fs.stat("ok.py").is_symlink
    with pytest.raises(ScoutError) as ei:  # a loop never resolves to a readable file
        fs.read_bytes("loop")
    assert ei.value.code in (ErrorCode.PATH_REFUSED, ErrorCode.NOT_FOUND)


def test_fifo_is_refused_not_blocked(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r", {"a.py": "1\n"})
    os.mkfifo(root / "pipe")
    assert stat.S_ISFIFO(os.lstat(root / "pipe").st_mode)
    with pytest.raises(ScoutError) as ei:
        LocalFileSystem(str(root)).read_bytes("pipe")
    assert ei.value.code is ErrorCode.PATH_REFUSED


# --- git ids ----------------------------------------------------------------------------------------
def test_git_ids_basic(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r")
    sh(root, "remote", "add", "origin", "git@GitHub.com:Acme/Repo.git")
    ids = LocalFileSystem(str(root)).git_ids()
    head = sh(root, "rev-parse", "HEAD")
    assert ids.head == head
    assert ids.root_commit == head
    assert ids.branch == "main"
    assert ids.default_branch == "main"
    assert ids.remote_norm == "github.com/acme/repo"
    assert not ids.detached
    assert not ids.shallow


def test_git_ids_root_commit_is_first_commit(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r")
    first = sh(root, "rev-parse", "HEAD")
    (root / "b.txt").write_text("b")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "second")
    ids = LocalFileSystem(str(root)).git_ids()
    assert ids.root_commit == first
    assert ids.head == sh(root, "rev-parse", "HEAD") != first


def test_git_ids_detached_head(tmp_path: Path) -> None:
    """Design section 15: detached HEAD -> branch `detached-<commit12>`."""
    root = init_repo(tmp_path / "r")
    head = sh(root, "rev-parse", "HEAD")
    sh(root, "checkout", "-q", "--detach")
    ids = LocalFileSystem(str(root)).git_ids()
    assert ids.detached
    assert ids.branch == f"detached-{head[:12]}"
    assert ids.head == head


def test_git_ids_branch_switch(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r")
    sh(root, "checkout", "-q", "-b", "feat/x")
    ids = LocalFileSystem(str(root)).git_ids()
    assert (ids.branch, ids.detached, ids.default_branch) == ("feat/x", False, "main")


def test_git_ids_shallow_clone_flag(tmp_path: Path) -> None:
    """Design section 15: a shallow clone is flagged so the caller can warn."""
    origin = init_repo(tmp_path / "origin")
    for i in range(3):
        (origin / f"f{i}.txt").write_text(str(i))
        sh(origin, "add", "-A")
        sh(origin, "commit", "-q", "-m", f"c{i}")
    clone = tmp_path / "clone"
    sh(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(clone))
    ids = LocalFileSystem(str(clone)).git_ids()
    assert ids.shallow
    assert ids.root_commit == ids.head  # the visible history is one commit
    full = tmp_path / "full"
    sh(tmp_path, "clone", "-q", origin.as_uri(), str(full))
    full_ids = LocalFileSystem(str(full)).git_ids()
    assert not full_ids.shallow
    assert full_ids.root_commit == sh(origin, "rev-list", "--max-parents=0", "HEAD")
    assert full_ids.remote_norm is None  # file:// remotes carry no shared identity
    assert full_ids.default_branch == "main"


def test_git_ids_empty_repo_has_no_head(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r", commit=False)
    ids = LocalFileSystem(str(root)).git_ids()
    assert ids.head is None
    assert ids.root_commit is None
    assert ids.branch == "main"
    assert ids.remote_norm is None


@pytest.mark.parametrize(
    ("url", "want"),
    [
        ("git@github.com:Acme/Repo.git", "github.com/acme/repo"),
        ("https://github.com/acme/repo.git", "github.com/acme/repo"),
        ("https://user:tok3n@github.com/acme/repo", "github.com/acme/repo"),
        ("ssh://git@host.example:2222/team/sub/repo.git/", "host.example/team/sub/repo"),
        ("HTTPS://GitHub.com/Acme/Repo/", "github.com/acme/repo"),
        ("git://example.org/x/y.git", "example.org/x/y"),
        ("file:///tmp/repo", None),
        ("/srv/repo", None),
        ("../repo", None),
        ("", None),
        ("https://github.com/", None),
    ],
)
def test_normalize_remote(url: str, want: str | None) -> None:
    got = git.normalize_remote(url)
    assert got == want
    assert got is None or "tok3n" not in got


# --- state dir and NOT_A_REPO -----------------------------------------------------------------------
def test_state_dir_plain_repo(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r")
    assert _norm(LocalFileSystem(str(root)).state_dir()) == _norm(root / ".git" / "scout")


def test_state_dir_from_subdirectory_root(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "r", {"pkg/m.py": "1\n"})
    fs = LocalFileSystem(str(root / "pkg"))
    assert _norm(fs.state_dir()) == _norm(root / ".git" / "scout")
    assert fs.list_files() == ["m.py"]


def test_state_dir_in_linked_worktree(tmp_path: Path) -> None:
    """AC-3.2: in a linked worktree `.git` is a file; state lives under the worktree's own git dir."""
    main = init_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    sh(main, "worktree", "add", "-q", "-b", "wtbranch", str(wt))
    assert (wt / ".git").is_file()
    fs = LocalFileSystem(str(wt))
    state = _norm(fs.state_dir())
    gitdir = _norm(sh(wt, "rev-parse", "--absolute-git-dir"))
    assert state == _norm(os.path.join(gitdir, "scout"))
    assert state == _norm(main / ".git" / "worktrees" / "wt" / "scout")
    assert state != _norm(wt / ".git" / "scout")
    assert state != _norm(main / ".git" / "scout")
    assert _norm(LocalFileSystem(str(main)).state_dir()) == _norm(main / ".git" / "scout")
    assert fs.git_ids().branch == "wtbranch"
    assert fs.list_files() == ["README.md"]


def test_not_a_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("x")
    fs = LocalFileSystem(str(plain))
    for call in (fs.state_dir, fs.git_ids, fs.list_files):
        with pytest.raises(ScoutError) as ei:
            call()
        assert ei.value.code is ErrorCode.NOT_A_REPO, call
    assert fs.read_bytes("a.txt") == b"x"  # plain file access does not need git
    with pytest.raises(ScoutError) as ei:
        FakeFileSystem(git=None).state_dir()
    assert ei.value.code is ErrorCode.NOT_A_REPO


# --- hostile .git/config ----------------------------------------------------------------------------
def _hostile_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Repo whose config would run commands that touch marker files if git ever honoured them."""
    root = init_repo(tmp_path / "r", {"a.py": "1\n"})
    markers = tmp_path / "markers"
    markers.mkdir()
    hook = tmp_path / "hook.sh"
    hook.write_text(f'#!/bin/sh\ntouch "{markers}/fsmonitor"\nexit 0\n')
    hook.chmod(0o755)
    pager = tmp_path / "pager.sh"
    pager.write_text(f'#!/bin/sh\ntouch "{markers}/pager"\ncat\n')
    pager.chmod(0o755)
    ext = tmp_path / "ext.sh"
    ext.write_text(f'#!/bin/sh\ntouch "{markers}/diff_external"\n')
    ext.chmod(0o755)
    cfg = {
        "core.fsmonitor": str(hook),
        "core.pager": str(pager),
        "pager.status": str(pager),
        "diff.external": str(ext),
        "core.sshCommand": f'touch "{markers}/ssh"; false',
        "core.editor": f'touch "{markers}/editor"',
        "core.hooksPath": str(tmp_path / "hooks"),
    }
    for k, v in cfg.items():
        sh(root, "config", k, v)
    sh(root, "config", "remote.origin.url", "https://example.com/acme/repo.git")
    (tmp_path / "hooks").mkdir()
    for name in ("post-checkout", "post-merge", "pre-commit", "reference-transaction"):
        h = tmp_path / "hooks" / name
        h.write_text(f'#!/bin/sh\ntouch "{markers}/hook_{name}"\n')
        h.chmod(0o755)
    return root, markers


def test_hostile_git_config_is_never_executed(tmp_path: Path) -> None:
    """Security: core.fsmonitor / core.pager / diff.external / hooks in `.git/config` must not run."""
    root, markers = _hostile_repo(tmp_path)
    (root / "untracked.py").write_text("2\n")

    # Control: plain `git status` DOES run the hostile fsmonitor here, so a silent marker below means
    # the adapter neutralised it (and this test cannot pass vacuously on a git that ignores the key).
    sh(root, "status", "--porcelain")
    assert (markers / "fsmonitor").exists(), "premise broken: hostile fsmonitor did not fire under raw git"
    (markers / "fsmonitor").unlink()
    assert not list(markers.iterdir())

    fs = LocalFileSystem(str(root))
    assert fs.list_files() == ["a.py", "untracked.py"]
    ids = fs.git_ids()
    assert ids.remote_norm == "example.com/acme/repo"
    assert fs.state_dir().endswith("scout")
    assert fs.read_bytes("a.py") == b"1\n"
    assert fs.read_text(".gitignore") is None
    assert sorted(p.name for p in markers.iterdir()) == [], "a hostile config command was executed"


def test_hostile_config_in_linked_worktree_is_never_executed(tmp_path: Path) -> None:
    root, markers = _hostile_repo(tmp_path)
    wt = tmp_path / "wt"
    # `git worktree add` itself may fire hooks; clear markers afterwards and test the adapter only
    sh(root, "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "worktree", "add", "-q", str(wt))
    for m in markers.iterdir():
        m.unlink()
    fs = LocalFileSystem(str(wt))
    fs.list_files()
    fs.git_ids()
    fs.state_dir()
    assert sorted(p.name for p in markers.iterdir()) == []


def test_git_call_shape_list_argv_fixed_config_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every git call: argument list, no shell, fixed -c config, timeout, no inherited GIT_* env."""
    root = init_repo(tmp_path / "r")
    head = sh(root, "rev-parse", "HEAD")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    real_run = subprocess.run

    def spy(argv: Any, **kw: Any) -> Any:
        calls.append((list(argv), kw))
        return real_run(argv, **kw)

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/bin/false")
    monkeypatch.setattr(subprocess, "run", spy)
    fs = LocalFileSystem(str(root))
    fs.list_files()
    fs.git_ids()
    fs.state_dir()
    assert len(calls) >= 6
    for argv, kw in calls:
        assert isinstance(argv, list)
        assert not kw.get("shell")
        assert kw["timeout"] > 0
        assert "GIT_DIR" not in kw["env"]
        assert "GIT_EXTERNAL_DIFF" not in kw["env"]
        assert kw["stdin"] is subprocess.DEVNULL
        assert "--no-pager" in argv
        for want in (
            "core.fsmonitor=false",
            "core.pager=cat",
            "diff.external=",
            f"core.hooksPath={os.devnull}",
        ):
            i = argv.index(want)
            assert argv[i - 1] == "-c"
    # inherited GIT_DIR did not redirect git away from `root`
    assert fs.git_ids().head == head


def test_git_timeout_is_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = init_repo(tmp_path / "r")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    slow = bindir / "git"
    slow.write_text("#!/bin/sh\nexec /bin/sleep 5\n")
    slow.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    with pytest.raises(ScoutError) as ei:
        git.run_git(str(root), ["rev-parse", "HEAD"], timeout=0.3)
    assert ei.value.code is ErrorCode.INTERNAL
    assert "timed out" in ei.value.message


def test_missing_git_binary_is_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = init_repo(tmp_path / "r")
    fs = LocalFileSystem(str(root))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with pytest.raises(ScoutError) as ei:
        fs.list_files()
    assert ei.value.code is ErrorCode.INTERNAL
