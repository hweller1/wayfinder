"""Local tools (task 1.7): ignore matcher, path guard, read_file, glob.

AC-2.3 (refusals), AC-2.4 (caps, 1-based inclusive), AC-2.5 (one matcher), AC-13.2 (denylist/size/binary).
"""

import os
from pathlib import Path

import pytest

from scout.adapters.fake_fs import FakeFileSystem
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ReadRangeRequest
from scout.core.local.glob import MAX_PATHS, glob
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import MAX_FILE_BYTES, Matcher
from scout.core.local.readfile import MAX_LINES, read_range

SECRETS = [
    ".env",
    ".env.local",
    "config/.env.production",
    "certs/server.pem",
    "home/id_rsa",
    "home/id_ed25519",
    "keys/signing.key",
    "aws/credentials",
    "aws/credentials.json",
    "CERTS/SERVER.PEM",
]


def _fs(extra: dict[str, str | bytes] | None = None, **kw: object) -> FakeFileSystem:
    files: dict[str, str | bytes] = {"src/a.py": "one\ntwo\nthree\n", "README.md": "hi\n", **(extra or {})}
    return FakeFileSystem(files, **kw)  # type: ignore[arg-type]


def _read(fs: FakeFileSystem, path: str, **kw: int | None) -> object:
    return read_range(fs, ReadRangeRequest(path=path, **kw))  # type: ignore[arg-type]


def _refused(fs: FakeFileSystem, path: str) -> None:
    with pytest.raises(ScoutError) as ei:
        _read(fs, path)
    assert ei.value.code is ErrorCode.PATH_REFUSED, path


# --- AC-2.3: path guard ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        "../x",
        "../../etc/passwd",
        "src/../../x",
        "src/../..",
        "/etc/passwd",
        "/repo/src/a.py",
        "~/x",
        "src\\..\\x",
        "a\x00b",
        "..",
    ],
)
def test_guard_refuses_traversal_and_absolute(bad: str) -> None:
    """AC-2.3: `..`, absolute, home, backslash and NUL paths raise PATH_REFUSED."""
    fs = _fs()
    with pytest.raises(ScoutError) as ei:
        resolve_in_root(fs, bad)
    assert ei.value.code is ErrorCode.PATH_REFUSED
    _refused(fs, bad)


def test_guard_allows_plain_and_dot_segments() -> None:
    assert resolve_in_root(_fs(), "src/./a.py") == "src/a.py"
    assert resolve_in_root(_fs(), "src//a.py") == "src/a.py"


def test_guard_symlink_escape_on_real_tmp_dir(tmp_path: Path) -> None:
    """AC-2.3: a real symlink resolving outside the root is refused, even for an adapter that follows it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret\n")
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src" / "ok.py").write_text("ok\n")
    os.symlink(outside / "secret.txt", root / "link_abs")  # absolute target outside
    os.symlink("../outside/secret.txt", root / "link_rel")  # relative target outside
    os.symlink(outside, root / "dir_link")  # directory link outside
    os.symlink("src/ok.py", root / "link_in")  # harmless link inside
    fs = FakeFileSystem({"src/ok.py": "ok\n", "dir_link/secret.txt": "top secret\n"}, root=str(root))
    for bad in ("link_abs", "link_rel", "dir_link", "dir_link/secret.txt"):
        with pytest.raises(ScoutError) as ei:
            resolve_in_root(fs, bad)
        assert ei.value.code is ErrorCode.PATH_REFUSED, bad
        _refused(fs, bad)
    assert resolve_in_root(fs, "link_in") == "src/ok.py"
    assert resolve_in_root(fs, "src/ok.py") == "src/ok.py"


def test_symlink_to_secret_inside_root_is_refused(tmp_path: Path) -> None:
    """AC-2.3: an in-root symlink named innocently but pointing at `.env` is refused (target is judged)."""
    root = tmp_path / "root"
    root.mkdir()
    (root / ".env").write_text("KEY=1\n")
    os.symlink(".env", root / "notes.txt")
    fs = FakeFileSystem({".env": "KEY=1\n", "notes.txt": "x"}, root=str(root), symlinks={"notes.txt": ".env"})
    _refused(fs, "notes.txt")


def test_fake_fs_symlink_escape() -> None:
    """AC-2.3: the in-memory adapter's own confinement also refuses an escaping link."""
    fs = _fs(symlinks={"link": "../outside", "abs": "/etc/passwd"})
    _refused(fs, "link")
    _refused(fs, "abs")


# --- AC-2.3 / AC-13.2: ignored and denylisted files ------------------------------------------------
@pytest.mark.parametrize("secret", SECRETS)
def test_denylist_refused_by_read_and_hidden_from_glob(secret: str) -> None:
    """AC-2.3, AC-13.2: denylist (.env*, *.pem, id_*, *.key, credentials*) never read or listed."""
    fs = _fs({secret: "TOP SECRET\n"})
    assert Matcher.from_fs(fs).is_excluded(secret)
    _refused(fs, secret)
    assert secret not in glob(fs, "**/*").paths
    assert fs.read_bytes(secret)  # the fs itself would serve it: the tool layer is the barrier


def test_denylist_cannot_be_unignored() -> None:
    """AC-13.2: a `!.env` negation in .scoutignore or .gitignore does not re-include a secret."""
    fs = _fs(
        {
            ".env": "K=1\n",
            ".scoutignore": "!.env\n!*.pem\n",
            ".gitignore": "!id_*\n",
            "id_rsa": "k",
            "a.pem": "k",
        }
    )
    for p in (".env", "id_rsa", "a.pem"):
        _refused(fs, p)


def test_gitignore_and_scoutignore_refused_and_hidden() -> None:
    """AC-2.3, AC-2.5: entries from both ignore files are refused by read and absent from glob."""
    fs = _fs(
        {
            ".gitignore": "build/\n*.log\n",
            ".scoutignore": "vendor/\ngenerated_*.py\n",
            "build/out.js": "x",
            "app.log": "x",
            "vendor/lib.py": "x",
            "generated_api.py": "x",
        }
    )
    for p in ("build/out.js", "app.log", "vendor/lib.py", "generated_api.py"):
        _refused(fs, p)
    listed = glob(fs, "**/*").paths
    assert listed == [".gitignore", ".scoutignore", "README.md", "src/a.py"]


def test_git_dir_is_refused() -> None:
    """AC-2.3: `.git/config` (may embed credentialed remote URLs) is refused."""
    _refused(_fs({".git/config": "[remote]\nurl=https://u:p@h/x\n"}), ".git/config")


def test_oversize_and_binary_excluded() -> None:
    """AC-13.2: files > 1 MiB and NUL-containing binaries are excluded; exactly 1 MiB is allowed."""
    fs = _fs(
        {
            "big.txt": b"a" * (MAX_FILE_BYTES + 1),
            "edge.txt": b"a" * MAX_FILE_BYTES,
            "bin.dat": b"abc\x00def",
        }
    )
    m = Matcher.from_fs(fs)
    assert m.is_excluded("big.txt")
    assert m.is_excluded("bin.dat")
    assert not m.is_excluded("edge.txt")
    _refused(fs, "big.txt")
    _refused(fs, "bin.dat")
    assert "edge.txt" in glob(fs, "*.txt").paths
    assert "big.txt" not in glob(fs, "*").paths
    assert "bin.dat" not in glob(fs, "*").paths


def test_one_matcher_shared_by_read_glob_and_sync() -> None:
    """AC-2.5: read_range, glob and an explicit Matcher agree on the same file set."""
    fs = _fs({".scoutignore": "secret_dir/\n", "secret_dir/x.py": "x", "pub/y.py": "y"})
    m = Matcher.from_fs(fs)
    for p in fs.list_files():
        refused = False
        try:
            _read(fs, p)
        except ScoutError:
            refused = True
        assert refused == m.is_excluded(p), p
        assert (p in glob(fs, "**/*", m).paths) == (not m.is_excluded(p)), p


# --- AC-2.4: read_file ------------------------------------------------------------------------------
def test_read_range_one_based_inclusive() -> None:
    """AC-2.4: lines are 1-based, end inclusive, `\\n`-split; a trailing newline adds no empty line."""
    fs = _fs({"f.txt": "l1\nl2\nl3\nl4\n"})
    r = read_range(fs, ReadRangeRequest(path="f.txt", start_line=2, end_line=3))
    assert (r.start_line, r.end_line, r.lines, r.truncated) == (2, 3, ["l2", "l3"], False)
    whole = read_range(fs, ReadRangeRequest(path="f.txt"))
    assert (whole.start_line, whole.end_line, whole.lines) == (1, 4, ["l1", "l2", "l3", "l4"])
    clamp = read_range(fs, ReadRangeRequest(path="f.txt", start_line=4, end_line=99))
    assert (clamp.end_line, clamp.lines, clamp.truncated) == (4, ["l4"], False)


def test_read_range_crlf_is_not_a_line_break() -> None:
    """AC-2.4: only `\\n` delimits lines; `\\r` stays in the line text."""
    r = read_range(_fs({"w.txt": "a\r\nb\r\n"}), ReadRangeRequest(path="w.txt"))
    assert r.lines == ["a\r", "b\r"]


def test_read_range_caps_at_400_lines() -> None:
    """AC-2.4: a range over 400 lines returns exactly 400, flagged truncated."""
    fs = _fs({"long.txt": "".join(f"line{i}\n" for i in range(1, 1001))})
    r = read_range(fs, ReadRangeRequest(path="long.txt", start_line=10, end_line=900))
    assert len(r.lines) == MAX_LINES == 400
    assert (r.start_line, r.end_line, r.truncated) == (10, 409, True)
    assert r.lines[0] == "line10"
    assert r.lines[-1] == "line409"
    open_ended = read_range(fs, ReadRangeRequest(path="long.txt"))
    assert (len(open_ended.lines), open_ended.end_line, open_ended.truncated) == (400, 400, True)
    exact = read_range(fs, ReadRangeRequest(path="long.txt", start_line=1, end_line=400))
    assert (len(exact.lines), exact.truncated) == (400, False)


def test_read_range_errors() -> None:
    fs = _fs()
    with pytest.raises(ScoutError) as ei:
        read_range(fs, ReadRangeRequest(path="src/a.py", start_line=5, end_line=2))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei:
        read_range(fs, ReadRangeRequest(path="src/a.py", start_line=50))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei:
        read_range(fs, ReadRangeRequest(path="nope.py"))
    assert ei.value.code is ErrorCode.NOT_FOUND
    _refused(fs, ".")  # a directory/root is not a readable file


# --- AC-2.4: glob -----------------------------------------------------------------------------------
def test_glob_semantics_sorted() -> None:
    fs = _fs({"src/b.py": "x", "src/sub/c.py": "x", "top.py": "x"})
    assert glob(fs, "**/*.py").paths == ["src/a.py", "src/b.py", "src/sub/c.py", "top.py"]
    assert glob(fs, "*.py").paths == ["top.py"]
    assert glob(fs, "src/*.py").paths == ["src/a.py", "src/b.py"]
    assert glob(fs, "src/**/*.py").paths == ["src/a.py", "src/b.py", "src/sub/c.py"]
    assert glob(fs, "src/?.py").paths == ["src/a.py", "src/b.py"]


def test_glob_caps_at_1000() -> None:
    """AC-2.4: glob returns <= 1,000 paths and says so when it truncated."""
    fs = FakeFileSystem({f"d/f{i:04d}.txt": "x" for i in range(MAX_PATHS + 50)})
    r = glob(fs, "**/*.txt")
    assert len(r.paths) == MAX_PATHS == 1000
    assert r.truncated
    assert r.paths == sorted(r.paths)
    exact = glob(FakeFileSystem({f"f{i:04d}.txt": "x" for i in range(MAX_PATHS)}), "*.txt")
    assert len(exact.paths) == 1000
    assert not exact.truncated


@pytest.mark.parametrize("bad", ["../*", "/etc/*", "src/../../*", "a\\b", ""])
def test_glob_refuses_escaping_patterns(bad: str) -> None:
    """AC-2.3: glob patterns cannot leave the root either."""
    with pytest.raises(ScoutError) as ei:
        glob(_fs(), bad)
    assert ei.value.code in (ErrorCode.PATH_REFUSED, ErrorCode.INVALID_ARGUMENT)
    if bad:
        assert ei.value.code is ErrorCode.PATH_REFUSED


def test_glob_hides_escaping_symlinks() -> None:
    """AC-2.3: a symlink resolving outside the root is not listed by glob."""
    fs = _fs(symlinks={"link": "../outside"})
    assert "link" not in glob(fs, "*").paths
