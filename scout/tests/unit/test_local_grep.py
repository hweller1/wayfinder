"""Local grep (task 1.8): AC-2.1 (pattern never an arg), AC-2.2 (caps, density order, early stop).

Also: never returns a line from an ignored/denylisted/oversize/binary/out-of-root file, deterministic ranking,
and early stop on pathological match volume.
"""

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fs_local import LocalFileSystem, rg
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import GrepRequest
from scout.core.local.grep import MAX_COLUMNS, MAX_FILES, MAX_MATCHES_PER_FILE, grep
from scout.core.local.ignore import MAX_FILE_BYTES
from scout.core.ports.fs import FileSystem, RgEvent, RgSpec

MakeFs = Callable[..., FileSystem]

# Mirrors the symlinked tree of test_fs_local_git: a link to a secret, to a file and a dir outside the root.
LINKS = {
    "link_in": "ok.py",
    "link_secret": ".env",
    "link_out": "../outside/secret.txt",
    "dir_out": "../outside",
}


@pytest.fixture(params=["fake", "local"])
def make_fs(request: pytest.FixtureRequest, tmp_path: Path) -> MakeFs:
    """Build the same tree as an in-memory fake or as a real git repo behind `LocalFileSystem` (real rg),
    so the fake's pure-python matcher cannot drift from ripgrep's behaviour."""

    def build(files: Mapping[str, str | bytes], symlinks: Mapping[str, str] | None = None) -> FileSystem:
        if request.param == "fake":
            return FakeFileSystem(files, symlinks=symlinks)
        return _local_repo(tmp_path, files, symlinks)

    return build


def _local_repo(
    tmp_path: Path, files: Mapping[str, str | bytes], symlinks: Mapping[str, str] | None = None
) -> LocalFileSystem:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside").mkdir(exist_ok=True)
    (tmp_path / "outside" / "secret.txt").write_text("needle\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607
    for rel, data in files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data.encode() if isinstance(data, str) else data)
    for rel, target in (symlinks or {}).items():
        os.symlink(target, repo / rel)
    return LocalFileSystem(str(repo))


def _lines(n: int, word: str = "needle") -> str:
    return "".join(f"{word} {i}\n" for i in range(n))


def _grep(fs: FileSystem, pattern: str = "needle", **kw: object):
    return grep(fs, GrepRequest(pattern=pattern, **kw))  # type: ignore[arg-type]


def test_ac_2_1_dash_pattern_is_a_pattern_not_an_argument() -> None:
    """AC-2.1: `--pre=/bin/sh` reaches the spec as the pattern; nothing else carries it."""
    fs = FakeFileSystem({"a.txt": "run --pre=/bin/sh now\n", "b.txt": "nothing\n"})
    res = _grep(fs, "--pre=/bin/sh", regex=False)
    assert [s.pattern for s in fs.grep_specs] == ["--pre=/bin/sh"]
    assert fs.grep_specs[0].regex is False
    assert fs.grep_specs[0].target == "."
    assert [f.path for f in res.files] == ["a.txt"]
    res = _grep(fs, "-e foo")
    assert fs.grep_specs[-1].pattern == "-e foo"
    assert res.files == []


def test_ac_2_2_300_files_capped_density_ordered_and_closed_early(make_fs: MakeFs) -> None:
    """AC-2.2: 300 matching files -> <= 200 files, <= 20 matches each, generator closed early."""
    files = {f"f{i:03d}.txt": _lines(25) for i in range(300)}
    fs = make_fs(files)
    res = _grep(fs)
    assert len(res.files) == MAX_FILES
    assert all(len(f.matches) == MAX_MATCHES_PER_FILE for f in res.files)
    assert res.truncated
    if not isinstance(fs, FakeFileSystem):
        return  # the real adapter's early stop is asserted by process in the rg tests below
    assert fs.grep_closed_early == 1
    assert fs.grep_events_yielded < 25 * 300  # stopped long before the stream ended
    assert fs.grep_events_yielded <= MAX_FILES * 25 + 1  # 200 saturated files, then the 201st match


def test_ranked_by_density_not_path_order(make_fs: MakeFs) -> None:
    fs = make_fs({"a.txt": _lines(1), "b.txt": _lines(5), "c.txt": _lines(3), "d.txt": _lines(25)})
    res = _grep(fs)
    assert [f.path for f in res.files] == ["d.txt", "b.txt", "c.txt", "a.txt"]
    assert [len(f.matches) for f in res.files] == [20, 5, 3, 1]
    assert res.truncated  # d.txt had 25 matches; 5 were dropped by the per-file cap


def test_ranking_is_deterministic_with_stable_tiebreak(make_fs: MakeFs) -> None:
    files = {f"{c}.txt": _lines(4) for c in "edcba"} | {"z.txt": _lines(9)}
    fs = make_fs(files)
    runs = [[f.path for f in _grep(fs).files] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == ["z.txt", "a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]


def test_tiebreak_is_by_path_not_adapter_stream_order() -> None:
    """Ripgrep is multi-threaded, so its file order varies run to run; equal density must sort by path."""

    class _Shuffled(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            for p in ("c.txt", "a.txt", "b.txt"):
                yield RgEvent(kind="match", path=p, line=1, text="needle")

    fs = _Shuffled({"a.txt": "needle\n", "b.txt": "needle\n", "c.txt": "needle\n"})
    assert [f.path for f in _grep(fs).files] == ["a.txt", "b.txt", "c.txt"]


def test_more_than_200_unsaturated_files_keeps_highest_density(make_fs: MakeFs) -> None:
    files = {f"f{i:03d}.txt": _lines(1 + (i % 2)) for i in range(250)}  # 125 with 1, 125 with 2
    res = _grep(make_fs(files))
    assert len(res.files) == MAX_FILES
    assert res.truncated
    assert all(len(f.matches) == 2 for f in res.files[:125])
    assert [f.path for f in res.files[:3]] == ["f001.txt", "f003.txt", "f005.txt"]


def test_line_text_truncated_to_300(make_fs: MakeFs) -> None:
    fs = make_fs({"a.txt": "needle" + "x" * 1000 + "\n"})
    res = _grep(fs)
    assert len(res.files[0].matches[0].text) <= 300 == MAX_COLUMNS
    if isinstance(fs, FakeFileSystem):
        assert fs.grep_specs[0].max_columns == MAX_COLUMNS


def test_context_lines_follow_request_and_stay_off_unmatched_files(make_fs: MakeFs) -> None:
    fs = make_fs({"a.txt": "before\nneedle\nafter\n", "b.txt": "nothing\n"})
    res = _grep(fs, context=1)
    assert [(m.line, m.text) for m in res.files[0].matches] == [(1, "before"), (2, "needle"), (3, "after")]
    assert [f.path for f in res.files] == ["a.txt"]


def test_never_returns_lines_from_ignored_denylisted_or_unreadable_files() -> None:
    """AC-2.5/AC-13.2: even if the adapter streams them, grep drops ignored/secret/binary/oversize files."""
    secrets = {
        ".env": "needle=1\n",
        ".env.local": "needle=2\n",
        "certs/a.pem": "needle\n",
        "home/id_rsa": "needle\n",
        "keys/s.key": "needle\n",
        "aws/credentials": "needle\n",
        ".git/config": "needle\n",
    }
    tree: dict[str, str | bytes] = {
        "ok.py": "needle\n",
        ".gitignore": "build/\n",
        ".scoutignore": "vendor/\n",
        "build/out.js": "needle\n",
        "vendor/lib.py": "needle\n",
        "bin.dat": b"needle\x00\x01",
        "big.txt": "needle\n" + "x" * MAX_FILE_BYTES,
        **secrets,
    }
    fs = FakeFileSystem(tree)

    class _Leaky(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:  # adapter that ignores every filter
            for path in sorted(self.files):
                for i, ln in enumerate(self.files[path].decode(errors="replace").split("\n")):
                    if "needle" in ln:
                        yield RgEvent(kind="match", path=path, line=i + 1, text=ln)

    leaky = _Leaky(dict(fs.files))
    for fs_ in (fs, leaky):
        res = _grep(fs_)
        assert [f.path for f in res.files] == ["ok.py"]


def test_real_rg_never_returns_ignored_denylisted_or_unreadable_files(make_fs: MakeFs) -> None:
    """Same tree through real ripgrep: rg's own filters plus the core matcher leave only `ok.py`."""
    fs = make_fs(
        {
            "ok.py": "needle\n",
            ".gitignore": "build/\n",
            ".scoutignore": "vendor/\n",
            "build/out.js": "needle\n",
            "vendor/lib.py": "needle\n",
            "bin.dat": b"needle\x00\x01",
            "big.txt": "needle\n" + "x" * MAX_FILE_BYTES,
            ".env": "needle=1\n",
            "certs/a.pem": "needle\n",
            "home/id_rsa": "needle\n",
        }
    )
    assert [f.path for f in _grep(fs).files] == ["ok.py"]


def test_escape_paths_streamed_by_adapter_are_dropped() -> None:
    class _Leaky(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            for p in ("../outside.txt", "/etc/passwd", "a/../../x", "a\\b", "ok.txt"):
                yield RgEvent(kind="match", path=p, line=1, text="needle")

    fs = _Leaky({".env": "needle\n", "ok.txt": "needle\n"})
    assert [f.path for f in _grep(fs).files] == ["ok.txt"]


def test_symlinks_to_secret_or_outside_root_never_match(make_fs: MakeFs) -> None:
    """A real symlink to a secret, to a file outside the root, or to an outside directory never yields lines
    (rg does not follow links; the fake models them as links too)."""
    fs = make_fs({"ok.py": "needle\n", ".env": "needle\n"}, LINKS)
    assert [f.path for f in _grep(fs).files] == ["ok.py"]
    with pytest.raises(ScoutError) as ei:
        _grep(fs, path="dir_out")
    assert ei.value.code is ErrorCode.PATH_REFUSED


def test_symlink_to_secret_or_outside_root_is_dropped_on_real_dir(tmp_path: Path) -> None:
    """A streamed path that resolves (real path) to a secret or outside the root never yields lines."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.txt").write_text("needle\n")
    (root / ".env").write_text("needle\n")
    (tmp_path / "outside.txt").write_text("needle\n")
    os.symlink(root / ".env", root / "link_env")
    os.symlink(tmp_path / "outside.txt", root / "link_out")

    class _Leaky(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            for p in ("link_env", "link_out", "ok.txt"):
                yield RgEvent(kind="match", path=p, line=1, text="needle")

    fs = _Leaky({"ok.txt": "needle\n", ".env": "needle\n"}, root=str(root))
    assert [f.path for f in _grep(fs).files] == ["ok.txt"]


def test_target_reaching_adapter_is_resolved_inside_root(tmp_path: Path) -> None:
    """An in-root symlinked directory pointing outside must be refused before rg ever runs on it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("needle\n")
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", root / "vendor")
    seen: list[str] = []

    class _Recording(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            seen.append(spec.target)
            return iter(())

    fs = _Recording({"src/a.py": "needle\n"}, root=str(root))
    with pytest.raises(ScoutError) as ei:
        _grep(fs, path="vendor")
    assert ei.value.code is ErrorCode.PATH_REFUSED
    assert seen == []
    _grep(fs, path="./src/")
    assert seen == ["src"]


@pytest.mark.parametrize("path", ["../x", "/etc", ".env", "config/.env.local", "a\x00b"])
def test_refuses_bad_or_secret_target(make_fs: MakeFs, path: str) -> None:
    fs = make_fs({"ok.py": "needle\n", ".env": "needle\n", "config/.env.local": "needle\n"})
    with pytest.raises(ScoutError) as ei:
        _grep(fs, path=path)
    assert ei.value.code is ErrorCode.PATH_REFUSED


def test_path_scopes_search_to_subtree(make_fs: MakeFs) -> None:
    fs = make_fs({"src/a.py": "needle\n", "docs/b.md": "needle\n", "src/sub/c.py": "needle\n"})
    assert [f.path for f in _grep(fs, path="src").files] == ["src/a.py", "src/sub/c.py"]
    assert [f.path for f in _grep(fs, path="./src/a.py").files] == ["src/a.py"]
    if isinstance(fs, FakeFileSystem):
        assert fs.grep_specs[0].target == "src"


def test_empty_pattern_and_bad_regex(make_fs: MakeFs) -> None:
    fs = make_fs({"a.txt": "x\n"})
    with pytest.raises(ScoutError) as ei:
        _grep(fs, "")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei:
        _grep(fs, "(")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_huge_match_volume_stops_early_within_caps() -> None:
    """A ~1 MB file of 500k matching lines then 300 more files: bounded work, generator closed, caps held."""
    fs = FakeFileSystem(
        {"a_big.txt": "x\n" * 500_000, **{f"b{i:03d}.txt": "x\n" for i in range(300)}},
    )
    res = grep(fs, GrepRequest(pattern="x"), max_events=50_000)
    assert [len(f.matches) for f in res.files[:1]] == [MAX_MATCHES_PER_FILE]
    assert res.files[0].path == "a_big.txt"
    assert len(res.files) <= MAX_FILES
    assert all(len(f.matches) <= MAX_MATCHES_PER_FILE for f in res.files)
    assert res.truncated
    assert fs.grep_closed_early == 1
    assert fs.grep_events_yielded == 50_001


def test_event_budget_stops_a_stream_that_never_saturates() -> None:
    fs = FakeFileSystem({f"f{i:05d}.txt": "needle\n" for i in range(5000)})
    res = grep(fs, GrepRequest(pattern="needle"), max_events=1000)
    assert res.truncated
    assert fs.grep_closed_early == 1
    assert fs.grep_events_yielded == 1001
    assert len(res.files) == MAX_FILES


def test_clock_deadline_stops_a_slow_stream() -> None:
    clock = FakeClock()

    class _Slow(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            for i in range(100):
                clock.advance(1.0)
                yield RgEvent(kind="match", path=f"f{i}.txt", line=1, text="needle")

    fs = _Slow({f"f{i}.txt": "needle\n" for i in range(100)})
    res = grep(fs, GrepRequest(pattern="needle"), clock=clock, timeout_s=5.0)
    assert res.truncated
    assert 5 <= len(res.files) <= 6


# --- the real adapter: option injection, early stop, error handling (AC-2.1, AC-2.2, FR-C5) ---------------
def _local(tmp_path: Path, files: Mapping[str, str | bytes]) -> LocalFileSystem:
    return _local_repo(tmp_path, files)


def _spec(pattern: str = "needle", **kw: object) -> RgSpec:
    return RgSpec(pattern=pattern, **kw)  # type: ignore[arg-type]


def _rg_pids(marker: str) -> list[int]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)  # noqa: S603, S607
    return [
        int(ln.split(None, 1)[0]) for ln in out.stdout.splitlines() if marker in ln and "ps -axo" not in ln
    ]


def test_ac_2_1_real_rg_pre_pattern_is_a_regex_and_runs_nothing(tmp_path: Path) -> None:
    """AC-2.1: a pattern `--pre=/bin/sh` and a file literally named `--pre=/bin/sh`: no program executes."""
    canary = tmp_path / "canary.sh"
    canary.write_text(f"#!/bin/sh\ntouch {tmp_path}/PWNED\n")
    canary.chmod(0o755)
    fs = _local(
        tmp_path,
        {
            "--pre=/bin/sh": "x --pre=/bin/sh y\n",
            "--pre": "needle --pre\n",
            "-e": "needle -e\n",
            "a.txt": f"--pre={canary}\nneedle\n",
        },
    )
    for pat in ("--pre=/bin/sh", f"--pre={canary}", "--pre", "-e", "--files", "-e foo"):
        got = list(fs.grep_stream(_spec(pat, regex=False)))
        assert all(ev.kind == "match" for ev in got)
    assert [ev.path for ev in fs.grep_stream(_spec("--pre=/bin/sh", regex=False))] == ["--pre=/bin/sh"]
    assert sorted(ev.path for ev in fs.grep_stream(_spec("needle"))) == ["--pre", "-e", "a.txt"]
    # a target that starts with `-` stays a path (after `--`)
    assert [ev.text for ev in fs.grep_stream(_spec("needle", target="--pre"))] == ["needle --pre"]
    assert [ev.text for ev in fs.grep_stream(_spec("needle", target="-e"))] == ["needle -e"]
    assert not (tmp_path / "PWNED").exists()
    # the same through the tool, which re-judges every event
    assert [f.path for f in _grep(fs, "--pre=/bin/sh", regex=False).files] == ["--pre=/bin/sh"]
    assert not (tmp_path / "PWNED").exists()


def test_argv_shape_pattern_only_after_regexp_target_only_after_dashdash(tmp_path: Path) -> None:
    argv = rg._argv("/usr/bin/rg", _spec("--pre=/bin/sh", target="-t"), ".scoutignore", "-t")  # noqa: SLF001
    assert argv[argv.index("--regexp") + 1] == "--pre=/bin/sh"
    assert argv[-2:] == ["--", "-t"]
    assert argv.index("--regexp") < argv.index("--")
    assert argv.count("--pre=/bin/sh") == 1
    assert argv.count("-t") == 1  # only as the target
    for banned in ("--pre", "--search-zip", "-z", "-L", "--follow", "--pcre2", "-P"):
        assert banned not in argv
    assert "--no-config" in argv
    assert "--no-follow" in argv
    assert "--json" in argv


def test_real_rg_ignores_config_file_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`RIPGREP_CONFIG_PATH` must not be able to add `--pre` or flip options."""
    canary = tmp_path / "canary.sh"
    canary.write_text(f"#!/bin/sh\ntouch {tmp_path}/PWNED\ncat\n")
    canary.chmod(0o755)
    cfg = tmp_path / "rgrc"
    cfg.write_text(f"--pre={canary}\n--follow\n--ignore-case\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(cfg))
    fs = _local(tmp_path, {"a.txt": "needle\nNEEDLE\n"})
    assert [ev.line for ev in fs.grep_stream(_spec())] == [1]
    assert not (tmp_path / "PWNED").exists()


def test_symlinks_are_never_followed_by_rg(tmp_path: Path) -> None:
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "x.txt").write_text("needle\n")
    fs = _local_repo(tmp_path, {"ok.txt": "needle\n"}, {"dir_out": "../outside", "f_out": "../outside/x.txt"})
    assert [ev.path for ev in fs.grep_stream(_spec())] == ["ok.txt"]


def test_early_close_terminates_rg_promptly_and_reaps_it(tmp_path: Path) -> None:
    """AC-2.2: closing the generator kills rg (process gone within 1 s, reaped, no zombie)."""
    marker = f"early-close-{tmp_path.name}"
    files = {f"d{i % 20}/f{i:04d}.txt": f"{marker}\n" * 30 for i in range(3000)}
    fs = _local_repo(tmp_path, files)
    gen = fs.grep_stream(_spec(marker, regex=False))
    first = next(gen)
    assert first.kind == "match"
    assert _rg_pids(marker), "rg should be running while the stream is open"
    gen.close()  # type: ignore[attr-defined]
    deadline = time.monotonic() + 1.0
    while _rg_pids(marker) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _rg_pids(marker) == []  # gone, not a zombie (ps lists zombies as `<defunct>`)


def test_core_grep_early_stop_kills_real_rg_on_a_large_tree(tmp_path: Path) -> None:
    marker = f"core-stop-{tmp_path.name}"
    fs = _local(tmp_path, {f"f{i:04d}.txt": f"{marker}\n" * 25 for i in range(1500)})
    t0 = time.monotonic()
    res = grep(fs, GrepRequest(pattern=marker, regex=False))
    assert len(res.files) == MAX_FILES
    assert res.truncated
    assert time.monotonic() - t0 < 5
    deadline = time.monotonic() + 1.0
    while _rg_pids(marker) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _rg_pids(marker) == []


def test_no_matches_is_not_an_error(tmp_path: Path) -> None:
    fs = _local(tmp_path, {"a.txt": "hello\n"})
    assert list(fs.grep_stream(_spec("zzz"))) == []
    assert _grep(fs, "zzz").files == []


def test_bad_target_is_a_typed_error(tmp_path: Path) -> None:
    fs = _local(tmp_path, {"a.txt": "needle\n"})
    with pytest.raises(ScoutError) as ei:
        list(fs.grep_stream(_spec(target="nope/missing")))
    assert ei.value.code is ErrorCode.NOT_FOUND
    with pytest.raises(ScoutError) as ei:
        list(fs.grep_stream(_spec(target="../x")))
    assert ei.value.code is ErrorCode.PATH_REFUSED


@pytest.mark.parametrize("pattern", ["(", "[", "(?=a)", "a\nb", "*"])
def test_invalid_regex_is_invalid_argument(tmp_path: Path, pattern: str) -> None:
    fs = _local(tmp_path, {"a.txt": "needle\n"})
    with pytest.raises(ScoutError) as ei:
        list(fs.grep_stream(_spec(pattern)))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_nul_in_pattern_is_invalid_argument(tmp_path: Path) -> None:
    fs = _local(tmp_path, {"a.txt": "needle\n"})
    with pytest.raises(ScoutError) as ei:
        list(fs.grep_stream(_spec("a\x00b")))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_unreadable_file_keeps_partial_results_not_a_crash(tmp_path: Path) -> None:
    """rg exits 2 when one file cannot be read; the other files' matches must still come back."""
    fs = _local(tmp_path, {"a.txt": "needle\n", "locked.txt": "needle\n", "z.txt": "needle\n"})
    locked = Path(fs.root) / "locked.txt"
    locked.chmod(0)
    try:
        probe = subprocess.run(  # noqa: S603
            ["rg", "--no-config", "--count", "needle", fs.root],  # noqa: S607
            capture_output=True,
            check=False,
        )
        rc = probe.returncode
        assert rc == 2  # premise: rg really does exit 2 here
        assert sorted(ev.path for ev in fs.grep_stream(_spec())) == ["a.txt", "z.txt"]
        assert [f.path for f in _grep(fs).files] == ["a.txt", "z.txt"]
    finally:
        locked.chmod(0o644)


def test_missing_scoutignore_is_fine_and_present_one_is_honoured(tmp_path: Path) -> None:
    fs = _local(tmp_path, {"a.txt": "needle\n", "vendor/b.txt": "needle\n"})
    assert sorted(ev.path for ev in fs.grep_stream(_spec())) == ["a.txt", "vendor/b.txt"]
    (Path(fs.root) / ".scoutignore").write_text("vendor/\n")
    assert [ev.path for ev in fs.grep_stream(_spec())] == ["a.txt"]
    assert sorted(ev.path for ev in fs.grep_stream(_spec(ignore_file=None))) == ["a.txt", "vendor/b.txt"]


def test_events_shape_context_columns_and_non_utf8(tmp_path: Path) -> None:
    fs = _local(
        tmp_path,
        {
            "a.txt": "before\nneedle one\nafter\n",
            "long.txt": "needle" + "y" * 500 + "\n",
            "latin.txt": b"caf\xe9 needle\n",
        },
    )
    evs = list(fs.grep_stream(_spec(context=1)))
    a = [(e.kind, e.line, e.text) for e in evs if e.path == "a.txt"]
    assert a == [("context", 1, "before"), ("match", 2, "needle one"), ("context", 3, "after")]
    long_ = next(e for e in evs if e.path == "long.txt")
    assert len(long_.text) <= 300
    latin = next(e for e in evs if e.path == "latin.txt")
    assert "needle" in latin.text
    paths = [e.path for e in evs]
    assert paths == sorted(paths)  # files contiguous and in stable path order


def test_missing_rg_binary_is_a_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _local(tmp_path, {"a.txt": "needle\n"})
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert shutil.which("rg") is None
    with pytest.raises(ScoutError) as ei:
        list(fs.grep_stream(_spec()))
    assert ei.value.code is ErrorCode.INTERNAL
    assert "rg" in ei.value.message


def test_dot_git_is_never_searched(tmp_path: Path) -> None:
    fs = _local(tmp_path, {"a.txt": "ok\n"})
    (Path(fs.root) / ".git" / "marker.txt").write_text("needle\n")
    assert list(fs.grep_stream(_spec())) == []


def test_nul_in_pattern_is_an_invalid_argument_not_an_internal_error() -> None:
    """A model-written pattern with a NUL used to reach rg (exit 2 -> INTERNAL) and abort the explore call."""
    called: list[str] = []

    class _Recording(FakeFileSystem):
        def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
            called.append(spec.pattern)
            return iter(())

    with pytest.raises(ScoutError) as ei:
        _grep(_Recording({"a.py": "x\n"}), pattern="foo\x00bar")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert called == []
