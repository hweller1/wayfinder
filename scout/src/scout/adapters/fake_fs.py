"""`FakeFileSystem`: in-memory repo with a pure-python ripgrep stand-in for client-side tests."""

import posixpath
import re
import threading
from collections.abc import Iterator, Mapping, Sequence

import pathspec

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.fs import FileStat, FsBatch, GitIds, RgEvent, RgSpec

_DEFAULT_GIT = GitIds(
    root_commit="0" * 40,
    remote_norm="example.com/acme/repo",
    branch="main",
    head="1" * 40,
    default_branch="main",
)


def _as_bytes(v: str | bytes) -> bytes:
    return v.encode() if isinstance(v, str) else v


class FakeFileSystem:
    """`files` maps repo-relative POSIX paths to content. `symlinks` maps a path to its link target
    (relative targets resolve from the link's directory); a target outside `root` is an escape.

    Observability for tests: `grep_specs` (every spec seen), `grep_events_yielded`, `grep_closed_early`
    (generators closed before exhaustion, i.e. rg would have been killed)."""

    def __init__(
        self,
        files: Mapping[str, str | bytes] | None = None,
        *,
        root: str = "/repo",
        git: GitIds | None = _DEFAULT_GIT,
        state_dir: str | None = None,
        symlinks: Mapping[str, str] | None = None,
        watch_batches: Sequence[FsBatch] = (),
    ) -> None:
        self.root = root
        self.files: dict[str, bytes] = {k: _as_bytes(v) for k, v in (files or {}).items()}
        self.symlinks: dict[str, str] = dict(symlinks or {})
        self._git = git
        self._state_dir = state_dir if state_dir is not None else f"{root}/.git/scout"
        self._watch_batches = list(watch_batches)
        self.grep_specs: list[RgSpec] = []
        self.grep_events_yielded = 0
        self.grep_closed_early = 0

    # --- confinement -------------------------------------------------------------------------
    def _confine(self, rel: str) -> str:
        if not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
            raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
        norm = posixpath.normpath(rel)
        if norm == ".." or norm.startswith("../"):
            raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
        seen: set[str] = set()
        while norm in self.symlinks:  # follow links; any hop leaving the root is refused
            if norm in seen:
                raise ScoutError(ErrorCode.PATH_REFUSED, "symlink loop")
            seen.add(norm)
            target = self.symlinks[norm]
            if target.startswith("/"):
                raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
            norm = posixpath.normpath(posixpath.join(posixpath.dirname(norm), target))
            if norm == ".." or norm.startswith("../"):
                raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
        return norm

    # --- FileSystem --------------------------------------------------------------------------
    def list_files(self) -> list[str]:
        return sorted({*self.files, *self.symlinks})

    def read_bytes(self, rel: str, *, max_bytes: int | None = None) -> bytes:
        data = self.files.get(self._confine(rel))
        if data is None:
            raise ScoutError(ErrorCode.NOT_FOUND, "file not found")
        return data if max_bytes is None else data[:max_bytes]

    def stat(self, rel: str) -> FileStat:
        real = self._confine(rel)
        data = self.files.get(real)
        if data is None:
            raise ScoutError(ErrorCode.NOT_FOUND, "file not found")
        return FileStat(
            size=len(data),
            mtime_ns=1_700_000_000_000_000_000,
            ino=sorted(self.files).index(real) + 1,
            is_symlink=rel in self.symlinks,
        )

    def read_text(self, rel: str) -> str | None:
        data = self.files.get(self._confine(rel))
        return None if data is None else data.decode("utf-8", errors="replace")

    def git_ids(self) -> GitIds:
        if self._git is None:
            raise ScoutError(ErrorCode.NOT_A_REPO, "not a git checkout")
        return self._git

    def state_dir(self) -> str:
        if self._git is None:
            raise ScoutError(ErrorCode.NOT_A_REPO, "not a git checkout")
        return self._state_dir

    def watch(self, stop: threading.Event, *, debounce_ms: int = 300) -> Iterator[FsBatch]:
        for batch in self._watch_batches:
            if stop.is_set():
                return
            yield batch

    # --- ripgrep stand-in --------------------------------------------------------------------
    def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
        """Mirrors `rg --regexp PAT -- TARGET`: the pattern is only ever a pattern (a leading `-` or
        `--pre=...` is just text), binary files are skipped, `spec.ignore_file` patterns are honored,
        and closing the generator before exhaustion is recorded as an early stop."""
        self.grep_specs.append(spec)
        try:
            rx = re.compile(spec.pattern if spec.regex else re.escape(spec.pattern))
        except re.error as e:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"invalid regex: {e}") from None
        return self._grep_gen(spec, rx)

    def _grep_gen(self, spec: RgSpec, rx: re.Pattern[str]) -> Iterator[RgEvent]:
        target = self._confine(spec.target) if spec.target not in (".", "") else "."
        ignore = self._ignore_spec(spec.ignore_file)
        exhausted = False
        try:
            for path in sorted(self.files):
                if target != "." and path != target and not path.startswith(target.rstrip("/") + "/"):
                    continue
                if ignore is not None and ignore.match_file(path):
                    continue
                data = self.files[path]
                if b"\x00" in data:
                    continue
                lines = data.decode("utf-8", errors="replace").split("\n")
                if lines and lines[-1] == "":
                    lines.pop()
                hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
                wanted: dict[int, str] = {}
                for i in hits:
                    wanted[i] = "match"
                    for j in range(max(0, i - spec.context), min(len(lines), i + spec.context + 1)):
                        wanted.setdefault(j, "context")
                for i in sorted(wanted):
                    self.grep_events_yielded += 1
                    kind = "match" if wanted[i] == "match" else "context"
                    yield RgEvent(kind=kind, path=path, line=i + 1, text=lines[i][: spec.max_columns])
            exhausted = True
        finally:
            if not exhausted:
                self.grep_closed_early += 1

    def _ignore_spec(self, ignore_file: str | None) -> pathspec.PathSpec | None:
        if ignore_file is None or ignore_file not in self.files:
            return None
        text = self.files[ignore_file].decode("utf-8", errors="replace")
        return pathspec.PathSpec.from_lines("gitignore", text.splitlines())
