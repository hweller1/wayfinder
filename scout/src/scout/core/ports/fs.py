"""FileSystem port (design section 3). CLIENT side, root-confined, sync; all paths are repo-relative POSIX."""

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class FileStat:
    size: int
    mtime_ns: int
    ino: int
    is_symlink: bool


@dataclass(frozen=True, slots=True)
class RgSpec:
    """Search request for `grep_stream`. The pattern is carried here and only here (AC-2.1)."""

    pattern: str
    target: str = "."  # repo-relative file or directory
    regex: bool = True  # False -> literal (fixed-strings)
    context: int = 0
    max_columns: int = 300
    ignore_file: str | None = ".scoutignore"


@dataclass(frozen=True, slots=True)
class RgEvent:
    """One matched or context line. Files are emitted contiguously, in a stable order."""

    kind: Literal["match", "context"]
    path: str
    line: int  # 1-based
    text: str


@dataclass(frozen=True, slots=True)
class GitIds:
    root_commit: str | None
    remote_norm: str | None
    branch: str  # branch name, or `detached-<commit12>` on a detached HEAD
    head: str | None
    default_branch: str | None
    detached: bool = False
    shallow: bool = False


@dataclass(frozen=True, slots=True)
class FsBatch:
    paths: tuple[str, ...]  # changed repo-relative paths (debounced)
    head_changed: bool = False  # git HEAD moved (branch switch / commit)


class FileSystem(Protocol):
    root: str

    def list_files(self) -> list[str]:
        """`git ls-files -co --exclude-standard -z`; NO scout filtering (core.local.ignore does that)."""
        ...

    def read_bytes(self, rel: str, *, max_bytes: int | None = None) -> bytes:
        """Refuses `..`, absolute, symlink escape (PATH_REFUSED). Missing -> NOT_FOUND."""
        ...

    def stat(self, rel: str) -> FileStat: ...

    def read_text(self, rel: str) -> str | None:
        """.scoutignore / .gitignore content for the matcher; None when the file is absent."""
        ...

    def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
        """`rg --json --no-config --max-columns N [--ignore-file .scoutignore] --regexp PAT -- TARGET`.
        Pattern is ALWAYS after --regexp and `--` (AC-2.1). Generator close kills rg (early stop)."""
        ...

    def git_ids(self) -> GitIds: ...

    def state_dir(self) -> str:
        """`git rev-parse --git-path scout` (worktree-safe). NOT_A_REPO if not a git checkout."""
        ...

    def watch(self, stop: threading.Event, *, debounce_ms: int = 300) -> Iterator[FsBatch]:
        """Batches of changed paths; also yields HEAD changes (explicit watch on git HEAD path)."""
        ...
