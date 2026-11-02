"""`glob` tool: sorted repo-relative paths matching a pathlib-style pattern, <= 1,000 (FR-C6, AC-2.4).

`*` and `?` stop at `/`; `**/` spans zero or more directories (as `pathlib.Path.glob`). Candidates come from
`fs.list_files()` and go through the same `Matcher` and root guard as `read_file`, so glob never names a file
that read_file would refuse.
"""

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import Matcher
from scout.core.ports.fs import FileSystem

MAX_PATHS = 1000


@dataclass(frozen=True, slots=True)
class GlobResult:
    paths: list[str] = field(default_factory=list[str])
    truncated: bool = False


def _compile(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob(fs: FileSystem, pattern: str, matcher: Matcher | None = None) -> GlobResult:
    pat = pattern.strip()
    if not pat:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "glob requires a non-empty pattern")
    if pat.startswith("/") or ".." in PurePosixPath(pat).parts or "\\" in pat or "\x00" in pat:
        raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
    rx = _compile(pat)
    m = matcher or Matcher.from_fs(fs)
    paths: list[str] = []
    for cand in sorted(fs.list_files()):
        if not rx.match(cand):
            continue
        try:
            if m.is_ignored(cand) or m.is_excluded(resolve_in_root(fs, cand)):
                continue
        except ScoutError as e:
            if e.code is ErrorCode.PATH_REFUSED:
                continue
            raise
        if len(paths) == MAX_PATHS:
            return GlobResult(paths, truncated=True)
        paths.append(cand)
    return GlobResult(paths)
