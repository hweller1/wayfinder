"""Indexable file set (FR-C8, AC-2.5, PRD 4.3): `FileSystem.list_files()` through the one shared `Matcher`.

Excluded: `.gitignore`/`.scoutignore`/denylist matches, files over 1 MiB, binaries, and anything the port
refuses (symlink escapes) or that vanished since listing. At most `MAX_FILES` are returned, in path order;
the overflow is reported in `Listing.warnings` instead of failing the sync.
"""

from dataclasses import dataclass, field

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.local.ignore import Matcher
from scout.core.ports.fs import FileSystem

MAX_FILES = 100_000


@dataclass(frozen=True, slots=True)
class Listing:
    paths: list[str]
    warnings: list[str] = field(default_factory=lambda: [])


def list_indexable(fs: FileSystem, matcher: Matcher | None = None, *, max_files: int = MAX_FILES) -> Listing:
    m = matcher if matcher is not None else Matcher.from_fs(fs)
    paths: list[str] = []
    beyond = 0
    for path in sorted(fs.list_files()):
        if m.is_ignored(path):
            continue
        if len(paths) >= max_files:
            beyond += 1  # path-level eligible only: content checks stop at the cap
            continue
        try:
            if m.is_excluded(path):
                continue
        except ScoutError as e:
            if e.code is ErrorCode.PATH_REFUSED:
                continue
            raise
        paths.append(path)
    warnings: list[str] = []
    if beyond:
        warnings.append(f"file cap {max_files} reached: up to {beyond} more files were not indexed")
    return Listing(paths, warnings)
