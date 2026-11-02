"""One exclusion matcher shared by sync, grep, read_file and glob (FR-C8, AC-2.5, AC-13.2).

A single `pathspec` gitignore-style spec is built from the repo's `.gitignore`, then `.scoutignore`, then the
built-in secret denylist. The denylist goes last so no user negation (`!.env`) can re-include a secret.
Size (> 1 MiB) and binary (NUL in the first 8 KiB) exclusions need content, so they live in `is_excluded`.
Nested `.gitignore` files are not read here: `git ls-files --exclude-standard` already applies them when
files are listed, and the root-level files are the only ones the matcher can load through the port.
"""

import pathspec

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.fs import FileSystem

MAX_FILE_BYTES = 1_048_576
BINARY_SNIFF_BYTES = 8192
IGNORE_FILES = (".gitignore", ".scoutignore")

# PRD 4.3 secret denylist (+ `.git/`, whose config can embed credentialed remote URLs).
SECRET_DENYLIST = (".env*", "*.pem", "id_*", "*.key", "credentials*")
_BUILTIN_DIRS = (".git/",)


def _case_insensitive(pattern: str) -> str:
    return "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in pattern)


_BUILTIN_PATTERNS = tuple(_case_insensitive(p) for p in SECRET_DENYLIST) + _BUILTIN_DIRS


class Matcher:
    def __init__(self, fs: FileSystem, user_lines: list[str]) -> None:
        self._fs = fs
        self._spec = pathspec.PathSpec.from_lines("gitignore", [*user_lines, *_BUILTIN_PATTERNS])

    @classmethod
    def from_fs(cls, fs: FileSystem) -> "Matcher":
        lines: list[str] = []
        for name in IGNORE_FILES:
            text = fs.read_text(name)
            if text:
                lines.extend(text.splitlines())
        return cls(fs, lines)

    def is_ignored(self, path: str) -> bool:
        """Path-only check: `.gitignore`, `.scoutignore`, secret denylist."""
        return self._spec.match_file(path)

    def is_excluded(self, path: str) -> bool:
        """`is_ignored` plus size > 1 MiB and binary sniff. A missing file is not excluded."""
        if self.is_ignored(path):
            return True
        try:
            if self._fs.stat(path).size > MAX_FILE_BYTES:
                return True
            head = self._fs.read_bytes(path, max_bytes=BINARY_SNIFF_BYTES)
        except ScoutError as e:
            if e.code is ErrorCode.NOT_FOUND:
                return False
            raise
        return b"\x00" in head
