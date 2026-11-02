"""Debounced change watcher for `LocalFileSystem.watch` (design section 3, PRD 4.3 item 7).

`watchfiles` batches added/modified/deleted events (a rename is added + deleted). Its default filter drops
everything under `.git`, so it is switched off here and filtered by hand: scout-ignored and denylisted paths
(the shared `Matcher`) never reach a batch, and the git HEAD file plus `refs/heads` (resolved through
`git rev-parse --git-path`, so linked worktrees work) set `head_changed` instead of listing paths. A
commit moves `refs/heads/<branch>`, a branch switch rewrites `HEAD`; both count as a HEAD change.
"""

import os
import posixpath
import threading
from collections.abc import Iterator

import watchfiles

from scout.adapters.fs_local import git
from scout.core.local.ignore import IGNORE_FILES, Matcher
from scout.core.ports.fs import FileSystem, FsBatch

_STEP_MS = 50  # poll interval of the stop event: `stop.set()` ends the iterator within a few steps


class _HeadPaths:
    """Absolute (realpath) git paths whose change means "HEAD moved"."""

    def __init__(self, root: str) -> None:
        self.head = os.path.realpath(git.git_path(root, "HEAD"))
        self.refs = os.path.realpath(git.git_path(root, "refs/heads"))
        self.packed = os.path.realpath(git.git_path(root, "packed-refs"))

    def matches(self, abs_path: str) -> bool:
        if abs_path.endswith(".lock"):
            return False
        return abs_path in (self.head, self.packed) or abs_path.startswith(self.refs + os.sep)

    def watch_roots(self, root: str) -> list[str]:
        """Extra dirs to watch: git locations outside the repo root (linked worktree, separate git dir)."""
        out: list[str] = []
        for p in (os.path.dirname(self.head), self.refs):
            inside = p == root or p.startswith(root + os.sep)
            if not inside and os.path.isdir(p) and p not in out:
                out.append(p)
        return out


def _rel(root: str, abs_path: str) -> str | None:
    if abs_path == root or not abs_path.startswith(root + os.sep):
        return None
    return abs_path[len(root) + 1 :].replace(os.sep, "/")


def watch(fs: FileSystem, stop: threading.Event, *, debounce_ms: int = 300) -> Iterator[FsBatch]:
    root = os.path.realpath(fs.root)
    head = _HeadPaths(root)
    matcher = Matcher.from_fs(fs)
    for changes in watchfiles.watch(
        root,
        *head.watch_roots(root),
        watch_filter=None,
        debounce=debounce_ms,
        step=_STEP_MS,
        stop_event=stop,
        raise_interrupt=False,
        ignore_permission_denied=True,
    ):
        head_changed = False
        paths: set[str] = set()
        for change, abs_path in changes:
            if head.matches(abs_path):
                head_changed = True
                continue
            rel = _rel(root, abs_path)
            if rel is None or rel == ".git" or rel.startswith(".git/"):
                continue
            if change is not watchfiles.Change.deleted and os.path.isdir(abs_path):
                continue
            paths.add(rel)
        if any(posixpath.basename(p) in IGNORE_FILES for p in paths):
            matcher = Matcher.from_fs(fs)  # an ignore file changed: later events use the new rules
        kept = tuple(sorted(p for p in paths if not matcher.is_ignored(p)))
        if kept or head_changed:
            yield FsBatch(paths=kept, head_changed=head_changed)
