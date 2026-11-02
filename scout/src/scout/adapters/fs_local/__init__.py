"""Local filesystem + git adapter for the `FileSystem` port (design section 3)."""

import errno
import os
import posixpath
import stat as stat_mod
import threading
from collections.abc import Iterator

from scout.adapters.fs_local import git, rg
from scout.adapters.fs_local import watch as fs_watch
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import MAX_FILE_BYTES
from scout.core.ports.fs import FileStat, FsBatch, GitIds, RgEvent, RgSpec

__all__ = ["LocalFileSystem"]


class LocalFileSystem:
    """Root-confined real filesystem. `root` must be an existing directory: `resolve_in_root` skips its
    realpath containment for a non-directory root (meant for the in-memory fake), so this adapter refuses
    to exist without one and re-checks before every access."""

    def __init__(self, root: str) -> None:
        if not os.path.exists(root):
            raise ScoutError(ErrorCode.NOT_FOUND, "root does not exist")
        if not os.path.isdir(root):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "root is not a directory")
        self.root = os.path.realpath(root)

    # --- confinement -------------------------------------------------------------------------
    def _resolve(self, rel: str) -> str:
        """Repo-relative, symlink-resolved path inside the root (PATH_REFUSED otherwise)."""
        if not os.path.isdir(self.root):  # root vanished: never fall into the guard's in-memory branch
            raise ScoutError(ErrorCode.NOT_FOUND, "root is gone")
        return resolve_in_root(self, rel)

    def _open_regular(self, real_rel: str) -> int:
        path = os.path.join(self.root, real_rel)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            raise ScoutError(ErrorCode.NOT_FOUND, "file not found") from None
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise ScoutError(ErrorCode.PATH_REFUSED, "path refused") from None
            if e.errno in (errno.ENOTDIR, errno.EISDIR):
                raise ScoutError(ErrorCode.NOT_FOUND, "file not found") from None
            raise ScoutError(ErrorCode.INTERNAL, f"cannot open file: {e.strerror or 'error'}") from None
        mode = os.fstat(fd).st_mode
        if not stat_mod.S_ISREG(mode):
            os.close(fd)
            if stat_mod.S_ISDIR(mode):
                raise ScoutError(ErrorCode.NOT_FOUND, "file not found")
            raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
        return fd

    # --- FileSystem --------------------------------------------------------------------------
    def list_files(self) -> list[str]:
        out: list[str] = []
        for name in git.list_files(self.root):
            try:
                if stat_mod.S_ISDIR(os.lstat(os.path.join(self.root, name)).st_mode):
                    continue  # submodule / nested repo entry
            except OSError:
                continue  # tracked but deleted
            out.append(name)
        return out

    def read_bytes(self, rel: str, *, max_bytes: int | None = None) -> bytes:
        fd = self._open_regular(self._resolve(rel))
        with os.fdopen(fd, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def stat(self, rel: str) -> FileStat:
        real = self._resolve(rel)
        fd = self._open_regular(real)
        try:
            st = os.fstat(fd)
        finally:
            os.close(fd)
        lexical = os.path.join(self.root, posixpath.normpath(rel.strip()))
        return FileStat(
            size=st.st_size, mtime_ns=st.st_mtime_ns, ino=st.st_ino, is_symlink=os.path.islink(lexical)
        )

    def read_text(self, rel: str) -> str | None:
        try:
            data = self.read_bytes(rel, max_bytes=MAX_FILE_BYTES)
        except ScoutError as e:
            if e.code is ErrorCode.NOT_FOUND:
                return None
            raise
        return data.decode("utf-8", errors="replace")

    def git_ids(self) -> GitIds:
        return git.git_ids(self.root)

    def state_dir(self) -> str:
        return git.state_dir(self.root)

    def grep_stream(self, spec: RgSpec) -> Iterator[RgEvent]:
        return rg.grep_stream(self.root, spec, self._resolve(spec.target))

    def watch(self, stop: threading.Event, *, debounce_ms: int = 300) -> Iterator[FsBatch]:
        return fs_watch.watch(self, stop, debounce_ms=debounce_ms)
