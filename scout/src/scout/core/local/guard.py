"""Root confinement for client-supplied paths (adapted from the benchmark's `resolve_in_corpus`; AC-2.3).

Refuses absolute paths, `..` components, backslashes, NUL, and symlinks that resolve outside the root. The
lexical checks run first. When the root exists on disk the real path is also resolved and checked, so the
guard holds even if a `FileSystem` adapter forgets to; the returned path is then the symlink-resolved
repo-relative target, which is what the ignore matcher must judge (a link to `.env` is a link to a secret).
"""

import os
import posixpath
from pathlib import PurePosixPath

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.fs import FileSystem


def _refused() -> ScoutError:
    return ScoutError(
        ErrorCode.PATH_REFUSED, "path refused: absolute, '..', and symlink escapes are not allowed"
    )


def resolve_in_root(fs: FileSystem, rel: str) -> str:
    text = rel.strip() or "."
    if "\x00" in text or "\\" in text or text.startswith(("/", "~")) or PurePosixPath(text).is_absolute():
        raise _refused()
    if ".." in PurePosixPath(text).parts:
        raise _refused()
    norm = posixpath.normpath(text)
    if not os.path.isdir(fs.root):  # in-memory fs: confinement is the adapter's job
        return norm
    real_root = os.path.realpath(fs.root)
    real = os.path.realpath(os.path.join(real_root, norm))
    if real != real_root and not real.startswith(real_root + os.sep):
        raise _refused()
    return posixpath.normpath(os.path.relpath(real, real_root).replace(os.sep, "/"))
