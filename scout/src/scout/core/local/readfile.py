"""`read_file` tool: 1-based inclusive line range, `\\n` lines, <= 400 lines (FR-C6, AC-2.3, AC-2.4)."""

import posixpath

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ReadRangeRequest, ReadRangeResult
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import MAX_FILE_BYTES, Matcher
from scout.core.ports.fs import FileSystem

MAX_LINES = 400


def read_range(fs: FileSystem, req: ReadRangeRequest, matcher: Matcher | None = None) -> ReadRangeResult:
    """Refuses (PATH_REFUSED) escapes, ignored, denylisted, oversize and binary files alike."""
    path = resolve_in_root(fs, req.path)
    m = matcher or Matcher.from_fs(fs)
    # judge the name as given and the symlink-resolved target (a link to `.env` is a link to a secret)
    if path == "." or m.is_ignored(posixpath.normpath(req.path.strip())) or m.is_excluded(path):
        raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
    start = req.start_line
    if req.end_line is not None and req.end_line < start:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "end_line must be >= start_line")
    text = fs.read_bytes(path, max_bytes=MAX_FILE_BYTES).decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if start > max(len(lines), 1):
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT, f"start_line is past the end of the file ({len(lines)} lines)"
        )
    wanted_end = len(lines) if req.end_line is None else min(req.end_line, len(lines))
    end = min(wanted_end, start + MAX_LINES - 1)
    chosen = lines[start - 1 : end]
    return ReadRangeResult(
        path=path,
        start_line=start,
        end_line=max(end, start - 1) if chosen else start,
        lines=chosen,
        truncated=end < wanted_end,
    )
