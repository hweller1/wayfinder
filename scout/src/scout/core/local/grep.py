"""`grep` tool: files ranked by match density, 200 files / 20 matches per file caps (FR-C5, AC-2.1, AC-2.2).

Ripgrep itself lives behind `FileSystem.grep_stream`; this module only consumes `RgEvent`s. The pattern
travels only in `RgSpec` (never as an argument), so a pattern like `--pre=/bin/sh` is just text (AC-2.1).

Ranking: density = `min(match_count, 20)`, descending; ties break by path, so the result does not depend on
ripgrep's (multi-threaded, nondeterministic) file order. Because density saturates at 20, a file with 20
matches can never be displaced by a later one, which is what allows an early stop: once 200 files are
saturated and another match arrives, the generator is closed (killing rg in the real adapter). Further
events from an already-saturated file are skipped, and a total event budget plus an optional clock deadline
bound pathological patterns that match nearly every line.

Every event is judged by the shared `Matcher` and root guard, so a line from an ignored, denylisted, binary,
oversize or out-of-root file is never returned, whatever the adapter streamed.
"""

import posixpath
from collections.abc import Generator

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import GrepFile, GrepMatch, GrepRequest, GrepResult
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import Matcher
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem, RgSpec

MAX_FILES = 200
MAX_MATCHES_PER_FILE = 20
MAX_COLUMNS = 300
MAX_EVENTS = 200_000  # events pulled from the stream (kept or not) before giving up
DEFAULT_TIMEOUT_S = 10.0


def _allowed(fs: FileSystem, m: Matcher, path: str, cache: dict[str, bool]) -> bool:
    ok = cache.get(path)
    if ok is None:
        try:
            resolved = resolve_in_root(fs, path)
            ok = (
                resolved != "." and not m.is_ignored(posixpath.normpath(path)) and not m.is_excluded(resolved)
            )
        except ScoutError as e:
            if e.code is not ErrorCode.PATH_REFUSED:
                raise
            ok = False
        cache[path] = ok
    return ok


def grep(
    fs: FileSystem,
    req: GrepRequest,
    matcher: Matcher | None = None,
    *,
    clock: Clock | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_events: int = MAX_EVENTS,
) -> GrepResult:
    if not req.pattern:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "grep requires a non-empty pattern")
    if "\x00" in req.pattern:  # rg exits 2 on NUL; reject here so the model gets a fixable tool error
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "grep pattern must not contain a NUL byte")
    m = matcher or Matcher.from_fs(fs)
    target = "."
    if req.path is not None and req.path.strip() not in ("", "."):
        target = resolve_in_root(fs, req.path)
        if m.is_ignored(posixpath.normpath(req.path.strip())) or m.is_excluded(target):
            raise ScoutError(ErrorCode.PATH_REFUSED, "path refused")
    spec = RgSpec(
        pattern=req.pattern, target=target, regex=req.regex, context=req.context, max_columns=MAX_COLUMNS
    )
    cache: dict[str, bool] = {}
    order: list[str] = []  # first-match order = tie-break
    lines: dict[str, list[GrepMatch]] = {}
    counts: dict[str, int] = {}
    saturated = 0  # files that reached 20 matches: nothing can outrank them or be displaced by later files
    truncated = False
    seen = 0
    deadline = None if clock is None else clock.monotonic() + timeout_s
    stream = fs.grep_stream(spec)
    try:
        for ev in stream:
            seen += 1
            if seen > max_events or (
                deadline is not None and clock is not None and clock.monotonic() > deadline
            ):
                truncated = True
                break
            if not _allowed(fs, m, ev.path, cache):
                continue
            n = counts.get(ev.path, 0)
            if n >= MAX_MATCHES_PER_FILE:  # saturated file: hidden density above the cap is by design
                if ev.kind == "match":
                    truncated = True
                continue
            if saturated >= MAX_FILES:  # 200 saturated files already outrank every other file
                truncated = True
                break
            if ev.kind == "match":
                if n == 0:
                    order.append(ev.path)
                counts[ev.path] = n + 1
                saturated += n + 1 == MAX_MATCHES_PER_FILE
            lines.setdefault(ev.path, []).append(GrepMatch(line=ev.line, text=ev.text))
    finally:
        if isinstance(stream, Generator):  # closing the generator is what kills rg (early stop)
            stream.close()
    ranked = sorted(order, key=lambda p: (-counts[p], p))  # path tie-break: rg's file order is not stable
    if len(ranked) > MAX_FILES:
        truncated = True
        ranked = ranked[:MAX_FILES]
    return GrepResult(files=[GrepFile(path=p, matches=lines[p]) for p in ranked], truncated=truncated)
