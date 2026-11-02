"""Candidate hits seen during an explore call, and the one validator every returned range goes through.

Scoring: a grep cluster scores its returned line count (match density; context lines are indistinguishable
in `GrepResult` and count too), a range the model read scores `READ_BONUS` plus the density of the grep and
semantic candidates it covers, and a semantic hit scores `SEMANTIC_BONUS` scaled by its score relative to the
best hit of the same call, plus the density of the grep clusters it covers. A grep cluster or semantic hit
that overlaps an already-read (or, for grep, semantic) range folds into it instead of competing with it.
Relocation of stale hits comes in 5.2: until then a hit whose file no longer matches the indexed hash is
dropped (`verify_hits`).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import GrepResult, Hit, ReadRangeResult, SnippetHit
from scout.core.local.guard import resolve_in_root
from scout.core.local.ignore import MAX_FILE_BYTES, Matcher
from scout.core.ports.fs import FileSystem

READ_BONUS = 3.0
SEMANTIC_BONUS = 2.0  # the best semantic hit of a call; the others scale down by their score
SNIPPET_LINES = 12
CLUSTER_GAP = 8  # matches further apart than this start a new range
CLUSTER_PAD = 2  # context lines around a cluster
WHY_MAX = 300


@dataclass(slots=True)
class Candidate:
    path: str
    start_line: int
    end_line: int
    source: str  # "grep" | "read" | "semantic"
    score: float
    why: str


def _overlaps(c: Candidate, path: str, start: int, end: int) -> bool:
    return c.path == path and c.start_line <= end and start <= c.end_line


class CandidateSet:
    def __init__(self) -> None:
        self._items: list[Candidate] = []

    def add_grep(self, pattern: str, res: GrepResult) -> None:
        for f in res.files:
            lines = sorted(m.line for m in f.matches)
            clusters: list[list[int]] = []
            for ln in lines:
                if clusters and ln - clusters[-1][-1] <= CLUSTER_GAP:
                    clusters[-1].append(ln)
                else:
                    clusters.append([ln])
            for c in clusters:
                start, end = max(1, c[0] - CLUSTER_PAD), c[-1] + CLUSTER_PAD
                read = next(
                    (
                        r
                        for r in self._items
                        if r.source in ("read", "semantic") and _overlaps(r, f.path, start, end)
                    ),
                    None,
                )
                if read is not None:
                    read.score += len(c)
                    continue
                self._items.append(
                    Candidate(
                        f.path,
                        start,
                        end,
                        "grep",
                        float(len(c)),
                        f"{len(c)} grep match(es) for {pattern!r}"[:WHY_MAX],
                    )
                )

    def add_read(self, res: ReadRangeResult) -> None:
        """A read range absorbs the grep and semantic candidates it covers (their scores add to its own)."""
        if not res.lines:
            return
        start, end = res.start_line, res.end_line
        score = READ_BONUS
        kept: list[Candidate] = []
        for c in self._items:
            if c.source in ("grep", "semantic") and _overlaps(c, res.path, start, end):
                score += c.score
            elif c.source == "read" and (c.path, c.start_line, c.end_line) == (res.path, start, end):
                score = max(score, c.score)  # re-read of the same range: one entry
            else:
                kept.append(c)
        kept.append(
            Candidate(res.path, start, end, "read", score, f"read by the explorer, lines {start}-{end}")
        )
        self._items = kept

    def add_semantic(self, hits: Sequence[SnippetHit]) -> None:
        """Verified semantic hits of one call, best first. A hit inside a read range or on an identical
        earlier semantic range folds into it; otherwise it absorbs the grep clusters it covers."""
        top = max((h.score or 0.0 for h in hits), default=0.0)
        for rank, h in enumerate(hits):
            base = SEMANTIC_BONUS * h.score / top if top > 0 and h.score else SEMANTIC_BONUS / (rank + 1)
            read = next(
                (
                    r
                    for r in self._items
                    if r.source == "read" and _overlaps(r, h.path, h.start_line, h.end_line)
                ),
                None,
            )
            if read is not None:
                read.score += base
                continue
            same = next(
                (
                    c
                    for c in self._items
                    if c.source == "semantic"
                    and (c.path, c.start_line, c.end_line) == (h.path, h.start_line, h.end_line)
                ),
                None,
            )
            if same is not None:
                same.score = max(same.score, base)
                continue
            score = base
            kept: list[Candidate] = []
            for c in self._items:
                if c.source == "grep" and _overlaps(c, h.path, h.start_line, h.end_line):
                    score += c.score
                else:
                    kept.append(c)
            kept.append(Candidate(h.path, h.start_line, h.end_line, "semantic", score, h.why))
            self._items = kept

    def source_for(self, path: str, start: int, end: int) -> str:
        """`read` when the model read any of these lines, else `semantic` when a semantic hit covers any,
        else `grep`."""
        for source in ("read", "semantic"):
            if any(c.source == source and _overlaps(c, path, start, end) for c in self._items):
                return source
        return "grep"

    def best(self, n: int) -> list[Candidate]:
        ranked = sorted(self._items, key=lambda c: (-c.score, c.path, c.start_line))
        return ranked[:n]


def line_count(fs: FileSystem, matcher: Matcher, path: str) -> tuple[str, int] | None:
    """(normalised path, number of `\\n`-split lines) when the file may be returned, else None. Same judgment
    as `read_file`: root guard, ignore/denylist, size, binary."""
    try:
        norm = resolve_in_root(fs, path)
        if norm == "." or matcher.is_ignored(norm) or matcher.is_excluded(norm):
            return None
        text = fs.read_bytes(norm, max_bytes=MAX_FILE_BYTES).decode("utf-8", errors="replace")
    except ScoutError as e:
        if e.code in (ErrorCode.PATH_REFUSED, ErrorCode.NOT_FOUND):
            return None
        raise
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return norm, len(lines)


def validated_hit(
    fs: FileSystem,
    matcher: Matcher,
    *,
    path: str,
    start_line: int,
    end_line: int,
    why: str,
    source: str,
    score: float | None = None,
) -> Hit | None:
    """Path guard, then clamp the range into the file. Dropped (None): refused, denylisted, ignored, missing,
    empty, or a range that starts past the end of the file."""
    found = line_count(fs, matcher, path)
    if found is None:
        return None
    norm, n = found
    start = max(1, start_line)
    if n == 0 or start > n:
        return None
    end = min(max(end_line, start), n)
    src = source if source in ("semantic", "grep", "read") else "grep"
    return Hit.model_validate(
        {
            "path": norm,
            "start_line": start,
            "end_line": end,
            "why": " ".join(why.split())[:WHY_MAX],
            "source": src,
            "score": score,
        }
    )


def read_checked(fs: FileSystem, matcher: Matcher, path: str) -> tuple[str, bytes] | None:
    """`(normalised path, bytes)` when the file may be returned (same judgment as `line_count`), else None.
    The bytes are read like the sync engine reads them, so their keyed hash can be compared to the index."""
    try:
        norm = resolve_in_root(fs, path)
        if norm == "." or matcher.is_ignored(norm) or matcher.is_excluded(norm):
            return None
        data = fs.read_bytes(norm, max_bytes=MAX_FILE_BYTES + 1)
    except ScoutError as e:
        if e.code in (ErrorCode.PATH_REFUSED, ErrorCode.NOT_FOUND):
            return None
        raise
    return None if len(data) > MAX_FILE_BYTES else (norm, data)


@dataclass(frozen=True, slots=True)
class RawHit:
    """A decrypted, scope-filtered search hit before any disk check."""

    path: str
    start_line: int
    end_line: int
    file_hash: str
    score: float | None = None


def verify_hits(
    fs: FileSystem,
    matcher: Matcher,
    leaf_hash: Callable[[bytes], str],
    raw: Sequence[RawHit],
    k: int,
    seen: dict[str, str] | None = None,
) -> tuple[list[SnippetHit], int]:
    """Turn raw hits into snippet hits carrying the CURRENT numbered lines (at most `SNIPPET_LINES`) and
    the number of hits dropped. A hit is dropped when the file is refused, missing, or its keyed hash no
    longer equals the indexed `file_hash` (the lines moved or changed: never returned unmarked). `seen`
    records the first verified hash per path, so a later check can tell the file changed after the search."""
    out: list[SnippetHit] = []
    dropped = 0
    loaded: dict[str, tuple[str, list[str], str] | None] = {}
    for h in raw:
        if len(out) >= k:
            break
        if h.path not in loaded:
            got = read_checked(fs, matcher, h.path)
            if got is None:
                loaded[h.path] = None
            else:
                norm, data = got
                lines = data.decode("utf-8", errors="replace").split("\n")
                if lines and lines[-1] == "":
                    lines.pop()
                loaded[h.path] = (norm, lines, leaf_hash(data))
        cur = loaded[h.path]
        if cur is None or cur[2] != h.file_hash or h.start_line > len(cur[1]):
            dropped += 1
            continue
        norm, lines, digest = cur
        end = min(h.end_line, len(lines))
        if any(x.path == norm and (x.start_line, x.end_line) == (h.start_line, end) for x in out):
            continue
        shown = min(end, h.start_line + SNIPPET_LINES - 1)
        snippet = "\n".join(f"{n}\t{lines[n - 1]}" for n in range(h.start_line, shown + 1))
        if shown < end:
            snippet += f"\n... {end - shown} more lines (read_file for the rest)"
        if seen is not None:
            seen.setdefault(norm, digest)
        out.append(
            SnippetHit(
                path=norm,
                start_line=h.start_line,
                end_line=end,
                why=f"semantic match, rank {len(out) + 1}",
                source="semantic",
                score=h.score,
                snippet=snippet,
            )
        )
    return out, dropped
