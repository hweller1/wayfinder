"""Deterministic backfill of an explore result to `explore.backfill_to` files (default 12).

The model's own `report` stays first and untouched. Files the model did not report are appended, one entry per
file, deduplicated against everything already in the list, in three tiers:

(a) files the model opened with `read_file` during the run, in first-seen order;
(b) the top `semantic_search` hits for the brief itself, best score first (only when the semantic backend is
    on; disk-verified like any semantic hit);
(c) files that contain identifiers named in the brief (backticked, snake_case, camelCase, dotted or path-like
    tokens), ranked by distinct identifiers matched, then match count, then path.

Every appended entry carries `verified=False`: the model did not judge it as an answer, so renderers show it
apart from the ranked list. A tier that fails with a `ScoutError` (rate limit, upstream) is skipped; the
result is never made worse by a failed backfill. `backfill_to <= 0` turns the whole thing off.
"""

import contextlib
import re
from collections.abc import Callable

from scout.core.contracts.errors import ScoutError
from scout.core.contracts.types import GrepRequest, Hit
from scout.core.explore.candidates import validated_hit, verify_hits
from scout.core.explore.tools import SemanticBackend
from scout.core.local.grep import grep
from scout.core.local.ignore import Matcher
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem

BRIEF_QUERY_CHARS = 2000  # the embedding query is the head of the brief
SEMANTIC_K = 30  # hits asked of the index; one entry per file is kept
MAX_IDENTIFIERS = 6
MIN_IDENT_LEN = 4
GREP_RANGE_LINES = 12  # a grep-backfilled entry covers the first match line and the lines after it

_TICKED = re.compile(r"`([^`\n]{2,80})`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.:/-][A-Za-z0-9_]+)*")
_COMMON = frozenset(
    [
        "the",
        "this",
        "that",
        "with",
        "from",
        "into",
        "when",
        "what",
        "where",
        "which",
        "should",
        "would",
        "could",
        "have",
        "using",
        "used",
        "uses",
        "use",
        "also",
        "does",
        "doesn",
        "make",
        "need",
        "needs",
        "file",
        "files",
        "code",
        "test",
        "tests",
        "function",
        "functions",
        "class",
        "method",
        "methods",
        "return",
        "true",
        "false",
        "none",
        "null",
        "error",
        "errors",
        "value",
        "values",
        "data",
        "type",
        "types",
        "name",
        "names",
        "list",
        "lists",
        "read",
        "write",
        "add",
        "added",
        "adds",
        "fix",
        "fixes",
        "support",
        "supports",
        "instead",
        "currently",
        "after",
        "before",
        "then",
        "than",
        "only",
        "which",
    ]
)


def _is_code_like(tok: str) -> bool:
    return (
        "_" in tok
        or "." in tok
        or "/" in tok
        or "::" in tok
        or (any(c.islower() for c in tok) and any(c.isupper() for c in tok[1:]))  # camelCase / PascalCase
        or (tok.isupper() and len(tok) >= 5)
    )


def brief_identifiers(brief: str, limit: int = MAX_IDENTIFIERS) -> list[str]:
    """Identifier-like tokens of the brief, first-seen order, backticked first. Plain words are skipped."""
    ticked = [m.group(1).strip() for m in _TICKED.finditer(brief)]
    words = [m.group(0).rstrip(".:/-") for m in _IDENT.finditer(brief)]
    out: list[str] = []
    for tok in [*(t for t in ticked if " " not in t), *(w for w in words if _is_code_like(w))]:
        if len(tok) < MIN_IDENT_LEN or tok.lower() in _COMMON or tok in out:
            continue
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def _from_opened(
    fs: FileSystem, matcher: Matcher, opened: dict[str, tuple[int, int]], changed: Callable[[str], bool]
) -> list[Hit]:
    hits: list[Hit] = []
    for path, (start, end) in opened.items():  # dict order = first-seen order
        hit = validated_hit(
            fs,
            matcher,
            path=path,
            start_line=start,
            end_line=end,
            why="opened by the explorer, not reported",
            source="read",
        )
        if hit is not None and not changed(hit.path):
            hits.append(hit.model_copy(update={"verified": False}))
    return hits


async def _from_semantic(fs: FileSystem, matcher: Matcher, backend: SemanticBackend, brief: str) -> list[Hit]:
    raw = await backend.search(brief[:BRIEF_QUERY_CHARS], None, SEMANTIC_K)
    verified, _ = verify_hits(fs, matcher, backend.leaf_hash, raw, SEMANTIC_K)
    hits: list[Hit] = []
    for h in sorted(
        verified, key=lambda x: (-round(x.score or 0.0, 9), x.path, x.start_line)
    ):  # ties by path
        hit = validated_hit(
            fs, matcher, path=h.path, start_line=h.start_line, end_line=h.end_line,
            why="semantic match for the brief", source="semantic", score=h.score,
        )  # fmt: skip
        if hit is not None:
            hits.append(hit.model_copy(update={"verified": False}))
    return hits


def _from_grep(fs: FileSystem, matcher: Matcher, clock: Clock | None, brief: str) -> list[Hit]:
    stats: dict[
        str, tuple[set[str], int, int, str]
    ] = {}  # path -> (identifiers, matches, first line, first id)
    for ident in brief_identifiers(brief):
        try:
            res = grep(fs, GrepRequest(pattern=ident, regex=False), matcher, clock=clock)
        except ScoutError:
            continue
        for f in res.files:
            if not f.matches:
                continue
            ids, n, first, first_id = stats.get(f.path, (set[str](), 0, f.matches[0].line, ident))
            ids.add(ident)
            stats[f.path] = (ids, n + len(f.matches), first, first_id)
    ranked = sorted(stats.items(), key=lambda kv: (-len(kv[1][0]), -kv[1][1], kv[0]))
    hits: list[Hit] = []
    for path, (ids, n, first, first_id) in ranked:
        why = f"{n} grep match(es) for {', '.join(repr(i) for i in sorted(ids))}" if len(ids) > 1 else (
            f"{n} grep match(es) for {first_id!r}"
        )  # fmt: skip
        hit = validated_hit(
            fs, matcher, path=path, start_line=max(1, first - 2), end_line=first + GREP_RANGE_LINES,
            why=why, source="grep", score=float(n),
        )  # fmt: skip
        if hit is not None:
            hits.append(hit.model_copy(update={"verified": False}))
    return hits


async def backfill(
    reported: list[Hit],
    *,
    target: int,
    brief: str,
    fs: FileSystem,
    matcher: Matcher,
    opened: dict[str, tuple[int, int]],
    changed: Callable[[str], bool],
    semantic: SemanticBackend | None,
    clock: Clock | None = None,
) -> list[Hit]:
    """`reported` followed by unverified entries until `target` distinct files; `reported` itself is returned
    untouched when it already has that many (or `target <= 0`)."""
    seen = {h.path for h in reported}
    if target <= 0 or len(seen) >= target:
        return list(reported)
    out = list(reported)

    def take(hits: list[Hit]) -> None:
        for h in hits:
            if len(seen) >= target:
                return
            if h.path not in seen:
                seen.add(h.path)
                out.append(h)

    take(_from_opened(fs, matcher, opened, changed))
    if len(seen) < target and semantic is not None:
        with contextlib.suppress(ScoutError):  # best effort: the model's list and tiers (a), (c) still stand
            take(await _from_semantic(fs, matcher, semantic, brief))
    if len(seen) < target:
        take(_from_grep(fs, matcher, clock, brief))
    return out
