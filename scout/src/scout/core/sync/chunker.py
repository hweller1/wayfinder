"""Byte-span chunker core (design section 8 "Chunker", section 13; FR-C10, AC-3.3).

A chunk is a byte span `[start_byte, end_byte)` of the original file bytes. Nothing is decoded, stripped or
re-joined, so the line numbers users see (`path:start-end`) agree with ripgrep and editors:

- `start_line = 1 + count(b"\\n", 0, start_byte)`; `end_line = 1 + count(b"\\n", 0, last_nonnewline_byte)`.
  Only `\\n` ends a line. `\\r\\n`, lone `\\r`, `\\x0c`, U+2028/U+2029 and a missing trailing newline never
  shift a number (the benchmark chunker used `splitlines()`, which does split on those).
- Whitespace is the ASCII set `bytes.strip()` uses. Every span is trimmed so its last byte is non-whitespace;
  a span that starts at a line start also drops leading blank lines but keeps its first line's indentation.
  Whitespace-only spans are dropped, so the chunks cover every non-whitespace byte exactly once and never
  overlap.
- Sizes are in characters, enforced on bytes safely. A character is one UTF-8 code point; a byte that is not
  part of a valid sequence counts as one character (`surrogateescape`). Cuts fall on line ends where possible
  and on code point boundaries otherwise, never inside a multibyte sequence.
- Pipeline: syntactic spans (tree-sitter, else 40-line windows) -> `_complete` (sorted, no overlap,
  gaps become spans of their own, so coverage holds whatever the source of spans) -> `_merge_small`
  (runs under `CODE_MIN_CHARS` merge forward as a contiguous byte union, a short tail merges back) -> `_cap`
  (spans over `CODE_MAX_CHARS` split on line ends, an over-long line on character boundaries).

Constants and the merge/cap/window behaviour are adapted from the benchmark chunker; nothing is imported from
it.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Node

CODE_MAX_CHARS = 6_000
CODE_MIN_CHARS = 200
FALLBACK_CHUNK_LINES = 40
TREE_SITTER_STRATEGY = "tree-sitter"

_MAX_UTF8_BYTES_PER_CHAR = 4
_MAX_DEPTH = 64  # recursion cap into oversize nodes; a deeper node is cut by `_cap` like any leaf
_NONWS = re.compile(rb"[^ \t\r\n\x0b\x0c]")

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """`[start_byte, end_byte)` of the file bytes; lines are 1-based, inclusive, counted by `\\n` only."""

    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


def chunk_file(path: str, data: bytes, *, strategy: str = TREE_SITTER_STRATEGY) -> list[ChunkSpan]:
    """Chunk one file into ordered, non-overlapping spans. Pure and deterministic."""
    if strategy != TREE_SITTER_STRATEGY:
        raise ValueError(f"unknown chunk strategy {strategy!r}")
    data = bytes(data)
    newlines = _newline_offsets(data)
    spans = _syntactic_spans(path, data)
    if spans is None:
        spans = _windows(data, newlines)
    spans = _cap(data, _merge_small(data, _complete(data, spans)))
    return [_to_chunk(data, newlines, a, b) for a, b in spans]


# suffix -> (grammar module, loader). Grammars are imported lazily so that importing this module (and so
# `scout mcp` start-up) never loads tree_sitter before the first chunking call.
_GRAMMARS: dict[str, tuple[str, str]] = {
    ".py": ("tree_sitter_python", "language"),
    ".go": ("tree_sitter_go", "language"),
    ".js": ("tree_sitter_javascript", "language"),
    ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".java": ("tree_sitter_java", "language"),
    ".rs": ("tree_sitter_rust", "language"),
    ".c": ("tree_sitter_c", "language"),
    ".h": ("tree_sitter_c", "language"),
    ".cpp": ("tree_sitter_cpp", "language"),
    ".cc": ("tree_sitter_cpp", "language"),
    ".cxx": ("tree_sitter_cpp", "language"),
    ".hpp": ("tree_sitter_cpp", "language"),
    ".hh": ("tree_sitter_cpp", "language"),
    ".hxx": ("tree_sitter_cpp", "language"),
    ".cs": ("tree_sitter_c_sharp", "language"),
}
_LANGUAGES: dict[str, Any] = {}


def _language(suffix: str) -> Any | None:
    spec = _GRAMMARS.get(suffix)
    if spec is None:
        return None
    lang = _LANGUAGES.get(suffix)
    if lang is None:
        import importlib

        from tree_sitter import Language

        capsule: object = getattr(importlib.import_module(spec[0]), spec[1])()
        lang = _LANGUAGES[suffix] = Language(capsule)
    return lang


def _syntactic_spans(path: str, data: bytes) -> list[Span] | None:
    """Spans at declaration boundaries; `None` (unknown suffix, nothing parsed) means "use windows".

    Every named child of the root is a unit that runs to the start of the next one, so a closing brace, a
    trailing `;` or a blank line rides with the unit before it. A unit over `CODE_MAX_CHARS` is replaced by
    its header (up to its first named child) and its named children, recursively, so a big class becomes
    its members. A unit with a syntax error that cannot be split further falls back to 40-line windows over
    its own bytes only; the rest of the file stays syntactic. Small units merge later in `_merge_small`.
    """
    dot = path.rfind(".")
    lang = _language(path[dot:].lower()) if dot != -1 else None
    if lang is None:
        return None
    from tree_sitter import Parser

    root = Parser(lang).parse(data).root_node
    if not root.named_children:
        return None
    out: list[Span] = []
    for kid in root.named_children:
        _unit(kid, data, out, 1)
    # Cuts snap to line starts (two units on one line stay together), so a chunk is always whole lines.
    starts = sorted({data.rfind(b"\n", 0, a) + 1 for a, _ in out})
    if not starts:  # named children but no units (e.g. only whitespace/stray tokens): fall back to windows
        return None
    return list(zip(starts, [*starts[1:], len(data)], strict=True))


def _unit(node: "Node", data: bytes, out: list[Span], depth: int) -> None:
    a, b = node.start_byte, node.end_byte
    kids = node.named_children
    if b - a > CODE_MAX_CHARS and kids and depth < _MAX_DEPTH and _chars(data, a, b) > CODE_MAX_CHARS:
        if kids[0].start_byte > a:
            out.append((a, kids[0].start_byte))
        for kid in kids:
            _unit(kid, data, out, depth + 1)
    else:
        out.extend(_range_windows(data, a, b) if node.has_error else [(a, b)])


def _range_windows(data: bytes, a: int, b: int) -> list[Span]:
    """`FALLBACK_CHUNK_LINES`-line windows over `[a, b)`, counted from the line holding `a`."""
    out: list[Span] = []
    start = a
    while start < b:
        end, count = start, 0
        while end < b and count < FALLBACK_CHUNK_LINES:
            nl = data.find(b"\n", end, b)
            end = b if nl == -1 else nl + 1
            count += 1
        out.append((start, end))
        start = end
    return out


def _newline_offsets(data: bytes) -> list[int]:
    out: list[int] = []
    i = data.find(b"\n")
    while i != -1:
        out.append(i)
        i = data.find(b"\n", i + 1)
    return out


def _lines_of(newlines: list[int], data: bytes, start: int, end: int) -> tuple[int, int]:
    start_line = 1 + bisect_left(newlines, start)
    last = end
    while last > start and data[last - 1] == 0x0A:
        last -= 1
    if last <= start:
        return start_line, start_line
    return start_line, 1 + bisect_left(newlines, last - 1)


def _to_chunk(data: bytes, newlines: list[int], a: int, b: int) -> ChunkSpan:
    start_line, end_line = _lines_of(newlines, data, a, b)
    return ChunkSpan(a, b, start_line, end_line)


def _chars(data: bytes, a: int, b: int) -> int:
    piece = data[a:b]
    return len(piece) if piece.isascii() else len(piece.decode("utf-8", "surrogateescape"))


def _windows(data: bytes, newlines: list[int]) -> list[Span]:
    """Fixed windows of `FALLBACK_CHUNK_LINES` `\\n`-lines (the last one may be short)."""
    n = len(data)
    if n == 0:
        return []
    line_count = len(newlines) + (0 if data.endswith(b"\n") else 1)
    out: list[Span] = []
    for first in range(0, line_count, FALLBACK_CHUNK_LINES):
        last = min(first + FALLBACK_CHUNK_LINES, line_count) - 1
        start = 0 if first == 0 else newlines[first - 1] + 1
        end = newlines[last] if last < len(newlines) else n
        out.append((start, end))
    return out


def _trim(data: bytes, a: int, b: int) -> Span | None:
    first = _NONWS.search(data, a, b)
    if first is None:
        return None
    p = first.start()
    at_line_start = a == 0 or data[a - 1] == 0x0A
    # whole blank lines go; the first kept line keeps its indentation
    a = (data.rfind(b"\n", a, p) + 1 or a) if at_line_start else p
    return a, a + len(data[a:b].rstrip())


def _complete(data: bytes, spans: list[Span]) -> list[Span]:
    """Sorted, clamped, non-overlapping, trimmed; gaps that hold non-whitespace become spans."""
    n = len(data)
    out: list[Span] = []
    cursor = 0
    for a, b in sorted((max(0, a), min(n, b)) for a, b in spans):
        a = max(a, cursor)
        if a >= b:
            continue
        if a > cursor:
            out.append((cursor, a))
        out.append((a, b))
        cursor = b
    if cursor < n:
        out.append((cursor, n))
    return [t for a, b in out if (t := _trim(data, a, b)) is not None]


def _merge_small(data: bytes, spans: list[Span]) -> list[Span]:
    """Merge runs of spans under `CODE_MIN_CHARS` into their successor (contiguous byte union)."""
    out: list[Span] = []
    held: Span | None = None
    for a, b in spans:
        held = (a, b) if held is None else (held[0], b)
        if _chars(data, *held) >= CODE_MIN_CHARS:
            out.append(held)
            held = None
    if held is not None:
        out.append(held if not out else (out.pop()[0], held[1]))
    return out


def _cap(data: bytes, spans: list[Span]) -> list[Span]:
    """Split any span over `CODE_MAX_CHARS` on line ends; an over-long line on character boundaries."""
    out: list[Span] = []
    for a, b in spans:
        if _chars(data, a, b) <= CODE_MAX_CHARS:
            out.append((a, b))
            continue
        pieces: list[Span] = []
        start, size, pos = a, 0, a
        while pos < b:
            nl = data.find(b"\n", pos, b)
            line_end = b if nl == -1 else nl + 1
            line_chars = _chars(data, pos, line_end)
            if line_chars > CODE_MAX_CHARS:
                if pos > start:
                    pieces.append((start, pos))
                pieces.extend(_split_chars(data, pos, line_end))
                start, size = line_end, 0
            else:
                if size and size + line_chars > CODE_MAX_CHARS:
                    pieces.append((start, pos))
                    start, size = pos, 0
                size += line_chars
            pos = line_end
        if start < b:
            pieces.append((start, b))
        out.extend(t for p, q in pieces if (t := _trim(data, p, q)) is not None)
    return out


def _split_chars(data: bytes, a: int, b: int) -> list[Span]:
    """Cut `[a, b)` into pieces of at most `CODE_MAX_CHARS` characters, on code point boundaries."""
    out: list[Span] = []
    pos = a
    while pos < b:
        window = data[pos : min(b, pos + _MAX_UTF8_BYTES_PER_CHAR * CODE_MAX_CHARS)]
        if window.isascii():
            n = min(len(window), CODE_MAX_CHARS)
        else:
            head = window.decode("utf-8", "surrogateescape")[:CODE_MAX_CHARS]
            n = len(head.encode("utf-8", "surrogateescape"))
        out.append((pos, pos + n))
        pos += n
    return out
