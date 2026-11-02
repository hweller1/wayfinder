"""Declaration chunker kept byte-faithful to benchmark arm g (design section 13; FR-C10, FR-G12, design Q7).

This exists only so the parity run reproduces the benchmark's chunk TEXT. It is a port of the sibling
`declaration` strategy (`_declaration_text`, `_python_spans`, `_brace_spans`, `_net_braces`, `_merge_small`,
`_cap`); nothing is imported from it and the algorithm is deliberately NOT improved:

- The file is decoded with `errors="replace"` and cut on `str.splitlines()` (which also splits on `\\r`,
  `\\x0c`, U+2028, ...), bodies are `.strip()`ped and re-joined with `\\n` / `\\n\\n`. So a chunk's `text` is
  NOT a byte span of the file, unlike the tree-sitter chunker.
- Every body carries, per body line, the source line it came from (`None` for the blank line `_merge_small`
  inserts). That lets each chunk report the `\\n`-based `start_line`/`end_line` and the byte range of the
  source lines it covers, without touching the text. Byte offsets are exact for valid UTF-8; inside one `\\n`
  line holding invalid UTF-8 and a `\\r`-style break they are approximate (line numbers stay exact).
- The benchmark embeds `f"{path}\\n{body}"` (`chunk_code_file` prepends the worktree-relative path to every
  chunk); `embedded_text` reproduces that. The product's chunk key rule is unchanged and lives elsewhere.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from scout.core.sync.chunker import CODE_MAX_CHARS, CODE_MIN_CHARS, FALLBACK_CHUNK_LINES, ChunkSpan

DECLARATION_STRATEGY = "declaration"

_PY_DECL_RE = re.compile(r"^(?:async\s+def|def|class)\s")
_PY_PREFIX_RE = re.compile(r"^(?:@|#)")
_BRACE_LANGS = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js", ".jsx", ".kt", ".php", ".rs",
        ".scala", ".swift", ".ts", ".tsx",
    }
)  # fmt: skip
_PY_LANGS = frozenset({".py"})

_Body = tuple[str, list[int | None]]  # (text, source line index per text line; None = inserted blank)


@dataclass(frozen=True, slots=True)
class DeclarationChunk(ChunkSpan):
    """A `ChunkSpan` over the source lines the chunk covers, plus the benchmark's exact body text."""

    text: str


def embedded_text(path: str, chunk: DeclarationChunk) -> str:
    """What the benchmark sends to the embedder: the path, a newline, then the body."""
    return f"{path}\n{chunk.text}"


def chunk_file(path: str, data: bytes, *, strategy: str = DECLARATION_STRATEGY) -> list[DeclarationChunk]:
    """Chunk one file the way arm g did. Pure and deterministic."""
    if strategy != DECLARATION_STRATEGY:
        raise ValueError(f"unknown chunk strategy {strategy!r}")
    data = bytes(data)
    suffix = ("." + path.rsplit(".", 1)[-1]).lower() if "." in path else ""
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    if suffix in _PY_LANGS:
        spans = _python_spans(lines)
    elif suffix in _BRACE_LANGS:
        spans = _brace_spans(lines)
    else:
        spans = [
            (i, min(i + FALLBACK_CHUNK_LINES, len(lines))) for i in range(0, len(lines), FALLBACK_CHUNK_LINES)
        ]
    bodies = [b for a, z in spans if (b := _strip("\n".join(lines[a:z]), list(range(a, z)))) is not None]
    starts, ends, numbers = _line_table(data, text)
    out: list[DeclarationChunk] = []
    for body, idx in _cap(_merge_small(bodies)):
        real = [i for i in idx if i is not None]
        lo, hi = real[0], real[-1]
        out.append(DeclarationChunk(starts[lo], ends[hi], numbers[lo], numbers[hi], body))
    return out


def _strip(text: str, idx: list[int | None]) -> _Body | None:
    """`text.strip()` that also drops the source-line entries of the blank lines it removed."""
    stripped = text.strip()
    if not stripped:
        return None
    lead = text[: len(text) - len(text.lstrip())].count("\n")
    trail = text[len(text.rstrip()) :].count("\n")
    return stripped, idx[lead : len(idx) - trail]


def _line_table(data: bytes, text: str) -> tuple[list[int], list[int], list[int]]:
    """Per `splitlines` line: start byte, end byte (before its terminator), 1-based `\\n` line number."""
    pieces = text.splitlines(keepends=True)
    newlines: list[int] = []
    i = data.find(b"\n")
    while i != -1:
        newlines.append(i)
        i = data.find(b"\n", i + 1)
    starts: list[int] = []
    ends: list[int] = []
    numbers: list[int] = []
    pos, seen = 0, 0
    for k, piece in enumerate(pieces):
        starts.append(pos)
        numbers.append(1 + seen)
        ends.append(pos + len(piece.splitlines()[0].encode("utf-8")))  # content only, no line terminator
        if piece.endswith("\n"):
            pos = newlines[seen] + 1
            seen += 1
        elif k == len(pieces) - 1:
            pos = len(data)
        else:
            pos += len(piece.encode("utf-8"))
    return starts, ends, numbers


def _python_spans(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Spans starting at each column-zero ``def``/``class``, decorators kept."""
    starts: list[int] = []
    for i, line in enumerate(lines):
        if line[:1].isspace() or not _PY_DECL_RE.match(line):
            continue
        start = i
        while start > 0 and _PY_PREFIX_RE.match(lines[start - 1].lstrip()) is not None:
            if lines[start - 1][:1].isspace():
                break
            start -= 1
        if not starts or start > starts[-1]:
            starts.append(start)
    if not starts:
        return [(0, len(lines))]
    if starts[0] != 0:
        starts.insert(0, 0)  # module header: imports and constants
    return list(zip(starts, [*starts[1:], len(lines)], strict=True))


def _brace_spans(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Spans that end where brace depth returns to zero."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for i, line in enumerate(lines):
        depth += _net_braces(line)
        # Unbalanced source (or a brace this stripper mis-saw): clamp rather
        # than let one bad line make every later boundary wrong.
        depth = max(depth, 0)
        if depth == 0 and "}" in line:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(lines):
        spans.append((start, len(lines)))
    return spans or [(0, len(lines))]


def _net_braces(line: str) -> int:
    """``{`` minus ``}`` outside string literals and ``//``/``#`` comments."""
    depth = 0
    quote: str | None = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "#" or (ch == "/" and i + 1 < n and line[i + 1] == "/"):
            break  # line comment: nothing after it counts
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


def _join(bodies: Sequence[_Body]) -> _Body:
    """`"\\n\\n".join`: one inserted blank line between bodies."""
    idx: list[int | None] = []
    for k, (_, part) in enumerate(bodies):
        if k:
            idx.append(None)
        idx.extend(part)
    return "\n\n".join(t for t, _ in bodies), idx


def _merge_small(bodies: Sequence[_Body]) -> list[_Body]:
    """Merge runs of sub-``CODE_MIN_CHARS`` bodies into their successor."""
    out: list[_Body] = []
    held: list[_Body] = []
    size = 0  # len of the "\n\n"-joined text of `held`
    for body in bodies:
        if not body[0]:
            continue
        size += len(body[0]) + (2 if held else 0)
        held.append(body)
        if size >= CODE_MIN_CHARS:
            out.append(_join(held))
            held, size = [], 0
    if held:
        joined = _join(held)
        if out:
            out[-1] = _join([out[-1], joined])
        else:
            out.append(joined)
    return out


def _cap(bodies: Sequence[_Body]) -> list[_Body]:
    """Slice any body over ``CODE_MAX_CHARS`` on line boundaries."""
    out: list[_Body] = []
    for text, idx in bodies:
        if len(text) <= CODE_MAX_CHARS:
            out.append((text, idx))
            continue
        lines = text.splitlines()
        pieces: list[tuple[int, int]] = []
        start = 0
        size = 0
        for pos, line in enumerate(lines):
            if size and size + len(line) + 1 > CODE_MAX_CHARS:
                pieces.append((start, pos))
                start, size = pos, 0
            size += len(line) + 1
        if start < len(lines):
            pieces.append((start, len(lines)))
        out.extend(b for a, z in pieces if (b := _strip("\n".join(lines[a:z]), idx[a:z])) is not None)
    return out
