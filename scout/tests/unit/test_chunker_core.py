"""Chunker core: byte spans, `\\n` line math, windows, merge/cap (FR-C10, AC-3.3)."""

import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scout.core.sync import chunker
from scout.core.sync.chunker import (
    CODE_MAX_CHARS,
    CODE_MIN_CHARS,
    FALLBACK_CHUNK_LINES,
    ChunkSpan,
    chunk_file,
)

PROP = settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)
WS = b" \t\r\n\x0b\x0c"

_TOKEN = st.sampled_from(
    [
        b"\n", b"\n", b"\n", b"\r\n", b"\r", b"\x0c", b"\x0b", "   ".encode(), " ".encode(),
        b" ", b"\t", b"    ", b"\x00", b"\xff", b"\xc3", b"foo", b"def f(x):", b"{", b"}",
        "é".encode(), "日本語".encode(), "😀".encode(), b"y" * 90, b"z" * 300,
    ]
)  # fmt: skip
_STRUCTURED = st.lists(_TOKEN, max_size=500).map(b"".join)
_RAW = st.binary(max_size=3000)
_BYTES = st.one_of(_STRUCTURED, _RAW, st.tuples(_STRUCTURED, _RAW).map(lambda t: t[0] + t[1]))
_UNICODE = st.text(
    alphabet=st.one_of(
        st.characters(blacklist_categories=("Cs",)), st.sampled_from(["\n", "\r", "\x0c", " ", " "])
    ),
    max_size=2500,
).map(lambda s: s.encode("utf-8"))
_LONG_LINES = st.lists(
    st.one_of(
        st.tuples(st.sampled_from(["x", "é", "日", "😀", "\x0c", "\u2028"]), st.integers(0, 14_000)).map(
            lambda t: t[0] * t[1]
        ),
        st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters="\n"), max_size=300),
    ),
    min_size=1,
    max_size=4,
).map(lambda ls: "\n".join(ls).encode("utf-8"))


def _nl_count(data: bytes, upto: int) -> int:
    return data[:upto].count(b"\n")


def _check_core(data: bytes, chunks: list[ChunkSpan]) -> None:
    """Invariants that hold for any input bytes."""
    cover = [0] * len(data)
    prev_end = 0
    for c in chunks:
        assert 0 <= c.start_byte < c.end_byte <= len(data)
        assert c.start_byte >= prev_end, "spans overlap or are out of order"
        prev_end = c.end_byte
        body = data[c.start_byte : c.end_byte]
        assert body[-1:] not in [bytes([w]) for w in WS], "trailing whitespace"
        assert body.strip(WS), "whitespace-only chunk"
        # line numbers = real `\n` span
        assert c.start_line == 1 + _nl_count(data, c.start_byte)
        last_nonnl = len(body.rstrip(b"\n")) + c.start_byte - 1
        assert c.end_line == 1 + _nl_count(data, last_nonnl)
        assert c.end_line >= c.start_line
        for i in range(c.start_byte, c.end_byte):
            cover[i] += 1
    # every non-whitespace byte exactly once; (whitespace may be inside a span, never twice)
    for i, byte in enumerate(data):
        assert cover[i] <= 1
        if byte not in WS:
            assert cover[i] == 1, f"byte {i} not covered"


def _oracle_view(data: bytes, c: ChunkSpan) -> tuple[bytes, int]:
    """(`\\n`-split lines start_line..end_line joined, byte offset of line start_line)."""
    lines = data.split(b"\n")
    line_start = sum(len(x) + 1 for x in lines[: c.start_line - 1])
    return b"\n".join(lines[c.start_line - 1 : c.end_line]), line_start


def _check_oracle(data: bytes, chunks: list[ChunkSpan], *, long_lines: bool) -> None:
    """The chunk is exactly the cited lines, except whitespace after its last non-newline byte and (only
    for a line cut by the character cap) a partial first/last line."""
    for c in chunks:
        joined, ls = _oracle_view(data, c)
        body = data[c.start_byte : c.end_byte]
        off = c.start_byte - ls
        assert joined[off : off + len(body)] == body
        head, tail = joined[:off], joined[off + len(body) :]
        if not long_lines:
            assert head == b"", "chunk starts mid-line"
            assert not tail.strip(WS), "cited lines hold bytes outside the chunk"


def _chars(b: bytes) -> int:
    return len(b.decode("utf-8", "surrogateescape"))


# --- properties (AC-3.3) -------------------------------------------------------------------------------


@PROP
@given(_BYTES)
def test_line_math_and_coverage_random_bytes(data: bytes) -> None:
    """AC-3.3: random bytes incl. \\x0c, U+2028, CRLF, no trailing newline, empty."""
    chunks = chunk_file("f.txt", data)
    _check_core(data, chunks)
    _check_oracle(data, chunks, long_lines=any(len(x) > CODE_MAX_CHARS for x in data.split(b"\n")))


@PROP
@given(_UNICODE)
def test_unicode_never_split_inside_a_character(data: bytes) -> None:
    """AC-3.3: valid UTF-8 in -> every chunk decodes strictly and is <= CODE_MAX_CHARS characters."""
    chunks = chunk_file("u.py", data)
    _check_core(data, chunks)
    for c in chunks:
        body = data[c.start_byte : c.end_byte]
        assert _chars(body) <= CODE_MAX_CHARS
        body.decode("utf-8")  # raises on a cut multibyte sequence


@PROP
@given(_LONG_LINES)
def test_long_lines_cut_on_character_boundaries(data: bytes) -> None:
    """A line over CODE_MAX_CHARS is cut, on code point boundaries, covering every non-ws byte once."""
    chunks = chunk_file("big.min.js", data)
    _check_core(data, chunks)
    _check_oracle(data, chunks, long_lines=True)
    for c in chunks:
        body = data[c.start_byte : c.end_byte]
        body.decode("utf-8")
        assert _chars(body) <= CODE_MAX_CHARS


@PROP
@given(_BYTES)
def test_max_size_holds_in_characters_for_arbitrary_bytes(data: bytes) -> None:
    for c in chunk_file("f.bin", data):
        assert _chars(data[c.start_byte : c.end_byte]) <= CODE_MAX_CHARS


@PROP
@given(st.lists(st.text("abc é{};=\t", max_size=100), max_size=300), st.sampled_from(["\n", "\r\n"]))
def test_min_size_merge_forward(lines: list[str], eol: str) -> None:
    """No chunk is under CODE_MIN_CHARS unless it is the only one (short lines: the cap never fires)."""
    data = eol.join(lines).encode()
    chunks = chunk_file("m.txt", data)
    _check_core(data, chunks)
    if len(chunks) > 1:
        assert all(_chars(data[c.start_byte : c.end_byte]) >= CODE_MIN_CHARS for c in chunks)


@PROP
@given(_BYTES, st.lists(st.tuples(st.integers(-5, 3200), st.integers(-5, 3200)), max_size=12))
def test_any_syntactic_spans_still_give_a_exact_cover(data: bytes, spans: list[tuple[int, int]]) -> None:
    """Spans from a (future) parser may overlap, leave gaps or run out of range: output stays a cover."""
    newlines = chunker._newline_offsets(data)
    done = chunker._cap(data, chunker._merge_small(data, chunker._complete(data, spans)))
    _check_core(data, [chunker._to_chunk(data, newlines, a, b) for a, b in done])


@PROP
@given(_BYTES)
def test_deterministic_and_input_type_agnostic(data: bytes) -> None:
    first = chunk_file("a/b.go", data)
    assert chunk_file("a/b.go", data) == first
    assert chunk_file("a/b.go", bytearray(data)) == first  # type: ignore[arg-type]


@PROP
@given(_BYTES)
def test_matches_independent_reference_line_numbers(data: bytes) -> None:
    """Line numbers via the `\\n`-split oracle equal the chunker's (str.splitlines would not)."""
    for c in chunk_file("f.rs", data):
        lines = data.split(b"\n")
        sl = next(i for i, _ in enumerate(lines, 1) if sum(len(x) + 1 for x in lines[:i]) > c.start_byte)
        el = next(i for i, _ in enumerate(lines, 1) if sum(len(x) + 1 for x in lines[:i]) > c.end_byte - 1)
        assert (c.start_line, c.end_line) == (sl, el)


# --- examples ------------------------------------------------------------------------------------------


def _body(data: bytes, c: ChunkSpan) -> bytes:
    return data[c.start_byte : c.end_byte]


def test_empty_and_whitespace_only() -> None:
    assert chunk_file("a.py", b"") == []
    assert chunk_file("a.py", b" \n\t\r\n\x0c\n\n") == []


def test_small_file_single_chunk_keeps_bytes_and_indent() -> None:
    data = b"\n\n    def f():\n        return 1\n\n\n"
    [c] = chunk_file("a.py", data)
    assert _body(data, c) == b"    def f():\n        return 1"
    assert (c.start_line, c.end_line) == (3, 4)


def test_no_trailing_newline_and_crlf_do_not_shift_lines() -> None:
    data = b"a = 1\r\nb = 2\r\nc = 3"
    [c] = chunk_file("a.py", data)
    assert _body(data, c) == data
    assert (c.start_line, c.end_line) == (1, 3)
    data = b"a = 1\r\nb = 2\r\nc = 3\r\n"
    [c] = chunk_file("a.py", data)
    assert _body(data, c) == data[:-2]
    assert (c.start_line, c.end_line) == (1, 3)


def test_lone_cr_ff_and_unicode_separators_are_not_line_breaks() -> None:
    data = "one\rtwo\x0cthree four five\nsix\n".encode()
    [c] = chunk_file("a.txt", data)
    assert (c.start_line, c.end_line) == (1, 2)
    assert len(data.decode().splitlines()) == 6  # what the benchmark chunker would have counted


def test_forty_line_windows_with_exact_numbers() -> None:
    data = (b"x" * 29 + b"\x0c\r\n") * 100  # form feeds and CRs are not line breaks
    chunks = chunk_file("notes.txt", data)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (41, 80), (81, 100)]
    assert FALLBACK_CHUNK_LINES == 40
    assert all(_chars(_body(data, c)) <= CODE_MAX_CHARS for c in chunks)


def test_small_tail_merges_back_and_small_head_merges_forward() -> None:
    line = b"y" * 99 + b"\n"  # 100 bytes
    data = line * 41  # windows 40 + 1; the 1-line tail (< 200) merges into the first
    [c] = chunk_file("a.txt", data)
    assert (c.start_line, c.end_line) == (1, 41)
    data = line * 43  # tail of 3 lines = 300 chars >= CODE_MIN_CHARS stays separate
    assert [(c.start_line, c.end_line) for c in chunk_file("a.txt", data)] == [(1, 40), (41, 43)]


def test_cap_splits_oversize_on_line_ends() -> None:
    data = (b"w" * 99 + b"\n") * 40  # 4,000 chars: under the cap
    assert len(chunk_file("a.txt", data)) == 1
    data = (b"w" * 299 + b"\n") * 40  # 12,000 chars in one window: split on line ends
    chunks = chunk_file("a.txt", data)
    assert len(chunks) == 2
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 20), (21, 40)]
    assert all(_chars(_body(data, c)) <= CODE_MAX_CHARS for c in chunks)


def test_overlong_single_line_is_cut_inside_the_line_never_inside_a_character() -> None:
    data = ("é" * 13_000).encode()  # one line, 26,000 bytes, 13,000 characters
    chunks = chunk_file("min.js", data)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 1)] * 3
    assert [_chars(_body(data, c)) for c in chunks] == [6000, 6000, 1000]
    assert b"".join(_body(data, c) for c in chunks) == data


def test_invalid_utf8_counts_one_char_per_byte_and_is_still_covered() -> None:
    data = b"\xff" * 7000
    chunks = chunk_file("blob", data)
    _check_core(data, chunks)
    assert [c.end_byte - c.start_byte for c in chunks] == [6000, 1000]


def test_chunk_span_is_frozen_and_strategy_validated() -> None:
    c = ChunkSpan(0, 1, 1, 1)
    with pytest.raises(AttributeError):
        c.start_byte = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="strategy"):
        chunk_file("a.py", b"x", strategy="nope")


def test_constants_match_the_design() -> None:
    assert (CODE_MAX_CHARS, CODE_MIN_CHARS, FALLBACK_CHUNK_LINES) == (6000, 200, 40)
    assert chunker.TREE_SITTER_STRATEGY == "tree-sitter"
    assert not re.search(r"splitlines|\.strip\(\)", Path(chunker.__file__).read_text().split('"""', 2)[2])
