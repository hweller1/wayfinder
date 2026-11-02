"""Declaration chunker parity with benchmark arm g (FR-C10, FR-G12, design Q7).

The golden (`tests/fixtures/declaration_golden.json`) is produced by `scripts/gen_declaration_golden.py` from
the sibling benchmark's own chunker; these tests require byte-equal chunk bodies on `repo_small` and on ~40
real llama.cpp files (C, C++, headers, Python).
"""

import json
from pathlib import Path

import pytest

from scout.core.sync.chunker import CODE_MAX_CHARS
from scout.core.sync.chunker_declaration import DeclarationChunk, chunk_file, embedded_text

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN: dict[str, dict[str, list[str]]] = json.loads((FIXTURES / "declaration_golden.json").read_text())
ROOTS = {"repo_small": FIXTURES / "repo_small", "real": FIXTURES / "declaration_real"}
CASES = [(name, rel) for name, files in GOLDEN.items() for rel in files]


def test_golden_shape() -> None:
    assert len(GOLDEN["real"]) >= 40
    assert len(GOLDEN["repo_small"]) >= 10
    assert {Path(p).suffix for p in GOLDEN["real"]} >= {".c", ".cpp", ".h", ".py"}


@pytest.mark.parametrize(("name", "rel"), CASES)
def test_bodies_match_benchmark_golden(name: str, rel: str) -> None:
    """AC-14.4: chunk bodies are byte-equal to the benchmark chunker's."""
    data = (ROOTS[name] / rel).read_bytes()
    chunks = chunk_file(rel, data)
    assert [c.text for c in chunks] == GOLDEN[name][rel]
    assert [embedded_text(rel, c) for c in chunks] == [f"{rel}\n{b}" for b in GOLDEN[name][rel]]


@pytest.mark.parametrize(("name", "rel"), CASES)
def test_spans_are_ordered_with_newline_line_math(name: str, rel: str) -> None:
    data = (ROOTS[name] / rel).read_bytes()
    prev_end = 0
    for c in chunk_file(rel, data):
        assert prev_end <= c.start_byte < c.end_byte <= len(data)
        assert c.start_line == 1 + data.count(b"\n", 0, c.start_byte)
        assert c.end_line == 1 + data.count(b"\n", 0, c.end_byte)
        assert c.start_line <= c.end_line
        prev_end = c.end_byte


def test_text_is_not_the_byte_span_when_merging() -> None:
    """Merged small bodies are joined with a blank line the file does not have; spans still cover them."""
    data = b"def a():\n    return 1\ndef b():\n    return 2\n"
    (chunk,) = chunk_file("m.py", data)
    assert chunk.text == "def a():\n    return 1\n\ndef b():\n    return 2"
    assert (chunk.start_byte, chunk.end_byte, chunk.start_line, chunk.end_line) == (0, len(data) - 1, 1, 4)


def test_crlf_and_lone_cr_use_splitlines_text_but_newline_line_numbers() -> None:
    data = b"x = 1\r\ny = 2\rz = 3\n" + b"# pad\n" * 40
    (chunk,) = chunk_file("c.py", data)
    assert chunk.text.startswith("x = 1\ny = 2\nz = 3\n")  # benchmark rewrote \r\n and \r to \n
    assert (chunk.start_line, chunk.start_byte) == (1, 0)
    assert chunk.end_line == 1 + data.rstrip().count(b"\n")


def test_cap_splits_and_keeps_line_ranges() -> None:
    data = b"".join(b"v%d = %d\n" % (i, i) for i in range(2000))
    chunks = chunk_file("big.py", data)
    assert len(chunks) > 1
    assert all(len(c.text) <= CODE_MAX_CHARS for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 2000
    assert all(a.end_line + 1 == b.start_line for a, b in zip(chunks, chunks[1:], strict=False))


def test_empty_whitespace_and_unknown_strategy() -> None:
    assert chunk_file("a.py", b"") == []
    assert chunk_file("a.go", b" \n\t\n\n") == []
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_file("a.py", b"x", strategy="tree-sitter")


def test_chunk_is_a_chunkspan_subclass() -> None:
    from scout.core.sync.chunker import ChunkSpan

    (chunk,) = chunk_file("a.py", b"x = 1\n")
    assert isinstance(chunk, ChunkSpan)
    assert isinstance(chunk, DeclarationChunk)
