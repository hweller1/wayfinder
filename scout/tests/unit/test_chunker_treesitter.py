"""Tree-sitter chunking: language boundaries, big-class recursion, subtree fallback (FR-C10, AC-3.3/3.4)."""

import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_chunker_core import _check_core, _check_oracle  # type: ignore[import-not-found]

from scout.core.sync import chunker
from scout.core.sync.chunker import CODE_MAX_CHARS, FALLBACK_CHUNK_LINES, chunk_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chunker"
PROP = settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)

# Exact (start_line, end_line) of every chunk of each fixture with the production constants: units under
# CODE_MIN_CHARS merge forward into the next declaration (imports/headers ride with the first one).
EXPECTED: dict[str, list[tuple[int, int]]] = {
    "sample.py": [(1, 13), (16, 27), (30, 34)],  # imports+def | class | decorated def
    "sample.go": [(1, 11), (12, 20), (22, 28)],  # package+import+type | NewCounter+Add | Report
    "sample.js": [(1, 10), (12, 31)],  # const+function | class+export function (189 chars merged back)
    "sample.ts": [(1, 7), (9, 22), (24, 27)],  # interface | type+class | export function
    "sample.tsx": [(1, 20), (22, 32)],  # import+type+function | class
    "Sample.java": [(1, 25), (27, 34)],  # package+imports+class | interface
    "sample.rs": [(1, 8), (10, 21), (23, 30)],  # use+struct | impl | fn
    "sample.c": [(1, 9), (11, 26)],  # includes+struct | two functions
    "sample.h": [(1, 9), (11, 19)],  # pragma+include+struct | typedef+prototype+inline function
    "sample.cpp": [(1, 18), (20, 29)],  # includes+class | template function
    "Sample.cs": [(1, 28)],  # whole file: ~600 bytes of units, merged down
}
# Same fixtures with CODE_MIN_CHARS=1: every top-level declaration is its own chunk, so the boundaries are
# exactly the declarations (functions, classes, methods, types).
EXPECTED_UNMERGED: dict[str, list[tuple[int, int]]] = {
    "sample.py": [(1, 1), (3, 3), (6, 13), (16, 27), (30, 34)],
    "sample.go": [(1, 1), (3, 3), (5, 5), (6, 9), (11, 11), (12, 14), (16, 16), (17, 20), (22, 22), (23, 28)],
    "sample.js": [(1, 1), (2, 2), (4, 10), (12, 24), (26, 31)],
    "sample.ts": [(1, 1), (2, 7), (9, 9), (11, 22), (24, 27)],
    "sample.tsx": [(1, 1), (2, 2), (4, 4), (6, 20), (22, 32)],
    "Sample.java": [(1, 1), (3, 3), (4, 4), (6, 6), (7, 25), (27, 27), (28, 34)],
    "sample.rs": [(1, 1), (2, 2), (4, 4), (5, 8), (10, 21), (23, 23), (24, 30)],
    "sample.c": [(1, 1), (2, 2), (3, 3), (5, 9), (11, 20), (22, 26)],
    "sample.h": [(1, 1), (2, 2), (4, 4), (6, 9), (11, 11), (13, 13), (15, 19)],
    "sample.cpp": [(1, 1), (2, 2), (3, 3), (5, 18), (20, 29)],
    "Sample.cs": [(1, 1), (2, 2), (3, 3), (5, 5), (7, 23), (25, 28)],
}


def _lines(data: bytes, chunks: list[chunker.ChunkSpan]) -> list[tuple[int, int]]:
    return [(c.start_line, c.end_line) for c in chunks]


def test_fixture_tables_cover_every_fixture_file() -> None:
    names = {p.name for p in FIXTURES.iterdir() if p.is_file()}
    assert names == set(EXPECTED) == set(EXPECTED_UNMERGED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_chunks_align_to_declarations(name: str) -> None:
    """AC-3.3: one fixture per language (py go js ts tsx java rs c h cpp c#): exact line ranges."""
    data = (FIXTURES / name).read_bytes()
    chunks = chunk_file(name, data)
    assert _lines(data, chunks) == EXPECTED[name]
    _check_core(data, chunks)
    _check_oracle(data, chunks, long_lines=False)


@pytest.mark.parametrize("name", sorted(EXPECTED_UNMERGED))
def test_every_top_level_declaration_is_its_own_chunk(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the small-chunk merge off, boundaries are exactly the top-level declarations."""
    monkeypatch.setattr(chunker, "CODE_MIN_CHARS", 1)
    data = (FIXTURES / name).read_bytes()
    chunks = chunk_file(name, data)
    assert _lines(data, chunks) == EXPECTED_UNMERGED[name]
    _check_core(data, chunks)
    # a chunk starts on a line start, so it begins with the declaration (no half-line chunks)
    for c in chunks:
        assert c.start_byte == 0 or data[c.start_byte - 1 : c.start_byte] == b"\n"


# --- big classes recurse into members (AC-3.4) --------------------------------------------------------------

_N = 30  # members of ~500 chars each: ~15 KB, over twice the cap


def _body(i: int, pad: str) -> str:
    return f"{pad}// member {i}: " + "x" * 400


def _py_class() -> str:
    members = "\n".join(
        f"    def m{i}(self):\n        # member {i}: {'x' * 400}\n        return {i}\n" for i in range(_N)
    )
    return f"class Big:\n    attr = 1\n\n{members}"


def _brace_class(head: str, member: str, tail: str = "}\n") -> str:
    members = "\n".join(member.format(i=i, pad="x" * 400) for i in range(_N))
    return f"{head} {{\n{members}\n{tail}"


BIG = {
    "big.py": _py_class(),
    "Big.java": _brace_class(
        "public class Big",
        "    public int m{i}() {{\n        // member {i}: {pad}\n        return {i};\n    }}\n",
    ),
    "big.ts": _brace_class(
        "export class Big",
        "    m{i}(): number {{\n        // member {i}: {pad}\n        return {i};\n    }}\n",
    ),
    "big.js": _brace_class(
        "class Big", "    m{i}() {{\n        // member {i}: {pad}\n        return {i};\n    }}\n"
    ),
    "Big.cs": _brace_class(
        "public class Big",
        "    public int M{i}()\n    {{\n        // member {i}: {pad}\n        return {i};\n    }}\n",
    ),
    "big.cpp": _brace_class(
        "class Big",
        "public:\n    int m{i}()\n    {{\n        // member {i}: {pad}\n        return {i};\n    }}\n",
        "};\n",
    ),
    "big.rs": _brace_class(
        "impl Big",
        "    fn m{i}(&self) -> i32 {{\n        // member {i}: {pad}\n        return {i};\n    }}\n",
    ),
}


@pytest.mark.parametrize("name", sorted(BIG))
def test_class_over_cap_splits_at_member_boundaries(name: str) -> None:
    """AC-3.4: a class > 6,000 chars is split by recursing into its body; cuts fall between members."""
    data = BIG[name].encode()
    assert len(data) > 2 * CODE_MAX_CHARS
    chunks = chunk_file(name, data)
    _check_core(data, chunks)
    assert len(chunks) >= 2
    assert all(c.end_byte - c.start_byte <= CODE_MAX_CHARS for c in chunks)
    # every chunk after the first starts on a member's first line, never mid-member
    for c in chunks[1:]:
        first = data[c.start_byte : data.index(b"\n", c.start_byte)]
        assert b"(" in first or first.strip() == b"public:", (name, first)
    # no member is cut: each "member N:" comment and its return line land in one chunk
    for i in range(_N):
        marker = f"member {i}:".encode()
        at = data.index(marker)
        owner = [c for c in chunks if c.start_byte <= at < c.end_byte]
        assert len(owner) == 1
        assert owner[0].end_byte > data.index(b"return", at), f"{name}: member {i} cut"


def test_big_python_class_exact_ranges() -> None:
    """Exact ranges: header + attr + m0 merge forward (members are ~430 chars), then one chunk each."""
    data = BIG["big.py"].encode()
    chunks = chunk_file("big.py", data)
    assert _lines(data, chunks) == [(1, 6)] + [(4 * k, 4 * k + 2) for k in range(2, _N + 1)]


def test_class_under_cap_is_one_unit() -> None:
    data = (FIXTURES / "sample.py").read_bytes()
    spans = chunker._syntactic_spans("sample.py", data)
    assert spans is not None
    assert any(b"class Registry" in data[a:b] and b"def dispatch" in data[a:b] for a, b in spans)


# --- syntax errors fall back for the affected subtree only (AC-3.4) -----------------------------------------


def _good(n: int) -> str:
    return f"def good_{n}(x):\n    return x + {n}\n"


def test_syntax_error_confines_fallback_to_its_subtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-3.4: only the broken declaration is cut into 40-line windows; neighbours keep boundaries."""
    monkeypatch.setattr(chunker, "CODE_MIN_CHARS", 1)
    broken = "def broken(x):\n    z = 1 +* )\n" + "".join(f"    y{i} = x * {i}\n" for i in range(98))
    data = (_good(1) + "\n\n" + broken + "\n\n" + _good(2)).encode()
    chunks = chunk_file("a.py", data)
    # good_1 = lines 1-2, broken = lines 5-104 (100 lines), good_2 = lines 107-108
    assert _lines(data, chunks) == [(1, 2), (5, 44), (45, 84), (85, 104), (107, 108)]
    assert chunks[1].end_line - chunks[1].start_line + 1 == FALLBACK_CHUNK_LINES
    _check_core(data, chunks)


def test_syntax_error_does_not_fall_back_whole_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole-file windows would start at 1, 41, 81; syntactic cuts at 5 and 107 prove otherwise."""
    monkeypatch.setattr(chunker, "CODE_MIN_CHARS", 1)
    broken = "def broken(x):\n    z = 1 +* )\n" + "".join(f"    y{i} = x * {i}\n" for i in range(98))
    data = (_good(1) + "\n\n" + broken + "\n\n" + _good(2)).encode()
    starts = {c.start_line for c in chunk_file("a.py", data)}
    assert {1, 5, 107} <= starts
    assert 41 not in starts
    assert 81 not in starts


def test_syntax_error_inside_a_big_class_hits_one_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recursion and fallback compose: one broken method leaves every other member syntactic."""
    monkeypatch.setattr(chunker, "CODE_MIN_CHARS", 1)
    text = BIG["big.py"].replace("def m5(self):", "def m5(self:", 1)
    data = text.encode()
    chunks = chunk_file("big.py", data)
    _check_core(data, chunks)
    lines = text.split("\n")
    starts = {c.start_line for c in chunks}
    for i in (0, 1, 2, 10, 23):
        assert 1 + lines.index(f"    def m{i}(self):") in starts


def test_grammar_error_with_no_good_neighbours_still_covers() -> None:
    data = b"def f(:\n    pass\n)))(((\nclass\n"
    _check_core(data, chunk_file("a.py", data))


# --- unknown suffixes, laziness -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["notes.txt", "Makefile", "a.unknown", "noext", "dir.py/readme"])
def test_unknown_suffix_uses_windows(name: str) -> None:
    data = b"".join(b"line number %d of the file, long enough\n" % i for i in range(100))
    assert [(c.start_line, c.end_line) for c in chunk_file(name, data)] == [(1, 40), (41, 80), (81, 100)]


def test_suffix_match_is_case_insensitive() -> None:
    data = (FIXTURES / "sample.go").read_bytes()
    assert chunk_file("SAMPLE.GO", data) == chunk_file("sample.go", data)


def test_import_is_lazy() -> None:
    """Importing the chunker, and chunking an unknown suffix, never imports tree_sitter."""
    code = (
        "import sys\n"
        "from scout.core.sync.chunker import chunk_file\n"
        "chunk_file('a.txt', b'hello\\n')\n"
        "assert not any(m == 'tree_sitter' or m.startswith('tree_sitter_') for m in sys.modules)\n"
        "chunk_file('a.py', b'x = 1\\n')\n"
        "assert 'tree_sitter' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


# --- the 2.18 properties re-run over real parses ------------------------------------------------------------

_SNIPPETS = st.sampled_from([p for p in sorted(FIXTURES.iterdir()) if p.is_file()]).map(
    lambda p: (p.name, p.read_bytes())
)


@st.composite
def _mutated_source(draw: st.DrawFn) -> tuple[str, bytes]:
    """A real fixture, then cut / duplicated / sprinkled with newlines, CRLF, \\x0c, U+2028, junk bytes."""
    name, data = draw(_SNIPPETS)
    a = draw(st.integers(0, len(data)))
    b = draw(st.integers(a, len(data)))
    out = data[a:b] if draw(st.booleans()) else data
    for _ in range(draw(st.integers(0, 4))):
        at = draw(st.integers(0, len(out)))
        piece = draw(
            st.sampled_from([b"\n", b"\r\n", b"\x0c", " ".encode(), b"{", b"}", b")", b"\xff", b"  "])
        )
        out = out[:at] + piece + out[at:]
    if draw(st.booleans()):
        out = out * draw(st.integers(1, 6))
    return name, out


@PROP
@given(_mutated_source())
def test_properties_hold_over_real_and_broken_source(case: tuple[str, bytes]) -> None:
    """AC-3.3 over tree-sitter output: exact cover, no overlap, `\\n` line math, size cap, per language."""
    name, data = case
    chunks = chunk_file(name, data, strategy="tree-sitter")
    _check_core(data, chunks)
    _check_oracle(data, chunks, long_lines=False)
    for c in chunks:
        assert len(data[c.start_byte : c.end_byte].decode("utf-8", "surrogateescape")) <= CODE_MAX_CHARS


@PROP
@given(
    st.binary(max_size=3000),
    st.sampled_from([".py", ".go", ".js", ".ts", ".tsx", ".java", ".rs", ".c", ".h", ".cpp", ".cs"]),
)
def test_random_bytes_under_every_grammar(data: bytes, suffix: str) -> None:
    """Garbage bytes parse to error trees; the cover and line invariants still hold."""
    chunks = chunk_file("f" + suffix, data)
    _check_core(data, chunks)
    _check_oracle(data, chunks, long_lines=any(len(x) > CODE_MAX_CHARS for x in data.split(b"\n")))


@PROP
@given(st.binary(max_size=3000), st.sampled_from([".txt", ".md", ".bin", "", ".json", ".yaml"]))
def test_random_bytes_unknown_extensions_use_windows(data: bytes, suffix: str) -> None:
    chunks = chunk_file("f" + suffix, data)
    _check_core(data, chunks)
    assert chunker._syntactic_spans("f" + suffix, data) is None
