"""Glob scope compiler (design section 7 table): AC-12.4, AC-12.6, FR-X4.

Exactness is checked end to end: rows sealed with the real crypto go into `FakeStore`, the compiled
`ScopeFilter` is turned into a store `Filter` (as the gateway whitelist does), and (server scope AND
post-filter) must select exactly what an independent fnmatch oracle selects on plaintext paths.
"""

import asyncio
import fnmatch
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import WireRow
from scout.core.crypto import ids, keys, paths, rows
from scout.core.crypto.scope import (
    MAX_FILTER_VALUE_BYTES,
    ScopeFilter,
    ScopePlan,
    UnsupportedGlobError,
    compile_scope,
)
from scout.core.ports.store import And, AttrSpec, Contains, Eq, Filter, Glob, NamespaceSchema, Row

SK = keys.derive(bytes(range(64)))
TENANT = "acme"
REPO = ids.repo_id(SK, "a" * 40, "github.com/acme/widgets")
CK = "ab" * 16
NS = "ns"
SCHEMA = NamespaceSchema(
    dims=2,
    attrs={
        "path_enc": AttrSpec("string", glob=True),
        "ext_tag": AttrSpec("string"),
        "dir_tags": AttrSpec("[]string"),
    },
)
CORPUS = [
    "README.md",
    "Makefile",
    ".env",
    ".py",
    "setup.py",
    "src/main.py",
    "src/util.go",
    "src/auth/session.py",
    "src/auth/token.go",
    "src/auth/deep/keys.py",
    "src/db/models.py",
    "tests/test_auth.py",
    "tests/test_db.go",
    "tests/unit/test_keys.py",
    "pkg/tests/test_pkg.py",
    "pkg/tests/data.json",
    "pkg/src/x.py",
]


def _comp(globs: list[str] | str | None) -> ScopePlan:
    return compile_scope(globs, SK, TENANT, REPO)


def _filter(scope: ScopeFilter) -> Filter | None:
    parts: list[Filter] = []
    if scope.path_glob_enc is not None:
        parts.append(Glob("path_enc", scope.path_glob_enc))
    if scope.ext_tag is not None:
        parts.append(Eq("ext_tag", scope.ext_tag))
    if scope.dir_tag is not None:
        parts.append(Contains("dir_tags", scope.dir_tag))
    return And(parts) if parts else None


def _store_row(r: WireRow) -> Row:
    attrs = {"path_enc": r.path_enc, "ext_tag": r.ext_tag, "dir_tags": r.dir_tags}
    return Row(id=r.id, attrs=attrs, vector=[1.0, 0.0])


def _store_select(files: list[str], plan: ScopePlan) -> set[str]:
    """Plaintext paths the server scope selects over `files`, then the post-filter, via FakeStore + open."""

    async def go() -> tuple[set[str], set[str]]:
        store = FakeStore()
        await store.ensure_namespace(NS, SCHEMA)
        by_enc: dict[str, str] = {}
        batch = []
        for i, f in enumerate(files):
            r = rows.seal_chunk_row(
                SK,
                TENANT,
                REPO,
                path=f,
                start=1,
                end=2,
                chunk_key=CK,
                occurrence=i,
                file_hash="cd" * 32,
                tokens=[],
            )
            by_enc[r.path_enc] = f
            batch.append(_store_row(r))
        await store.upsert(NS, batch)
        got = await store.query_ann(NS, [1.0, 0.0], top_k=1000, flt=_filter(plan.server), attrs=["path_enc"])
        server = {dec for r in got if (dec := by_enc[r.attrs["path_enc"]])}
        return server, {p for p in server if plan.post(p)}

    server, post = asyncio.run(go())
    return server if not plan.needs_post else post


def _server_only(files: list[str], plan: ScopePlan) -> set[str]:
    full = _store_select(files, ScopePlan(plan.server, plan.post, plan.overfetch, needs_post=False))
    return full


# --- independent oracle: per-segment fnmatchcase -------------------------------------------------------


def _oracle(glob: str, path: str) -> bool:
    g = glob
    while g.startswith("./"):
        g = g[2:]
    if g.endswith("/"):
        g += "**"
    gsegs = g.split("/")
    if len(gsegs) == 1 and gsegs[0] != "**":
        gsegs = ["**", *gsegs]
    return _walk(gsegs, path.split("/"))


def _seg_ok(pat: str, name: str) -> bool:
    if re.fullmatch(r"\*+\.[A-Za-z0-9_]+", pat) and name == pat.lstrip("*"):
        return False  # dotfiles have no extension: `*.py` does not match a file named `.py`
    return fnmatch.fnmatchcase(name, pat)


def _walk(gs: list[str], ps: list[str]) -> bool:
    if not gs:
        return not ps
    if gs[0] == "**":
        if len(gs) == 1:
            return len(ps) >= 1
        return any(_walk(gs[1:], ps[i:]) for i in range(len(ps) + 1))
    return bool(ps) and _seg_ok(gs[0], ps[0]) and _walk(gs[1:], ps[1:])


def _oracle_set(glob: str, files: list[str]) -> set[str]:
    return {f for f in files if _oracle(glob, f)}


# --- AC-12.4: every table row of section 7 ----------------------------------------------------------------


def test_dir_prefix_glob_is_whole_segment_ciphertext_ac_12_4():
    """AC-12.4: `src/**` -> `enc(src)/**`, nothing else, no post-filter."""
    p = _comp("src/**")
    assert p.server.path_glob_enc == paths.enc_prefix_glob(SK, TENANT, REPO, "src")
    assert p.server.ext_tag is None
    assert p.server.dir_tag is None
    assert not p.needs_post
    assert p.overfetch == 1


def test_nested_children_glob_ac_12_4():
    """AC-12.4: `src/auth/*` -> `enc(src/auth)/*` (the enc(src)/enc(auth) chain), exact."""
    p = _comp("src/auth/*")
    glob = p.server.path_glob_enc
    assert glob is not None
    assert glob == paths.enc_path(SK, TENANT, REPO, "src/auth") + "/*"
    assert glob.startswith(paths.enc_path(SK, TENANT, REPO, "src") + "/")
    assert not p.needs_post
    assert _store_select(CORPUS, p) == {"src/auth/session.py", "src/auth/token.go"}


@pytest.mark.parametrize("glob", ["*.py", "**.py", "**/*.py"])
def test_extension_globs_use_ext_tag_ac_12_4(glob: str):
    """AC-12.4: `*.py`, `**.py`, `**/*.py` -> `ext_tag = HMAC(.py)` only, exact."""
    p = _comp(glob)
    assert p.server == ScopeFilter(ext_tag=ids.ext_tag(SK, ".py"))
    assert not p.needs_post
    assert p.overfetch == 1
    assert _store_select(CORPUS, p) == {f for f in CORPUS if f.endswith(".py") and f != ".py"}


@pytest.mark.parametrize("glob", ["src/**/*.py", "src/*.py"])
def test_prefix_and_ext_no_post_ac_12_4(glob: str):
    """Table row `dir/**/*.py`: prefix glob AND ext_tag, no post-filter."""
    p = _comp(glob)
    assert p.server.path_glob_enc is not None
    assert p.server.ext_tag == ids.ext_tag(SK, ".py")
    assert not p.needs_post
    assert _store_select(CORPUS, p) == _oracle_set(glob, CORPUS)


def test_directory_scope_trailing_slash():
    """`tests/` is the root-anchored directory, same as `tests/**`; not `pkg/tests/`."""
    p = _comp("tests/")
    assert p.server == _comp("tests/**").server
    want = {"tests/test_auth.py", "tests/test_db.go", "tests/unit/test_keys.py"}
    assert _store_select(CORPUS, p) == want


def test_dir_tag_when_prefix_too_long_ac_12_6():
    """AC-12.6: a deep prefix whose ciphertext passes 4 KiB falls back to `dir_tag` (exact for `dir/**`)."""
    deep = "/".join(f"{c}" + "x" * 250 for c in "abcdefgh")
    assert len(paths.enc_prefix_glob(SK, TENANT, REPO, deep)) > MAX_FILTER_VALUE_BYTES
    p = _comp(f"{deep}/**")
    assert p.server == ScopeFilter(dir_tag=ids.dir_tag(SK, deep))
    assert not p.needs_post
    files = [f"{deep}/f.py", f"{deep}/m/g.py", "other/h.py"]
    assert _store_select(files, p) == {f"{deep}/f.py", f"{deep}/m/g.py"}


def test_deep_prefix_under_limit_stays_ciphertext_ac_12_6():
    """AC-12.6: a long but legal prefix stays a path glob below 4096 bytes."""
    deep = "/".join("y" * 60 for _ in range(10))
    p = _comp(f"{deep}/**")
    assert p.server.path_glob_enc is not None
    assert len(p.server.path_glob_enc) <= MAX_FILTER_VALUE_BYTES


def test_too_long_children_glob_widens_to_dir_tag_with_post_ac_12_6():
    """AC-12.6: `deep/*` cannot be exact via dir_tag (descendants) so the post-filter is required."""
    deep = "/".join("z" * 250 for _ in range(8))
    p = _comp(f"{deep}/*")
    assert p.server == ScopeFilter(dir_tag=ids.dir_tag(SK, deep))
    assert p.needs_post
    assert p.overfetch == 3
    files = [f"{deep}/f.py", f"{deep}/m/g.py", "other/h.py"]
    assert _store_select(files, p) == {f"{deep}/f.py"}
    assert _server_only(files, p) == {f"{deep}/f.py", f"{deep}/m/g.py"}  # server scope alone is wider


def test_too_long_literal_path_falls_back_to_parent_dir_tag():
    deep = "/".join("q" * 250 for _ in range(8))
    p = _comp(f"{deep}/f.py")
    assert p.server == ScopeFilter(dir_tag=ids.dir_tag(SK, deep))
    assert p.needs_post
    files = [f"{deep}/f.py", f"{deep}/g.py"]
    assert _store_select(files, p) == {f"{deep}/f.py"}


def test_literal_file_path_is_exact_ciphertext():
    p = _comp("src/main.py")
    assert p.server == ScopeFilter(path_glob_enc=paths.enc_path(SK, TENANT, REPO, "src/main.py"))
    assert not p.needs_post
    assert _store_select(CORPUS, p) == {"src/main.py"}


def test_wildcard_below_literal_prefix_is_superset_plus_post():
    """`**/tests/*`: `dir_tag` hashes the root-anchored ancestor, so a nested `tests` has no tag."""
    p = _comp("**/tests/*")
    assert p.server == ScopeFilter()
    assert p.needs_post
    assert p.overfetch == 3
    assert _store_select(CORPUS, p) == {
        "tests/test_auth.py",
        "tests/test_db.go",
        "pkg/tests/test_pkg.py",
        "pkg/tests/data.json",
    }


def test_mid_segment_wildcard_post_filtered_with_overfetch_3():
    p = _comp("test_*")
    assert p.server == ScopeFilter()
    assert p.needs_post
    assert p.overfetch == 3
    assert _store_select(CORPUS, p) == _oracle_set("test_*", CORPUS)
    assert {"tests/test_auth.py", "tests/unit/test_keys.py", "pkg/tests/test_pkg.py"} <= _store_select(
        CORPUS, p
    )


@pytest.mark.parametrize(
    "glob", ["src/a*/x.py", "src/a?th/*.py", "*/tests/*.py", "pkg/[ab]*/**", "**/t*/*.py", "src/**/s*.py"]
)
def test_mid_segment_wildcards_never_widen_the_result(glob: str):
    """A glob the server can only over-approximate must carry a post-filter; the final set equals the glob."""
    p = _comp(glob)
    assert p.needs_post
    assert p.overfetch == 3
    assert _store_select(CORPUS, p) == _oracle_set(glob, CORPUS)
    assert {f for f in CORPUS if p.post(f)} == _oracle_set(glob, CORPUS)


def test_literal_prefix_then_wildcard_narrows_server_side_but_posts():
    """`src/auth/te*`: server scope is `enc(src/auth)/**` (superset), post-filter makes it exact."""
    p = _comp("src/auth/to*")
    assert p.server == ScopeFilter(path_glob_enc=paths.enc_prefix_glob(SK, TENANT, REPO, "src/auth"))
    assert p.needs_post
    assert p.overfetch == 3
    assert _store_select(CORPUS, p) == {"src/auth/token.go"}


def test_wildcard_with_literal_extension_adds_ext_tag_superset():
    p = _comp("src/**/s*.py")
    assert p.server.ext_tag == ids.ext_tag(SK, ".py")
    assert p.server.path_glob_enc is not None
    assert p.needs_post
    assert _store_select(CORPUS, p) == {"src/auth/session.py"}


def test_dotfile_has_no_extension():
    """`*.py` never selects a file named `.py` (no extension, matching ext_tag), exactness depends on it."""
    assert _store_select(CORPUS, _comp("*.py")) >= {"setup.py"}
    assert ".py" not in _store_select(CORPUS, _comp("*.py"))
    assert ".py" in _store_select(CORPUS, _comp(".py"))


def test_union_of_globs_post_filters_everything():
    p = _comp(["src/**", "tests/*.go"])
    assert p.server == ScopeFilter()
    assert p.needs_post
    assert p.overfetch == 3
    want = _oracle_set("src/**", CORPUS) | _oracle_set("tests/*.go", CORPUS)
    assert _store_select(CORPUS, p) == want


def test_no_globs_means_no_restriction():
    p = _comp(None)
    assert p.server == ScopeFilter()
    assert not p.needs_post
    assert p.post("any/file")
    assert p.overfetch == 1
    assert _comp([]).server == ScopeFilter()


# --- rejection: typed, never widened --------------------------------------------------------------


@pytest.mark.parametrize(
    "glob",
    [
        "",
        "/etc/passwd",
        "../x",
        "src/../x",
        "src//x",
        "src/./x/*.py",
        "a\\b",
        "a\0b",
        "!src/**",
        "*.{py,go}",
        "src/[ab",
        "src/[z-a]/*",
        "a/[a--]",
        "x" * 5000,
        "/",
    ],
)
def test_unsupported_globs_rejected_with_typed_error(glob: str):
    with pytest.raises(UnsupportedGlobError) as ei:
        _comp(glob)
    assert isinstance(ei.value, ScoutError)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


def test_one_bad_glob_in_a_union_rejects_the_whole_scope():
    with pytest.raises(UnsupportedGlobError):
        _comp(["src/**", "../x"])


# --- no plaintext in the compiled scope ---------------------------------------------------------------

NAMES = ["payments", "src", "tests", "hyphen-dir", "Zebra", "model.py", ".py", "go"]
SCAN_GLOBS = [
    "payments/**",
    "payments/src/*",
    "payments/src/Zebra/**/*.py",
    "**/*.go",
    "*.py",
    "hyphen-dir/model.py",
    "tests/",
    "**/tests/*",
    "payments/te*",
    "payments/**/Z*.go",
    "src/[a-m]*.py",
]


def test_compiled_scope_contains_no_plaintext_names():
    """Byte scan: no segment, extension or directory name appears anywhere in the serialized scope."""
    for glob in SCAN_GLOBS:
        p = _comp(glob)
        blob = p.server.model_dump_json().encode()
        assert re.fullmatch(rb'[0-9a-f{}":,/*_a-z]*', blob), blob
        for name in NAMES + ["py", "Zebra", "payments"]:
            assert name.encode() not in blob, (glob, name)
        for lit in re.split(r"[/*\[\]?]", glob):
            if lit and lit not in ("a-m", "te", "Z", "s"):
                assert lit.encode() not in blob, (glob, lit)


def test_post_filter_closure_holds_no_key_material_in_scope():
    p = _comp("payments/src/*.py")
    assert SK.path not in p.server.model_dump_json().encode()


# --- property: server scope AND post-filter == plaintext fnmatch ------------------------------------------

_SEG = st.sampled_from(["src", "auth", "tests", "pkg", "a", "b", "deep", "x1"])
_FILE = st.sampled_from(
    ["a.py", "b.go", "test_a.py", "test_b.go", "m.py", "x", ".py", ".env", "k.tar.py", "z.py.go"]
)
_PATH = st.builds(lambda d, f: "/".join([*d, f]), st.lists(_SEG, max_size=4), _FILE)
_TREES = st.lists(_PATH, min_size=1, max_size=9, unique=True).filter(
    lambda fs: not any(b.startswith(a + "/") for a in fs for b in fs)  # a name is a file xor a directory
)

_LIT = st.sampled_from(["src", "auth", "tests", "pkg", "a", "b", "deep", "x1"])
_LAST = st.sampled_from(
    [
        "*.py",
        "*.go",
        "**.py",
        "test_*",
        "test_*.py",
        "*",
        "a.py",
        "m.py",
        "?.py",
        "[ab].py",
        "*.tar.py",
        "x",
        "*.py.go",
        ".py",
        "t*.go",
    ]
)
_MID = st.sampled_from(["**", "*", "a*", "?", "[ab]*", "t*"])
_GLOB = st.one_of(
    st.builds(
        lambda lits, mids, last: "/".join([*lits, *mids, last]),
        st.lists(_LIT, max_size=3),
        st.lists(_MID, max_size=2),
        _LAST,
    ),
    st.builds(lambda lits: "/".join(lits) + "/**", st.lists(_LIT, min_size=1, max_size=3)),
    st.builds(lambda lits: "/".join(lits) + "/*", st.lists(_LIT, min_size=1, max_size=3)),
    st.builds(lambda lits: "/".join(lits) + "/", st.lists(_LIT, min_size=1, max_size=3)),
    st.builds(
        lambda mids, last: "/".join([*mids, last]),
        st.lists(st.sampled_from(["**", "*"]), min_size=1, max_size=2),
        _LAST,
    ),
)


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(files=_TREES, globs=st.lists(_GLOB, min_size=1, max_size=1) | st.lists(_GLOB, min_size=2, max_size=3))
def test_server_scope_plus_post_equals_plaintext_selection(files: list[str], globs: list[str]):
    """Property (AC-12.4): selection through FakeStore + post-filter equals fnmatch on plaintext paths, and
    the server scope alone is never narrower (it may only widen, then `needs_post` and overfetch 3)."""
    plan = _comp(globs)
    want = {f for f in files if any(_oracle(g, f) for g in globs)}
    assert _store_select(files, plan) == want, (globs, plan)
    server_only = _server_only(files, plan)
    assert want <= server_only
    if not plan.needs_post:
        assert server_only == want
    else:
        assert plan.overfetch == 3


@settings(max_examples=200, deadline=None)
@given(files=_TREES, glob=_GLOB)
def test_post_filter_matches_oracle_on_plaintext(files: list[str], glob: str):
    plan = _comp(glob)
    assert {f for f in files if plan.post(f)} == _oracle_set(glob, files)
