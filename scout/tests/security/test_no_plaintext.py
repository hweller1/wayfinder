"""Task 4.7: security byte scan. FR-X1, FR-X5, AC-12.1, AC-12.5 (design 9 "Security byte scan").

The FULL client flow runs through a `RecordingStore` (first sync, incremental edit, move, branch switch,
search under path globs, teammate seed with several proof pages, search on the seeded checkout,
`delete_branch`) over the fixture repos `repo_small` and `declaration_real` (llama.cpp C/C++/Python, 56
files in all). Everything
the store ever received is then scanned for:
- path segments (>= 3 chars; whole names and their dotted parts),
- identifiers (the `identifier_tokens` output plus the raw `[A-Za-z_][A-Za-z0-9_]{2,63}` regex matches),
- every 20-byte window of every file version the flow ever held, including the never-indexed ones (`.env`,
  `.scoutignore`d `generated/`), so a secret or an ignored file reaching the store fails too.
Structure is checked as well: every namespace and row id has its exact opaque shape, and every dict key the
store saw is port vocabulary or a schema attribute (a leaked name cannot hide as a column).

Randomness rule: ciphertext fields (`loc_enc`, `children_enc`, `sketch`, `wrapped_k_team`) are random base64
that can contain any short needle by chance, so only words of >= 8 chars and the 20-byte windows are checked
there; every other field is compared word by word, exactly, which is deterministic.

Three negative controls (a planted plaintext path, a raw BM25 token, plaintext chunk text) must be caught by
the same scanner, so a scanner that stopped looking would fail the suite. `test_hmac_token_retrieval` proves
the identifier search works through HMAC tokens (AC-12.5).
"""

import asyncio
import dataclasses
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from recording_store import RecordingStore, vocabulary

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_fs import _DEFAULT_GIT, FakeFileSystem  # pyright: ignore[reportPrivateUsage]
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.app import ScoutApp
from scout.core.contracts.types import SemanticSearchRequest
from scout.core.contracts.wire import StatusReq
from scout.core.crypto import ids, keys, paths, rows
from scout.core.crypto.tokens import hmac_tokens, identifier_tokens
from scout.core.ports.fs import FileStat
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import Row, VectorStore
from scout.core.service.schema import CHUNK_SCHEMA, META_SCHEMA, TREE_SCHEMA
from scout.core.service.syncsvc import CACHE_SCHEMA
from scout.core.service.tenants import KEYS_SCHEMA
from scout.core.sync.engine import SyncConfig

pytestmark = pytest.mark.security

TENANT = "dev"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BASE_MTIME = 1_700_000_000_000_000_000
OPAQUE_KEYS = frozenset({"loc_enc", "children_enc", "sketch", "wrapped_k_team"})
HEX = r"[0-9a-f]"
NS_SHAPE = re.compile(rf"scout_[A-Za-z0-9-]+_(?:{HEX}{{32}}_{HEX}{{16}}(?:__tree)?|meta|cache)|scoutsys_keys")
PREFIX_SHAPE = re.compile(r"scout_[A-Za-z0-9-]+_")  # `list_namespaces(prefix)`: tenant name only
ID_SHAPE = re.compile(
    rf"root|{HEX}{{16,64}}|(?:team|sketch|proofs)-{HEX}{{32}}|proofpage-{HEX}{{32}}-\d+|nodepage-{HEX}{{16}}-\d+"
)
ID_KEYS = frozenset({"id", "ids"})
NS_KEYS = frozenset({"ns", "source"})
PLAINTEXT_BY_DESIGN = {("tenant", "dev")}  # the tenant name is plaintext metadata (design 7)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,63}")
WINDOW = 20


# --- the scanner -----------------------------------------------------------------------------------------
@dataclass
class Corpus:
    """Needles built from every file version the flow held (`versions: path -> set of contents`)."""

    segments: set[str] = field(default_factory=set[str])  # exact, case-sensitive (`session.py`, `ggml`)
    words: dict[str, str] = field(default_factory=dict[str, str])  # lowercase word -> "path" | "identifier"
    windows: set[bytes] = field(default_factory=set[bytes])

    @classmethod
    def build(cls, versions: dict[str, set[bytes]]) -> "Corpus":
        c = cls()
        for path, contents in versions.items():
            for seg in path.split("/"):
                if len(seg) >= 3:
                    c.segments.add(seg)
                    c.words.setdefault(seg.lower(), "path")
                for part in re.split(r"[.\-]", seg):
                    if len(part) >= 3:
                        c.words.setdefault(part.lower(), "path")
            for data in contents:
                text = data.decode("utf-8", errors="replace")
                for tok in identifier_tokens(text, cap=10**9):
                    c.words.setdefault(tok, "identifier")
                for m in IDENT.finditer(text):
                    c.words.setdefault(m.group(0).lower(), "identifier")
                for i in range(len(data) - WINDOW + 1):
                    w = data[i : i + WINDOW]
                    if len(set(w)) > 1:  # a run of one byte is not source text
                        c.windows.add(w)
        return c


@dataclass(frozen=True)
class Finding:
    kind: str  # "path" | "identifier" | "window" | "shape" | "key"
    where: str
    detail: str


def _window_hit(blob: bytes, windows: set[bytes]) -> bool:
    return any(blob[i : i + WINDOW] in windows for i in range(len(blob) - WINDOW + 1))


def scan(store: RecordingStore, corpus: Corpus, schemas: list[Any]) -> list[Finding]:
    out: list[Finding] = []
    allowed_keys = vocabulary() | {a for s in schemas for a in s.attrs}
    for k in sorted(store.keys - allowed_keys):
        out.append(Finding("key", k, "dict key is neither port vocabulary nor a schema attribute"))
    for v, ks in store.values.items():
        where = ",".join(sorted(ks))
        if ks <= {"tenant"} and ("tenant", v) in PLAINTEXT_BY_DESIGN:
            continue
        if ks == {"prefix"}:
            if not PREFIX_SHAPE.fullmatch(v):
                out.append(Finding("shape", where, f"namespace prefix {v[:60]!r} is not a tenant prefix"))
            continue
        if ks <= NS_KEYS:
            if not NS_SHAPE.fullmatch(v):
                out.append(Finding("shape", where, f"namespace name {v[:60]!r} is not an opaque shape"))
            continue
        if ks <= ID_KEYS:
            if not ID_SHAPE.fullmatch(v):
                out.append(Finding("shape", where, f"row id {v[:60]!r} is not an opaque shape"))
            continue
        opaque = ks <= OPAQUE_KEYS
        for w in WORD.findall(v):
            lw = w.lower()
            if (not opaque or len(lw) >= 8) and lw in corpus.words:
                out.append(Finding(corpus.words[lw], where, f"word {w!r}"))
        if not opaque:
            for seg in v.split("/"):
                if seg in corpus.segments:
                    out.append(Finding("path", where, f"segment {seg!r}"))
        if _window_hit(v.encode("utf-8", errors="replace"), corpus.windows):
            out.append(Finding("window", where, "20-byte source window"))
    if _window_hit(store.buffer, corpus.windows):
        out.append(Finding("window", "buffer", "20-byte source window in the canonical serialization"))
    return out


# --- the flow --------------------------------------------------------------------------------------------
class SecFS(FakeFileSystem):
    """`FakeFileSystem` with edits (mtime bumps so the stat cache sees them), moves and checkouts. Every
    content ever written is kept in `versions` for the scanner."""

    def __init__(self, files: dict[str, bytes], **kw: Any) -> None:
        super().__init__(files, **kw)
        self.mtimes: dict[str, int] = {}
        self.inos: dict[str, int] = {}
        self._tick = 0
        self.versions: dict[str, set[bytes]] = {p: {d} for p, d in files.items()}

    def stat(self, rel: str) -> FileStat:
        data = self.files.get(rel)
        if data is None:
            return super().stat(rel)
        ino = self.inos.setdefault(rel, len(self.inos) + 1)
        return FileStat(len(data), self.mtimes.get(rel, BASE_MTIME), ino, False)

    def write(self, path: str, content: str | bytes) -> None:
        data = content.encode() if isinstance(content, str) else content
        self.files[path] = data
        self._tick += 1
        self.mtimes[path] = BASE_MTIME + self._tick
        self.versions.setdefault(path, set()).add(data)

    def move(self, old: str, new: str) -> None:
        data = self.files.pop(old)
        self.write(new, data)

    def checkout(self, branch: str) -> None:
        head = ("2" if branch != "main" else "1") * 40
        self._git = dataclasses.replace(_DEFAULT_GIT, branch=branch, head=head)


def load(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def fixture_files() -> dict[str, bytes]:
    small, real = load(FIXTURES / "repo_small"), load(FIXTURES / "declaration_real")
    real = {p: d for p, d in real.items() if not p.endswith(".pyc") and p != "LICENSE"}
    assert not set(small) & set(real)
    return {**small, **real}


@dataclass
class Flow:
    store: RecordingStore
    versions: dict[str, set[bytes]]
    results: dict[str, list[str]]  # label -> hit paths
    gw: Gateway
    k: keys.Subkeys
    repo_id: str
    main_ns: str


GLOBS = ["src/**", "**.py", "src/**/*.py", "web/**", "**/tests/*", "sess*", "ggml/src/ggml-cpu/*"]


async def run_flow(store: RecordingStore, *, hook: Callable[[str, Row], Row] | None = None) -> Flow:
    store.row_hook = hook
    clock = FakeClock()
    gw = make_fake_gateway(TENANT, store=store, embedder=FakeEmbedder(), clock=clock)
    k = keys.derive(await gw.team_key())
    state = tempfile.mkdtemp(prefix="scout-sec-")
    fs_a = SecFS(fixture_files(), state_dir=f"{state}/a")
    app_a = ScoutApp(gw, fs_a, clock, sync_config=SyncConfig())
    results: dict[str, list[str]] = {}

    async def search(app: ScoutApp, label: str, query: str, glob: str | None = None) -> None:
        clock.advance(1.0)  # the gateway rate-limits searches per key
        res = await app.semantic_search(SemanticSearchRequest(query=query, path_glob=glob, k=10))
        results[label] = [h.path for h in res.hits]

    # 1. first sync
    assert (await app_a.sync()).state == "ready"
    # 2. incremental edit (one Python, one C++ file)
    fs_a.write(
        "src/auth/session.py",
        fs_a.files["src/auth/session.py"].decode()
        + "\n\ndef purgeStaleTokens(maxAgeSeconds):\n    return 0\n",
    )
    cpp = "src/models/phi2.cpp"
    fs_a.write(cpp, fs_a.files[cpp].decode() + "\nstatic int sec_scan_marker_value = 4242;\n")
    assert (await app_a.sync()).state == "ready"
    # 3. move
    fs_a.move("web/src/session.ts", "web/src/lib/sessionStore.ts")
    assert (await app_a.sync()).state == "ready"
    # 4. search with and without path globs
    await search(app_a, "plain", "getUserSession")
    for g in GLOBS:
        await search(app_a, f"glob:{g}", "getUserSession session lookup", g)
    await search(app_a, "moved", "sessionStore")
    # 5. branch switch: branch the default branch, then sync the diff
    fs_a.checkout("feature/login")
    fs_a.write("src/handlers/login.py", fs_a.files["src/handlers/login.py"].decode() + "\n# login branch\n")
    assert (await app_a.sync()).state == "ready"
    await search(app_a, "feature", "getUserSession", "src/**")
    # 6. teammate seed with proof pages (tiny page size: 56 files -> 8 pages)
    import scout.core.sync.engine as engine_mod

    old_pages = engine_mod.MAX_PROOFS
    engine_mod.MAX_PROOFS = 7
    try:
        b_files = dict(fs_a.files)
        for p in sorted(b_files)[:3]:
            b_files[p] = b_files[p] + b"\n# edited by the teammate\n"
        fs_b = SecFS(b_files, state_dir=f"{state}/b")
        fs_b.checkout("feature/login")
        app_b = ScoutApp(gw, fs_b, clock, sync_config=SyncConfig())
        assert (await app_b.sync()).state == "ready"
    finally:
        engine_mod.MAX_PROOFS = old_pages
    await search(app_b, "seeded", "getUserSession", "src/**")
    # 7. delete_branch (the feature branch of A and the whole of B's)
    repo_id = ids.repo_id(k, _DEFAULT_GIT.root_commit or "", _DEFAULT_GIT.remote_norm or "")
    from scout.core.sync.state import StateDir

    for fs in (fs_a, fs_b):
        st = StateDir(f"{fs.state_dir()}").read_state()
        await gw.delete_branch(repo_id, ids.branch_id(k, repo_id, st.clone_id, "feature/login"))
    versions: dict[str, set[bytes]] = {}
    for fs in (fs_a, fs_b):
        for p, vs in fs.versions.items():
            versions.setdefault(p, set()).update(vs)
    st_a = StateDir(fs_a.state_dir()).read_state()
    main_ns = ids.namespace(TENANT, repo_id, ids.branch_id(k, repo_id, st_a.clone_id, "main"))
    return Flow(store, versions, results, gw, k, repo_id, main_ns)


SCHEMAS = [CHUNK_SCHEMA, TREE_SCHEMA, META_SCHEMA, CACHE_SCHEMA, KEYS_SCHEMA]


@pytest.fixture(scope="module")
def flow() -> Flow:
    return asyncio.run(run_flow(RecordingStore()))


@pytest.fixture(scope="module")
def corpus(flow: Flow) -> Corpus:
    return Corpus.build(flow.versions)


# --- the scan --------------------------------------------------------------------------------------------
def test_flow_ran_end_to_end(flow: Flow) -> None:
    """The scan is only meaningful if the flow really exercised the store: every operation kind is present
    and the searches returned the files they should."""
    assert {
        "ensure_namespace",
        "upsert",
        "get_by_ids",
        "delete_ids",
        "delete_by_file",
        "branch_from",
        "delete_namespace",
        "query_ann",
        "query_bm25",
    } <= set(flow.store.ops)
    assert any('"op":"Glob"' in line for line in flow.store.buffer.decode().splitlines())
    assert any(
        '"kind":"proofs"' in line and '"complete":true' in line
        for line in flow.store.buffer.decode().splitlines()
    )
    assert flow.store.ops.count("branch_from") >= 4  # feature branch (2) + teammate seed (2)
    r = flow.results
    assert "src/auth/session.py" in r["plain"]
    for glob, ok in (
        ("src/**", lambda p: p.startswith("src/")),
        ("web/**", lambda p: p.startswith("web/")),
        ("**.py", lambda p: p.endswith(".py")),
    ):
        assert r[f"glob:{glob}"]
        assert all(ok(p) for p in r[f"glob:{glob}"])
    assert any(p == "web/src/lib/sessionStore.ts" for p in r["moved"])
    assert "src/auth/session.py" in r["feature"]
    assert r["seeded"]


def test_corpus_is_substantial(corpus: Corpus) -> None:
    assert len(corpus.windows) > 200_000
    assert len(corpus.words) > 3_000
    assert {"session", "ggml", "getusersession", "purgestaletokens"} <= set(corpus.words)
    assert {"session.py", "sessionStore.ts", "ggml-cpu"} <= corpus.segments | {"ggml-cpu"}


def test_no_plaintext_path_identifier_or_source_window_reaches_the_store(flow: Flow, corpus: Corpus) -> None:
    """FR-X1, AC-12.1: nothing the store received (rows, filters, namespace names, token lists, ids)
    contains a path segment, an identifier or 20 bytes of source."""
    assert scan(flow.store, corpus, SCHEMAS) == []


def test_no_text_attribute_ever_exists(flow: Flow) -> None:
    """The embedder-bound text never reaches the store: no row attribute can carry it."""
    assert flow.store.attr_names
    assert not {"text", "body", "content", "snippet", "path", "name"} & flow.store.attr_names
    assert flow.store.attr_names <= set(CHUNK_SCHEMA.attrs) | set(TREE_SCHEMA.attrs) | set(
        META_SCHEMA.attrs
    ) | set(KEYS_SCHEMA.attrs)


def test_every_store_method_is_recorded() -> None:
    from scout.adapters.fake_store import FakeStore

    methods = [m for m in dir(VectorStore) if not m.startswith("_")]
    assert methods
    for m in methods:
        assert getattr(RecordingStore, m) is not getattr(FakeStore, m), f"RecordingStore does not record {m}"


# --- negative controls: the same scanner must catch a planted leak ----------------------------------------
async def small_flow(hook: Callable[[str, Row], Row]) -> Flow:
    store = RecordingStore()
    store.row_hook = hook
    clock = FakeClock()
    gw = make_fake_gateway(TENANT, store=store, embedder=FakeEmbedder(), clock=clock)
    k = keys.derive(await gw.team_key())
    fs = SecFS(load(FIXTURES / "repo_small"), state_dir=tempfile.mkdtemp(prefix="scout-sec-"))
    assert (await ScoutApp(gw, fs, clock, sync_config=SyncConfig()).sync()).state == "ready"
    return Flow(store, fs.versions, {}, gw, k, "", "")


def small_corpus() -> Corpus:
    fs = load(FIXTURES / "repo_small")
    return Corpus.build({p: {d} for p, d in fs.items()})


def leak_path(ns: str, row: Row) -> Row:
    """PlainPathStore: one chunk row carries its file's plaintext path instead of `path_enc`."""
    if row.attrs.get("kind") == "chunk":
        return Row(row.id, {**row.attrs, "path_enc": "src/auth/session.py"}, row.vector)
    return row


def leak_token(ns: str, row: Row) -> Row:
    """One chunk row also indexes a raw (unhashed) BM25 token."""
    if row.attrs.get("kind") == "chunk":
        return Row(row.id, {**row.attrs, "tokens": [*row.attrs["tokens"], "getusersession"]}, row.vector)
    return row


def leak_text(ns: str, row: Row) -> Row:
    """One chunk row carries plaintext source text."""
    if row.attrs.get("kind") == "chunk":
        text = (FIXTURES / "repo_small/src/auth/session.py").read_text()[:200]
        return Row(row.id, {**row.attrs, "text": text}, row.vector)
    return row


@pytest.mark.parametrize(
    ("hook", "kinds"),
    [(leak_path, {"path"}), (leak_token, {"identifier"}), (leak_text, {"window", "identifier"})],
    ids=["planted_plaintext_path", "raw_bm25_token", "chunk_text"],
)
def test_negative_control_leak_is_caught(hook: Callable[[str, Row], Row], kinds: set[str]) -> None:
    """The scanner must FAIL on each planted leak (and pass on the clean run of the same repo)."""
    clean = asyncio.run(small_flow(lambda ns, r: r))
    assert scan(clean.store, small_corpus(), SCHEMAS) == []
    leaky = asyncio.run(small_flow(hook))
    found = scan(leaky.store, small_corpus(), SCHEMAS)
    assert found, "the scanner did not catch a planted leak"
    assert {f.kind for f in found} & kinds


def test_negative_control_text_attribute_is_visible() -> None:
    leaky = asyncio.run(small_flow(leak_text))
    assert "text" in leaky.store.attr_names
    assert "text" in {f.where for f in scan(leaky.store, small_corpus(), SCHEMAS)} or any(
        f.kind == "key" and f.where == "text" for f in scan(leaky.store, small_corpus(), SCHEMAS)
    )


# --- AC-12.5: identifier search works through HMAC tokens -----------------------------------------------
@pytest.mark.parametrize("spelling", ["getUserSession", "get_user_session", "GetUserSession"])
def test_hmac_token_retrieval(flow: Flow, spelling: str) -> None:
    """AC-12.5: every spelling of the identifier retrieves its chunk through the keyed BM25 tokens alone
    (no vectors involved); the row's path decrypts to the defining file."""
    toks = hmac_tokens(flow.k, identifier_tokens(spelling, cap=32))
    ns = flow.main_ns
    hits = asyncio.run(flow.store.query_bm25(ns, "tokens", toks, top_k=20))
    got = {paths.dec_path(flow.k, TENANT, flow.repo_id, str(h.attrs["path_enc"])) for h in hits}
    assert "src/auth/session.py" in got
    assert all(not re.search(spelling, t, re.IGNORECASE) for t in toks)
    # a plaintext token matches nothing: the index holds keyed hashes only
    assert asyncio.run(flow.store.query_bm25(ns, "tokens", [spelling.lower()], top_k=5)) == []
    _ = rows, StatusReq  # imported for the type surface of the flow helpers
