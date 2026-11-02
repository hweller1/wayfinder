"""Task 4.8: in-process end to end, sync then semantic search over the fake system, for every scope shape.

FR-X4, AC-12.4, AC-12.5 (design 5.6, 7). `ScoutApp.sync` indexes `repo_small` through the real
`GatewayService` over fakes, then `raw_search` (decrypted hits BEFORE the disk check) must return the right
file and a line range that covers the definition, for each glob shape of the design 7 table: whole-segment
prefix, extension, prefix plus extension, post-filtered (`**/tests/*`) and mid-segment wildcards (overfetch
3). The scoped result must equal the plaintext selection over the same unscoped hits, and the facade's
verified `semantic_search` must agree.
"""

import fnmatch
import tempfile
from pathlib import Path

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_fs import _DEFAULT_GIT as _GIT  # pyright: ignore[reportPrivateUsage]
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.app import ScoutApp
from scout.core.contracts.types import SemanticSearchRequest
from scout.core.contracts.wire import StatusReq
from scout.core.crypto import keys
from scout.core.crypto.ids import branch_id, repo_id
from scout.core.explore.semantic import raw_search
from scout.core.sync.state import StateDir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TENANT = "dev"
IDENT = "getUserSession"
QUERY = f"{IDENT} session lookup"


def repo_small() -> dict[str, str | bytes]:
    root = FIXTURES / "repo_small"
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class World:
    def __init__(
        self, app: ScoutApp, fs: FakeFileSystem, clock: FakeClock, gw: object, k: keys.Subkeys
    ) -> None:
        self.app, self.fs, self.clock, self.gw, self.k = app, fs, clock, gw, k

    async def raw(self, glob: str | None, k: int = 30) -> list[tuple[str, int, int]]:
        self.clock.advance(1.0)  # the gateway rate-limits searches per key
        st = StateDir.open(self.fs).read_state()
        rid = repo_id(self.k, _GIT.root_commit or "", _GIT.remote_norm or "")
        req = StatusReq(repo_id=rid, branch_id=branch_id(self.k, rid, st.clone_id, _GIT.branch))
        hits = await raw_search(self.gw, self.k, TENANT, req, QUERY, glob, k)  # type: ignore[arg-type]
        return [(h.path, h.start_line, h.end_line) for h in hits]


@pytest.fixture
async def world() -> World:
    clock = FakeClock()
    gw = make_fake_gateway(TENANT, clock=clock)
    fs = FakeFileSystem(repo_small(), state_dir=tempfile.mkdtemp(prefix="scout-app-search-"))
    app = ScoutApp(gw, fs, clock)
    assert (await app.sync()).state == "ready"
    return World(app, fs, clock, gw, keys.derive(await gw.team_key()))


def covers_definition(w: World, path: str, start: int, end: int, needle: str) -> bool:
    lines = w.fs.files[path].decode().split("\n")
    return any(needle in line for line in lines[start - 1 : end])


async def test_unscoped_search_returns_the_definition_with_a_correct_range(world: World) -> None:
    hits = await world.raw(None)
    by_path = {p: (s, e) for p, s, e in hits}
    start, end = by_path["src/auth/session.py"]
    assert 1 <= start <= end <= len(world.fs.files["src/auth/session.py"].decode().split("\n"))
    assert covers_definition(world, "src/auth/session.py", start, end, f"def {IDENT}")
    assert ".env" not in by_path
    assert not any(p.startswith("generated/") for p in by_path)  # `.scoutignore`d, never indexed
    # the facade's verified layer agrees: same top file, snippet lines are the CURRENT disk lines
    sem = await world.app.semantic_search(SemanticSearchRequest(query=QUERY, k=30))
    assert sem.note is None
    top = {h.path: h for h in sem.hits}["src/auth/session.py"]
    first = world.fs.files[top.path].decode().split("\n")[top.start_line - 1]
    assert top.snippet.splitlines()[0] == f"{top.start_line}\t{first}"


# (glob, shape from the design 7 table, path predicate, paths that must be found)
CASES = [
    ("src/**", "dir prefix", lambda p: p.startswith("src/"), {"src/auth/session.py"}),
    ("src/auth/*", "nested children", lambda p: p.startswith("src/auth/"), {"src/auth/session.py"}),
    ("src/auth/session.py", "literal file", lambda p: p == "src/auth/session.py", {"src/auth/session.py"}),
    ("**.py", "extension", lambda p: p.endswith(".py"), {"src/auth/session.py"}),
    ("*.py", "extension", lambda p: p.endswith(".py"), {"src/auth/session.py"}),
    ("**/*.py", "extension", lambda p: p.endswith(".py"), {"src/auth/session.py"}),
    (
        "src/**/*.py",
        "prefix and extension",
        lambda p: p.startswith("src/") and p.endswith(".py"),
        {"src/auth/session.py"},
    ),
    (
        "web/**/*.ts",
        "prefix and extension",
        lambda p: p.startswith("web/") and p.endswith(".ts"),
        {"web/src/session.ts"},
    ),
    (
        "**/tests/*",
        "post-filtered",
        lambda p: fnmatch.fnmatch(p, "tests/*") or "/tests/" in p,
        {"tests/session_spec.py"},
    ),
    (
        "session*",
        "mid-segment wildcard",
        lambda p: p.rsplit("/", 1)[-1].startswith("session"),
        {"src/auth/session.py", "web/src/session.ts"},
    ),
]


@pytest.mark.parametrize(("glob", "shape", "pred", "must"), CASES, ids=[f"{c[0]} ({c[1]})" for c in CASES])
async def test_every_scope_shape_returns_exactly_the_scoped_files_with_correct_ranges(
    world: World, glob: str, shape: str, pred: object, must: set[str]
) -> None:
    assert callable(pred)
    everything = {p for p, _s, _e in await world.raw(None)}
    scoped = await world.raw(glob)
    got = {p for p, _s, _e in scoped}
    assert got, shape
    assert got == {p for p in everything if pred(p)}  # scope narrows the hits, never invents or loses one
    assert must <= got
    for path, start, end in scoped:  # every range is a real range of the file as indexed
        assert 1 <= start <= end <= len(world.fs.files[path].decode().split("\n"))
    session = {p: (s, e) for p, s, e in scoped}.get("src/auth/session.py")
    if session:
        assert covers_definition(world, "src/auth/session.py", *session, f"def {IDENT}")
    sem = await world.app.semantic_search(SemanticSearchRequest(query=QUERY, path_glob=glob, k=30))
    assert {h.path for h in sem.hits} == got
