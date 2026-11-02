"""Task 6.8: the startup guard (design section 2, FR-S8, NFR-1, AC-6.3).

`python -X importtime -m scout mcp` is driven by the MCP client over stdio against the fake profile. Before
the first tool call no scout module may import a heavy dependency (`turbopuffer`, `voyageai`, `tree_sitter`,
`watchfiles`, `cryptography.hazmat`, `httpx`) and the app, adapters and wiring must not be loaded; then
`initialize` plus the first `index_status` call must take under one second.

The `mcp` SDK itself imports `watchfiles` (through `sse_starlette` and `uvicorn`) and `cryptography.hazmat`
(through its request-state signing) when it loads; scout cannot avoid that and it costs milliseconds. So the
import log is read as a tree and a forbidden module counts against scout only when scout's own code is its
nearest importer, that is, a `scout.*` module appears on its import chain before any `mcp.*` module does.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests" / "fixtures" / "poc_explore_case.json"
FORBIDDEN = ("turbopuffer", "voyageai", "tree_sitter", "watchfiles", "cryptography.hazmat", "httpx")
MUST_NOT_BE_LOADED = (
    "scout.core.app",
    "scout.wiring",
    "scout.adapters",
    "scout.core.sync",
    "scout.core.explore.loop",
)
BOUND_S = 1.0
ATTEMPTS = 3  # the bound is on the best of a few runs: one noisy sample on a busy machine is no regression


def _env(home: Path) -> dict[str, str]:
    return {
        **os.environ,
        "SCOUT_PROFILE": "fake",
        "SCOUT_FAKE_CASE": str(CASE),
        "SCOUT_KEYSTORE": "file",
        "HOME": str(home),
    }


def matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def import_chains(log: str) -> list[list[str]]:
    """Each module of an `-X importtime` log with its importers, innermost first (`[mod, parent, ..., root]`).

    A module's line is printed after its children and indented two spaces per nesting level, so its
    importer is the next later line one level up."""
    rows: list[tuple[int, str]] = []
    for line in log.splitlines():
        parts = line.split("|")
        if not line.startswith("import time:") or len(parts) < 3 or not parts[1].strip().isdigit():
            continue
        name = parts[2]
        rows.append(((len(name) - len(name.lstrip(" "))) // 2, name.strip()))
    chains: list[list[str]] = []
    for i, (depth, name) in enumerate(rows):
        chain, want = [name], depth - 1
        for d, n in rows[i + 1 :]:
            if d == want:
                chain.append(n)
                want -= 1
                if want < 0:
                    break
        chains.append(chain)
    return chains


def scout_caused(chains: list[list[str]]) -> list[str]:
    """The forbidden modules scout's own code imports eagerly, as `module <- importer <- ...` strings.

    Walking up a chain from the module: a `scout.*` importer first means scout did it; an `mcp.*` importer
    first means the SDK did it as part of loading itself (not counted)."""
    out: set[str] = set()
    for chain in chains:
        if not any(matches(chain[0], f) for f in FORBIDDEN):
            continue
        for parent in chain[1:]:
            if parent.startswith("scout"):
                out.add(" <- ".join(chain[:6]))
                break
            if matches(parent, "mcp"):
                break
    return sorted(out)


def _read(path: str) -> str:
    return Path(path).read_text()


async def first_call(*, importtime: bool, home: Path, log_path: Path) -> tuple[float, str, bool]:
    """`(seconds to initialize + first index_status, import log written before that call, call succeeded)`.

    With `importtime` the log is read once `scout.surfaces.mcp.server` has finished importing (the line is
    printed after its children, so everything it pulled in is already there) and before any tool runs."""
    args = ["-X", "importtime"] if importtime else []
    params = StdioServerParameters(command=sys.executable, args=[*args, "-m", "scout", "mcp"], env=_env(home))
    before = ""
    with log_path.open("w+") as log:
        started = time.monotonic()
        async with Client(stdio_client(params, errlog=log)) as client:
            waited = 0.0
            if importtime:
                t = time.monotonic()
                for _ in range(100):
                    log.flush()
                    before = await asyncio.to_thread(_read, str(log_path))
                    if "scout.surfaces.mcp.server" in before:
                        break
                    await asyncio.sleep(0.05)
                waited = time.monotonic() - t
            result = await client.call_tool("index_status", {})
            elapsed = time.monotonic() - started - waited
    return elapsed, before, not result.is_error


async def test_no_heavy_import_before_the_first_tool_call(tmp_path: Path) -> None:
    """AC-6.3: scout's own startup path imports none of the heavy modules, and not the app or the adapters."""
    _, log, ok = await first_call(importtime=True, home=tmp_path, log_path=tmp_path / "import.log")
    assert ok
    chains = import_chains(log)
    loaded = {c[0] for c in chains}
    assert any(matches(m, "scout.surfaces.mcp.server") for m in loaded), "the log is not the server's startup"
    assert any(matches(m, "mcp.server.mcpserver") for m in loaded)

    eager = scout_caused(chains)
    assert not eager, "scout imports heavy modules at startup:\n" + "\n".join(eager)
    # the strictest of them are absent outright, not merely not scout's doing
    for module in ("turbopuffer", "voyageai", "tree_sitter", "httpx"):
        assert not any(matches(m, module) for m in loaded), f"{module} was imported at startup"

    pulled = sorted(m for m in loaded if any(matches(m, bad) for bad in MUST_NOT_BE_LOADED))
    assert not pulled, f"the app side is loaded before the first call: {pulled}"


def test_the_guard_sees_an_eager_import() -> None:
    """The scanner is itself tested: scout importing a heavy module is reported, the SDK doing it is not."""
    log = (
        "import time:       100 |        100 |       httpx\n"
        "import time:       100 |        300 |     scout.leaky\n"
        "import time:       100 |        100 |             cryptography.hazmat\n"
        "import time:       100 |        200 |           mcp.server.request_state\n"
        "import time:       100 |        200 |         mcp.server.mcpserver\n"
        "import time:       100 |        200 |       scout.surfaces.mcp.server\n"
        "import time:       100 |        100 |   watchfiles\n"
        "import time:       100 |        200 | scout.adapters.fs_local.watch\n"
    )
    flagged = scout_caused(import_chains(log))
    assert len(flagged) == 2
    assert flagged[0].startswith("httpx <- scout.leaky")
    assert flagged[1].startswith("watchfiles <- scout.adapters.fs_local.watch")
    assert not any("cryptography" in f for f in flagged)  # reached through `mcp.*` first: the SDK's doing


async def test_initialize_and_first_call_take_under_a_second(tmp_path: Path) -> None:
    """NFR-1, AC-6.3: `initialize` + the first `index_status` < 1 s on the fake profile (best of three)."""
    timings = []
    for i in range(ATTEMPTS):
        elapsed, _, ok = await first_call(
            importtime=False, home=tmp_path / str(i), log_path=tmp_path / f"{i}.log"
        )
        assert ok
        timings.append(elapsed)
        if elapsed < BOUND_S:
            break
    assert min(timings) < BOUND_S, (
        f"initialize + first call took {min(timings):.2f}s (bound {BOUND_S}s): {timings}"
    )
