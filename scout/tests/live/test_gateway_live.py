"""Task 8.8: the deployed gateway, end to end, from a client that holds only a scout key.

FR-G1, FR-G2, AC-9.2, AC-14.6. A scratch HOME and the production default URL (`DEFAULT_GATEWAY_URL`, or
`SCOUT_GATEWAY_URL`): `scout login` equivalent (`ScoutApp.login`), healthz and whoami, a stable team key,
a full `sync` of a small PUBLIC repo through the real `HttpGateway`, `list_namespaces` on real turbopuffer
showing `{ns}` and `{ns}__tree`, `semantic_search` returning a verified range, an `explore` whose chat goes
through the gateway to the deployed A1-35B, then tenant isolation with a second key, and cleanup in `finally`.

Keys: env `SCOUT_LIVE_DEV_KEY` / `SCOUT_LIVE_EVAL_KEY`, else the 0600 file
`~/.config/scout/live-smoke-keys.txt` (`dev=...` / `eval=...` lines, written by the key-issuing step). The
test needs no vendor key except TURBOPUFFER_API_KEY, which only the namespace assertions and the cleanup
use. Skips (not fails) without keys. Writes `artifacts/live/gateway_smoke.json` (no key material). Never
prints a key.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from scout.adapters.gateway_http import HttpGateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import Budget, ExploreRequest, SemanticSearchRequest
from scout.core.contracts.wire import NodesReq, PlanReq, SearchReq, StatusReq
from scout.core.crypto import ids
from scout.core.sync.state import StateDir
from scout.demo.env import load_env
from scout.wiring import DEFAULT_GATEWAY_URL, Config, build_app

pytestmark = [pytest.mark.live]

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "live" / "gateway_smoke.json"
KEY_FILE = Path.home() / ".config" / "scout" / "live-smoke-keys.txt"
PUBLIC_REPO = os.environ.get("SCOUT_LIVE_REPO_URL", "https://github.com/gorilla/mux")
QUERY = "where does the router match an incoming request to a route and extract path variables"


def _keys() -> dict[str, str]:
    out = {
        "dev": os.environ.get("SCOUT_LIVE_DEV_KEY", ""),
        "eval": os.environ.get("SCOUT_LIVE_EVAL_KEY", ""),
    }
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name in out and value and not out[name]:
                out[name] = value.strip()
    return out


@pytest.fixture(scope="module")
def keys() -> dict[str, str]:
    k = _keys()
    if not (k["dev"] and k["eval"]):
        pytest.skip(
            "no live gateway keys (SCOUT_LIVE_DEV_KEY / SCOUT_LIVE_EVAL_KEY or the live-smoke-keys file)"
        )
    return k


@pytest.fixture(scope="module")
def tpuf() -> Any:
    env = load_env()
    if not env.get("TURBOPUFFER_API_KEY"):
        pytest.skip("TURBOPUFFER_API_KEY is not set")
    from scout.adapters.turbopuffer import TurbopufferStore

    return TurbopufferStore.from_env(env)


def _clone(dest: Path) -> Path:
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", "--quiet", PUBLIC_REPO, str(dest)],  # noqa: S607
        check=True,
        capture_output=True,
        timeout=180,
    )
    return dest


async def test_deployed_gateway_end_to_end_and_isolation(
    tmp_path: Path, keys: dict[str, str], tpuf: Any
) -> None:
    url = os.environ.get("SCOUT_GATEWAY_URL") or DEFAULT_GATEWAY_URL
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "SCOUT_KEYSTORE": "file"}
    repo = _clone(tmp_path / "repo")
    result: dict[str, Any] = {"gateway_url": url, "repo": PUBLIC_REPO}
    dev = HttpGateway(url, keys["dev"])
    eval_gw = HttpGateway(url, keys["eval"])
    created: list[str] = []
    app = build_app(Config(root=str(repo), gateway_url=url), env)
    try:
        # --- liveness and identity -------------------------------------------------------------
        t0 = time.monotonic()
        await dev.ping()
        result["first_request_s"] = round(time.monotonic() - t0, 2)  # includes a cold container start
        who = await dev.whoami()
        assert who.tenant == "dev"
        k1, k2 = await dev.team_key(), await dev.team_key()
        assert k1 == k2
        assert len(k1) == 64
        assert (await eval_gw.whoami()).tenant == "eval"
        assert await eval_gw.team_key() != k1

        # --- login as a user would, then sync a public repo through the gateway --------------------
        await app.login(keys["dev"])
        assert (home / ".config" / "scout" / "credentials").stat().st_mode & 0o777 == 0o600
        t0 = time.monotonic()
        status = await app.sync()
        result["sync_s"] = round(time.monotonic() - t0, 1)
        assert (status.state, status.coverage) == ("ready", 1.0)
        assert status.files > 20
        result["files"] = status.files

        state = StateDir.open(app._fs).read_state()  # pyright: ignore[reportPrivateUsage]
        assert state.repo_id
        assert state.last_branch
        repo_id, branch_id = state.repo_id, state.last_branch
        ns = ids.namespace("dev", repo_id, branch_id)
        tree = ids.tree_namespace(ns)
        created += [ns, tree, "scout_dev_cache", "scout_dev_meta"]
        listed = await tpuf.list_namespaces("scout_dev_")
        assert ns in listed
        assert tree in listed
        assert ns.startswith("scout_dev_")

        # --- semantic search and explore -------------------------------------------------------
        sem = await app.semantic_search(SemanticSearchRequest(query=QUERY, k=5))
        assert sem.hits
        for h in sem.hits:
            lines = (repo / h.path).read_text(errors="replace").splitlines()
            assert 1 <= h.start_line <= h.end_line <= len(lines)
        result["semantic_top"] = [f"{h.path}:{h.start_line}-{h.end_line}" for h in sem.hits[:3]]
        t0 = time.monotonic()
        explored = await app.explore(
            ExploreRequest(brief=QUERY, budget=Budget(max_turns=25, max_seconds=300))
        )
        result["explore_s"] = round(time.monotonic() - t0, 1)
        result["explore_hits"] = [
            f"{h.path}:{h.start_line}-{h.end_line} ({h.source})" for h in explored.results
        ]
        result["explore_usage"] = {
            "turns": explored.usage.turns,
            "tool_calls": explored.usage.tool_calls,
            "tokens_in": explored.usage.model_tokens.in_,
            "tokens_out": explored.usage.model_tokens.out,
        }
        assert explored.results, explored.gaps
        assert explored.usage.turns >= 1
        assert explored.index.semantic_enabled
        for h in explored.results:
            lines = (repo / h.path).read_text(errors="replace").splitlines()
            assert 1 <= h.start_line <= h.end_line <= len(lines)

        # --- tenant isolation: the second key sees nothing of dev's index ------------------------------
        sreq = StatusReq(repo_id=repo_id, branch_id=branch_id)
        assert (await eval_gw.index_status(sreq)).state == "absent"
        assert (await dev.index_status(sreq)).state == "ready"
        blobs = (
            await eval_gw.sync_nodes(NodesReq(repo_id=repo_id, branch_id=branch_id, ids=[repo_id]))
        ).blobs
        assert blobs == {}
        with pytest.raises(ScoutError) as ei:
            await eval_gw.search(SearchReq(repo_id=repo_id, branch_id=branch_id, query=QUERY))
        assert ei.value.code is ErrorCode.NOT_FOUND
        miss = await eval_gw.sync_plan(PlanReq(repo_id=repo_id, branch_id=branch_id, chunk_keys=[repo_id]))
        assert miss.missing == [repo_id]
        # The eval tenant owns no namespace for dev's repo (other eval work may legitimately be running).
        assert not [n for n in await tpuf.list_namespaces("scout_eval_") if repo_id in n]
        stranger = HttpGateway(url, "sk_scout_" + "Z" * 43)
        with pytest.raises(ScoutError) as ei2:
            await stranger.whoami()
        assert ei2.value.code is ErrorCode.UNAUTHENTICATED
        result["isolation"] = (
            "eval key: index absent, no node blobs, search NOT_FOUND, no scout_eval_ namespace for the repo"
        )

        # --- delete_branch removes both namespaces ---------------------------------------------------
        await dev.delete_branch(repo_id, branch_id)
        after = await tpuf.list_namespaces("scout_dev_")
        assert ns not in after
        assert tree not in after
        result["passed"] = True
    finally:
        for n in created:
            try:
                await tpuf.delete_namespace(n)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("cleanup_errors", []).append(f"{n}: {type(exc).__name__}")
        await asyncio.gather(dev.aclose(), eval_gw.aclose())
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
