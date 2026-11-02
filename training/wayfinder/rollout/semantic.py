"""Semantic search for wayfinder rollouts: scout's REAL secure-indexing path, namespaced per repo@commit.

Nothing of scout is copied or edited. `scout.demo.runtime.DemoRuntime` (real TurbopufferStore, real VoyageEmbedder
voyage-code-4, in-process `GatewayService`, Merkle sync, encrypted paths/line ranges, HMAC tokens) is subclassed here to
change two things only:

1. Namespaces. The demo maps every namespace to `scout_dev_demo_<name>`. Wayfinder tasks pin a base_commit, so the
   chunk/tree namespaces are keyed by repo AND commit, while the embed cache, keys and meta namespaces are keyed by repo
   (all commits of a repo share one team key, so an unchanged chunk is embedded once across commits):

       chunks  scout_dev_wf_<repo-slug>_<commit8>        tree  scout_dev_wf_<repo-slug>_<commit8>__tree
       cache   scout_dev_wf_<repo-slug>__cache           meta  scout_dev_wf_<repo-slug>__meta
       keys    scout_dev_wf_<repo-slug>__keys

   `<repo-slug>` = lowercase `owner__name` with non-alphanumerics folded to `_`; `<commit8>` = first 8 hex of base_commit.
2. Keys. The gateway master key is not a local file but `WF_INDEX_MASTER_KEY_HEX` (Modal secret `wayfinder-index-keys`;
   local copy ~/.scout-wayfinder/master.key). The per-repo team key is wrapped under it in `..__keys`, as in the product.

The checkout's `clone_id` (scout derives the branch namespace id from it) is made deterministic per repo@commit
(`clone_id_for`) and written into `.git/scout/state.json` before indexing AND before every rollout, so a fresh rollout
checkout addresses exactly the namespace the indexer wrote. Search uses scout's `ScoutApp.explore` unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PREFIX = "scout_dev_wf_"
MIN_COVERAGE = 0.8  # scout offers semantic_search only at >= this coverage


def slug(repo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", repo.replace("/", "__").lower()).strip("_")


def ns_base(repo: str, commit: str) -> str:
    return f"{PREFIX}{slug(repo)}_{commit[:8]}"


def clone_id_for(repo: str, commit: str) -> str:
    return hashlib.sha256(f"wf-clone:{repo}@{commit}".encode()).hexdigest()[:12]


def prepare_state(root: str, repo: str, commit: str) -> None:
    """Pin scout's clone_id for this checkout (idempotent; keeps an existing valid state.json's other fields)."""
    d = Path(root) / ".git" / "scout"
    (d / "trees").mkdir(parents=True, exist_ok=True)
    p = d / "state.json"
    cid = clone_id_for(repo, commit)
    try:
        cur = json.loads(p.read_text())
    except (OSError, ValueError):
        cur = {}
    if cur.get("clone_id") != cid:
        cur = {"schema": "1.0", "clone_id": cid}
        p.write_text(json.dumps(cur, sort_keys=True, separators=(",", ":")))


def load_env_keys() -> dict[str, str]:
    """Process env (Modal secret) first; on the laptop fall back to the same files the secret was built from."""
    env = dict(os.environ)
    if env.get("TURBOPUFFER_API_KEY") and env.get("VOYAGE_API_KEY") and env.get("WF_INDEX_MASTER_KEY_HEX"):
        return env
    from scout.demo.env import load_env

    env = load_env(base=os.environ)
    kf = Path.home() / ".scout-wayfinder" / "master.key"
    if not env.get("WF_INDEX_MASTER_KEY_HEX") and kf.exists():
        env["WF_INDEX_MASTER_KEY_HEX"] = kf.read_bytes().hex()
    return env


def _classes() -> Any:
    from scout.demo.runtime import DemoRuntime
    from scout.demo.store import _CHUNK, _SUFFIX, DemoNamespaceStore  # noqa: PLC2701

    from scout.core.contracts.errors import ErrorCode, ScoutError

    class WfNamespaceStore(DemoNamespaceStore):
        def __init__(self, inner: Any, repo: str, commit: str) -> None:
            super().__init__(inner, "wf")
            self._chunk_base = ns_base(repo, commit)
            self._family = f"{PREFIX}{slug(repo)}"

        def real(self, ns: str) -> str:
            tree = ns.endswith("__tree")
            core = ns[: -len("__tree")] if tree else ns
            if _CHUNK.match(core):
                return self._chunk_base + ("__tree" if tree else "")
            if not tree and core in _SUFFIX:
                return self._family + _SUFFIX[core]
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "namespace is outside the wayfinder family")

    import scout.demo.runtime as _demo_rt
    from scout.adapters.fs_local import LocalFileSystem as _LFS

    class TolerantFS(_LFS):
        """The demo's pre-index size scan stats every listed path and aborts on a dangling symlink / vanished file;
        scout's sync itself skips those. Same filesystem, `stat` of an unreadable path is size 0."""

        def stat(self, rel: str) -> Any:
            try:
                return super().stat(rel)
            except ScoutError:
                from scout.core.ports.fs import FileStat

                return FileStat(size=0, mtime_ns=0, ino=0, is_symlink=False)

    _demo_rt.LocalFileSystem = TolerantFS
    # scout's batcher can hand one chunk-key group (an identical chunk repeated in >500 files, e.g. a license header or
    # empty __init__) to a single EmbedReq with >MAX_FILES `replace` entries -> pydantic ValidationError. Capping rows per
    # group at 500 (<= MAX_FILES distinct files) makes its own flush rules hold; EmbedReq allows 2000 rows.
    import scout.core.sync.engine as _eng

    _eng.MAX_ROWS = 500  # wrap, do not edit, the demo module (its `index` builds its own fs)

    class WfRuntime(DemoRuntime):
        """DemoRuntime keyed by repo@commit, master key from the environment."""

        def __init__(self, repo: str, commit: str, *, env: Mapping[str, str], home: Path, log: Any) -> None:
            super().__init__(env=env, home=home, log=log)
            self.repo, self.commit = repo, commit

        def _master_key(self) -> bytes:
            key = bytes.fromhex(self.env["WF_INDEX_MASTER_KEY_HEX"])
            if len(key) != 32:
                raise ValueError("WF_INDEX_MASTER_KEY_HEX must be 32 bytes")
            return key

        def _gateway(self, name: str, chat: Any, embedder: Any) -> Any:
            from scout.core.service.gateway import GatewayService
            from scout.core.service.ratelimit import Limits, RateLimiter
            from scout.demo.store import KEYS_NS, TENANT

            store = WfNamespaceStore(self._real_store(), self.repo, self.commit)
            limits = Limits(embed_tokens_per_min=500_000_000.0, search_per_s=200.0, search_burst=400.0)
            svc = GatewayService(
                store, self.clock, self._master_key(), keys_namespace=KEYS_NS, embedder=embedder, chat=chat,
                limiter=RateLimiter(self.clock, limits),
            )
            return svc.for_tenant(TENANT)

    return WfRuntime


def runtime_for(repo: str, commit: str, home: str, log: Any = lambda _m: None) -> Any:
    return _classes()(repo, commit, env=load_env_keys(), home=Path(home), log=log)


def tpuf_bytes_for(client: Any, base_ns: str) -> dict[str, Any]:
    """Read-only post-write metadata of one repo@commit's tpuf namespaces: `{base_ns, base_ns__tree}` approx_logical_bytes/rows.
    Per-repo shared namespaces (__cache/__meta/__keys) are excluded (shared across commits). Never raises: failures give {}.
    Bytes are the namespace's logical size right after this index run, a proxy for bytes written (cheap: one metadata call each)."""
    out: dict[str, Any] = {}
    try:
        for tag, ns in (("chunks", base_ns), ("tree", base_ns + "__tree")):
            m = client.namespace(ns).metadata()
            out[f"tpuf_{tag}_bytes"] = int(getattr(m, "approx_logical_bytes", 0) or 0)
            out[f"tpuf_{tag}_rows"] = int(getattr(m, "approx_row_count", 0) or 0)
        out["tpuf_bytes"] = out["tpuf_chunks_bytes"] + out["tpuf_tree_bytes"]
    except Exception:  # noqa: BLE001
        return {}
    return out


def _tpuf_client(env: Mapping[str, str]) -> Any:
    import turbopuffer

    return turbopuffer.Turbopuffer(api_key=env["TURBOPUFFER_API_KEY"], region=env.get("TURBOPUFFER_REGION") or "gcp-us-central1", max_retries=2)


async def index_checkout(root: str, repo: str, commit: str, home: str, log: Any = lambda _m: None) -> dict[str, Any]:
    """Index the checkout at `root` (a wayfinder `checkout()` of repo@commit) into its wf namespaces."""
    prepare_state(root, repo, commit)
    rt = runtime_for(repo, commit, home, log)
    t0 = time.monotonic()
    rep = await rt.index(root, "wf")
    d = rep.to_dict()
    d.update(repo=repo, base_commit=commit, namespace=ns_base(repo, commit), seconds=round(time.monotonic() - t0, 1))
    for k in ("name", "root"):
        d.pop(k, None)
    try:  # record tpuf bytes for the cost tables (read-only metadata; failure must not fail indexing)
        import asyncio

        d.update(await asyncio.to_thread(tpuf_bytes_for, _tpuf_client(load_env_keys()), d["namespace"]))
    except Exception:  # noqa: BLE001
        pass
    return d


class SemanticGateway:
    """The real tenant gateway for everything (index_status, search, keys) except `complete`: chat goes straight to
    the rollout's `CompatChat`, exactly like the grep-only `DirectGateway`."""

    def __init__(self, inner: Any, chat: Any) -> None:
        self._inner = inner
        self._chat = chat

    async def complete(self, req: Any) -> Any:
        return await self._chat.complete(req)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SemanticStack:
    """Per-rollout semantic wiring over the real TurbopufferStore + VoyageEmbedder (query embeds only)."""

    def __init__(self, repo: str, commit: str, root: str, home: str = "/tmp/wf_home") -> None:
        self.repo, self.commit, self.root = repo, commit, root
        prepare_state(root, repo, commit)
        self.rt = runtime_for(repo, commit, home)
        self.embedder = None
        self._gw: Any = None

    def gateway(self) -> Any:
        if self._gw is None:
            from scout.demo.runtime import MeteredEmbedder

            self.embedder = MeteredEmbedder(self.rt._real_embedder(), lambda _m: None)  # noqa: SLF001
            self._gw = self.rt._gateway("wf", None, self.embedder)  # noqa: SLF001
        return self._gw

    async def status(self) -> Any:
        """IndexStatus of this checkout's namespace (absent when never indexed)."""
        from scout.core.app import ScoutApp

        app = ScoutApp(self.gateway(), _fs(self.root), self.rt.clock)
        return await app.index_status()

    def app(self, chat: Any, fs: Any, clock: Any, profile: Any) -> Any:
        from scout.core.app import ScoutApp

        return ScoutApp(SemanticGateway(self.gateway(), chat), fs, clock, profile=profile, backfill_to=0)  # backfill off: see run.py

    @property
    def query_tokens(self) -> int:
        return self.embedder.query_tokens if self.embedder else 0


def _fs(root: str) -> Any:
    from scout.adapters.fs_local import LocalFileSystem

    return LocalFileSystem(root)
