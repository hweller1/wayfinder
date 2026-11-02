"""Shared fixtures for the service contract tests: a write-logging `RecordingStore`, `Clone` (one teammate
checkout with the real keyed Merkle tree, real MinHash and real sync calls) and small file/namespace helpers.
Moved out of `test_service_seed.py` (task 3.9) so the tenant-isolation suite reuses them."""

import base64
import copy
import hashlib
from collections.abc import Sequence
from typing import Any

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodeUpsert,
    ReplaceFile,
    SearchReq,
    SeedReq,
    SyncBeginReq,
    WireRow,
)
from scout.core.crypto import ids, keys, minhash, paths
from scout.core.ports.store import Filter, NamespaceSchema, Row
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.schema import ROOT_ID
from scout.core.service.syncsvc import meta_namespace, proofs_row_id
from scout.core.sync import merkle

MASTER = b"m" * 32
REPO = "a" * 32
SRC_BR = "b" * 16
NEW_BR = "c" * 16
Call = tuple[str, ...]


class RecordingStore(FakeStore):
    """Logs every write/copy; can fail one call once and run a hook after a call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.touched: list[tuple[str, str]] = []  # (op, namespace) of EVERY store call, reads included
        self.fail_once: dict[Call, Exception] = {}
        self.after: dict[Call, Any] = {}

    async def _enter(self, *call: str) -> None:
        self.touched.append((call[0], call[1]))
        if call[0] == "branch_from":
            self.touched.append((call[0], call[2]))
        self.calls.append(call)
        err = self.fail_once.pop(call, None)
        if err is not None:
            raise err

    async def _exit(self, *call: str) -> None:
        hook = self.after.pop(call, None)
        if hook is not None:
            await hook()

    async def branch_from(self, ns: str, source: str) -> None:
        await self._enter("branch_from", ns, source)
        await super().branch_from(ns, source)
        await self._exit("branch_from", ns, source)

    async def upsert(self, ns: str, rows: Sequence[Row], **kw: Any) -> int:
        label = ("upsert", ns, ",".join(sorted({r.id.split("-")[0] for r in rows})))
        await self._enter(*label)
        out = await super().upsert(ns, rows, **kw)
        await self._exit(*label)
        return out

    async def delete_namespace(self, ns: str) -> None:
        await self._enter("delete_namespace", ns)
        await super().delete_namespace(ns)

    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        self.touched.append(("ensure_namespace", ns))
        await super().ensure_namespace(ns, schema)

    async def namespace_exists(self, ns: str) -> bool:
        self.touched.append(("namespace_exists", ns))
        return await super().namespace_exists(ns)

    async def list_namespaces(self, prefix: str) -> list[str]:
        self.touched.append(("list_namespaces", prefix))  # a prefix, not a namespace name
        return await super().list_namespaces(prefix)

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        self.touched.append(("delete_ids", ns))
        return await super().delete_ids(ns, ids)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        self.touched.append(("delete_by_filter", ns))
        return await super().delete_by_filter(ns, flt)

    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        self.touched.append(("get_by_ids", ns))
        return await super().get_by_ids(ns, ids, include_vectors=include_vectors, attrs=attrs)

    async def query_ann(
        self,
        ns: str,
        vector: Sequence[float],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        self.touched.append(("query_ann", ns))
        return await super().query_ann(ns, vector, top_k=top_k, flt=flt, attrs=attrs)

    async def query_bm25(
        self,
        ns: str,
        field: str,
        tokens: Sequence[str],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        self.touched.append(("query_bm25", ns))
        return await super().query_bm25(ns, field, tokens, top_k=top_k, flt=flt, attrs=attrs)

    def ops(self, *names: str) -> list[Call]:
        return [c for c in self.calls if c[0] in names]

    def rows(self, ns: str) -> dict[str, Any]:
        cur = self._ns.get(ns)  # pyright: ignore[reportPrivateUsage]
        return cur.rows if cur is not None else {}

    def names(self) -> list[str]:
        """Every namespace name (not logged), sorted."""
        return sorted(self._ns)  # pyright: ignore[reportPrivateUsage]

    def dump(self, ns: str) -> dict[str, Any]:
        """Deep copy of one namespace's full contents: schema, vector-ness and every row (attrs + vector)."""
        cur = self._ns[ns]  # pyright: ignore[reportPrivateUsage]
        return {
            "schema": cur.schema,
            "has_vector": cur.has_vector,
            "rows": {i: (copy.deepcopy(r.attrs), r.vector) for i, r in sorted(cur.rows.items())},
        }


def ns_pair(tenant: str, branch: str, repo: str = REPO) -> tuple[str, str]:
    ns = ids.namespace(tenant, repo, branch)
    return ns, ids.tree_namespace(ns)


def make(
    store: RecordingStore | None = None, **kw: Any
) -> tuple[RecordingStore, GatewayService, FakeEmbedder]:
    store, emb = store or RecordingStore(), FakeEmbedder()
    return store, GatewayService(store, FakeClock(), MASTER, embedder=emb, **kw), emb


def src(n: int, salt: str = "s") -> dict[str, str]:
    return {f"src/m{i}.py": f"def fn_{salt}_{i}():\n    return {i}\n" for i in range(n)}


def edited(files: dict[str, str], count: int) -> dict[str, str]:
    """`files` with the first `count` files rewritten (so those leaves differ)."""
    out = dict(files)
    for p in sorted(out)[:count]:
        out[p] = out[p] + "# edited\n"
    return out


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class Clone:
    """One teammate checkout of one tenant: real keyed tree, real MinHash, real sync calls."""

    def __init__(self, svc: GatewayService, store: RecordingStore, tenant: str, branch: str, k: keys.Subkeys):
        self.gw: TenantGateway = svc.for_tenant(tenant)
        self.store, self.tenant, self.branch, self.k = store, tenant, branch, k
        self.ns, self.tree_ns = ns_pair(tenant, branch)

    @classmethod
    async def create(
        cls, svc: GatewayService, store: RecordingStore, tenant: str = "acme", branch: str = NEW_BR
    ) -> "Clone":
        gw = svc.for_tenant(tenant)
        return cls(svc, store, tenant, branch, keys.derive(await gw.team_key()))

    def tree(self, files: dict[str, str]) -> merkle.Tree:
        leaves = [(p, merkle.leaf_hash(self.k, t.encode())) for p, t in files.items()]
        return merkle.build_tree(leaves, self.k, self.tenant, REPO)

    def leaves(self, files: dict[str, str]) -> list[str]:
        return sorted(self.tree(files).leaves.values())

    def sketch_b64(self, leaves: Sequence[str]) -> str:
        return base64.b64encode(minhash.sketch(self.k, [bytes.fromhex(x) for x in leaves])).decode()

    def seed_req(self, files: dict[str, str], **kw: Any) -> SeedReq:
        leaves = self.leaves(files)
        return SeedReq(
            repo_id=REPO,
            branch_id=self.branch,
            minhash_b64=kw.pop("minhash_b64", self.sketch_b64(leaves)),
            proofs=kw.pop("proofs", leaves),
            **kw,
        )

    def remote_nodes(self, node_ids: Sequence[str]) -> dict[str, merkle.SealedNode]:
        rows = self.store.rows(self.tree_ns)
        return {
            i: merkle.SealedNode(rows[i].attrs["children_enc"], rows[i].attrs["node_hash"])
            for i in node_ids
            if i in rows and rows[i].attrs.get("kind") == "node"
        }

    async def begin(self, tree: merkle.Tree, base: str | None = None) -> str | None:
        res = await self.gw.sync_begin(
            SyncBeginReq(
                repo_id=REPO,
                branch_id=self.branch,
                base_branch_id=base,
                root=tree.root_hash,
                expected_chunks=len(tree.leaves),
            )
        )
        return res.server_root

    def wire_row(self, path: str, text: str) -> WireRow:
        fid, ck = ids.file_id(self.k, path), text_key(text)
        return WireRow(
            id=ids.row_id(self.k, fid, ck, 0),
            file_id=fid,
            path_enc=paths.enc_path(self.k, self.tenant, REPO, path),
            ext_tag=ids.ext_tag(self.k, ".py"),
            dir_tags=[ids.dir_tag(self.k, "src")],
            file_hash=merkle.leaf_hash(self.k, text.encode()),
            chunk_key=ck,
            loc_enc="bG9j",
            tokens=["t" * 32],
        )

    async def sync(self, files: dict[str, str], *, base: str | None = None) -> merkle.Delta:
        """begin, diff against the server tree, embed the delta, commit with this clone's sketch."""
        tree = self.tree(files)
        server_root = await self.begin(tree, base)
        delta = merkle.diff(tree, server_root, self.remote_nodes, self.k)
        todo = [*delta.added, *delta.changed]
        rows = [self.wire_row(p, files[p]) for p in todo]
        await self.gw.embed(
            EmbedReq(
                repo_id=REPO,
                branch_id=self.branch,
                texts={text_key(files[p]): files[p] for p in todo},
                rows=rows,
                replace=[ReplaceFile(file_id=r.file_id, file_hash=r.file_hash) for r in rows],
                delete_files=[ids.file_id(self.k, p) for p in delta.removed],
            )
        )
        sealed = merkle.seal_tree(tree, self.k)
        stored = {i for i, r in self.store.rows(self.tree_ns).items() if r.attrs.get("kind") == "node"}
        await self.gw.sync_commit(
            CommitReq(
                repo_id=REPO,
                branch_id=self.branch,
                expected_root=server_root,
                new_root=tree.root_hash,
                node_upserts=[
                    NodeUpsert(id=i, node_hash=n.node_hash, children_enc=n.children_enc)
                    for i, n in sealed.items()
                ],
                node_deletes=sorted(stored - sealed.keys()),
                written_chunks=len(tree.leaves),
                expected_chunks=len(tree.leaves),
                minhash_b64=self.sketch_b64(sorted(tree.leaves.values())),
            )
        )
        return delta

    async def hits(self, k: int = 30) -> set[str]:
        res = await self.gw.search(SearchReq(repo_id=REPO, branch_id=self.branch, query="fn", k=k))
        return {h.file_hash for h in res.hits}

    def file_hashes(self, files: dict[str, str]) -> set[str]:
        return set(self.leaves(files))


async def source_clone(
    svc: GatewayService, store: RecordingStore, files: dict[str, str], tenant: str = "acme"
) -> Clone:
    c = await Clone.create(svc, store, tenant, SRC_BR)
    await c.sync(files)
    return c


def server_root(store: RecordingStore, tree_ns: str) -> str | None:
    row = store.rows(tree_ns).get(ROOT_ID)
    return row.attrs["root_hash"] if row else None


def proofs_row(store: RecordingStore, tenant: str, ns: str) -> Row | None:
    return store.rows(meta_namespace(tenant)).get(proofs_row_id(ns))
