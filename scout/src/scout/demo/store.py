"""`DemoNamespaceStore`: maps the gateway's namespace names onto the demo's `scout_dev_demo_<name>` family.

The real `GatewayService` names its namespaces `scout_{tenant}_{repo_id}_{branch_id}` (+ `__tree`) and
`scout_{tenant}_cache` / `scout_{tenant}_meta`. The demo runs one tenant (`dev`) and one namespace family per
repo name, so the owner can find, query and delete each repo's data by a readable name:

    chunks  scout_dev_demo_<name>            tree   scout_dev_demo_<name>__tree
    cache   scout_dev_demo_<name>__cache     meta   scout_dev_demo_<name>__meta     keys  ..._<name>__keys

Every name that is not one of those shapes is refused, so nothing outside `scout_dev_demo_<name>` can be
touched through this store. The wrapped store is the real turbopuffer adapter in the demo and a fake in tests.
"""

import re
from collections.abc import Sequence

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import Filter, NamespaceSchema, Row, VectorStore

TENANT = "dev"
KEYS_NS = "scout_dev_keys"  # the `keys_namespace` handed to GatewayService
PREFIX = "scout_dev_demo_"
_CHUNK = re.compile(r"^scout_dev_[0-9a-f]{32}_[0-9a-f]{16}$")
_SUFFIX = {
    "scout_dev_cache": "__cache",
    "scout_dev_meta": "__meta",
    KEYS_NS: "__keys",
}


def namespace_for(name: str) -> str:
    return f"{PREFIX}{name}"


class DemoNamespaceStore:
    def __init__(self, inner: VectorStore, name: str) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,48}", name):
            raise ValueError("demo repo name must match [a-z0-9_]{1,48}")
        self._inner = inner
        self._base = namespace_for(name)

    def real(self, ns: str) -> str:
        tree = ns.endswith("__tree")
        core = ns[: -len("__tree")] if tree else ns
        if _CHUNK.match(core):
            return self._base + ("__tree" if tree else "")
        if not tree and core in _SUFFIX:
            return self._base + _SUFFIX[core]
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "namespace is outside the demo family")

    # --- VectorStore -----------------------------------------------------------------------
    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        await self._inner.ensure_namespace(self.real(ns), schema)

    async def namespace_exists(self, ns: str) -> bool:
        return await self._inner.namespace_exists(self.real(ns))

    async def list_namespaces(self, prefix: str) -> list[str]:
        return []  # teammate seeding scans other branches' namespaces; the demo has none

    async def delete_namespace(self, ns: str) -> None:
        await self._inner.delete_namespace(self.real(ns))

    async def branch_from(self, ns: str, source: str) -> None:
        await self._inner.branch_from(self.real(ns), self.real(source))

    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        return await self._inner.upsert(self.real(ns), rows, if_absent=if_absent, condition=condition)

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        return await self._inner.delete_ids(self.real(ns), ids)

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        return await self._inner.delete_by_file(self.real(ns), file_id, except_file_hash=except_file_hash)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        return await self._inner.delete_by_filter(self.real(ns), flt)

    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        return await self._inner.get_by_ids(self.real(ns), ids, include_vectors=include_vectors, attrs=attrs)

    async def query_ann(
        self,
        ns: str,
        vector: Sequence[float],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        return await self._inner.query_ann(self.real(ns), vector, top_k=top_k, flt=flt, attrs=attrs)

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
        return await self._inner.query_bm25(self.real(ns), field, tokens, top_k=top_k, flt=flt, attrs=attrs)
