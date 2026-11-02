"""VectorStore port (design section 3). GATEWAY side only: the client process never touches it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class Row:
    id: str  # <= 64 bytes
    attrs: Mapping[str, Any] = field(default_factory=dict[str, Any])
    vector: Sequence[float] | None = None
    score: float | None = None  # set on query results (higher = better)


@dataclass(frozen=True, slots=True)
class Eq:
    attr: str
    value: str | int | bool


@dataclass(frozen=True, slots=True)
class In:
    attr: str
    values: Sequence[str]


@dataclass(frozen=True, slots=True)
class Glob:
    attr: str  # must be schema glob=True
    pattern: str


@dataclass(frozen=True, slots=True)
class Contains:
    attr: str  # attr is []string
    value: str


@dataclass(frozen=True, slots=True)
class And:
    parts: Sequence["Filter"]


@dataclass(frozen=True, slots=True)
class Not:
    part: "Filter"


# Filter AST: the adapter translates it to turbopuffer tuples; FakeStore evaluates it.
type Filter = Eq | In | Glob | Contains | And | Not


@dataclass(frozen=True, slots=True)
class AttrSpec:
    type: Literal["string", "int", "[]string", "bool"]
    filterable: bool = True
    glob: bool = False
    fts: Literal["none", "pre_tokenized_array"] = "none"


@dataclass(frozen=True, slots=True)
class NamespaceSchema:
    dims: int | None  # None = vector-less namespace (tree, meta, keys); int = vector (chunks, cache)
    attrs: Mapping[str, AttrSpec]
    distance: Literal["cosine_distance"] = "cosine_distance"


class VectorStore(Protocol):
    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        """Idempotent create. Existing ns with different schema -> INTERNAL. `dims=None` creates a
        vector-less namespace (added after spike 0a-A; the only port change for the tree split)."""
        ...

    async def namespace_exists(self, ns: str) -> bool: ...

    async def list_namespaces(self, prefix: str) -> list[str]: ...

    async def delete_namespace(self, ns: str) -> None:
        """Idempotent (missing ns is OK). Used by branch GC, tests, janitor."""
        ...

    async def branch_from(self, ns: str, source: str) -> None:
        """Copy-on-write clone of ONE namespace. Schema and rows inherited. source missing -> NOT_FOUND;
        ns exists -> CONFLICT. Edits to ns never touch source. Callers copy `{ns}__tree` before `{ns}` (5)."""
        ...

    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        """Returns rows affected. if_absent=True: insert-if-absent per id (race-safe cache writes);
        returns count actually inserted. `condition` (turbopuffer `upsert_condition`): evaluated per row
        against the EXISTING row with the same id; a row whose existing version fails it is skipped (not
        counted), a row with no existing version always passes (the upsert proceeds). The condition's
        attributes must be filterable. A conditional single-row upsert is the cross-process
        compare-and-set (affected == 0 means the CAS lost). A row without `vector` is allowed only in a
        vector-less namespace
        (dims=None); in a vector namespace it is INVALID_ARGUMENT, as turbopuffer returns HTTP 400
        (spike 0a-A). A row with `vector` in a vector-less namespace is INVALID_ARGUMENT too (port-level
        guard so a tree/meta namespace never becomes a vector namespace; not verified on turbopuffer)."""
        ...

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int: ...

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        """delete_by_filter(file_id==X [AND file_hash!=except]). Returns rows deleted. Missing ns -> 0."""
        ...

    async def delete_by_filter(self, ns: str, flt: Filter) -> int: ...

    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        """Missing ids omitted. <= 5,000 ids/call (adapter chunks)."""
        ...

    async def query_ann(
        self,
        ns: str,
        vector: Sequence[float],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]: ...

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
        """field uses pre_tokenized_array; tokens are the exact pre-hashed strings (OR-ranked).
        No stemming, case-sensitive. Missing ns -> NOT_FOUND.
        Errors: UPSTREAM_UNAVAILABLE, UPSTREAM_TIMEOUT."""
        ...
