"""`RecordingStore` for the security byte scan (task 4.7, design 9 "Security byte scan"; AC-12.1).

A `FakeStore` that records EVERY call it receives: the operation, namespace names (chunk, `{ns}__tree`, cache,
meta, keys), every row (id and attributes), every filter (the whole `Eq/In/Glob/Contains/And/Not` tree), BM25
field and token list, id lists, list prefixes and the schema attribute names. Every `VectorStore` method is
overridden (`test_every_store_method_is_recorded` fails when the port grows a method this class forgets).

Three views of the same traffic:
- `buffer`: canonical JSON (sorted keys, no ASCII escaping), one document per call, newline separated.
- `values`: every DATA string, decoded (no JSON escaping, so a newline of leaked source stays one character),
  mapped to the dict keys it appeared under (`ns`, `id`, `tokens`, `path_enc`, ...).
- `structural`: store vocabulary that is not data (operation names, a row's `kind`, attribute names in
  filters and projections, everything inside a schema), kept apart so a scan for a source identifier such as
  `string` does not trip over `{"type": "string"}`.
- `keys`: every dict key; `vocabulary()` plus the schemas' attribute names is the closed set it must stay in.

Vectors are not text: only their dimension is recorded (`vector_dims`). A float array cannot carry a path or
an identifier; the residual risk that embeddings leak content is stated in PRD section 5 and is not what
this gate checks. The text handed to the `Embedder` is not recorded: the gateway intentionally sees it, and
no store method has a text parameter, so the only way it could reach the store is inside a row attribute,
which this recorder captures (`attr_names` lets a test assert no `text` attribute ever exists).

`row_hook` runs on each upserted row BEFORE it is recorded and forwarded; the negative controls use it to
inject a leak and prove the scanner sees it.
"""

import dataclasses
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scout.adapters.fake_store import FakeStore
from scout.core.ports.store import (
    And,
    AttrSpec,
    Contains,
    Eq,
    Filter,
    Glob,
    In,
    NamespaceSchema,
    Not,
    Row,
    VectorStore,
)


def vocabulary() -> frozenset[str]:
    """Every dict key the recorder itself can emit: port parameter names, filter/row/spec field names and the
    fixed document fields. Row attribute names are the only other keys (compare them to the schemas)."""
    names: set[str] = {"op", "vector_dims", "dims", "schema", "ns", "flt"}
    for cls in (Eq, In, Glob, Contains, And, Not, AttrSpec, Row):
        names.update(f.name for f in dataclasses.fields(cls))
    for method in dir(VectorStore):
        if not method.startswith("_"):
            names.update(inspect.signature(getattr(VectorStore, method)).parameters)
    names.discard("self")
    return frozenset(names)


def _plain(value: Any) -> Any:
    """JSON-able form of one argument; vectors become their dimension, filters keep their operator name."""
    if isinstance(value, Row):
        return {
            "id": value.id,
            "attrs": {k: _plain(v) for k, v in value.attrs.items()},
            "vector_dims": None if value.vector is None else len(value.vector),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"op": type(value).__name__}
        for f in dataclasses.fields(value):
            out[f.name] = _plain(getattr(value, f.name))
        return out
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


class RecordingStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self._lines: list[bytes] = []
        self._values: dict[str, set[str]] = {}  # every data string -> the dict keys it appeared under
        self.structural: set[str] = (
            set()
        )  # store vocabulary strings (ops, kinds, schema contents, attr names)
        self.keys: set[str] = set()  # every dict key of every call (attribute and field names)
        self.ops: list[str] = []
        self.attr_names: set[str] = set()  # every attribute name of every row ever upserted
        self.row_hook: Callable[[str, Row], Row] | None = None

    @property
    def buffer(self) -> bytes:
        """Everything the store received, one canonical JSON document per call, newline separated."""
        return b"\n".join(self._lines)

    @property
    def values(self) -> dict[str, set[str]]:
        """Every distinct data string any call carried (no JSON escaping) -> the keys it appeared under."""
        return self._values

    def _collect(self, node: Any, *, structural: bool = False, key: str | None = None) -> None:
        if isinstance(node, dict):
            kind_filter = node.get("op") in ("Eq", "In") and node.get("attr") == "kind"  # pyright: ignore[reportUnknownMemberType]
            for k, v in node.items():  # pyright: ignore[reportUnknownVariableType]
                name = str(k)  # pyright: ignore[reportUnknownArgumentType]
                self.keys.add(name)
                vocab = (
                    structural
                    or name in ("op", "attr", "field", "schema")
                    or (name == "kind")
                    or (kind_filter and name in ("value", "values"))
                )
                self._collect(v, structural=vocab, key=name)
        elif isinstance(node, list):
            vocab = structural or key == "attrs"  # `attrs=[...]` is a projection; a row's `attrs` is a dict
            for v in node:  # pyright: ignore[reportUnknownVariableType]
                self._collect(v, structural=vocab, key=key)
        elif isinstance(node, str):
            if structural:
                self.structural.add(node)
            else:
                self._values.setdefault(node, set()).add(key or "")

    def _rec(self, op: str, **fields: Any) -> None:
        doc = {"op": op, **{k: _plain(v) for k, v in fields.items()}}
        self._collect(doc)
        self._lines.append(
            json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        )
        self.ops.append(op)

    # --- namespaces ---------------------------------------------------------------------------------
    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        self._rec("ensure_namespace", ns=ns, dims=schema.dims, schema=dict(schema.attrs))
        await super().ensure_namespace(ns, schema)

    async def namespace_exists(self, ns: str) -> bool:
        self._rec("namespace_exists", ns=ns)
        return await super().namespace_exists(ns)

    async def list_namespaces(self, prefix: str) -> list[str]:
        self._rec("list_namespaces", prefix=prefix)
        return await super().list_namespaces(prefix)

    async def delete_namespace(self, ns: str) -> None:
        self._rec("delete_namespace", ns=ns)
        await super().delete_namespace(ns)

    async def branch_from(self, ns: str, source: str) -> None:
        self._rec("branch_from", ns=ns, source=source)
        await super().branch_from(ns, source)

    # --- writes -------------------------------------------------------------------------------------
    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        sent = [self.row_hook(ns, r) for r in rows] if self.row_hook is not None else list(rows)
        for r in sent:
            self.attr_names.update(r.attrs)
        self._rec("upsert", ns=ns, rows=sent, if_absent=if_absent, condition=condition)
        return await super().upsert(ns, sent, if_absent=if_absent, condition=condition)

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        self._rec("delete_ids", ns=ns, ids=list(ids))
        return await super().delete_ids(ns, ids)

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        self._rec("delete_by_file", ns=ns, file_id=file_id, except_file_hash=except_file_hash)
        return await super().delete_by_file(ns, file_id, except_file_hash=except_file_hash)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        self._rec("delete_by_filter", ns=ns, flt=flt)
        return await super().delete_by_filter(ns, flt)

    # --- reads --------------------------------------------------------------------------------------
    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        self._rec("get_by_ids", ns=ns, ids=list(ids), include_vectors=include_vectors, attrs=attrs)
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
        self._rec("query_ann", ns=ns, vector_dims=len(vector), top_k=top_k, flt=flt, attrs=attrs)
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
        self._rec("query_bm25", ns=ns, field=field, tokens=list(tokens), top_k=top_k, flt=flt, attrs=attrs)
        return await super().query_bm25(ns, field, tokens, top_k=top_k, flt=flt, attrs=attrs)
