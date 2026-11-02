"""In-memory stand-in for the turbopuffer SDK (`AsyncTurbopuffer`) used by the adapter tests (task 7.1).

It speaks the SDK's wire shapes (filter tuples, `rank_by` tuples, `upsert_condition`, schema dicts) and
mirrors the behaviours spike 0a-A verified (artifacts/spike/0a_tpuf.json): a vector-less row in a vector
namespace is a whole-batch HTTP 400, vector-less namespaces copy through `branch_from`, `upsert_condition`
is checked against the EXISTING row (a missing row passes, `("id","Eq",None)` is insert-if-absent), BM25 on
a `pre_tokenized_array` attribute is exact-token OR, and `$dist` is cosine distance for ANN and the BM25
score for BM25. Its filter evaluator is independent of the adapter's translation and of FakeStore.
It records every call in `calls` and can be told to fail the next N calls with a given error.
"""

import copy
import math
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


class APIStatusError(Exception):
    """Lookalike of `turbopuffer.APIStatusError` (the adapter keys on class name and `status_code`)."""

    def __init__(self, status_code: int, body: str = "") -> None:
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code = status_code
        self.body = body


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


VECTORLESS_400 = "\U0001f494 each row must have a `vector` value; to delete a row, use `deletes` instead"


@dataclass
class _Ns:
    schema: dict[str, Any]
    vector: bool
    rows: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])


def _glob(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c))
        i += 1
    return re.compile("".join(out), re.DOTALL)


def matches(flt: Any, row: Mapping[str, Any]) -> bool:
    op = flt[0]
    if op == "And":
        return all(matches(p, row) for p in flt[1])
    if op == "Not":
        return not matches(flt[1], row)
    attr, kind, value = flt
    got = row.get(attr)
    if kind == "Eq":
        return got == value
    if kind == "In":
        return got in value
    if kind == "Glob":
        return isinstance(got, str) and _glob(value).fullmatch(got) is not None
    if kind == "Contains":
        return isinstance(got, list) and value in got
    raise AssertionError(f"stub: unsupported filter {flt!r}")


def _cos_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (na * nb) if na and nb else 0.0)


class _NamespaceHandle:
    def __init__(self, tp: "StubTurbopuffer", name: str) -> None:
        self.tp, self.name = tp, name

    async def _enter(self, op: str, **kw: Any) -> None:
        self.tp.calls.append((op, self.name, kw))
        if self.tp.fail:
            exc = self.tp.fail.pop(0)
            raise exc

    def _need(self) -> _Ns:
        ns = self.tp.data.get(self.name)
        if ns is None:
            raise APIStatusError(404, "namespace not found")
        return ns

    async def exists(self) -> bool:
        await self._enter("exists")
        return self.name in self.tp.data

    async def schema(self) -> dict[str, Any]:
        await self._enter("schema")
        return copy.deepcopy(self._need().schema)

    async def delete_all(self) -> None:
        await self._enter("delete_all")
        self._need()
        del self.tp.data[self.name]

    async def branch_from(self, *, source_namespace: str) -> None:
        await self._enter("branch_from", source_namespace=source_namespace)
        src = self.tp.data.get(source_namespace)
        if src is None:
            raise APIStatusError(404, "source not found")
        if self.name in self.tp.data:
            raise APIStatusError(409, "exists")
        self.tp.data[self.name] = copy.deepcopy(src)

    async def write(self, **kw: Any) -> SimpleNamespace:
        await self._enter("write", **kw)
        ns = self.tp.data.get(self.name)
        if "delete_by_filter" in kw:
            ns = self._need()
            doomed = [i for i, r in ns.rows.items() if matches(kw["delete_by_filter"], r)]
            for i in doomed:
                del ns.rows[i]
            return SimpleNamespace(rows_affected=len(doomed))
        rows: list[dict[str, Any]] = kw.get("upsert_rows", [])
        if ns is None:
            if not kw.get("create_namespace") and not rows:
                raise APIStatusError(404, "namespace not found")
            schema = copy.deepcopy(kw.get("schema", {}))
            has_vec = "vector" in schema or any("vector" in r for r in rows)
            if has_vec and "vector" not in schema:
                schema["vector"] = {"type": f"[{len(next(r['vector'] for r in rows if 'vector' in r))}]f32"}
            ns = _Ns(schema=_server_schema(schema, rows), vector=has_vec)
            created = True
        else:
            created = False
        for r in rows:
            if ns.vector and "vector" not in r:
                raise APIStatusError(400, VECTORLESS_400)
        written = 0
        cond = kw.get("upsert_condition")
        staged: dict[str, dict[str, Any]] = {}
        for r in rows:
            old = ns.rows.get(r["id"])
            if (
                old is not None
                and cond is not None
                and (cond == ("id", "Eq", None) or not matches(cond, old))
            ):
                continue
            staged[r["id"]] = copy.deepcopy(r)
            written += 1
        if created:
            self.tp.data[self.name] = ns
        ns.rows.update(staged)
        return SimpleNamespace(rows_affected=written)

    async def query(self, **kw: Any) -> SimpleNamespace:
        await self._enter("query", **kw)
        ns = self._need()
        pool = [r for r in ns.rows.values() if "filters" not in kw or matches(kw["filters"], r)]
        rank = kw.get("rank_by")
        scored: list[tuple[float, dict[str, Any]]]
        if rank is None:
            scored = [(0.0, r) for r in sorted(pool, key=lambda r: r["id"])]
        elif rank[1] == "ANN":
            if not ns.vector:
                raise APIStatusError(400, "no vector attribute")
            scored = sorted(
                ((_cos_dist(rank[2], r["vector"]), r) for r in pool), key=lambda t: (t[0], t[1]["id"])
            )
        else:
            scored = self._bm25(ns, rank[0], rank[2], pool)
        include = kw.get("include_attributes", True)
        out: list[_Row] = []
        for d, r in scored[: kw["top_k"]]:
            row = {k: v for k, v in r.items() if include is True or k in include or k == "id"}
            if include is True:
                row.pop("vector", None)  # the real service omits vectors unless named
            if rank is not None:
                row["$dist"] = d
            out.append(_Row(row))
        return SimpleNamespace(rows=out)

    def _bm25(
        self, ns: _Ns, field_: str, tokens: list[str], pool: list[dict[str, Any]]
    ) -> list[tuple[float, dict[str, Any]]]:
        cfg = ns.schema.get(field_, {})
        if not (isinstance(cfg, dict) and cfg.get("full_text_search")):  # pyright: ignore[reportUnknownMemberType]
            raise APIStatusError(400, "attribute has no full-text index")
        assert isinstance(tokens, list)
        docs = [r for r in ns.rows.values() if isinstance(r.get(field_), list)]
        n = len(docs)
        avg = sum(len(r[field_]) for r in docs) / n if n else 1.0
        out: list[tuple[float, dict[str, Any]]] = []
        for r in pool:
            toks = set(r.get(field_) or ())
            sc = 0.0
            for t in tokens:
                if t in toks:
                    df = sum(1 for d in docs if t in d[field_])
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    norm = 1 - 0.75 + 0.75 * len(toks) / avg
                    sc += idf * 2.2 / (1 + 1.2 * norm)
            if sc > 0:
                out.append((sc, r))
        return sorted(out, key=lambda t: (-t[0], t[1]["id"]))


class _Row(dict[str, Any]):
    """SDK `Row` lookalike: attribute access plus `model_extra` (the adapter reads either)."""

    @property
    def id(self) -> Any:
        return self["id"]

    @property
    def vector(self) -> Any:
        return self.get("vector")

    @property
    def model_extra(self) -> dict[str, Any]:
        return {k: v for k, v in self.items() if k not in ("id", "vector")}


def _server_schema(schema: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What `ns.schema()` returns: declared attributes plus inferred ones (string/int/bool/[]string)."""
    out: dict[str, Any] = {k: (v if isinstance(v, dict) else {"type": v}) for k, v in schema.items()}
    out.setdefault("id", {"type": "string"})
    for r in rows:
        for k, v in r.items():
            if k in out or k == "vector":
                continue
            out[k] = {"type": _infer(v)}
    return out


def _infer(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, list):
        return "[]string"
    return "string"


class StubTurbopuffer:
    """`AsyncTurbopuffer` lookalike. `fail` is a FIFO of exceptions raised by the next calls."""

    def __init__(self) -> None:
        self.data: dict[str, _Ns] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail: list[Exception] = []

    def namespace(self, name: str) -> _NamespaceHandle:
        return _NamespaceHandle(self, name)

    def namespaces(self, *, prefix: str = "") -> AsyncIterator[SimpleNamespace]:
        names = sorted(n for n in self.data if n.startswith(prefix))

        async def gen() -> AsyncIterator[SimpleNamespace]:
            self.calls.append(("namespaces", prefix, {}))
            for n in names:
                yield SimpleNamespace(id=n)

        return gen()

    def calls_of(self, op: str) -> list[dict[str, Any]]:
        return [kw for o, _, kw in self.calls if o == op]


def no_sleep() -> tuple[Callable[[float], Any], list[float]]:
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    return sleep, slept
