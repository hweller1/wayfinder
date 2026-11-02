"""The resolved retrieval spec as the core sees it (the retrieval spec, not in this release).

The Claude Code plugin (or any host) resolves the three layers (system defaults, harness defaults, the repo's
`.scout/spec.toml`) and writes `<state_dir>/spec_resolved.json`. The core only reads that file, per call, so
an edit shows up on the next search. Nothing here fetches anything: a missing, unreadable or disabled spec is
`None` and the explorer behaves exactly as before.

What the spec does inside scout (instructed retrieval):
- `addendum`: instructions, the index description and the mined or curated examples are appended to the
  explorer's system prompt, so the model plans its queries with them;
- `Filters.apply`: path excludes drop semantic hits, path boosts and the recency boost rerank them (a score
  adjustment, nothing is retrained and the gateway is unchanged);
- `k`: results per semantic search (a cap on what the model or caller asks for);
- `index_enabled=False`: semantic search is off, so the explorer is grep-only.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pathspec

from scout.core.explore.candidates import RawHit

SPEC_FILE = "spec_resolved.json"
MAX_ADDENDUM_CHARS = 6000
K_MIN, K_MAX = 1, 30


@dataclass(frozen=True, slots=True)
class RetrievalSpec:
    instructions: str = ""
    description: str = ""
    examples: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exclude: tuple[str, ...] = ()
    boost: tuple[str, ...] = ()
    boost_weight: float = 0.3
    recency_boost: float = 0.0
    k: int = 5
    max_turns: int = 0  # 0: no spec cap (the request or profile limit applies)
    index_enabled: bool = True
    layers: tuple[str, ...] = ("system",)

    @property
    def filters(self) -> "Filters":
        return Filters(self.exclude, self.boost, self.boost_weight, self.recency_boost)

    def addendum(self) -> str:
        """Text appended to the explorer's system prompt; empty when the spec says nothing."""
        parts: list[str] = []
        if self.instructions.strip():
            parts.append(
                "Instructions from the repository owner (follow them over your defaults):\n"
                + self.instructions.strip()
            )
        if self.description.strip():
            parts.append("Index description (what is searchable here):\n" + self.description.strip())
        if self.examples:
            lines = [f"- {q} -> {', '.join(files)}" for q, files in self.examples]
            parts.append(
                "Examples of how past changes map to files (description -> files touched):\n"
                + "\n".join(lines)
            )
        if self.boost:
            parts.append("Prefer these paths when results are close: " + ", ".join(self.boost) + ".")
        if self.exclude:
            parts.append(
                "Never report files matching: "
                + ", ".join(self.exclude)
                + ". Search results already omit them."
            )
        parts.append(
            f"Semantic search returns {self.k} results per call; use several differently worded queries "
            "rather than a larger k."
        )
        text = "\n\n".join(parts)
        return text[:MAX_ADDENDUM_CHARS]


class Filters:
    """Path excludes, path boosts and the recency boost over raw semantic hits."""

    def __init__(
        self, exclude: tuple[str, ...], boost: tuple[str, ...], boost_weight: float, recency: float
    ) -> None:
        self._exclude = pathspec.PathSpec.from_lines("gitignore", exclude) if exclude else None
        self._boost = pathspec.PathSpec.from_lines("gitignore", boost) if boost else None
        self._weight = boost_weight
        self._recency = recency

    def excluded(self, path: str) -> bool:
        return bool(self._exclude and self._exclude.match_file(path))

    def apply(self, hits: list[RawHit], mtime_ns: Callable[[str], int | None] | None = None) -> list[RawHit]:
        """Drop excluded paths, then rerank:
        `adjusted = rank_score * (1 + boost_weight if boosted) + recency`.
        `rank_score` is the hit's score when every hit has one, else a descending rank score; the recency
        term is `recency_boost * (mtime - oldest) / (newest - oldest)` over the hits. Stable on ties."""
        kept = [h for h in hits if not self.excluded(h.path)]
        if len(kept) < 2 or not (self._boost or self._recency):
            return kept
        n = len(kept)
        scores = [h.score for h in kept]
        base = (
            [sc for sc in scores if sc is not None]
            if all(sc is not None for sc in scores)
            else [float(n - i) for i in range(n)]
        )
        top = max(abs(float(b)) for b in base) or 1.0
        norm = [float(b) / top for b in base]
        rec = [0.0] * n
        if self._recency and mtime_ns is not None:
            ts = [mtime_ns(h.path) for h in kept]
            known = [t for t in ts if t is not None]
            if known and max(known) > min(known):
                lo, hi = min(known), max(known)
                rec = [self._recency * ((t - lo) / (hi - lo)) if t is not None else 0.0 for t in ts]
        adj = [
            norm[i] * (1 + self._weight if self._boost and self._boost.match_file(kept[i].path) else 1.0)
            + rec[i]
            for i in range(n)
        ]
        order = sorted(range(n), key=lambda i: (-adj[i], i))
        return [kept[i] for i in order]


def _strs(v: Any) -> tuple[str, ...]:
    items: list[Any] = cast("list[Any]", v) if isinstance(v, list) else []
    return tuple(x for x in items if isinstance(x, str))


def from_dict(raw: Any) -> RetrievalSpec | None:
    """Tolerant parse of `spec_resolved.json`; anything unusable is None (no spec)."""
    if not isinstance(raw, dict):
        return None
    raw = cast("dict[str, Any]", raw)
    if raw.get("version") != 1:
        return None
    raw_layers = raw.get("layers")
    layers: dict[str, Any] = cast("dict[str, Any]", raw_layers) if isinstance(raw_layers, dict) else {}
    if layers.get("disabled"):
        return None
    examples: list[tuple[str, tuple[str, ...]]] = []
    raw_examples: list[Any] = cast("list[Any]", raw.get("examples") or [])
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        e = cast("dict[str, Any]", item)
        if isinstance(e.get("query"), str) and _strs(e.get("files")):
            examples.append((e["query"], _strs(e["files"])))
    k = raw.get("k")
    w, rb = raw.get("boost_weight"), raw.get("recency_boost")
    names = (
        ["system"] + (["harness"] if layers.get("harness") else []) + (["user"] if layers.get("user") else [])
    )
    return RetrievalSpec(
        instructions=raw["instructions"] if isinstance(raw.get("instructions"), str) else "",
        description=raw["description"] if isinstance(raw.get("description"), str) else "",
        examples=tuple(examples),
        exclude=_strs(raw.get("exclude")),
        boost=_strs(raw.get("boost")),
        boost_weight=float(w) if isinstance(w, int | float) and not isinstance(w, bool) else 0.3,
        recency_boost=float(rb) if isinstance(rb, int | float) and not isinstance(rb, bool) else 0.0,
        max_turns=raw["max_turns"]
        if isinstance(raw.get("max_turns"), int)
        and not isinstance(raw.get("max_turns"), bool)
        and raw["max_turns"] > 0
        else 0,
        k=max(K_MIN, min(K_MAX, k)) if isinstance(k, int) and not isinstance(k, bool) else 5,
        index_enabled=raw.get("index_enabled") is not False,
        layers=tuple(names),
    )


def load(state_dir: str) -> RetrievalSpec | None:
    try:
        with open(os.path.join(state_dir, SPEC_FILE), encoding="utf-8") as f:
            return from_dict(json.load(f))
    except (OSError, ValueError):
        return None
