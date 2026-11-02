"""Plain-text rendering of an `ExploreResult` under a token cap (FR-C2, AC-1.2, NFR-6).

Layout is `summary`, then one `path:start-end  why` line per result in rank order, then gaps. When the cap is
exceeded the lowest-ranked result lines go first (whole lines only, never cut mid-line), then gap lines from
the end, then summary lines from the end. A dropped-results notice is added once anything is dropped.
"""

import math
from collections.abc import Callable

from scout.core.contracts.types import ExploreResult, Hit

FURTHER_HEADING = "Further candidates (unverified):"
TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """Conservative estimate: `ceil(utf8_bytes / 3.5)`. No tokenizer dependency in core."""
    return math.ceil(len(text.encode("utf-8")) / 3.5)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _hit_line(hit: Hit) -> str:
    where = f"{hit.path}:{hit.start_line}-{hit.end_line}"
    suffix = (" (stale)" if hit.stale else "") + ("" if hit.verified else " (unverified)")
    why = _one_line(hit.why)
    return f"{where}{suffix}  {why}".rstrip()


def _lines(text: str) -> list[str]:
    return [ln for ln in (_one_line(p) for p in text.splitlines()) if ln]


def render_text(result: ExploreResult, max_tokens: int, count: TokenCounter = approx_tokens) -> str:
    head = _lines(result.summary)
    if result.budget_exhausted:
        head.append("(budget exhausted: results are the best candidates found so far)")
    ranked = [h for h in result.results if h.verified]
    extra = [h for h in result.results if not h.verified]
    hits = [_hit_line(h) for h in ranked]
    if extra:  # one heading, then the backfilled entries in order; the heading goes before the first of them
        hits += [FURTHER_HEADING, *(_hit_line(h) for h in extra)]
    gaps = [f"gap: {g}" for g in (_one_line(g) for g in result.gaps) if g]

    n_hits, n_gaps, n_head = len(hits), len(gaps), len(head)
    total_hits = len(hits)

    def build() -> str:
        shown = hits[:n_hits]
        if shown and shown[-1] == FURTHER_HEADING:
            shown.pop()  # never end the list on a heading with nothing under it
        out = head[:n_head] + shown + gaps[:n_gaps]
        if n_hits < total_hits:
            out.append(f"... {total_hits - n_hits} lower-ranked results omitted")
        return "\n".join(out)

    text = build()
    while count(text) > max_tokens and (n_hits or n_gaps or n_head):
        if n_hits:
            n_hits -= 1
        elif n_gaps:
            n_gaps -= 1
        else:
            n_head -= 1
        text = build()
    return text if count(text) <= max_tokens else ""
