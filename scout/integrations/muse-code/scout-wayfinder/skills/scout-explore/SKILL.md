---
name: scout-explore
description: When and how to delegate code exploration to the scout-explorer subagent (scout + Wayfinder over an indexed repository). Use before planning or editing in a repository you have not read yet, and for broad where/how questions; skip it for exact-string lookups and files you already have open.
---

# scout-explore

The `scout-wayfinder` plugin adds a read-only subagent, `scout-wayfinder/scout-explorer`. It calls scout first
(one Wayfinder exploration over a semantic and lexical index of the repository), reads the top ranges to check
them, fills gaps with search, and hands back up to 12 ranked `path:start-end  why` lines: checked files first,
then scout's remaining files marked `(unchecked)`.

## Delegate to it

Spawn it with `subagent_spawn`, `subagent_type: "scout-wayfinder/scout-explorer"`, then
`subagent_wait` with `timeout_ms: 300000`, when:

- the user asks where something is implemented, how a feature works end to end, or which files a change touches;
- you are about to plan a change in a repository (or a part of one) you have not read yet;
- the question is phrased in behaviour, not identifiers ("where do we retry failed uploads"), so grep has
  nothing exact to match;
- the answer probably spans several files (definition, callers, config, tests).

Write the objective as a task for a colleague: what to find and why, in one paragraph, with any names the user
gave. Do not pre-search before delegating; the subagent searches. It usually takes one to three minutes (one
scout call, then verification reads). Wait for it: if `subagent_wait` times out, call it again; do not cancel it
and do not redo the search yourself in the meantime. Use its list as the reading plan: open the top ranges
yourself before you edit. When the user wants a list of files, your list starts with the subagent's list:
every entry, in its order, including the `(unchecked)` and `(search)` ones. Add files of your own only after
them, and do not drop or reorder its entries because you did not open them yourself.

## The retrieval spec

The session may carry the repository's retrieval spec (`.scout/spec.toml`, `docs/retrieval-spec.md`): its
instructions, an index description and examples of past changes. Follow the instructions, and word the objective
with the description's package names and the examples' vocabulary. A reminder on each prompt says to delegate
before reading or editing code; until you have delegated, a first `search` may be denied with the same pointer.

## Do it yourself instead

- An exact identifier, error string or file name is known: one `search` call is faster.
- The relevant file is already open or was named by the user.
- A follow-up question about code you already read in this session.
- Trivial edits (typo, rename in one file, version bump).

## Status and fallback

The session-start reminder says whether scout is ready for this checkout. If it says not indexed, the subagent
still works but falls back to plain search; offer the user `/scout-wayfinder:index` (or `scout sync --wait`).
If the subagent reports `scout: unavailable`, trust its list anyway (it searched without scout) and mention once
that scout was unavailable. The first scout call after a long idle can take a few minutes while Wayfinder starts.
