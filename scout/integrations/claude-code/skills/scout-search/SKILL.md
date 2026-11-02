---
name: scout-search
description: When to delegate code search to the scout-explorer subagent (Wayfinder over a semantic index, verified reads) and when not to. Use when a question about the current repository is broad - "where is X handled", "how does Y work", "what would I change to do Z", a flow across several files, or an unfamiliar codebase - and scout is indexed for it.
---

# Searching code with scout

The scout plugin adds the `scout:scout-explorer` subagent. It calls Wayfinder, a code-search model running
over a semantic and lexical index of this repository, then reads and verifies the ranges it returns and fills
gaps with Grep. It hands back at most 12 ranked `path:start-end` ranges with a one-line reason each.

## Delegate to scout-explorer when

- The question is broad: "where is authentication enforced", "how does a request reach the handler", "what
  needs to change to add a flag like --foo".
- The repository is unfamiliar, or the answer spans several files (a flow, a registry plus its users, a
  feature with its config and tests).
- You would otherwise run many Grep/Glob/Read rounds guessing at names.

Call it with the Agent tool, `subagent_type: "scout:scout-explorer"`, and a brief written like a task for a
colleague: what to find and why, plus any names, error strings or behaviours you already know. A plain
Explore call is also routed to it automatically when scout is ready. `/scout:find <question>` does the same.

## Do not delegate when

- You already know the file or the symbol: Read it or Grep it yourself.
- The question is a single exact identifier or string: one Grep is faster.
- You only need to list files or read a config you can name.
- scout is not indexed here (`/scout:status`): use Explore or your own search.

## Cost and latency

A scout-explorer call takes about 1 to 3 minutes (the search model is slow per turn; the first call after
idle may wait for a cold start, which the session-start warm-up usually hides). Issue it once per question,
keep doing independent work meanwhile if you can, and do not re-ask with a reworded brief. Trust the ranges
it returns: they were read and verified.
