---
description: Find where something lives in this repository (delegates to scout-explorer)
argument-hint: <what to find>
---
Spawn the `scout-wayfinder/scout-explorer` subagent with `subagent_spawn` and this objective, then
`subagent_wait` for it with `timeout_ms: 300000` (wait again if it times out; it takes one to three minutes):

$ARGUMENTS

Return its ranked `path:start-end  why` list as is (at most 12 lines), followed by its `Gaps:` line. Do not
search yourself first and do not add files it did not list.
