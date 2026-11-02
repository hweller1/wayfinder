---
description: Find where something lives in this repo with scout (scout-explorer subagent, verified ranges)
argument-hint: <question about the code>
---
Delegate this question to the `scout:scout-explorer` subagent with the Agent tool (subagent_type
`scout:scout-explorer`, run_in_background false: the user is waiting for this answer). Pass the question
below as the prompt, verbatim, followed by one line of context about why the user needs it if the
conversation makes that clear. Do not search the repository yourself first.

Question: $ARGUMENTS

When the subagent returns, show its ranked list to the user unchanged (keep every `path:start-end` and
reason), then add at most two sentences of your own if they help answer the question.
