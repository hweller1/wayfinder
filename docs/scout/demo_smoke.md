# Demo lane smoke test (2026-10-26)

Run from `voyage-scout/` with keys loaded from the `.env` files (never printed). Real turbopuffer (gcp-us-central1),
real Voyage voyage-code-4 (document/query, 1024 dims), real in-process `GatewayService` (encrypted paths and line
ranges, HMAC tokens, Merkle sync), model Agents-A1-35B on Modal (`A1_BASE_URL`, H100, vLLM).

## Index

| Repo | Command | Files | Chunks | Voyage tokens | Time | Est. cost |
|---|---|---|---|---|---|---|
| gorilla/mux (Go) | `uv run scout-demo index https://github.com/gorilla/mux --name gomux` | 27 | 408 | 72,669 | 17 s | $0.013 |
| pallets/flask (Python) | `uv run scout-demo index https://github.com/pallets/flask --name pyflask` | 230 | 1,375 | 464,285 | 27 s | $0.084 |

Cost is tokens x $0.18/M (the voyage-code-3 list price; voyage-code-4 pricing is doc-only, so treat it as an
indexes read `state=ready coverage=1.00`.

Namespaces left in place for the author:

- `scout_dev_demo_gomux`, `scout_dev_demo_gomux__tree`, `__cache`, `__meta`, `__keys`
- `scout_dev_demo_pyflask`, `scout_dev_demo_pyflask__tree`, `__cache`, `__meta`, `__keys`

Delete with `ops/ns_janitor.py` (prefix `scout_dev_demo_`) when done.

## Explore (A1-35B, 3 questions per repo)

Command shape: `uv run scout-demo explore "<question>" --repo <name> --preset a1`. Wall time excludes a 0.5-0.9 s
warmup (the Modal app was already warm). Cost is wall time x $3.95/h H100, a single-stream upper bound.

| Repo | Question (short) | Turns | Tool calls (sem/grep/read) | Tokens in/out | Wall | Est. cost |
|---|---|---|---|---|---|---|
| gomux | route variables like `{id}`: parsing and matching | 16 | 2 / 5 / 8 | 115k / 2.6k | 31.0 s | $0.034 |
| gomux | how the router picks a handler; where 405 comes from | 9 | 1 / 2 / 5 | 44k / 2.2k | 21.2 s | $0.023 |
| gomux | CORS middleware; where subrouters with a prefix are made | 12 | 1 / 4 / 6 | 61k / 3.3k | 29.5 s | $0.032 |
| pyflask | which view handles a request; `<int:id>` converters | 11 | 8 / 6 / 10 | 160k / 5.4k | 48.9 s | $0.054 |
| pyflask | config from env vars or a Python file | 15 | 2 / 2 / 10 | 160k / 5.1k | 44.5 s | $0.049 |
| pyflask | exception handling, error handlers, default 500 | 29 | 2 / 9 / 15 | 344k / 8.9k | 80.1 s | $0.088 |

Quality impressions (read against the repos, not scored): every question got the right files with tight line
ranges, a direct summary and, where relevant, named gaps. Examples: the 405 question returned
`mux.go:188-222` (ServeHTTP), `route.go:47-114` (Route.Match sets ErrMethodMismatch) and `mux.go:670-676` (the
405 handler), and the A1 run flagged that the `Allow` header is not written by the default handler. The Flask
config answer listed `from_envvar`, `from_prefixed_env`, `from_pyfile` and `from_object` with their line ranges
in `src/flask/config.py`. Results are mostly `read` sourced: the model reads the code before reporting, as the
prompt asks. Semantic search was used in every run, between 1 and 8 times. The exception question was the
slowest (29 turns, 80 s), which is within the 120 s product budget but above the median.

Earlier the same pipeline was proven with Baseten `meta-models/muse-spark-open` (`--preset baseten`): 7 turns,
19.7 s on the `{id}` question, same top files.

## MCP

`uv run scout-demo mcp` serves `explore`, `index_repo`, `list_repos`. Verified with the MCP Python client
(initialize in about 1 s, no network at startup; `list_repos` and `explore` return) and with Claude Code 2.1.304
using a project-scoped config in `/tmp/mcptest` and `claude -p --mcp-config ... --strict-mcp-config`: Claude saw
all three tools, called `list_repos` and `explore` on `gomux`, and reported the top ranges. `claude mcp list` in
that directory shows the server as "Pending approval" until the project `.mcp.json` is approved interactively.
The global Claude config was not changed.

## Known gaps

- Demo master key and team key are local (`~/.scout-demo/master.key`); there is no scout key auth or HTTP gateway
  here. Teammate seeding is off (the demo store reports no sibling namespaces).
- `explore` over MCP uses the default 240 s budget and warms nothing; the first call after Modal scale-to-zero can
  take a minute or more.
- No file watcher: re-run `index` (incremental, embeds only changed files) after the checkout changes.
- Cost for explore is an estimate from wall time, not a Modal bill.
