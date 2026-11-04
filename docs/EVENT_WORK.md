# What was built during the event and what was not

The event ran October 26 to November 4, 2026. Only work from the event counts, so this section says what existed the day before. `scout/` descends from a repository whose history starts on October 25, 2026: 42 commits that day, then 167 more from October 26 to November 1 (this release keeps part of that tree). Of the 408 `scout/` files in this repository, 131 were first added on October 25. The table is derived from `git log` of that repository: first-add date per file, and commits and lines changed (added plus deleted, in the files kept here) from October 26 on. "Files" is files kept in this release; "line counts" compare the October 25 snapshot with the release.

| Component (`scout/`) | State on Oct 25 | Work during the event |
|---|---|---|
| Wire schemas (`schema/v1`) | 38 of 40 JSON schemas generated | 2 added; 5 commits, 214 lines changed |
| Typed contracts, port interfaces, local file layer (`grep`, `glob`, `read_file`, ignore rules), `app.py` facade (21 files) | All 21 files existed (about 1,400 lines) | Small edits: 13 commits, 308 lines changed |
| Explore loop and prompts (`core/explore`, 12 files) | 9 files, about 790 lines: loop, tools, profiles, budget, candidates, render, two prompts | 3 files added (`backfill.py`, `semantic.py`, `spec.py`); 6 commits, 859 lines changed; about 1,550 lines now |
| Adapters (26 files) | 12 files, about 540 lines: in-memory fakes for chat, clock, filesystem, gateway and store, plus empty packages for the Voyage, turbopuffer, keychain, OpenAI-compatible chat and gateway HTTP adapters | The real adapters: 17 commits, 2,443 lines changed; about 2,760 lines now |
| Encryption (`core/crypto`, 8 files) | An empty package | Written during the event: keys, path and row encryption, tokens, scope, MinHash; about 880 lines |
| Sync engine and teammate seed (`core/sync`, 8 files) | An empty package | Written during the event; about 2,320 lines |
| Gateway service (`core/service`, 10 files) | An empty package | Written during the event: auth, tenants, indexing, search, chat; about 1,710 lines |
| CLI, daemon, MCP and gateway HTTP surfaces, SDK (15 files) | Empty packages for the four surfaces, plus a 167-line `sdk.py` | Written during the event: 12 commits, 2,013 lines changed; about 2,130 lines now |
| Wiring, package root and demo runtime (11 files) | `wiring.py` (83 lines) and the package init | The 9-file demo runtime added; `wiring.py` grew to 551 lines; 18 commits, 1,635 lines changed |
| Muse Code and Claude Code plugins (`integrations/`, 53 files) | None | All event work: 22 commits, 6,564 lines changed |
| Evaluation harness (`eval/harness`, 16 files) | None | All event work: 17 commits, 2,851 lines changed |
| Tests (174 files) | 35 files, about 2,850 lines: unit tests for the contracts, explore loop, fakes, local tools and wire schemas, a proof-of-concept contract test, the small fixture repository | 139 files added, including the security, conformance and live suites, the declaration fixtures and the service, sync and crypto tests; 95 commits, 36,452 lines changed (fixtures and golden files are about 10,000 of those) |
| Build, scripts, ops, deploy (14 files) | `Makefile`, `pyproject.toml`, `uv.lock`, schema generator, namespace janitor | 9 files added, including the Modal gateway app: 20 commits, 1,361 lines changed |

Everything outside `scout/` is event work: all of `training/` and `eval/` (their history starts on October 26) and the deployment kits in [deploy/modal](../deploy/modal) and [deploy/gcp](../deploy/gcp). The model, the training recipe, the held-out evaluations, the end-to-end study and the Muse Code study are event work. [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) lists the October 25 files.

**Existing things this work uses and does not claim.** Muse Glimmer 30B (the base model) and Muse Spark Open (the teacher), Agents-A1-35B and its public transcripts, the `voyage-code-4` embedding model and API, turbopuffer, vLLM, Muse Code and Claude Code, and a pre-existing private benchmark of coding tasks from public repositories, from which the training pool and the held-out sets were drawn. Some evaluation and mining scripts here are adapted from tooling for that benchmark. The benchmark, its task records, and the teacher trajectories are not included.
