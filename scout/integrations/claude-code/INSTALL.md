# Install the scout plugin in Claude Code

Three commands from zero to working, on a public repository (Voyage zero-retention is not confirmed yet, so
index public or synthetic code only). Needs Claude Code 2.1.292 or later, `uv`, `python3` (3.9 or later) and a
scout key.

```bash
uv tool install /path/to/voyage-scout && scout login --key <scout key>     # scout on PATH; key in the Keychain
SCOUT_INTEGRATIONS_DIR=/path/to/voyage-scout/integrations scout init --claude-code   # validate, add, install, verify
cd /path/to/public/checkout && scout sync --wait && claude                  # first index, then start Claude Code
```

`scout init --claude-code` checks the Claude Code version and runs `install.py`. The installer validates the
plugin, adds the `voyage-scout` marketplace (or updates it if it is already added) and installs
`scout@voyage-scout`. It then verifies the install: it lists the plugin and runs the plugin's doctor, which
checks python, the `scout` binary, the key and the gateway's `/v1/healthz`. It prints one JSON report and never
the key. `--dry-run` prints the steps without running them. `CLAUDE_CONFIG_DIR=/tmp/x` scopes the install to a
scratch config directory.

In Claude Code, `/scout:status` should show `index ready 100%, at HEAD` and `routing mode hybrid: Explore ->
scout:scout-explorer`. The index can also be built from inside Claude Code with `/scout:index` instead of
`scout sync --wait`.

The same steps by hand:

```bash
claude plugin marketplace add /path/to/voyage-scout/integrations/claude-code
claude plugin install scout@voyage-scout
```

## Upgrading from super-explore 0.1

The plugin was renamed from `super-explore` to `scout` in 0.2.0. Remove the old one first:
`claude plugin uninstall super-explore@super-explore && claude plugin marketplace remove super-explore`.

## Demo lane (no scout key)

Without a key, the MCP launcher runs the demo server (`scout-demo mcp`). It needs this checkout built with
`uv sync` and the demo keys in `voyage-scout/.env`. Index a repo once with
`uv run --project /path/to/voyage-scout scout-demo index /path/to/checkout --name myrepo`; Explore is then
routed in that checkout the same way. The demo lane has no watcher, so re-run the index after changes.

## Try without installing

`claude --plugin-dir /path/to/voyage-scout/integrations/claude-code` loads the plugin for one session. Its data
directory is then `~/.claude/plugins/data/scout-inline/` instead of `.../scout-voyage-scout/`.

## Uninstall

`claude plugin uninstall scout@voyage-scout && claude plugin marketplace remove voyage-scout`. The index stays
until you delete it. `scout logout` removes the key.
