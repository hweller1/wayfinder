#!/bin/sh
# Starts the scout MCP server for the plugin. Called by .mcp.json (never rely on plugin bin/ PATH).
#
# Lanes (the hook, hooks/rewrite_explore.py, applies the same rule):
#   production  `scout mcp ...` against the hosted gateway (default https://scout-gateway.example.com,
#               override with SCOUT_GATEWAY_URL). Used when a scout key is configured: $SCOUT_KEY / $SCOUT_API_KEY,
#               `~/.config/scout/credentials` (`scout login`, file store) or, on macOS, the Keychain item that
#               `scout login` writes. Binary: $SCOUT_BIN, `scout` on PATH, ~/.local/bin/scout.
#   demo        `scout-demo mcp` over real turbopuffer/Voyage, found via $SCOUT_DEMO_PROJECT, a `scout-demo`
#               binary, ~/.scout-demo/project (a one-line pointer file), or this script's own checkout.
#               Fallback when no key is configured or no `scout` binary exists.
# $SCOUT_LANE=demo|production forces a lane (production still needs the binary). No network at startup. Prints a
# hint on stderr and exits 1 when nothing is found.
PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin"
export PATH

# `scout` reads SCOUT_KEY; SCOUT_API_KEY is accepted as an alias.
if [ -z "${SCOUT_KEY:-}" ] && [ -n "${SCOUT_API_KEY:-}" ]; then
  SCOUT_KEY="$SCOUT_API_KEY"
  export SCOUT_KEY
fi

have_key() {
  if [ -n "${SCOUT_KEY:-}" ] || [ -f "$HOME/.config/scout/credentials" ]; then
    return 0
  fi
  # `scout login` stores the key in the macOS Keychain (service scout, account scout_key). Attributes only:
  # no -w/-g, so the secret is never read and no prompt can appear.
  if [ "${SCOUT_KEYSTORE:-}" != "file" ] && [ -z "${SCOUT_NO_KEYCHAIN_PROBE:-}" ] && [ "$(uname)" = "Darwin" ]; then
    security find-generic-password -s scout -a scout_key >/dev/null 2>&1 && return 0
  fi
  return 1
}

scout_bin() {
  if [ -n "${SCOUT_BIN:-}" ] && [ -x "$SCOUT_BIN" ]; then
    printf '%s\n' "$SCOUT_BIN"
  elif command -v scout >/dev/null 2>&1; then
    command -v scout
  elif [ -x "$HOME/.local/bin/scout" ]; then
    printf '%s\n' "$HOME/.local/bin/scout"
  else
    return 1
  fi
}

production=0
case "${SCOUT_LANE:-}" in
  demo) production=0 ;;
  production) production=1 ;;
  *) have_key && production=1 ;;
esac

if [ "$production" = 1 ]; then
  # /scout:model glimmer|v3|a1 picks the gateway per project; SCOUT_GATEWAY_URL set by the user still wins.
  if [ -z "${SCOUT_GATEWAY_URL:-}" ] && command -v python3 >/dev/null 2>&1; then
    url=$(python3 "$(dirname "$0")/../lib/scout_cc.py" gateway-url 2>/dev/null) || url=""
    if [ -n "$url" ]; then
      SCOUT_GATEWAY_URL="$url"
      export SCOUT_GATEWAY_URL
    fi
  fi
  if bin=$(scout_bin); then
    exec "$bin" "$@"
  fi
  if [ "${SCOUT_LANE:-}" = "production" ]; then
    echo "scout plugin: SCOUT_LANE=production but no scout binary found (uv tool install, or set SCOUT_BIN)." >&2
    exit 1
  fi
fi

# Demo lane: the demo server takes no flags.
project="${SCOUT_DEMO_PROJECT:-}"
if [ -z "$project" ] && [ -r "$HOME/.scout-demo/project" ]; then
  project=$(head -n 1 "$HOME/.scout-demo/project")
fi
if [ -z "$project" ]; then
  here=$(cd "$(dirname "$0")/../../.." 2>/dev/null && pwd)
  if [ -f "$here/pyproject.toml" ] && grep -q 'scout-demo' "$here/pyproject.toml" 2>/dev/null; then
    project="$here"
  fi
fi
# Prefer the project venv's own entry point: no `uv run` re-sync, so a venv that another process is rebuilding
# cannot make the server fail to start (Claude Code skips a stdio server that failed in the last 15 minutes).
if [ -n "$project" ] && [ -x "$project/.venv/bin/scout-demo" ]; then
  exec "$project/.venv/bin/scout-demo" mcp
fi
if [ -n "$project" ] && command -v uv >/dev/null 2>&1; then
  exec uv run --quiet --project "$project" scout-demo mcp
fi
if command -v scout-demo >/dev/null 2>&1; then
  exec scout-demo mcp
fi

# No demo either: a `scout` without a key still starts (its tools answer LOGIN_REQUIRED, which says what to do).
if bin=$(scout_bin); then
  exec "$bin" "$@"
fi

echo "scout plugin: scout not found. Install it (uv tool install ...) or set SCOUT_BIN / SCOUT_DEMO_PROJECT." >&2
exit 1
