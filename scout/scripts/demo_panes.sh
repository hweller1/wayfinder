#!/usr/bin/env bash
# Two isolated Muse Code setups for the side-by-side demo, so neither touches ~/.config/muse or ~/.local/share/muse.
#
#   scripts/demo_panes.sh setup  <owner/repo> <commit> [--key-file F]  clone twice, install the plugin in the scout pane, sync
#   scripts/demo_panes.sh stock  ["prompt"]                            left pane : Muse Code without Scout (TUI)
#   scripts/demo_panes.sh scout  ["prompt"]                            right pane: Muse Code with the Scout plugin and spec
#   ... stock|scout --headless "prompt"                                the same session without the TUI (JSON events)
#   scripts/demo_panes.sh status                                       index state and what the plugin sees
#   scripts/demo_panes.sh teardown                                     remove both panes' state (not the clones' history)
#
# Environment: SCOUT_GATEWAY_URL (required for setup and status), and optional DEMO_DIR (default ~/demo-muse), SCOUT_KEY_FILE (a scout key, default
# ~/.config/scout/demo.key), META_KEY_FILE (the Muse/Meta API key, default ~/.config/scout/meta.key). Keys are read
# from files into the child environment and never printed. Public repositories only: Muse sends the code it reads to Meta.
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-$HOME/demo-muse}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SCOUT_KEY_FILE="${SCOUT_KEY_FILE:-$HOME/.config/scout/demo.key}"
META_KEY_FILE="${META_KEY_FILE:-$HOME/.config/scout/meta.key}"
# The gateway URL has no default: it is the HTTPS name of your VM (deploy/gcp/README.md), e.g. https://34-1-2-3.sslip.io
GATEWAY="${SCOUT_GATEWAY_URL:-}"
need_gateway() { [ -n "$GATEWAY" ] || { echo "demo_panes: set SCOUT_GATEWAY_URL to your gateway's https URL (see deploy/gcp/README.md)" >&2; exit 3; }; }
SCOUT="${SCOUT_BIN:-$(command -v scout || echo "$HERE/.venv/bin/scout")}"
MUSE="${MUSE_BIN:-$(command -v muse)}"

xdg() {  # xdg stock|scout -> exports that pane's isolated XDG dirs
  export XDG_CONFIG_HOME="$DEMO_DIR/$1/xdg/config" XDG_DATA_HOME="$DEMO_DIR/$1/xdg/data"
  export XDG_STATE_HOME="$DEMO_DIR/$1/xdg/state" XDG_CACHE_HOME="$DEMO_DIR/$1/xdg/cache"
  mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
}

need_key() { [ -r "$1" ] || { echo "demo_panes: missing key file $1" >&2; exit 3; }; }

case "${1:-}" in
  setup)
    repo="$2"; commit="$3"; shift 3
    while [ $# -gt 0 ]; do case "$1" in --key-file) SCOUT_KEY_FILE="$2"; shift 2 ;; *) shift ;; esac; done
    need_key "$SCOUT_KEY_FILE"; need_gateway
    for pane in stock scout; do
      d="$DEMO_DIR/$pane/repo"
      if [ ! -d "$d/.git" ]; then
        mkdir -p "$DEMO_DIR/$pane"
        git clone --quiet --filter=blob:none --no-checkout "https://github.com/$repo.git" "$d"
      fi
      git -C "$d" checkout --quiet --force --detach "$commit"
    done
    xdg scout
    SCOUT_KEYSTORE=file python3 "$HERE/integrations/muse-code/install.py" --muse "$MUSE" --lane production \
      --key-file "$SCOUT_KEY_FILE" --scout-bin "$SCOUT" >"$DEMO_DIR/scout/install.json"
    python3 - "$XDG_DATA_HOME/muse/plugins/data/scout-wayfinder/scout.env" "$GATEWAY" <<'PYEOF'
import sys
open(sys.argv[1], "a").write(f"SCOUT_GATEWAY_URL={sys.argv[2]}\nSCOUT_NO_WARM=1\n")
PYEOF
    SCOUT_KEY="$(cat "$SCOUT_KEY_FILE")" SCOUT_KEYSTORE=file SCOUT_GATEWAY_URL="$GATEWAY" \
      "$SCOUT" --root "$DEMO_DIR/scout/repo" sync --wait
    echo "demo_panes: ready. stock repo: $DEMO_DIR/stock/repo   scout repo: $DEMO_DIR/scout/repo"
    ;;
  stock|scout)
    pane="$1"; shift
    need_key "$META_KEY_FILE"
    xdg "$pane"
    export META_API_KEY="$(cat "$META_KEY_FILE")"
    cd "$DEMO_DIR/$pane/repo"
    if [ "${1:-}" = "--headless" ]; then  # same session without the TUI: transcript to stdout as JSON events (rehearsal, replays)
      shift
      exec "$MUSE" exec --json --trust-workspace --no-foreign-personal-context --disable-write --disable-shell "$@"
    fi
    exec "$MUSE" --trust-workspace --no-foreign-personal-context "$@"
    ;;
  status)
    need_gateway
    xdg scout
    SCOUT_KEY="$(cat "$SCOUT_KEY_FILE")" SCOUT_KEYSTORE=file SCOUT_GATEWAY_URL="$GATEWAY" "$SCOUT" --root "$DEMO_DIR/scout/repo" status
    "$MUSE" plugins list --json | python3 -c "
import json, sys
print('plugins in the scout pane:', [(p['record']['id'], 'enabled' if p['record']['enabled'] else 'disabled') for p in json.load(sys.stdin)['plugins']])"
    ;;
  teardown)
    pkill -f "scout watch --ensure --root $DEMO_DIR" 2>/dev/null || true
    rm -rf "$DEMO_DIR/stock/xdg" "$DEMO_DIR/scout/xdg"
    echo "demo_panes: removed the isolated Muse state of both panes (clones kept in $DEMO_DIR)."
    ;;
  *)
    echo "usage: $0 setup <owner/repo> <commit> | stock | scout | status | teardown" >&2; exit 2 ;;
esac
