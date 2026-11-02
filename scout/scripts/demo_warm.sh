#!/usr/bin/env bash
# Start and stop the demo VM (wayfinder-glimmer + the Scout gateway on one A100, deploy/gcp/), with a teardown watcher.
#
#   scripts/demo_warm.sh on [HOURS]   start the VM; arm a watcher in the same step that stops it after HOURS (default 4)
#   scripts/demo_warm.sh off          stop the VM and cancel the watcher (idempotent)
#   scripts/demo_warm.sh status       VM state, watcher, and the gateway's /v1/healthz
#   scripts/demo_warm.sh wait         block until a real Scout call returns (the model answered); 15 min cap
#
# Environment: GCP_PROJECT (required for on/off/status), GCP_ZONE (default us-central1-a), VM_NAME (default
# wayfinder-demo), SCOUT_GATEWAY_URL (https URL of the gateway; required for status/wait), SCOUT_KEY (a scout key,
# for `wait`). A stopped VM costs only its disk; a running one bills the A100 (deploy/gcp/README.md has the rate).
# An auth canary runs first and aborts without touching anything. Nothing here uses Modal.
set -euo pipefail

VM_NAME="${VM_NAME:-wayfinder-demo}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
STATE_DIR="${DEMO_WARM_STATE:-${TMPDIR:-/tmp}/scout-demo-warm}"
PIDFILE="$STATE_DIR/watcher.pid"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

gc() { gcloud --project "${GCP_PROJECT:?set GCP_PROJECT}" "$@"; }
canary() { gc compute instances list --limit 1 >/dev/null 2>&1 || { echo "demo_warm: gcloud is not authenticated for $GCP_PROJECT; nothing changed" >&2; exit 3; }; }
vm_state() { gc compute instances describe "$VM_NAME" --zone "$GCP_ZONE" --format='value(status)' 2>/dev/null || echo UNKNOWN; }
stop_watcher() { if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi; }

case "${1:-}" in
  on)
    hours="${2:-4}"
    canary
    mkdir -p "$STATE_DIR"
    stop_watcher
    gc compute instances start "$VM_NAME" --zone "$GCP_ZONE"
    # Teardown watcher, armed in the same step as the GPU: after HOURS it runs `off` whatever else happens.
    nohup bash -c "sleep $(( ${hours%.*} * 3600 )); GCP_PROJECT=\"$GCP_PROJECT\" VM_NAME=\"$VM_NAME\" GCP_ZONE=\"$GCP_ZONE\" \"$HERE/scripts/demo_warm.sh\" off" >"$STATE_DIR/watcher.log" 2>&1 &
    echo $! >"$PIDFILE"
    echo "demo_warm: ON. The VM boots, vLLM loads the weights (3 to 8 minutes from a cold disk cache); use: demo_warm.sh wait"
    echo "demo_warm: watcher pid $(cat "$PIDFILE") stops the VM in ${hours} h. Run 'scripts/demo_warm.sh off' when done."
    ;;
  off)
    canary
    stop_watcher
    gc compute instances stop "$VM_NAME" --zone "$GCP_ZONE"
    echo "demo_warm: OFF (VM stopped; the disk and the static IP still bill a little)."
    ;;
  status)
    canary
    echo "VM $VM_NAME ($GCP_ZONE): $(vm_state)"
    if [ -f "$PIDFILE" ]; then
      echo "watcher pid $(cat "$PIDFILE") (alive: $(kill -0 "$(cat "$PIDFILE")" 2>/dev/null && echo yes || echo no))"
    else
      echo "no watcher"
    fi
    if [ -n "${SCOUT_GATEWAY_URL:-}" ]; then
      curl -fs -m 8 "$SCOUT_GATEWAY_URL/v1/healthz" >/dev/null && echo "gateway: healthy" || echo "gateway: not answering"
    fi
    ;;
  wait)
    : "${SCOUT_GATEWAY_URL:?set SCOUT_GATEWAY_URL}"
    deadline=$(( $(date +%s) + 900 ))
    until SCOUT_KEYSTORE="${SCOUT_KEYSTORE:-}" "${SCOUT_BIN:-scout}" warm >/dev/null 2>&1; do
      [ "$(date +%s)" -lt "$deadline" ] || { echo "demo_warm: gateway/model not ready after 15 min (deploy/gcp/healthcheck.sh local)" >&2; exit 4; }
      sleep 5  # retry delay inside the script, between two blocking warm calls
    done
    echo "demo_warm: Wayfinder answered; ready."
    ;;
  *)
    echo "usage: $0 on [HOURS] | off | status | wait" >&2; exit 2 ;;
esac
