#!/usr/bin/env bash
# GCE startup-script (metadata key `startup-script`). Runs as root on every boot. Idempotent.
# First boot: install. Every boot: refresh secrets from Secret Manager (if the secrets exist) and start the services.
# Instance metadata used: scout-repo-url, scout-hostname, scout-quant (all optional except scout-hostname).
set -euo pipefail
md() { curl -fs -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
export REPO_URL="$(md scout-repo-url)" SCOUT_HOSTNAME="$(md scout-hostname)" WAYFINDER_QUANT="$(md scout-quant)"
: "${WAYFINDER_QUANT:=bf16}"; export WAYFINDER_QUANT
D=/opt/wayfinder/repo/deploy/gcp
if [ ! -f /opt/wayfinder/.installed ]; then
  mkdir -p /opt/wayfinder
  if [ -n "$REPO_URL" ] && [ ! -d /opt/wayfinder/repo/.git ]; then git clone --depth 1 "$REPO_URL" /opt/wayfinder/repo; fi
  bash "$D/install.sh" && touch /opt/wayfinder/.installed
fi
if gcloud secrets describe scout-vllm-api-key >/dev/null 2>&1; then bash "$D/fetch_secrets.sh"; fi
systemctl enable --now wayfinder-vllm scout-gateway caddy
systemctl reload-or-restart caddy   # pick up the Caddyfile install.sh wrote
