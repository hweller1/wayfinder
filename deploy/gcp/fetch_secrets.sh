#!/usr/bin/env bash
# Write /etc/scout/{gateway,vllm}.env (mode 600, root) from GCP Secret Manager. Run as root on the VM.
# Secrets expected in the project (create them once, see README): scout-turbopuffer-api-key, scout-voyage-api-key,
# scout-master-key, scout-pepper, scout-vllm-api-key, and optionally scout-hf-token. The VM's service account needs
# roles/secretmanager.secretAccessor on them. Values are never printed.
set -euo pipefail
umask 077
install -d -m 700 /etc/scout
get() { gcloud secrets versions access latest --secret="$1" 2>/dev/null || true; }
need() { local v; v="$(get "$1")"; [ -n "$v" ] || { echo "fetch_secrets: secret $1 missing or not readable" >&2; exit 3; }; printf '%s' "$v"; }

# Plain assignments, so `set -e` stops the script if a required secret is missing (it would not inside a heredoc).
TPUF="$(need scout-turbopuffer-api-key)"
VOY="$(need scout-voyage-api-key)"
MASTER="$(need scout-master-key)"
PEPPER="$(need scout-pepper)"
VLLM_KEY="$(need scout-vllm-api-key)"
HF="$(get scout-hf-token)"
cat >/etc/scout/gateway.env <<EOT
TURBOPUFFER_API_KEY=${TPUF}
VOYAGE_API_KEY=${VOY}
SCOUT_MASTER_KEY=${MASTER}
SCOUT_PEPPER=${PEPPER}
SCOUT_LLM_URL=http://127.0.0.1:8000/v1
SCOUT_LLM_API_KEY=${VLLM_KEY}
SCOUT_LLM_MODEL=wayfinder-glimmer
SCOUT_LLM_SAMPLING=sft
EOT
cat >/etc/scout/vllm.env <<EOT
VLLM_API_KEY=${VLLM_KEY}
HF_TOKEN=${HF}
WAYFINDER_QUANT=${WAYFINDER_QUANT:-bf16}
HF_HOME=/opt/wayfinder/hf
HF_HUB_DISABLE_TELEMETRY=1
EOT
chmod 600 /etc/scout/gateway.env /etc/scout/vllm.env
chown root:root /etc/scout/gateway.env /etc/scout/vllm.env
echo "fetch_secrets: wrote /etc/scout/gateway.env and /etc/scout/vllm.env (mode 600)"
