#!/usr/bin/env bash
# Install wayfinder-glimmer (vLLM 0.32.0) and the Scout gateway on a fresh GCP Deep Learning VM (Debian or Ubuntu,
# CUDA 12.x driver already present). Idempotent; run as root:
#
#   sudo REPO_URL=https://github.com/<owner>/<repo>.git SCOUT_HOSTNAME=34-1-2-3.sslip.io ./install.sh
#
# or, to deploy a local checkout without a Git remote, copy the repo to /opt/wayfinder/repo first
# (gcloud compute scp --recurse) and leave REPO_URL unset.
set -euo pipefail

: "${SCOUT_HOSTNAME:?set SCOUT_HOSTNAME to the DNS name of this VM (e.g. 34-1-2-3.sslip.io)}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"      # the version the Modal serving app and every evaluation used
PY_VERSION="${PY_VERSION:-3.12}"            # scout needs >= 3.12; the Deep Learning image python is older
BASE=/opt/wayfinder
REPO="$BASE/repo"

log() { printf '[install] %s\n' "$*"; }

log "driver check"
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found: use a Deep Learning VM image with the driver installed" >&2; exit 2; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "${drv:-0}" -ge 565 ] || log "WARNING: driver $drv is old for the CUDA 12.9 wheels vLLM $VLLM_VERSION pulls; use an image with driver 570+ if vllm fails to start"

log "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ca-certificates debian-keyring debian-archive-keyring apt-transport-https gnupg build-essential

if ! command -v caddy >/dev/null; then
  log "caddy"
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -qq && apt-get install -y -qq caddy
fi

id scout >/dev/null 2>&1 || useradd --system --create-home --home-dir "$BASE/home" --shell /usr/sbin/nologin scout
install -d -o scout -g scout "$BASE" "$BASE/hf"

if ! command -v uv >/dev/null; then
  log "uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi

if [ -n "${REPO_URL:-}" ]; then
  log "repo from $REPO_URL"
  if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else git clone --depth 1 "$REPO_URL" "$REPO"; fi
fi
[ -f "$REPO/deploy/gcp/gateway_vm.py" ] || { echo "no checkout at $REPO (set REPO_URL or copy the repo there)" >&2; exit 2; }
chown -R scout:scout "$REPO"

log "vLLM $VLLM_VERSION venv"
export UV_PYTHON_INSTALL_DIR="$BASE/python" UV_CACHE_DIR="$BASE/uv-cache"
[ -x "$BASE/vllm-venv/bin/python" ] || uv venv --python "$PY_VERSION" "$BASE/vllm-venv"
uv pip install --python "$BASE/vllm-venv/bin/python" "vllm==$VLLM_VERSION" huggingface_hub
"$BASE/vllm-venv/bin/vllm" download-kernels >/dev/null 2>&1 || log "vllm download-kernels unavailable; FlashInfer will JIT on first start"

log "gateway venv (scout[gateway], pinned by uv.lock)"
( cd "$REPO/scout" && UV_PROJECT_ENVIRONMENT="$BASE/gateway-venv" uv sync --frozen --no-dev --extra gateway )

log "config and units"
install -d -m 700 /etc/scout
for f in gateway vllm; do
  [ -f "/etc/scout/$f.env" ] || install -m 600 -o root -g root "$REPO/deploy/gcp/env/$f.env.example" "/etc/scout/$f.env"
done
install -m 644 "$REPO/deploy/gcp/systemd/wayfinder-vllm.service" /etc/systemd/system/
install -m 644 "$REPO/deploy/gcp/systemd/scout-gateway.service" /etc/systemd/system/
sed "s/{\$SCOUT_HOSTNAME}/$SCOUT_HOSTNAME/" "$REPO/deploy/gcp/Caddyfile.template" > /etc/caddy/Caddyfile
systemctl daemon-reload

log "pre-fetching weights (needs HF_TOKEN in /etc/scout/vllm.env while the repo is private)"
set +u; set -a; . /etc/scout/vllm.env; set +a; set -u
runuser -u scout -- env HF_HOME="$BASE/hf" HF_TOKEN="${HF_TOKEN:-}" WAYFINDER_MODEL="${WAYFINDER_MODEL:-}" "$BASE/vllm-venv/bin/python" - <<'PY' || log "weight pre-fetch failed; vLLM will download on first start"
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ.get("WAYFINDER_MODEL") or "hweller1/wayfinder-glimmer-30b", token=os.environ.get("HF_TOKEN") or None)
PY

log "done. Next: fill /etc/scout/*.env (or run fetch_secrets.sh), then: systemctl enable --now wayfinder-vllm scout-gateway caddy"
