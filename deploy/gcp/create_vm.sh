#!/usr/bin/env bash
# The owner runs this once. It creates billable resources (an A100 80 GB VM, a static IP, a firewall rule) in the
# project and zone below. Nothing here has been run by the author of the repo. Set CONFIRM=yes to run it for real;
# without it the commands are only printed.
#
#   PROJECT=my-project CONFIRM=yes ./create_vm.sh
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-us-central1-a}"                       # a2-ultragpu-1g capacity is per zone; use the zone you reserved in
VM="${VM:-wayfinder-demo}"
MACHINE="${MACHINE:-a2-ultragpu-1g}"                # 1x A100 80 GB, 12 vCPU, 170 GB RAM
IMAGE_PROJECT="deeplearning-platform-release"
# List current families with:  gcloud compute images list --project deeplearning-platform-release --no-standard-images | grep cu12
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-570}"   # CUDA 12.x + driver preinstalled (verify the name)
DISK_GB="${DISK_GB:-300}"                           # 60 GB weights + caches + OS
REPO_URL="${REPO_URL:-}"                            # https clone URL of the public repo, or empty if you scp it
QUANT="${QUANT:-bf16}"
MAX_RUN="${MAX_RUN:-12h}"                           # the VM stops itself after this long (backstop against a forgotten GPU)
RESERVATION="${RESERVATION:-}"                      # name of your reservation, if you made one
SA="${SA:-}"                                        # service account email with secretmanager.secretAccessor

run() { echo "+ $*"; [ "${CONFIRM:-no}" = yes ] && "$@" || true; }

run gcloud compute addresses create "$VM-ip" --project "$PROJECT" --region "${ZONE%-*}"
IP="$( [ "${CONFIRM:-no}" = yes ] && gcloud compute addresses describe "$VM-ip" --project "$PROJECT" --region "${ZONE%-*}" --format='value(address)' || echo 1.2.3.4)"
HOSTNAME_="${SCOUT_HOSTNAME:-${IP//./-}.sslip.io}"
echo "gateway hostname: $HOSTNAME_  (static IP $IP)"

# Public: 80 and 443 only (ACME challenge and HTTPS to Caddy). vLLM (8000) and the gateway (8080) listen on localhost.
run gcloud compute firewall-rules create "$VM-web" --project "$PROJECT" --network default \
  --allow tcp:80,tcp:443 --target-tags "$VM" --direction INGRESS --source-ranges 0.0.0.0/0
# SSH only through Identity-Aware Proxy (gcloud compute ssh --tunnel-through-iap).
run gcloud compute firewall-rules create "$VM-iap-ssh" --project "$PROJECT" --network default \
  --allow tcp:22 --target-tags "$VM" --direction INGRESS --source-ranges 35.235.240.0/20

args=(gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE"
  --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced
  --maintenance-policy TERMINATE --address "$VM-ip" --tags "$VM"
  --max-run-duration "$MAX_RUN" --instance-termination-action STOP
  --scopes cloud-platform
  --metadata "scout-hostname=$HOSTNAME_,scout-quant=$QUANT,scout-repo-url=$REPO_URL,install-nvidia-driver=True"
  --metadata-from-file "startup-script=$(dirname "$0")/startup.sh")
[ -n "$SA" ] && args+=(--service-account "$SA")
[ -n "$RESERVATION" ] && args+=(--reservation-affinity specific --reservation "$RESERVATION")
run "${args[@]}"
echo "next: see deploy/gcp/README.md (secrets, health checks)"
