# Run wayfinder-glimmer and the Scout gateway on one GCP VM

This kit puts the model and the gateway on a single `a2-ultragpu-1g` (1x A100 80 GB) as an alternative to the Modal path in `deploy/modal`. Nothing in this repository creates a cloud resource by itself: `create_vm.sh` prints its commands unless you run it with `CONFIRM=yes`, and the scripts here were written and checked offline only (see "What was and was not tested").

```
laptop (Muse Code / Claude Code + Scout plugin)
      |  HTTPS 443 (Caddy, automatic certificate)
      v
 VM  Caddy  ->  scout-gateway (uvicorn, 127.0.0.1:8080)  ->  vLLM 0.32.0 (127.0.0.1:8000, wayfinder-glimmer)
                      |
                      +--> turbopuffer (vectors, encrypted paths)      Voyage API (voyage-code-4 embeddings)
```

Only ports 80 and 443 are open to the internet (80 is for the certificate challenge). vLLM and the gateway listen on localhost. SSH goes through Identity-Aware Proxy.

## Files

| File | Purpose |
|---|---|
| `create_vm.sh` | Static IP, firewall rules, the VM with the startup script. Prints the commands; `CONFIRM=yes` runs them |
| `startup.sh` | GCE startup script: first boot runs `install.sh`, every boot refreshes secrets and starts the services |
| `install.sh` | Installs vLLM 0.32.0 (own venv), the gateway (`uv sync --frozen` from `scout/uv.lock`), Caddy, the units, and pre-fetches the weights |
| `vllm_args.py` | Renders the `vllm serve` command from the environment (flags identical to the Modal app; a test checks that) |
| `gateway_vm.py` | Gateway entrypoint: `serve`, `issue-key --tenant T`, `check-config` |
| `systemd/` | `wayfinder-vllm.service`, `scout-gateway.service` |
| `Caddyfile.template` | HTTPS reverse proxy to the gateway only |
| `fetch_secrets.sh` | Writes `/etc/scout/*.env` (mode 600) from Secret Manager |
| `env/*.env.example` | The variable names, all values empty |
| `healthcheck.sh` | `local` (on the VM) and `remote` (from your laptop) checks |
| `tests/` | Offline tests: flag parity with the Modal app, config parsing, gateway import and healthz, no secrets in the examples |

## 1. Secrets (once, from your laptop)

Secrets never enter the repository or the VM image. Put them in Secret Manager, or skip Secret Manager and write `/etc/scout/gateway.env` and `/etc/scout/vllm.env` by hand on the VM (copy the `env/*.example` files, fill them in, `chmod 600`).

```bash
export PROJECT=<your-project>
gcloud services enable secretmanager.googleapis.com compute.googleapis.com iap.googleapis.com --project $PROJECT
mk() { printf '%s' "$2" | gcloud secrets create "$1" --project $PROJECT --data-file=- ; }
mk scout-turbopuffer-api-key "$TURBOPUFFER_API_KEY"
mk scout-voyage-api-key      "$VOYAGE_API_KEY"
mk scout-master-key          "$SCOUT_MASTER_KEY"   # base64, 32 bytes
mk scout-pepper              "$SCOUT_PEPPER"       # base64, 32 bytes
mk scout-vllm-api-key        "$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
mk scout-hf-token            "$HF_TOKEN"           # only while hweller1/wayfinder-glimmer-30b is private
```

`SCOUT_MASTER_KEY` and `SCOUT_PEPPER` decide whether the VM can read an index that an earlier gateway wrote. Reuse the values of that gateway and its `SCOUT_KEYS_NS` to keep existing scout keys and indexes working. New values mean an empty index and new keys (`gateway_vm.py issue-key`), and every repo is re-embedded on its first sync.

Give the VM a service account that can read them:

```bash
gcloud iam service-accounts create wayfinder-vm --project $PROJECT
for s in scout-turbopuffer-api-key scout-voyage-api-key scout-master-key scout-pepper scout-vllm-api-key scout-hf-token; do
  gcloud secrets add-iam-policy-binding $s --project $PROJECT \
    --member serviceAccount:wayfinder-vm@$PROJECT.iam.gserviceaccount.com --role roles/secretmanager.secretAccessor
done
```

## 2. Create the VM (the author runs this)

```bash
cd deploy/gcp
PROJECT=$PROJECT ZONE=us-central1-a \
  SA=wayfinder-vm@$PROJECT.iam.gserviceaccount.com \
  REPO_URL=https://github.com/<author>/<repo>.git \
  ./create_vm.sh                      # prints the commands; review them
CONFIRM=yes PROJECT=$PROJECT ZONE=us-central1-a SA=... REPO_URL=... ./create_vm.sh
```

The script reserves a static IP, opens 80 and 443 to the VM's network tag, opens SSH to the IAP range only, and creates the VM from a Deep Learning VM image with the GPU driver preinstalled. Details to check before you run it:

- The image family default (`common-cu129-ubuntu-2204-nvidia-570`) is a guess at the current name. List the real ones with `gcloud compute images list --project deeplearning-platform-release --no-standard-images | grep cu12` and pass `IMAGE_FAMILY=...`. Pick one whose driver is 570 or newer: the CUDA 12.9 wheels that vLLM 0.32.0 pulls need it.
- The `default` network ships a `default-allow-ssh` rule open to the whole internet. Delete it (`gcloud compute firewall-rules delete default-allow-ssh`) or restrict it, so SSH is reachable only through IAP.
- Use the zone of your reservation (`RESERVATION=<name>` adds the reservation affinity).
- If you have no domain, the hostname defaults to `<ip-with-dashes>.sslip.io`, which resolves to your IP and gets a Let's Encrypt certificate from Caddy on first request. Pass `SCOUT_HOSTNAME=demo.example.com` to use your own DNS name (point an A record at the static IP first).
- If you copy the repository to the VM instead of cloning it, leave `REPO_URL` empty and run `gcloud compute scp --recurse --tunnel-through-iap . wayfinder-demo:/tmp/repo` then move it to `/opt/wayfinder/repo` before the first boot finishes (or run `install.sh` by hand).

## 3. Startup and health checks

First boot installs everything and downloads about 60 GB of weights, which takes 15 to 25 minutes. Watch it with `gcloud compute ssh wayfinder-demo --tunnel-through-iap -- sudo journalctl -u google-startup-scripts -f`. Later boots only start the services; vLLM needs 3 to 8 minutes to load the weights and capture CUDA graphs before `/health` answers.

```bash
gcloud compute ssh wayfinder-demo --zone us-central1-a --tunnel-through-iap -- sudo /opt/wayfinder/repo/deploy/gcp/healthcheck.sh local
SCOUT_GATEWAY_URL=https://<host> deploy/gcp/healthcheck.sh remote
```

`local` checks the three services, the GPU, vLLM `/health`, that the model is listed, that a real tool-call turn parses (the `muse_glimmer` parser), the gateway's `/v1/healthz`, and that vLLM and the gateway listen on loopback only. `remote` checks the gateway over HTTPS, that an unauthenticated call is refused, and that ports 8000 and 8080 are closed from the outside.

Issue a scout key for the demo tenant (printed once, only its HMAC tag is stored):

```bash
sudo bash -c 'set -a; . /etc/scout/gateway.env; set +a; /opt/wayfinder/gateway-venv/bin/python /opt/wayfinder/repo/deploy/gcp/gateway_vm.py issue-key --tenant demo'
```

## 4. Point the Muse Code plugin at the gateway

The plugin and the CLI read `SCOUT_GATEWAY_URL` and the key; nothing else is needed.

```bash
export SCOUT_GATEWAY_URL=https://<host>
printf '%s\n' "<the scout key>" > ~/.config/scout/demo.key && chmod 600 ~/.config/scout/demo.key
scout login            # or: SCOUT_KEY_FILE / --key-file as in docs/demo.md
scout --root /path/to/checkout sync --wait
scripts/demo_panes.sh setup <author/repo> <commit>      # the side-by-side runbook, docs/demo.md
```

`scripts/demo_warm.sh on|off|status|wait` starts and stops the VM and arms a watcher that stops it after N hours (set `GCP_PROJECT`, `GCP_ZONE`, `VM_NAME`).

## bf16 and FP8

Default is bf16, the precision every evaluation used (on an H100). The weights are about 60 GB, so at `--gpu-memory-utilization 0.92` of 80 GB roughly 13 GB is left for the KV cache. That fits the 65,536-token context with a handful of concurrent sequences, which is what a demo needs; a long, heavily concurrent run may see vLLM preempt sequences, and the log says so.

`WAYFINDER_QUANT=fp8` (in `vllm.env`, or the `scout-quant` instance metadata key) adds `--quantization fp8`. On an A100 this is weight-only FP8 through vLLM's Marlin kernels, since the card has no FP8 tensor cores: weights shrink to about 30 GB and the KV cache grows to about 40 GB, at some accuracy risk. wayfinder-glimmer's recall was never measured in FP8, so use bf16 unless memory is the problem, and re-run a quick check (the held-out retrieval harness) if you switch.

The same VM is slower than the H100 the results were measured on: an A100 has about 60 percent of an H100's memory bandwidth, which is what bounds decoding. Expect Scout searches to take longer than the 2 to 3 minutes reported on the H100.

## Cost at list price

| Item | Rate | Note |
|---|---|---|
| `a2-ultragpu-1g` on demand, us-central1 | about $5.07 per hour | Figure from memory of Google's public price list; check the pricing page and your reservation terms. 8 hours is about $41 |
| Boot disk, 300 GB `pd-balanced` | about $30 per month | Bills while the VM is stopped |
| Static IP | free while attached to a running VM, a few cents a day when reserved and unused | |
| Modal H100 (what the evaluations used) | about $3.95 per hour | for comparison |

A reservation bills whether or not the VM runs. Stopping the VM stops the GPU line of the bill, not the disk or a reservation.

## Teardown

```bash
scripts/demo_warm.sh off                       # stop the VM (keeps the disk, so the next start skips the install)
# or remove everything:
gcloud compute instances delete wayfinder-demo --zone us-central1-a --project $PROJECT
gcloud compute addresses delete wayfinder-demo-ip --region us-central1 --project $PROJECT
gcloud compute firewall-rules delete wayfinder-demo-web wayfinder-demo-iap-ssh --project $PROJECT
gcloud secrets delete scout-vllm-api-key --project $PROJECT   # and the others, if you are done with them
```

`create_vm.sh` sets `--max-run-duration` (default 12 h) with termination action STOP as a backstop if you forget. If your `gcloud` does not know that flag, update it or remove the two lines.

## What was and was not tested

Tested offline, on a CPU laptop: the vLLM flag list against the Modal serving app's (`tests/test_gcp_kit.py`), environment parsing and the fail-closed error that names missing variables without their values, the gateway app builds from the entrypoint and answers `/v1/healthz`, the key cache, the example env files hold no secrets, only the gateway is proxied, and every shell script parses (`bash -n`). Run them with `cd scout && uv run --extra gateway python -m pytest ../deploy/gcp/tests -q`.

Not tested: anything on a real VM. The install script, the systemd units, Caddy certificate issuance, the Secret Manager fetch, vLLM 0.32.0 on an A100 with CUDA 12.9 wheels, FP8, and the latency on an A100 are all unrun. Treat the first boot as a dry run and expect to fix something small. The Modal app used an H100 and the CUDA 12.9 base image; `install.sh` warns if the driver is older than 565.
