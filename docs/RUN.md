This is the full deployment runbook moved out of the README. It covers wayfinder-glimmer and the Scout gateway; wayfinder-qwen is served the same way with `deploy/modal/serve_qwen.py` (see [MODEL_CARD_QWEN.md](MODEL_CARD_QWEN.md)).

# How to run (wayfinder-glimmer, Modal and GCP)

Everything below that touches a cloud service needs your own keys and your own account. There are two ways to serve the model and the gateway: Modal (first) and one GCP VM (second).

**Tests (offline).**

```bash
cd scout && uv sync --frozen --extra gateway
uv run pytest -q                                            # contract, unit, conformance and security tests; no network
uv run python -m pytest ../deploy/gcp/tests ../deploy/modal/tests -q   # the deployment kits' offline checks
cd ../training && PYTHONPATH=.:../scout/src python3 -m pytest tests -q
```

## Option 1: Run on Modal (your own account)

This deploys two apps in your Modal workspace: wayfinder-glimmer on one H100 ([deploy/modal/serve_glimmer.py](../deploy/modal/serve_glimmer.py)) and the Scout gateway on a small CPU container ([scout/deploy/modal/gateway.py](../scout/deploy/modal/gateway.py)). Both scale to zero when idle. The Modal path is written and tested offline; it has not been deployed from a fresh account (see [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md)), so expect to read a Modal error or two the first time.

**Cost.** While the H100 is warm it bills about $3.95 per hour. When the app has scaled to zero (after 10 idle minutes by default) it costs nothing, and the gateway's CPU container is a small fraction of that. The first request after idle waits several minutes while a container starts and vLLM loads the model; the very first start also downloads about 60 GB of weights and compiles kernels, and takes longer. Embedding is billed by Voyage, about $1.72 per million lines indexed. `max_containers` is 1, so the GPU bill cannot exceed one H100.

**1. Prerequisites.** A [Modal](https://modal.com) account with GPU access (H100), [uv](https://docs.astral.sh/uv/), Python 3.12, and:

```bash
uv tool install modal && modal setup          # logs the CLI in to your workspace
modal profile list                            # note the workspace name, used below as <workspace>
export VOYAGE_API_KEY=...                     # https://dash.voyageai.com
export TURBOPUFFER_API_KEY=...                # https://turbopuffer.com
```

**2. Create the secrets.** One command creates `wayfinder-glimmer-key` (the model's bearer key) and `scout-gateway` (your Voyage and turbopuffer keys, a newly generated master key and pepper, and the model's URL and key). Values go through a 0600 temporary file, never on a command line or to the terminal.

```bash
python3 deploy/modal/make_secrets.py --workspace <workspace>
```

The generated keys are kept in `~/.config/scout/modal-keys.json` (mode 0600), so a re-run keeps them. Back that file up: the master key and pepper decide whether a gateway can read an index it wrote earlier. The script refuses to overwrite an existing secret without `--force`. If you deploy into a non-default Modal environment, pass the URL that step 3 prints with `--llm-url` instead of `--workspace`.

**3. Deploy the model app.** From the repository root. No Hugging Face token is needed; the weights are public.

```bash
modal deploy deploy/modal/serve_glimmer.py
```

It serves `hweller1/wayfinder-glimmer-30b` as `wayfinder-glimmer` (vLLM 0.32.0, the flags of every published result, a `/warm` endpoint) at `https://<workspace>--wayfinder-glimmer-serve.modal.run`. Nothing starts until the first request. Weights and compile caches are kept in two Modal Volumes. Knobs (environment variables at deploy time): `WAYFINDER_MIN_CONTAINERS` (0), `WAYFINDER_MAX_CONTAINERS` (1), `WAYFINDER_SCALEDOWN_S` (600).

**4. Deploy the gateway.**

```bash
cd scout
modal deploy deploy/modal/gateway.py
```

The deploy builds a wheel of this checkout with uv and prints the gateway URL, `https://<workspace>--scout-gateway-api.modal.run`. That is your `SCOUT_GATEWAY_URL`.

**5. Issue a Scout key.** Still in `scout/`:

```bash
modal run deploy/modal/gateway.py::issue_key --tenant me
```

It prints `tenant=me` and `key=...` once; the gateway stores only a tag of the key, so save it now. Run it again with another `--tenant` for each person; tenants cannot see each other's index.

**6. Install the plugin and point it at your gateway.** Install the scout CLI and a plugin, with the key from step 5:

```bash
export SCOUT_GATEWAY_URL=https://<workspace>--scout-gateway-api.modal.run
uv tool install ./scout && scout login --key <scout key>
export SCOUT_INTEGRATIONS_DIR=$PWD/scout/integrations      # where `scout init` finds the plugins
scout init --muse-code          # or: scout init --claude-code
```

Details: [scout/integrations/muse-code/INSTALL.md](../scout/integrations/muse-code/INSTALL.md), [scout/integrations/claude-code/README.md](../scout/integrations/claude-code/README.md). Keep `SCOUT_GATEWAY_URL` exported in the shell that starts Muse Code or Claude Code.

**7. Warm up.** The plugin's session hook does this for you, but you can do it by hand: `scout warm --trigger` starts the model loading and returns in about a second, and `scout warm` waits until the model answers and prints how long that took (several minutes from zero). To check the model endpoint directly without printing the key:

```bash
KEY=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.config/scout/modal-keys.json")))["WAYFINDER_API_KEY"])')
curl -s -m 5 -H "Authorization: Bearer $KEY" https://<workspace>--wayfinder-glimmer-serve-warm.modal.run
```

**8. First search.** In a checkout, build the index, then ask for a search:

```bash
cd /path/to/your/repo
scout sync --wait                                # embeds the code with voyage-code-4 (the Voyage cost above)
scout explore "where is the retry backoff computed?" --text
```

Then start Muse Code or Claude Code in the same checkout; "explore" tasks go to the Scout subagent. Index public code until you have confirmed the Voyage data-retention setting (see Privacy above).

**Stop paying.** Nothing to do: the apps scale to zero. To stop a warm GPU at once, `modal app stop wayfinder-glimmer-serve`; to remove everything, stop both apps and delete the Volumes `wayfinder-glimmer-weights` and `wayfinder-glimmer-cache`.

## Option 2: one GCP VM

Serve the model and the gateway on one A100 80 GB VM: [deploy/gcp/README.md](../deploy/gcp/README.md). The kit was written and tested offline only; it has not been run on a real VM. It installs vLLM 0.32.0, serves `hweller1/wayfinder-glimmer-30b` as `wayfinder-glimmer` with the `muse_glimmer` tool-call and reasoning parsers, runs the gateway under uvicorn behind Caddy (HTTPS), and keeps vLLM on localhost. To serve the model alone: the command in [docs/MODEL_CARD.md](MODEL_CARD.md).

**Use Scout from Muse Code or Claude Code with a GCP gateway.** Set `SCOUT_GATEWAY_URL` to your gateway and `scout login` with a key from `gateway_vm.py issue-key`; then `scout sync --wait` in a checkout and install the plugin as in step 6 above. The side-by-side demo is [docs/demo.md](demo.md); the pane and VM scripts are `scout/scripts/demo_panes.sh` and `demo_warm.sh`.

**Train and evaluate.** [training/README.md](../training/README.md) (the recipe, and how to regenerate the teacher data) and [eval/README.md](../eval/README.md).
