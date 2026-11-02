"""The vLLM command line for wayfinder-glimmer, rendered from the environment (no vLLM import, so it runs anywhere).

Flags are the ones of the Modal serving app (training/serve/wayfinder_v3.py, `VLLM_ARGS`): bf16, 65,536 context,
92% GPU memory, prefix caching, language model only, auto tool choice with the `muse_glimmer` tool-call and
reasoning parsers, 32 concurrent sequences, and the checkpoint's own chat template (no `--chat-template`). Differences
on the VM: the host is localhost, the bearer key comes from VLLM_API_KEY, the served name is `wayfinder-glimmer`, and
`WAYFINDER_QUANT=fp8` adds `--quantization fp8` (weight-only FP8 on an A100, which has no FP8 tensor cores).

    python vllm_args.py            # prints the full `vllm serve ...` command, one shell-quoted line
    python vllm_args.py --exec     # exec's it (what the systemd unit runs)
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Mapping

DEFAULTS = {
    "WAYFINDER_MODEL": "hweller1/wayfinder-glimmer-30b",
    "WAYFINDER_SERVED_NAME": "wayfinder-glimmer",
    "WAYFINDER_MAX_MODEL_LEN": "65536",
    "WAYFINDER_GPU_UTIL": "0.92",
    "WAYFINDER_MAX_SEQS": "32",
    "WAYFINDER_QUANT": "bf16",
    "WAYFINDER_PORT": "8000",
}
QUANT = ("bf16", "fp8")


def render(env: Mapping[str, str]) -> list[str]:
    cfg = {k: (env.get(k) or v) for k, v in DEFAULTS.items()}
    if cfg["WAYFINDER_QUANT"] not in QUANT:
        raise SystemExit(f"WAYFINDER_QUANT must be one of {', '.join(QUANT)}")
    if not env.get("VLLM_API_KEY"):
        raise SystemExit("VLLM_API_KEY is not set (the bearer key vLLM and the gateway share)")
    args = [
        "vllm", "serve", cfg["WAYFINDER_MODEL"],
        "--served-model-name", cfg["WAYFINDER_SERVED_NAME"],
        "--dtype", "bfloat16",
        "--max-model-len", cfg["WAYFINDER_MAX_MODEL_LEN"],
        "--gpu-memory-utilization", cfg["WAYFINDER_GPU_UTIL"],
        "--enable-prefix-caching",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "muse_glimmer",
        "--reasoning-parser", "muse_glimmer",
        "--max-num-seqs", cfg["WAYFINDER_MAX_SEQS"],
        "--host", "127.0.0.1",
        "--port", cfg["WAYFINDER_PORT"],
    ]  # fmt: skip
    if cfg["WAYFINDER_QUANT"] == "fp8":
        args += ["--quantization", "fp8"]
    return args


def main(argv: list[str]) -> int:
    args = render(os.environ)
    if "--exec" in argv:
        # The bearer key goes through the environment (VLLM_API_KEY is read by vLLM), never on the command line.
        os.execvp(args[0], args)
    print(shlex.join(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
