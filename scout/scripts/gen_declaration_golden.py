"""One-off: regenerate the declaration-chunker golden from the sibling benchmark chunker (task 2.20).

Reads nothing from `../benchmark-repo` except through a read-only subprocess
(`uv run --project ../benchmark-repo python -`) that imports `asa.corpus.chunking` and prints JSON. Nothing in
`src/scout` imports it and the sibling is never modified.

Two input sets, both chunked with `chunk_code_file(..., strategy="declaration")`:
- `repo_small`: every file of `tests/fixtures/repo_small`.
- `real`: ~40 files sampled (fixed seed, sorted) from the sibling's vendored llama.cpp (C/C++/Python), copied
  to `tests/fixtures/declaration_real/` with the same relative paths.

Output `tests/fixtures/declaration_golden.json`: `{set: {path: [chunk body, ...]}}`. Bodies are the text the
benchmark embeds minus the leading `<path>\\n` header, which this script asserts is present on every chunk.

Usage: uv run python scripts/gen_declaration_golden.py [--llama ../benchmark-repo/vendor/llama.cpp]
"""

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEED = 20261026
QUOTA = {".py": 10, ".cpp": 10, ".h": 10, ".c": 5, ".hpp": 5}  # 40 files
MIN_BYTES, MAX_BYTES = 800, 40_000

_SIBLING = r"""
import json, sys
from asa.corpus.chunking import chunk_code_file
out = {}
for name, root in json.load(sys.stdin).items():
    files = {}
    for rel in root["paths"]:
        text = open(root["base"] + "/" + rel, "rb").read().decode("utf-8", errors="replace")
        chunks = chunk_code_file("scope", rel, text, strategy="declaration")
        bodies = []
        for c in chunks:
            assert c.text.startswith(rel + "\n"), rel
            bodies.append(c.text[len(rel) + 1:])
        files[rel] = bodies
    out[name] = files
json.dump(out, sys.stdout)
"""


def _sample_real(llama: Path) -> list[str]:
    tracked = subprocess.run(  # noqa: S603
        ["git", "-C", str(llama), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout.decode()
    by_suffix: dict[str, list[str]] = {s: [] for s in QUOTA}
    for rel in sorted(p for p in tracked.split("\0") if p):
        suffix = Path(rel).suffix.lower()
        path = llama / rel
        if suffix in by_suffix and path.is_file() and MIN_BYTES <= path.stat().st_size <= MAX_BYTES:
            by_suffix[suffix].append(rel)
    rng = random.Random(SEED)  # noqa: S311  (deterministic sampling, not crypto)
    picked = [rel for s, n in QUOTA.items() for rel in rng.sample(by_suffix[s], n)]
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", type=Path, default=ROOT.parent / "benchmark-repo" / "vendor" / "llama.cpp")
    args = ap.parse_args()

    real_dir = FIXTURES / "declaration_real"
    if real_dir.exists():
        shutil.rmtree(real_dir)
    real = _sample_real(args.llama)
    for rel in real:
        (real_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.llama / rel, real_dir / rel)
    shutil.copyfile(args.llama / "LICENSE", real_dir / "LICENSE")  # MIT

    small_root = FIXTURES / "repo_small"
    small = sorted(p.relative_to(small_root).as_posix() for p in small_root.rglob("*") if p.is_file())
    request = {
        "repo_small": {"base": str(small_root), "paths": small},
        "real": {"base": str(real_dir), "paths": real},
    }
    proc = subprocess.run(  # noqa: S603
        ["uv", "run", "--project", str(ROOT.parent / "benchmark-repo"), "python", "-c", _SIBLING],  # noqa: S607
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=True,
    )
    golden = json.loads(proc.stdout)
    (FIXTURES / "declaration_golden.json").write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    for name, files in golden.items():
        print(name, len(files), "files", sum(len(v) for v in files.values()), "chunks")


if __name__ == "__main__":
    main()
