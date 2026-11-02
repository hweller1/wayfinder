#!/usr/bin/env python3
"""Keep the Muse Code plugin's copy of the retrieval-spec code identical to the Claude Code plugin's.

Both hosts read one spec through one implementation. `integrations/claude-code/lib/scout_spec.py` is the
source and `integrations/muse-code/scout-wayfinder/lib/scout_spec.py` is a verbatim copy (a Muse plugin is
installed on its own and cannot import across directories), likewise `spec/harness_defaults.toml`.

    python3 scripts/sync_muse_spec.py           # copy
    python3 scripts/sync_muse_spec.py --check   # exit 1 when they differ (a unit test runs this)
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "integrations" / "claude-code"
MUSE = ROOT / "integrations" / "muse-code" / "scout-wayfinder"
PAIRS = (
    (CC / "lib" / "scout_spec.py", MUSE / "lib" / "scout_spec.py"),
    (CC / "spec" / "harness_defaults.toml", MUSE / "spec" / "harness_defaults.toml"),
)


def drifted() -> list[str]:
    return [
        str(dst.relative_to(ROOT))
        for src, dst in PAIRS
        if not dst.exists() or dst.read_bytes() != src.read_bytes()
    ]


def main(argv: list[str]) -> int:
    if "--check" in argv:
        bad = drifted()
        if bad:
            print("out of sync with the Claude Code copy (run scripts/sync_muse_spec.py): " + ", ".join(bad))
        return 1 if bad else 0
    for src, dst in PAIRS:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print("synced " + ", ".join(str(d.relative_to(ROOT)) for _, d in PAIRS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
