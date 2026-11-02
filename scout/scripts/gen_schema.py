"""Write `schema/v1/*.json` from the contract models, or `--check` that the committed files are current."""

import argparse
import sys
import tempfile
from pathlib import Path

from scout.core.contracts.schema import export_schemas

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "schema" / "v1"


def drift(committed: Path, fresh: Path) -> list[str]:
    names = {p.name for p in committed.glob("*.json")} | {p.name for p in fresh.glob("*.json")}
    out: list[str] = []
    for name in sorted(names):
        a, b = committed / name, fresh / name
        if not a.exists():
            out.append(f"missing: {name}")
        elif not b.exists():
            out.append(f"stale (no model): {name}")
        elif a.read_bytes() != b.read_bytes():
            out.append(f"differs: {name}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on any byte difference")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="committed schema directory")
    args = parser.parse_args(argv)
    if not args.check:
        for stale in args.dir.glob("*.json") if args.dir.exists() else []:
            stale.unlink()
        paths = export_schemas(args.dir)
        print(f"wrote {len(paths)} files to {args.dir}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        export_schemas(Path(tmp))
        problems = drift(args.dir, Path(tmp))
    if problems:
        print("schema drift (run `uv run python scripts/gen_schema.py`):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
