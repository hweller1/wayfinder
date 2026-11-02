"""turbopuffer namespace janitor. Stdlib only (turbopuffer imported lazily); never imported by scout.core.

Only ever touches namespaces whose name starts with `scout_dev_` or `scout_eval_`. The `--prefix` must itself
start with one of those (so empty, `scout_`, `scout_dev` without the underscore, and any foreign prefix are
refused), and every listed name is re-checked against both the allowed prefix and `--prefix` before any
delete, so a client that ignores the prefix filter still cannot cause an out-of-scope delete.

Dry-run by default (lists what it would delete). `--delete` is explicit. `--verify-empty` re-lists (after
deleting when `--delete` is also given) and exits 1 with the leftover list if anything outside `--keep`
remains.

Exit codes: 0 ok, 1 leftovers or failed deletes, 2 refused/usage/configuration.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from typing import Protocol

ALLOWED_PREFIXES = ("scout_dev_", "scout_eval_")
_PREFIX_RE = re.compile(r"^scout_(?:dev|eval)_[A-Za-z0-9_.-]*$")
EXIT_OK, EXIT_LEFTOVER, EXIT_REFUSED = 0, 1, 2


class NamespaceClient(Protocol):
    def list_namespaces(self, prefix: str) -> list[str]: ...
    def delete_namespace(self, name: str) -> None: ...


class PrefixRefusedError(ValueError):
    pass


def validate_prefix(prefix: str) -> str:
    if not prefix.startswith(ALLOWED_PREFIXES) or not _PREFIX_RE.fullmatch(prefix):
        raise PrefixRefusedError(
            f"refusing prefix {prefix!r}: must start with one of {ALLOWED_PREFIXES} (underscore included)"
        )
    return prefix


def in_scope(name: str, prefix: str) -> bool:
    return name.startswith(ALLOWED_PREFIXES) and name.startswith(prefix)


class FakeClient:
    """In-memory client (`--backend fake` and unit tests). Mirrors prefix filtering of the real API."""

    def __init__(self, namespaces: Sequence[str] = ()) -> None:
        self.namespaces: set[str] = set(namespaces)
        self.deleted: list[str] = []

    def list_namespaces(self, prefix: str) -> list[str]:
        return sorted(n for n in self.namespaces if n.startswith(prefix))

    def delete_namespace(self, name: str) -> None:
        self.namespaces.discard(name)
        self.deleted.append(name)


class TurbopufferClient:
    """Real client; `turbopuffer` imported lazily and the key read from the environment only."""

    def __init__(self) -> None:
        key = os.environ.get("TURBOPUFFER_API_KEY")
        if not key:
            raise RuntimeError("TURBOPUFFER_API_KEY is not set (load .env)")
        import turbopuffer

        # SDK 3.0.0 requires a region; gcp-us-central1 verified live (spike 0a-A) with our key
        region = os.environ.get("TURBOPUFFER_REGION") or "gcp-us-central1"
        self._tpuf = turbopuffer.Turbopuffer(api_key=key, region=region)

    def list_namespaces(self, prefix: str) -> list[str]:
        return sorted(ns.id for ns in self._tpuf.namespaces(prefix=prefix))

    def delete_namespace(self, name: str) -> None:
        self._tpuf.namespace(name).delete_all()


def run(
    client: NamespaceClient,
    prefix: str,
    *,
    keep: Sequence[str] = (),
    delete: bool = False,
    verify_empty: bool = False,
    out=print,
) -> int:
    """Core logic (client injected). Returns the process exit code."""
    try:
        validate_prefix(prefix)
    except PrefixRefusedError as exc:
        out(f"REFUSED: {exc}")
        return EXIT_REFUSED
    keep_set = set(keep)
    listed = client.list_namespaces(prefix)
    skipped = [n for n in listed if not in_scope(n, prefix)]
    for n in skipped:
        out(f"SKIP out-of-scope namespace returned by client (never touched): {n}")
    targets = [n for n in listed if in_scope(n, prefix) and n not in keep_set]
    kept = [n for n in listed if in_scope(n, prefix) and n in keep_set]
    for n in kept:
        out(f"keep {n}")
    failures: list[str] = []
    if delete:
        for n in targets:
            try:
                client.delete_namespace(n)
                out(f"deleted {n}")
            except Exception as exc:  # noqa: BLE001 - report and continue, exit non-zero below
                failures.append(n)
                out(f"FAILED to delete {n}: {type(exc).__name__}")
    else:
        for n in targets:
            out(f"would delete {n}")
        out(f"dry-run: {len(targets)} namespace(s) would be deleted; pass --delete to delete")
    if not verify_empty:
        return EXIT_LEFTOVER if failures else EXIT_OK
    leftover = [n for n in client.list_namespaces(prefix) if in_scope(n, prefix) and n not in keep_set]
    if leftover:
        out(f"LEFTOVER {len(leftover)} namespace(s) with prefix {prefix}: {', '.join(leftover)}")
        return EXIT_LEFTOVER
    out(f"verified empty: no namespaces with prefix {prefix} (excluding --keep)")
    return EXIT_LEFTOVER if failures else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ns_janitor.py", description=__doc__.split("\n")[0])
    p.add_argument("--prefix", required=True, help="scout_dev_... or scout_eval_...")
    p.add_argument("--keep", nargs="*", default=[], metavar="NS", help="namespaces to leave alone")
    p.add_argument("--delete", action="store_true", help="actually delete (default is dry-run)")
    p.add_argument("--verify-empty", action="store_true", help="exit 1 if any namespace remains")
    p.add_argument("--backend", choices=("tpuf", "fake"), default="tpuf")
    p.add_argument("--fake-ns", action="append", default=[], help="seed the fake backend (repeatable)")
    return p


def main(argv: Sequence[str] | None = None, client: NamespaceClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_prefix(args.prefix)  # refuse before any client (or key) is touched
    except PrefixRefusedError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    if client is None:
        if args.backend == "fake":
            client = FakeClient(args.fake_ns)
        else:
            try:
                client = TurbopufferClient()
            except (RuntimeError, ImportError) as exc:
                print(f"cannot create turbopuffer client: {exc}", file=sys.stderr)
                return EXIT_REFUSED
    return run(client, args.prefix, keep=args.keep, delete=args.delete, verify_empty=args.verify_empty)


if __name__ == "__main__":
    sys.exit(main())
