"""`scout`: the CLI entry point. argparse only, and nothing heavier than `scout.core.contracts.errors` is
imported before parsing: the command module (which pulls in the app, adapters and, for `mcp`, the MCP SDK) is
imported after the arguments are known, so `scout --help` and a typo cost milliseconds (NFR-1).

JSON on stdout by default, `--text` for the plain rendering; a failure prints the error body on stdout, one
human line on stderr, and exits with the code of design section 4.
"""

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.surfaces.cli.exit import report_error


class _Parser(argparse.ArgumentParser):
    """A usage error is an `INVALID_ARGUMENT` like any other (exit 2, error body on stdout)."""

    def error(self, message: str) -> NoReturn:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, message)


def _common() -> _Parser:
    """Options every command accepts, before or after the command name (`SUPPRESS`: an absent flag must not
    overwrite one given on the other side)."""
    p = _Parser(add_help=False)
    s = argparse.SUPPRESS
    p.add_argument("--profile", default=s, help="composition: default (hosted gateway), fake, embedded, demo")
    p.add_argument("--root", default=s, help="repository root (default: the working directory)")
    p.add_argument("--gateway-url", default=s, help="gateway base URL (SCOUT_GATEWAY_URL)")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    p = _Parser(prog="scout", description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, parents=[common])
        sp.add_argument("--json", action="store_true", help="JSON on stdout (the default)")
        return sp

    ex = add("explore", "delegated search: ranked path:start-end ranges for a brief")
    ex.add_argument("brief")
    ex.add_argument(
        "--paths", nargs="+", default=None, metavar="GLOB", help="limit the search to these globs"
    )
    ex.add_argument("--max-results", type=int, default=None)
    ex.add_argument("--render-max-tokens", type=int, default=None)
    ex.add_argument("--max-turns", type=int, default=None)
    ex.add_argument("--max-seconds", type=int, default=None)
    ex.add_argument("--text", action="store_true", help="the plain-text rendering instead of JSON")

    se = add("search", "semantic search over the index (current file contents)")
    se.add_argument("query")
    se.add_argument("--path-glob", default=None)
    se.add_argument("-k", type=int, default=None)
    se.add_argument("--text", action="store_true")

    gr = add("grep", "ripgrep over the checkout")
    gr.add_argument("pattern")
    gr.add_argument("--path", default=None)
    gr.add_argument("--literal", action="store_true", help="treat the pattern as a literal string")
    gr.add_argument("--context", type=int, default=None)
    gr.add_argument("--text", action="store_true")

    rd = add("read", "read a line range of a file")
    rd.add_argument("path")
    rd.add_argument("start", type=int, nargs="?", default=None)
    rd.add_argument("end", type=int, nargs="?", default=None)
    rd.add_argument("--text", action="store_true")

    st = add("status", "index status of the checkout")
    st.add_argument("--text", action="store_true")

    wm = add(
        "warm", "wake the explore model behind the gateway (one 1-token call); prints the seconds it took"
    )
    wm.add_argument("--trigger", action="store_true",
                    help="only start the cold start (gateway /v1/warm, ~1 s); no waiting")  # fmt: skip

    mc = add(
        "mcp", "MCP server over stdio (or streamable HTTP with --http): the explore tool and the primitives"
    )
    mc.add_argument(
        "--tools",
        action="append",
        default=None,
        metavar="NAMES",
        help="comma list of tools to expose: explore, semantic_search, grep, read_range, index_status",
    )
    mc.add_argument(
        "--render-ceiling",
        type=int,
        default=None,
        metavar="N",
        help="cap, in tokens, on the plain-text rendering of any explore call",
    )
    mc.add_argument(
        "--structured", action="store_true", help="tool results are the JSON of the contract type"
    )
    mc.add_argument(
        "--http", type=int, default=None, metavar="PORT", help="serve streamable HTTP on loopback"
    )

    sv = add("serve", "local HTTP daemon: POST /explore /search /grep /read /status (loopback, bearer token)")
    sv.add_argument("--port", type=int, default=8765, help="port to listen on (0: pick a free one)")
    sv.add_argument("--host", default="127.0.0.1", help="loopback address to bind (anything else is refused)")
    sv.add_argument("--token-file", default=None, help="bearer token file, mode 0600 (created if absent)")

    lg = add("login", "verify a gateway key, store it (and the team key) in the key store")
    lg.add_argument(
        "--key", default=None, metavar="K", help="the key, or - to read it from stdin (SCOUT_KEY)"
    )
    lg.add_argument("--no-daemon", action="store_true", help="do not install the launchd watch daemon")

    add("logout", "delete the stored key and team key (the watch daemon stays; see `scout daemon`)")

    dm = add("daemon", "install or remove the launchd agent that runs `scout watch --all`")
    dm.add_argument("action", choices=["install", "uninstall"])

    sy = add("sync", "index the checkout")
    sy.add_argument(
        "--wait", action="store_true", help="run to completion (default: start in the background)"
    )
    sy.add_argument("--full", action="store_true", help="re-hash every file")
    sy.add_argument("--text", action="store_true")

    wa = add("watch", "keep the index fresh; one watcher per checkout")
    grp = wa.add_mutually_exclusive_group()
    grp.add_argument("--ensure", action="store_true", help="start unless one already runs (the default)")
    grp.add_argument("--add", default=None, metavar="ROOT", help="register a checkout for the daemon")
    grp.add_argument("--remove", default=None, metavar="ROOT", help="unregister a checkout")
    grp.add_argument("--all", action="store_true", help="watch every registered checkout (the daemon)")

    ini = add("init", "check a host and print its install commands; --override writes its override file")
    ini.add_argument("--host", default=None, metavar="NAME", help="a directory name under integrations/")
    ini.add_argument("--claude-code", dest="host", action="store_const", const="claude-code",
                     help="alias for --host claude-code")  # fmt: skip
    ini.add_argument("--muse-code", dest="host", action="store_const", const="muse-code",
                     help="alias for --host muse-code (installs and approves the plugin)")  # fmt: skip
    ini.add_argument("--override", action="store_true", help="write the host's override file (opt-in)")
    ini.add_argument("--dry-run", action="store_true", help="print the install steps instead of running them")

    sc = add("schema", "print the JSON schema of a contract type (no name: list them)")
    sc.add_argument("name", nargs="?", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        from scout.surfaces.cli import commands  # lazy: parsing never pays for the app

        return commands.run(args)
    except ScoutError as e:
        return report_error(e)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - the CLI never prints a traceback: type name only (text may carry a path)
        return report_error(ScoutError(ErrorCode.INTERNAL, f"internal error: {type(e).__name__}"))


if __name__ == "__main__":
    sys.exit(main())
