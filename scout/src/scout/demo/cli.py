"""`scout-demo`: index a repo into real turbopuffer, explore it, or serve the same over MCP stdio."""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

from scout.core.contracts.errors import ScoutError, cli_exit
from scout.demo.runtime import DemoRuntime, render_report, resolve_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scout-demo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ix = sub.add_parser(
        "index", help="clone or open a repo, chunk, embed (voyage-code-4) and write to turbopuffer"
    )
    ix.add_argument("source", help="git URL (shallow-cloned) or local checkout path")
    ix.add_argument("--name", help="demo name; namespace is scout_dev_demo_<name> (default: repo basename)")
    ix.add_argument("--full", action="store_true", help="re-hash every file")
    ix.add_argument("--json", action="store_true")

    ex = sub.add_parser("explore", help="run the scout explore loop over an indexed repo")
    ex.add_argument("question")
    ex.add_argument("--repo", required=True, help="name given to `index`")
    ex.add_argument("--json", action="store_true")
    ex.add_argument(
        "--preset",
        choices=["a1", "v3", "baseten", "scout"],
        help="model preset (default: $SCOUT_DEMO_PRESET, else a1 if A1_BASE_URL set; v3 = wayfinder-glimmer)",
    )
    ex.add_argument("--model-base-url", help="OpenAI-compatible base URL including /v1")
    ex.add_argument("--model", help="model name sent to the endpoint")
    ex.add_argument("--api-key", help="bearer token (prefer env; flags show up in `ps`)")
    ex.add_argument("--max-results", type=int, default=12)
    ex.add_argument("--max-turns", type=int, default=40)
    ex.add_argument("--max-seconds", type=int, default=240)
    ex.add_argument("--no-warmup", action="store_true", help="skip the cold-start warmup call")

    ls = sub.add_parser("list", help="list indexed demo repos")
    ls.add_argument("--json", action="store_true")

    sub.add_parser("mcp", help="MCP stdio server: explore(brief, repo), index_repo(source), list_repos()")

    wm = sub.add_parser("warm", help="one 1-token call that wakes the demo model (scale-to-zero cold start)")
    wm.add_argument("--preset", choices=["a1", "v3", "baseten", "scout"], help="as for explore")
    wm.add_argument(
        "--trigger", action="store_true", help="only start the cold start (Modal /warm) and return"
    )
    return p


def _fail(e: ScoutError, as_json: bool) -> int:
    if as_json:
        print(e.to_body().model_dump_json(indent=2))
    else:
        print(f"error: {e.code}: {e.message}", file=sys.stderr)
    return cli_exit(e.code)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)  # the Voyage SDK leaves an aiohttp session open
    as_json: bool = getattr(args, "json", False)
    try:
        rt = DemoRuntime()
        if args.cmd == "index":
            rep = asyncio.run(rt.index(args.source, args.name, full=args.full))
            if as_json:
                print(json.dumps(rep.to_dict(), indent=2))
            else:
                print(
                    f"indexed {rep.name}: state={rep.state} coverage={rep.coverage:.2f} files={rep.files} "
                    f"chunks={rep.chunks} tokens={rep.voyage_tokens:,} est_cost=${rep.est_cost_usd:.4f} "
                    f"namespace={rep.namespace} ({rep.seconds:.1f}s)"
                )
            return 0
        if args.cmd == "warm":
            cfg = resolve_model(rt.env, preset=args.preset)
            if args.trigger:
                mode, seconds = asyncio.run(rt.warm_start(cfg))
            else:
                mode, seconds = "completion", asyncio.run(rt.warmup(cfg))
            done = mode == "completion"
            print(json.dumps({"warm": done, "warming": not done, "mode": mode, "seconds": round(seconds, 2)}))
            return 0
        if args.cmd == "explore":
            cfg = resolve_model(
                rt.env,
                preset=args.preset,
                base_url=args.model_base_url,
                model=args.model,
                api_key=args.api_key,
            )
            report = asyncio.run(
                rt.explore(
                    args.question,
                    args.repo,
                    cfg,
                    max_results=args.max_results,
                    max_turns=args.max_turns,
                    max_seconds=args.max_seconds,
                    warmup=not args.no_warmup,
                )
            )
            print(json.dumps(report.to_dict(), indent=2) if as_json else render_report(report))
            return 0
        if args.cmd == "list":
            repos = rt.list_repos()
            if as_json:
                print(json.dumps([r.__dict__ if hasattr(r, "__dict__") else {} for r in repos], indent=2))
            for r in repos:
                if not as_json:
                    print(
                        f"{r.name}  {r.namespace}  {r.state}@{r.coverage:.2f}  "
                        f"files={r.files} chunks={r.chunks}  {r.root}"
                    )
            return 0
        from scout.demo.mcp_server import serve

        serve(rt)
        return 0
    except ScoutError as e:
        return _fail(e, as_json)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
