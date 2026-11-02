#!/usr/bin/env python3
"""Turn a recorded Muse run (eval/harness/muse_spec.py transcripts) into a readable replay with wall times.

    python3 scripts/demo_replay.py artifacts/live/muse_spec/transcripts/<task10>__<arm>.sessions [-o out.md]

Merges the parent thread (`main.jsonl.gz`) and every subagent thread (`subN.jsonl.gz`) into one timeline:
seconds since the first event, thread, what was called (arguments cut to one line) and what came back (first
line). The last block is the parent's final answer. It reads session logs only; nothing here talks to a
network or a model.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def events(path: Path, thread: str) -> list[tuple[float, str, dict]]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ev = (e.get("payload") or {}).get("event") if isinstance(e.get("payload"), dict) else None
            if isinstance(ev, dict):
                out.append((e.get("recorded_at", 0) / 1e6, thread, ev))
    return out


def one_line(text: object, n: int = 150) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render(sessions: Path) -> str:
    evs: list[tuple[float, str, dict]] = []
    files = sorted(sessions.glob("*.jsonl.gz"))
    for f in files:
        evs += events(
            f, "parent" if f.name.startswith("main") else f.name.split(".")[0].replace("sub", "child ")
        )
    evs.sort(key=lambda x: x[0])
    t0 = evs[0][0] if evs else 0.0
    lines = ["| t (s) | thread | step |", "|---|---|---|"]
    final = ""
    for t, thread, ev in evs:
        kind = ev.get("kind")
        if kind == "assistant_tool_calls_committed":
            for c in ev["tool_calls"]:
                tool, args = c["name"].split("__")[-1], one_line(c.get("args", ""), 110)
                lines.append(f"| {t - t0:6.1f} | {thread} | call `{tool}` {args} |")
        elif kind == "tool_result_batch_committed":
            for r in ev["results"]:
                lines.append(f"| {t - t0:6.1f} | {thread} | result: {one_line(r.get('text', ''), 130)} |")
        elif kind == "assistant_message_committed" and ev.get("phase") != "commentary" and thread == "parent":
            final = ev.get("text") or final
    wall = (evs[-1][0] - t0) if evs else 0.0
    head = f"Replay of `{sessions.name}`: {len(evs)} events, {wall:.0f} s from first to last event.\n\n"
    return head + "\n".join(lines) + "\n\nParent's final answer:\n\n" + (final or "(none recorded)") + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args(argv)
    text = render(a.sessions)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
