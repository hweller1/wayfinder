"""Convert Agents-A1-35B Inspect rollouts (benchmark tool names) into Muse Glimmer SFT examples (scout tool surface).

Pure library (no Modal): `convert_eval_file` is called by wayfinder/sft/convert_modal.py inside Modal CPU containers and
by the local tests. Mapping and loss decisions are documented in docs/sft_dryrun.md; the short version:

  system   scout product prompt (serve/scout_product_prompt.md), byte-identical to scout's prompts/product.md
  tools    scout tool_specs: arm g -> semantic_search, grep, glob, read_file, report; arm b2 (no index tool) -> the
           4-tool grep-only set scout offers when the index is absent. Arm c (search only, never reads a file) is dropped.
  user     "<brief>\\n\\nReport at most 12 results." (scout loop `_user_message`)
  calls    search -> semantic_search{query}; grep/glob/read_file unchanged (arg names are identical; grep path "." is
           dropped because null already means the whole repo)
  results  re-rendered from the benchmark `<results>` envelope into scout's plain-text formats (grep `path:line: text`,
           glob one path per line, read_file `path lines a-b` + numbered lines, semantic_search `rank. path` + snippet)
  report   synthesized from the final free-text answer: cited files that were actually opened with read_file, spans from
           the read ranges (or the line range the answer cites inside them), why/summary/gaps lifted from the answer text
  loss     assistant tokens only: everything the model must generate after `<|start|>assistant` (the routing header
           ` to=<tool>` included) up to and including `<|eom|>`/`<|eot|>`; system/user/tool tokens are masked
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

MAX_TOKENS = 32768
MAX_RESULTS = 12
SUMMARY_MAX = 1200
WHY_MAX = 280
GAP_MAX = 5
SEARCH_SNIPPET_LINES = 12
SEARCH_SNIPPET_CHARS = 1800
KEEP_STATUS = {"success", "retrieval_miss"}  # others never reached a final answer
USER_SUFFIX = f"\n\nReport at most {MAX_RESULTS} results."

_RESULT_HDR = re.compile(r'<result rank="(\d+)" id="(.*?)" source="(.*?)" score="(.*?)" truncated="(true|false)">\n?')
_ENV_HDR = re.compile(r'^<results tool="([^"]*)" backend="([^"]*)" query="(.*)" k="(\d+)" returned="(\d+)">')
_MATCH_LINE = re.compile(r"^(\d+):(.*)$")


# --------------------------------------------------------------------------------------------------------------
# envelope parsing and scout-format rendering
# --------------------------------------------------------------------------------------------------------------
def parse_envelope(text: str) -> tuple[int, list[dict[str, Any]]] | None:
    """Return (returned, results[{id, source, truncated, body}]) or None when `text` is not an envelope."""
    if not text.startswith("<results "):
        return None
    head, _, rest = text.partition("\n")
    m = _ENV_HDR.match(head)
    if not m:
        return None
    returned = int(m.group(5))
    if rest.rstrip().endswith("</results>"):
        rest = rest.rstrip()[: -len("</results>")]
    hdrs = list(_RESULT_HDR.finditer(rest))
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hdrs):
        end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(rest)
        body = rest[h.end() : end]
        if body.endswith("\n</result>\n"):
            body = body[: -len("\n</result>\n")]
        elif body.endswith("\n</result>"):
            body = body[: -len("\n</result>")]
        out.append({"id": h.group(2), "source": h.group(3), "truncated": h.group(5) == "true", "body": body})
    return returned, out


def render_grep(res: list[dict[str, Any]]) -> tuple[str, bool]:
    lossy = False
    out: list[str] = []
    for r in res:
        lossy |= r["truncated"]
        for ln in r["body"].split("\n"):
            m = _MATCH_LINE.match(ln)
            if m:
                out.append(f"{r['id']}:{m.group(1)}: {m.group(2)}")
            elif ln.strip():
                out.append(f"{r['id']}: {ln}")  # continuation of a long/flattened match line
    return ("\n".join(out) if out else "no matches"), lossy


def render_glob(res: list[dict[str, Any]]) -> tuple[str, bool]:
    return ("\n".join(r["id"] for r in res) if res else "no files match"), False


def render_read(res: list[dict[str, Any]], start: int, span_end: int | None) -> tuple[str, bool, int | None]:
    """Scout `_read_text`. Returns (text, lossy, delivered_end). A truncated envelope body is the first 512
    whitespace tokens with newlines flattened: line numbers are unrecoverable, so it is rendered flat and flagged."""
    r = res[0]
    path = r["id"]
    body = r["body"]
    if r["truncated"]:
        end = span_end
        head = f"{path} lines {start}-{end}" if end else f"{path} lines {start}-"
        return f"{head}\n{body}\n... output truncated (line numbering unavailable in the source transcript)", True, end
    lines = body.split("\n") if body != "" else []
    if body == "":
        return f"{path} lines {start}-{start - 1}", False, None
    end = start + len(lines) - 1
    out = [f"{path} lines {start}-{end}"] + [f"{start + i}\t{ln}" for i, ln in enumerate(lines)]
    return "\n".join(out), False, end


def render_search(res: list[dict[str, Any]]) -> str:
    if not res:
        return "no results"
    parts = []
    for i, r in enumerate(res, 1):
        body = r["body"]
        first, _, snippet = body.partition("\n")
        if first.strip() != r["id"]:  # defensive: the envelope normally starts the chunk with its path
            snippet = body
        lines = snippet.split("\n")
        if len(lines) > 1:
            snippet = "\n".join(lines[:SEARCH_SNIPPET_LINES])
        if len(snippet) > SEARCH_SNIPPET_CHARS:
            snippet = snippet[:SEARCH_SNIPPET_CHARS].rstrip() + " ..."
        parts.append(f"{i}. {r['id']}\n{snippet}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------------------------
# report synthesis
# --------------------------------------------------------------------------------------------------------------
def _clean(s: str) -> str:
    s = re.sub(r"[`*_#>]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -|:\t")
    return s


def _why(path: str, answer: str) -> str:
    base = path.rsplit("/", 1)[-1]
    for needle in (path, base):
        for ln in answer.split("\n"):
            if needle in ln:
                if "|" in ln:  # markdown table row: the longest cell without the path
                    cells = [_clean(c) for c in ln.split("|")]
                    cells = [c for c in cells if c and needle not in c]
                    if cells:
                        return max(cells, key=len)[:WHY_MAX]
                txt = _clean(ln.replace(path, " ").replace(base, " "))
                txt = re.sub(r"\(lines? [\d,\s\-–and]+\)|:\d+(?:[-–]\d+)?", " ", txt)
                txt = _clean(txt)
                if len(txt) >= 12:
                    return txt[:WHY_MAX]
                return _clean(ln)[:WHY_MAX]
    return "opened and read while searching for the brief"


def _summary(answer: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
    paras = [p for p in paras if not p.lstrip().startswith(("#", "|", "---"))]
    s = ""
    for p in paras:
        s = (s + "\n\n" + p).strip() if s else p
        if len(s) >= 200:
            break
    s = s or _clean(answer)
    if len(s) > SUMMARY_MAX:
        cut = s[:SUMMARY_MAX]
        k = max(cut.rfind(". "), cut.rfind("\n"))
        s = cut[: k + 1] if k > 400 else cut
    return s.strip()


_GAP_HDR = re.compile(r"(could not|couldn't|cannot|can't|unable|not establish|remain|gap|unsupported|not found|did not find|missing)", re.I)


def _gaps(answer: str) -> list[str]:
    out: list[str] = []
    lines = answer.split("\n")
    i = 0
    while i < len(lines) and len(out) < GAP_MAX:
        ln = lines[i]
        is_hdr = bool(re.match(r"^\s*(#+\s|\*\*|[A-Z][^.]{0,60}:\s*$)", ln)) and _GAP_HDR.search(ln)
        if is_hdr:
            j = i + 1
            while j < len(lines) and len(out) < GAP_MAX:
                b = lines[j]
                if re.match(r"^\s*#+\s", b):
                    break
                if re.match(r"^\s*([-*+]|\d+[.)])\s+", b):
                    c = _clean(re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", b))
                    if c:
                        out.append(c[:200])
                j += 1
            i = j
        else:
            i += 1
    return out


def _cited_range(path: str, answer: str) -> tuple[int, int] | None:
    m = re.search(re.escape(path) + r"[`'\"]?\s*(?:[:(,]\s*(?:lines?\s*)?|\s+lines?\s+)(\d+)(?:\s*[-–]\s*(\d+))?", answer)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (a, b) if 0 < a <= b else None


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def synthesize_report(
    answer: str, reads: dict[str, list[tuple[int, int | None]]], seen: set[str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return (report_args, info). `reads` maps a file to its delivered (start, end) ranges (end None = unknown)."""
    info: dict[str, Any] = {"cited_unread": 0, "cited_read": 0}
    pos: dict[str, int] = {}
    for p in set(seen) | set(reads):
        i = answer.find(p)
        if i >= 0:
            pos[p] = i
    # basename fallback only for read files with a unique basename
    bases = Counter(p.rsplit("/", 1)[-1] for p in reads)
    for p in reads:
        if p in pos:
            continue
        b = p.rsplit("/", 1)[-1]
        if bases[b] == 1:
            m = re.search(r"(?<![\w/.-])" + re.escape(b) + r"(?![\w-])", answer)
            if m:
                pos[p] = m.start()
    cited = sorted(pos, key=lambda p: pos[p])
    info["cited_unread"] = sum(1 for p in cited if p not in reads)
    hits: list[dict[str, Any]] = []
    for p in cited:
        if p not in reads:
            continue
        info["cited_read"] += 1
        known = [(a, b) for a, b in reads[p] if b is not None]
        cr = _cited_range(p, answer) if p in answer else None
        if not known:
            # only truncated whole-file reads: the delivered extent is unknown. Accept an explicit range cited in the
            # answer (flagged span_unverified), otherwise the file cannot be reported with an honest span.
            if cr is None:
                continue
            info["span_unverified"] = info.get("span_unverified", 0) + 1
            hits.append({"path": p, "start_line": cr[0], "end_line": cr[1], "why": _why(p, answer)})
            continue
        if cr and any(a <= cr[0] and cr[1] <= b for a, b in known):
            spans = [cr]
        elif cr and any(cr[0] <= b and a <= cr[1] for a, b in known):  # partially inside a delivered range: clamp
            lo = max(cr[0], min(a for a, b in known if cr[0] <= b and a <= cr[1]))
            hi = min(cr[1], max(b for a, b in known if cr[0] <= b and a <= cr[1]))
            spans = [(lo, hi)]
        else:
            spans = _merge(known)
        why = _why(p, answer)
        for a, b in spans:
            hits.append({"path": p, "start_line": a, "end_line": b, "why": why})
    hits = hits[:MAX_RESULTS]
    if not hits:
        return None, info
    report: dict[str, Any] = {"results": hits, "summary": _summary(answer)}
    gaps = _gaps(answer)
    if gaps:
        report["gaps"] = gaps
    return report, info


# --------------------------------------------------------------------------------------------------------------
# rollout -> messages
# --------------------------------------------------------------------------------------------------------------
class Drop(Exception):
    pass


def reasoning_by_turn(sample: Any) -> dict[str, str]:
    """Map assistant turn key -> reasoning text. Key = first tool_call id, or 'FINAL' for the closing answer."""
    out: dict[str, str] = {}
    for ev in sample.events:
        if type(ev).__name__ != "ModelEvent":
            continue
        msg = ev.output.choices[0].message
        parts = msg.content if isinstance(msg.content, list) else []
        txt = []
        for p in parts:
            if getattr(p, "type", "") == "reasoning":
                r = p.reasoning or ""
                if r.startswith("attachment://"):
                    r = sample.attachments.get(r[len("attachment://") :], "")
                txt.append(r.strip())
        key = msg.tool_calls[0].id if msg.tool_calls else "FINAL"
        out[key] = "\n".join(t for t in txt if t)
    return out


def build_messages(rec: dict[str, Any], reasoning: dict[str, str], system_prompt: str, with_semantic: bool
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (scout-format messages, info). Raises Drop(reason) when the mapping would be lossy beyond repair."""
    src = rec["messages"]
    if len(src) < 3 or src[0]["role"] != "system" or src[1]["role"] != "user":
        raise Drop("bad_shape")
    final = src[-1]
    if final["role"] != "assistant" or final.get("tool_calls") or not (final.get("content") or "").strip():
        raise Drop("no_final_answer")
    spans_lookup: dict[tuple[str, int], int] = {}
    for s in rec.get("retrieved_spans") or []:
        k = (s["path"], s["start_line"])
        spans_lookup[k] = max(spans_lookup.get(k, 0), s["end_line"])

    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": src[1]["content"] + USER_SUFFIX},
    ]
    pending: dict[str, dict[str, Any]] = {}
    reads: dict[str, list[tuple[int, int | None]]] = {}
    seen: set[str] = set()
    info: dict[str, Any] = {"lossy": [], "n_turns": 0, "tool_counts": Counter(), "reasoning_chars": 0}
    for m in src[2:-1]:
        if m["role"] == "assistant":
            tcs = m.get("tool_calls") or []
            if not tcs:
                raise Drop("mid_turn_without_tool_call")
            calls = []
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    raise Drop("bad_tool_args") from None
                if not isinstance(args, dict):
                    raise Drop("bad_tool_args")
                if name == "search":
                    name = "semantic_search"
                    if not with_semantic:
                        raise Drop("semantic_in_grep_only_arm")
                    args = {"query": str(args.get("query", ""))}
                elif name == "grep":
                    args = {k: v for k, v in args.items() if k in ("pattern", "path", "regex", "context")}
                    if args.get("path") in (".", "", "./"):
                        args.pop("path")
                elif name == "glob":
                    args = {"pattern": str(args.get("pattern", ""))}
                elif name == "read_file":
                    args = {k: v for k, v in args.items() if k in ("path", "start_line", "end_line") and v is not None}
                    s, e = args.get("start_line", 1), args.get("end_line")
                    if e is not None and e - s + 1 > 400:
                        raise Drop("read_range_over_400")
                else:
                    raise Drop(f"unmappable_tool:{name}")
                info["tool_counts"][name] += 1
                pending[tc["id"]] = {"name": name, "args": args}
                calls.append({"id": tc["id"], "type": "function", "function": {"name": name, "arguments": args}})
            r = reasoning.get(tcs[0]["id"], "")
            content = (m.get("content") or "").strip()
            if content:  # the ATEM template drops `content` next to tool calls; keep it as part of the reasoning
                r = (r + "\n" + content).strip()
            am: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": calls}
            if r:
                am["reasoning_content"] = r
                info["reasoning_chars"] += len(r)
            msgs.append(am)
            info["n_turns"] += 1
        elif m["role"] == "tool":
            p = pending.get(m["tool_call_id"])
            if p is None:
                raise Drop("orphan_tool_result")
            env = parse_envelope(m["content"])
            if env is None:
                raise Drop("non_envelope_tool_result")
            _, res = env
            name, args = p["name"], p["args"]
            lossy = False
            if name == "grep":
                text, lossy = render_grep(res)
                seen.update(r["id"] for r in res)
            elif name == "glob":
                text, lossy = render_glob(res)
                seen.update(r["id"] for r in res)
            elif name == "semantic_search":
                text = render_search(res)
                seen.update(r["id"] for r in res)
            else:  # read_file
                if not res:
                    text = f"error: NOT_FOUND: {args.get('path')} is not a readable file"
                    lossy = True
                else:
                    ranged = "start_line" in args or "end_line" in args
                    start = int(args.get("start_line", 1))
                    span_end = spans_lookup.get((res[0]["id"], start)) if ranged else None
                    text, lossy, end = render_read(res, start, span_end)
                    if not (ranged is False and lossy):  # whole-file truncated read: end unknown
                        pass
                    if res[0]["body"] != "":
                        reads.setdefault(res[0]["id"], []).append((start, end))
                    seen.add(res[0]["id"])
            if lossy:
                info["lossy"].append(name)
            tool_msg = {"role": "tool", "tool_call_id": m["tool_call_id"], "content": text}
            msgs.append(tool_msg)
        else:
            raise Drop("unexpected_role")
    if any(len(pending) == 0 for _ in [0]):
        raise Drop("no_tool_calls")
    report, rinfo = synthesize_report(final["content"], reads, seen)
    info.update(rinfo)
    if report is None:
        raise Drop("no_cited_read_file_for_report")
    rid = "report_" + (src[2]["tool_calls"][0]["id"][:8] if src[2].get("tool_calls") else "0")
    r = reasoning.get("FINAL", "")
    am = {"role": "assistant", "content": "", "tool_calls": [
        {"id": rid, "type": "function", "function": {"name": "report", "arguments": report}}]}
    if r:
        am["reasoning_content"] = r
        info["reasoning_chars"] += len(r)
    msgs.append(am)
    info["n_turns"] += 1
    info["report"] = report
    return msgs, info


# --------------------------------------------------------------------------------------------------------------
# tokenization and loss mask
# --------------------------------------------------------------------------------------------------------------
_SEG = re.compile(r"<\|start\|>assistant")
_END = re.compile(r"<\|(eom|eot)\|>")


def assistant_char_spans(text: str) -> list[tuple[int, int]]:
    """Char spans the model must generate. A segment starts after `<|start|>assistant` (already in the generation
    prompt) when it opens a turn, or at `<|start|>` when it follows an `<|eom|>` inside the same turn."""
    spans = []
    prev_end_kind = None
    prev_was_assistant = False
    pos = 0
    for m in _SEG.finditer(text):
        e = _END.search(text, m.end())
        if not e:
            raise ValueError("unterminated assistant segment")
        cont = prev_was_assistant and prev_end_kind == "eom"
        start = m.start() if cont else m.end()
        spans.append((start, e.end()))
        prev_end_kind = e.group(1)
        prev_was_assistant = True
        pos = e.end()
        # any non-assistant segment between assistant segments resets continuation
        nxt = text.find("<|start|>", pos)
        if nxt >= 0 and not text.startswith("<|start|>assistant", nxt):
            prev_was_assistant = False
    return spans


def tokenize_example(tok: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    offs = enc["offset_mapping"]
    spans_c = assistant_char_spans(text)
    tspans: list[list[int]] = []
    ti = 0
    for a, b in spans_c:
        while ti < len(offs) and offs[ti][1] <= a:
            ti += 1
        s = ti
        while ti < len(offs) and offs[ti][0] < b:
            ti += 1
        tspans.append([s, ti])
    return {"input_ids": ids, "assistant_spans": tspans, "n_tokens": len(ids)}


# --------------------------------------------------------------------------------------------------------------
# one .eval file
# --------------------------------------------------------------------------------------------------------------
def file_recall(files: set[str], gold: list[str]) -> float:
    g = list(dict.fromkeys(gold))
    return sum(1 for x in g if x in files) / len(g) if g else 0.0


def convert_eval_file(path: str, splits: dict[str, Any], system_prompt: str, tools_all: list[dict[str, Any]],
                      tok: Any, include_lossy: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (examples, drops). Each drop = {file, task_id, arm, reason}. Examples carry meta.tier in {strict, lossy_render}."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    ex_out: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    tools_no_sem = [t for t in tools_all if t["function"]["name"] != "semantic_search"]
    for s in log.samples or []:
        rec = (s.store or {}).get("asa/task_record")
        md = s.metadata or {}
        native = md.get("native_id") or ""
        base = {"file": path.rsplit("/", 1)[-1], "task_id": native, "arm": (rec or {}).get("arm")}
        if not rec:
            drops.append({**base, "reason": "no_task_record"})
            continue
        t = splits["tasks"].get(native)
        if t is None:
            drops.append({**base, "reason": "task_not_in_splits"})
            continue
        if t["split"] != "train":
            drops.append({**base, "reason": f"split_{t['split']}"})
            continue
        arm = rec["arm"]
        if arm not in ("g", "b2"):
            drops.append({**base, "reason": f"arm_{arm}_unmappable"})
            continue
        if rec["status"] not in KEEP_STATUS:
            drops.append({**base, "reason": f"status_{rec['status']}"})
            continue
        with_semantic = arm == "g"
        try:
            msgs, info = build_messages(rec, reasoning_by_turn(s), system_prompt, with_semantic)
        except Drop as d:
            drops.append({**base, "reason": f"drop:{d}"})
            continue
        tools = tools_all if with_semantic else tools_no_sem
        tier = "lossy_render" if info["lossy"] else "strict"
        if tier == "lossy_render" and not include_lossy:
            drops.append({**base, "reason": "lossy_render"})
            continue
        tk = tokenize_example(tok, msgs, tools)
        if tk["n_tokens"] > MAX_TOKENS:
            drops.append({**base, "reason": "over_32k_tokens", "n_tokens": tk["n_tokens"]})
            continue
        report = info["report"]
        rfiles = {h["path"] for h in report["results"]}
        gold = t["gold_files"]
        recall_report = file_recall(rfiles, gold)
        seen_recall = file_recall(set(rec.get("retrieved_docids") or []), gold)
        meta = {
            "task_id": native, "repo": t["repo_full"], "arm": arm, "category": t.get("category"), "split": "train",
            "a1_status": rec["status"], "tier": tier, "lossy_tools": dict(Counter(info["lossy"])),
            "recall_report": round(recall_report, 4), "all_gold_found": recall_report == 1.0,
            "recall_seen": round(seen_recall, 4), "all_gold_seen": seen_recall == 1.0,
            "n_gold": len(set(gold)), "turns": info["n_turns"], "tokens": tk["n_tokens"],
            "assistant_tokens": sum(b - a for a, b in tk["assistant_spans"]),
            "tool_counts": dict(info["tool_counts"]), "n_report_hits": len(report["results"]),
            "cited_unread": info["cited_unread"], "span_unverified": info.get("span_unverified", 0), "reasoning_chars": info["reasoning_chars"],
            "evidence_covered": True, "a1_total_tokens": rec.get("total_tokens"), "source_file": base["file"],
            "tools_variant": "scout5" if with_semantic else "scout4_grep_only",
        }
        ex_out.append({"id": f"{native}|{arm}|{base['file'][-26:-5]}", "input_ids": tk["input_ids"],
                       "assistant_spans": tk["assistant_spans"], "n_tokens": tk["n_tokens"],
                       "tools_variant": meta["tools_variant"], "messages": msgs, "meta": meta})
    return ex_out, drops
