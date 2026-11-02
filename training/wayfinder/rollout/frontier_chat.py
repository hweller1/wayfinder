"""`FrontierChat`: drives Sonnet 5.5 (Anthropic Messages route) and GPT-6.1 Sol (OpenAI Responses route) through the LLM gateway
inside scout's own explore loop, the same loop, prompt, tools and nudge as the v3 and A1 rows.

Only the model endpoint changes. The loop still builds OpenAI-style messages and tool specs; this class translates each request
to the provider's native wire format and the response back to a `ChatTurn`. Native reasoning is retained across turns:

- Sonnet: the assistant turn's native content blocks (thinking with signature, text, tool_use) are stored per turn and replayed
  verbatim. Adaptive thinking, `output_config.effort = "high"`, thinking display "summarized" (so transcripts have text).
  Anthropic fixes temperature at 1 when thinking is on, so no temperature or top_p is sent.
- GPT: stateless Responses calls (`store: false`, `include: reasoning.encrypted_content`); the turn's native output items
  (reasoning, message, function_call) are replayed verbatim. Default reasoning (effort medium), provider-default sampling.

Every call is billed from its token buckets (fresh, cache read, cache write, output) times the rate table below and appended to a
ledger by `SpendGuard`, which also enforces a hard cap before each call. The tokens-times-rate cost is what the repo records as
"billed"; the LLM gateway does not return a price. Keys come from the environment (LLM_GATEWAY_API_KEY) and are never logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
from scout.adapters.openai_chat import client as _scout_client
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatRequest, ChatTurn

from wayfinder.rollout.chat import CompatChat, Sampling

GATEWAY = os.environ.get("LLM_GATEWAY_URL", "").rstrip("/")  # base URL of your LLM API gateway (Anthropic and OpenAI-compatible routes)
ANTHROPIC_URL = f"{GATEWAY}/anthropic/v1/messages"
RESPONSES_URL = f"{GATEWAY}/openai/v1/responses"

# USD per 1M tokens: (fresh input, cache read, cache write, output). Copied from benchmark-repo/scripts/score_frontier.py RATES
# (sonnet, the LLM gateway catalog) and scripts/e2e/run_harness.py EXTRA_RATES (gpt-6.1-sol: ASSUMPTION, models.dev list price 2.00/0.10/2.50/10.00
# scaled by the 0.74 the LLM gateway/list factor of gpt-6-sol; not confirmed against the catalog).
RATES = {
    "claude-sonnet-5-5": (2.00, 0.20, 2.50, 10.00),
    "gpt-6.1-sol": (1.48, 0.074, 1.85, 7.40),
}
MAX_TOKENS = 16000  # per call, thinking included (the open-model rows used 8192 with a cheaper reasoning budget); recorded in meta
SETTINGS = {
    "claude-sonnet-5-5": {"route": "anthropic /v1/messages", "thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high",
                          "temperature": "1 (fixed by the API when thinking is on; not sent)", "top_p": "not sent", "top_k": "not sent",
                          "presence_penalty": "not sent", "seed": "not supported (seed value is a label only)", "max_tokens": MAX_TOKENS,
                          "thinking_retained": "native thinking blocks with signature replayed every turn",
                          "prompt_caching": "cache_control ephemeral on the system block and the last message block"},
    "gpt-6.1-sol": {"route": "openai /v1/responses", "reasoning": "default (response reported effort=medium, mode=standard)", "temperature": "provider default (1.0 reported)",
                    "top_p": "provider default (0.98 reported)", "seed": "not supported (seed value is a label only)", "max_output_tokens": MAX_TOKENS,
                    "thinking_retained": "encrypted reasoning items replayed every turn (store=false, include reasoning.encrypted_content)",
                    "prompt_caching": "provider automatic prefix caching"},
}


class SpendCapReached(RuntimeError):
    pass


class SpendGuard:
    """Per-call ledger plus a hard cap. Single event loop: check-and-append is not awaited in between, so it is atomic."""

    def __init__(self, ledger: Path, cap_usd: float, headroom_usd: float = 0.0, phase: str = "") -> None:
        self.ledger, self.cap, self.headroom, self.phase = ledger, cap_usd, headroom_usd, phase
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.total = 0.0
        if ledger.exists():
            for ln in ledger.read_text().splitlines():
                if ln.strip():
                    self.total += json.loads(ln)["usd"]
        self.by_model: dict[str, float] = {}
        self.calls = 0
        self.stopped = False

    def check(self, est_usd: float = 0.0) -> None:
        if self.stopped or self.total + est_usd + self.headroom >= self.cap:
            self.stopped = True
            raise SpendCapReached(f"llm_key spend cap: ${self.total:.2f} + headroom ${self.headroom:.2f} reaches ${self.cap:.2f}")

    def record(self, rec: dict[str, Any]) -> None:
        self.total += rec["usd"]
        self.calls += 1
        self.by_model[rec["model"]] = self.by_model.get(rec["model"], 0.0) + rec["usd"]
        with self.ledger.open("a") as f:
            f.write(json.dumps({**rec, "phase": self.phase, "cum_usd": round(self.total, 6)}) + "\n")


def price(model: str, fresh: int, cache_read: int, cache_write: int, out: int) -> float:
    i, cr, cw, o = RATES[model]
    return (fresh * i + cache_read * cr + cache_write * cw + out * o) / 1e6


def _load_gateway_key() -> str:
    if os.environ.get("LLM_GATEWAY_API_KEY"):
        return os.environ["LLM_GATEWAY_API_KEY"]
    from scout.demo.env import parse_env_text

    for p in (Path(__file__).resolve().parents[2] / ".env",):
        try:
            v = parse_env_text(p.read_text()).get("LLM_GATEWAY_API_KEY")
        except OSError:
            continue
        if v:
            return v
    raise RuntimeError("LLM_GATEWAY_API_KEY not found in the environment or .env files")


def _to_anthropic(messages: list[dict[str, Any]], natives: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    ai = 0
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append({"type": "text", "text": m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif role == "assistant":
            blocks = [dict(b) for b in natives[ai]] if ai < len(natives) else []
            ai += 1
            if not blocks:
                blocks = [{"type": "text", "text": m.get("content") or " "}]
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            blk = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": str(m["content"])}
            if out and out[-1]["role"] == "user" and out[-1]["content"] and out[-1]["content"][0]["type"] == "tool_result":
                out[-1]["content"].append(blk)
            else:
                out.append({"role": "user", "content": [blk]})
    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
    if out:
        out[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return system, out


def _to_responses(messages: list[dict[str, Any]], natives: list[list[dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    instructions = ""
    items: list[dict[str, Any]] = []
    ai = 0
    for m in messages:
        role = m["role"]
        if role == "system":
            instructions = m["content"]
        elif role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            items.extend(dict(x) for x in (natives[ai] if ai < len(natives) else []))
            ai += 1
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": str(m["content"])})
    return instructions, items


class FrontierChat(CompatChat):
    def __init__(self, model: str, *, guard: SpendGuard, key: str, seed: int | None = None, timeout: float = 600.0, max_retries: int = 6,
                 tag: dict[str, Any] | None = None) -> None:
        super().__init__(GATEWAY, None, model, sampling=Sampling(max_tokens=MAX_TOKENS), seed=seed, timeout=timeout, max_retries=max_retries,
                         retain_reasoning=True)
        assert model in RATES
        self.guard, self._key, self.tag = guard, key, tag or {}
        self._natives: list[list[dict[str, Any]]] = []
        self.buckets = {"fresh": 0, "cache_read": 0, "cache_write": 0, "out": 0}
        self.cost = 0.0

    def cost_usd(self) -> float:
        return self.cost

    async def complete(self, req: ChatRequest) -> ChatTurn:
        self._snapshot(req.messages)
        msgs = [dict(m) for m in req.messages]
        tools = list(req.tools)
        anth = self.model.startswith("claude")
        if anth:
            system, am = _to_anthropic(msgs, self._natives)
            body: dict[str, Any] = {"model": self.model, "max_tokens": MAX_TOKENS, "system": system, "messages": am,
                                    "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"},
                                    "tools": [{"name": t["function"]["name"], "description": t["function"]["description"],
                                               "input_schema": t["function"]["parameters"]} for t in tools],
                                    "tool_choice": {"type": "auto"}}
            url, headers = ANTHROPIC_URL, {"api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        else:
            instr, items = _to_responses(msgs, self._natives)
            body = {"model": self.model, "instructions": instr, "input": items, "store": False, "include": ["reasoning.encrypted_content"],
                    "max_output_tokens": MAX_TOKENS, "tool_choice": "auto",
                    "tools": [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
                               "parameters": t["function"]["parameters"]} for t in tools]}
            url, headers = RESPONSES_URL, {"api-key": self._key, "content-type": "application/json"}
        entry: dict[str, Any] = {"ok": False, "retries": 0, "statuses": []}
        t0 = time.monotonic()
        res: httpx.Response | None = None
        est = price(self.model, max(self._last_in, 2000), 0, 0, 1500)  # worst-case one call before the cap check
        for attempt in range(self.max_retries + 1):
            self.guard.check(est)
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    res = await client.post(url, headers=headers, json=body)
                status: object = res.status_code
            except httpx.TimeoutException:
                res, status = None, "timeout"
            except httpx.HTTPError:
                res, status = None, "conn"
            entry["statuses"].append(status)
            if status == 200:
                break
            if status == 429:
                self.n429 += 1
            if status not in ("timeout", "conn", 408, 429, 500, 502, 503, 504, 529) or attempt == self.max_retries:
                break
            self.n_retries += 1
            entry["retries"] += 1
            wait = min(3.0 * 2**attempt, 40.0) * (0.5 + random.random())
            if res is not None and status == 429:
                try:
                    wait = max(wait, float(res.headers["retry-after"]))
                except (KeyError, ValueError):
                    pass
            await asyncio.sleep(wait)
            self._excl(wait)
        entry["latency_s"] = round(time.monotonic() - t0, 3)
        if res is None:
            self.turn_log.append(entry)
            raise ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the chat model timed out or was unreachable")
        if res.status_code != 200:
            entry["error_body"] = res.text[:300]
            self.turn_log.append(entry)
            raise _scout_client._status_error(res)
        data = res.json()
        if anth:
            turn, native, u = self._parse_anthropic(data)
        else:
            turn, native, u = self._parse_responses(data)
        self._natives.append(native)
        usd = price(self.model, u["fresh"], u["cache_read"], u["cache_write"], u["out"])
        self.cost += usd
        for k in self.buckets:
            self.buckets[k] += u[k]
        self._last_in = u["fresh"] + u["cache_read"] + u["cache_write"]
        self.guard.record({"ts": round(time.time(), 1), "model": self.model, **self.tag, "fresh": u["fresh"], "cache_read": u["cache_read"],
                           "cache_write": u["cache_write"], "out": u["out"], "usd": round(usd, 6), "request_id": res.headers.get("x-request-id") or data.get("id")})
        entry.update(ok=True, usage={"in": self._last_in, "out": u["out"], "cached": u["cache_read"], "reasoning": u["reasoning"]},
                     billed={k: u[k] for k in ("fresh", "cache_read", "cache_write", "out")}, usd=round(usd, 6), finish=turn.finish,
                     reasoning_content=turn.reasoning, stop=u.get("stop"))
        self.turn_log.append(entry)
        return turn

    def _parse_anthropic(self, data: dict[str, Any]) -> tuple[ChatTurn, list[dict[str, Any]], dict[str, int]]:
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None
        thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking" and b.get("thinking")) or None
        try:
            calls = tuple(_scout_client._parse_call(i, {"id": b["id"], "function": {"name": b["name"], "arguments": b.get("input") or {}}})
                          for i, b in enumerate(x for x in blocks if x.get("type") == "tool_use"))
        except (KeyError, TypeError):
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model returned a malformed response") from None
        us = data.get("usage") or {}
        cr, cw = int(us.get("cache_read_input_tokens") or 0), int(us.get("cache_creation_input_tokens") or 0)
        fresh, out = int(us.get("input_tokens") or 0), int(us.get("output_tokens") or 0)
        stop = data.get("stop_reason")
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(stop, str(stop))
        turn = ChatTurn(content=text, tool_calls=calls, tokens_in=fresh + cr + cw, tokens_out=out, finish=finish, reasoning=thinking)
        return turn, blocks, {"fresh": fresh, "cache_read": cr, "cache_write": cw, "out": out,
                              "reasoning": int((us.get("output_tokens_details") or {}).get("thinking_tokens") or 0), "stop": stop}

    def _parse_responses(self, data: dict[str, Any]) -> tuple[ChatTurn, list[dict[str, Any]], dict[str, int]]:
        if data.get("error"):
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model returned an error")
        items = data.get("output") or []
        text = "".join(c.get("text", "") for it in items if it.get("type") == "message" for c in it.get("content", []) if c.get("type") == "output_text") or None
        summ = "\n".join(s.get("text", "") for it in items if it.get("type") == "reasoning" for s in (it.get("summary") or [])) or None
        try:
            calls = tuple(_scout_client._parse_call(i, {"id": it["call_id"], "function": {"name": it["name"], "arguments": it.get("arguments")}})
                          for i, it in enumerate(x for x in items if x.get("type") == "function_call"))
        except (KeyError, TypeError):
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model returned a malformed response") from None
        us = data.get("usage") or {}
        det = us.get("input_tokens_details") or {}
        cr, cw = int(det.get("cached_tokens") or 0), int(det.get("cache_write_tokens") or 0)
        tin, out = int(us.get("input_tokens") or 0), int(us.get("output_tokens") or 0)
        status = data.get("status")
        finish = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
        turn = ChatTurn(content=text, tool_calls=calls, tokens_in=tin, tokens_out=out, finish=finish, reasoning=summ)
        return turn, items, {"fresh": max(tin - cr - cw, 0), "cache_read": cr, "cache_write": cw, "out": out,
                             "reasoning": int((us.get("output_tokens_details") or {}).get("reasoning_tokens") or 0), "stop": status}


__all__ = ["FrontierChat", "SpendGuard", "SpendCapReached", "RATES", "SETTINGS", "MAX_TOKENS", "price", "replace"]
