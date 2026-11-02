"""`CompatChat`: scout's `OpenAIChat` plus what rollouts need and scout's adapter does not carry.

Reuses scout's request body, status mapping and response parsing (no copy), and adds: a configurable auth
header (Baseten wants `Authorization: Api-Key <key>`), sampling overrides (the loop builds `ChatRequest` with
A1 defaults, so the harness pins the teacher's sampling here), retry with backoff on 429/5xx/timeouts
(counted), raw usage incl. `cached_tokens`, the model's `reasoning_content`, per-turn latency, and a full
message log (the loop elides old tool results in place when over budget; we copy each message the first
time it is sent so the stored transcript is never elided).

The same class drives any OpenAI-compatible endpoint (Baseten, a self-hosted vLLM Spark Open or Glimmer on
Modal): only `base_url`, `model`, key and sampling change.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from scout.adapters.openai_chat import client as _scout_client
from scout.adapters.openai_chat.client import OpenAIChat
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.chat import ChatRequest, ChatTurn


@dataclass(frozen=True, slots=True)
class Sampling:
    temperature: float = 0.85
    top_p: float = 0.95
    top_k: int = 0  # 0 = do not send
    presence_penalty: float = 0.0
    max_tokens: int = 8192


@dataclass(frozen=True, slots=True)
class Prices:
    """USD per million tokens."""

    input: float = 1.25
    output: float = 4.25
    cached_input: float = 0.15

    def cost(self, tokens_in: int, tokens_out: int, cached: int) -> float:
        fresh = max(tokens_in - cached, 0)
        return (fresh * self.input + cached * self.cached_input + tokens_out * self.output) / 1e6


SPARK_OPEN_BASETEN = {
    "base_url": "https://inference.baseten.co/v1",
    "model": "meta-models/muse-spark-open",
    "auth_scheme": "Api-Key",
}


class TokenGate:
    """Process-wide sliding-window limiter on prompt tokens per minute, shared by every concurrent rollout.
    Baseten's Spark Open limit is 1M tokens/min and (observed) cached prompt tokens count against it, so
    concurrency alone cannot be tuned: long multi-turn rollouts re-send their whole prefix every turn. Each call
    reserves an estimate (the previous prompt size of that rollout + headroom) before sending and is trued up to
    the real `prompt_tokens + completion_tokens` after."""

    def __init__(self, tokens_per_min: int, window_s: float = 60.0, ceiling: int | None = None) -> None:
        self.limit = float(tokens_per_min)
        self.ceiling = float(ceiling or tokens_per_min)  # adaptive: limit ramps up to this, backs off on 429
        self.floor = 0.85 * self.limit
        self.account_limit = 1_000_000.0  # x-ratelimit-limit-tokens; header-driven account-wide view (see observe/reserve)
        self.margin = 100_000.0
        self._hdr: tuple[float, float] | None = None  # (monotonic t, remaining tokens) from the latest response
        self._since_hdr = 0.0  # tokens this gate reserved since that response
        self.window = window_s
        self._events: list[list[float]] = []  # [t, tokens]
        self._lock: asyncio.Lock | None = None
        self.waited_s = 0.0
        self.n429 = 0
        self.n_ok = 0
        self._cooldown_until = 0.0
        self._last429 = 0.0
        self._last_adjust = time.monotonic()

    def observe(self, status: object, headers: Any = None) -> None:
        """Feedback from the endpoint: a 429 pauses every rollout briefly and lowers the limit 8% (not below half the
        start); a minute without one raises it 3% (not above the ceiling)."""
        now = time.monotonic()
        try:
            if headers is not None and headers.get("x-ratelimit-remaining-tokens") is not None:
                self._hdr = (now, float(headers["x-ratelimit-remaining-tokens"]))
                self._since_hdr = 0.0
                self.account_limit = float(headers.get("x-ratelimit-limit-tokens") or self.account_limit)
        except (TypeError, ValueError):
            pass
        if status == 429:
            self.n429 += 1
            self._last429 = now
            if now >= self._cooldown_until:
                self._cooldown_until = now + 8.0
                self.limit = max(self.limit * 0.92, self.floor)
                self._last_adjust = now
        elif status == 200:
            self.n_ok += 1
            if now - self._last429 > 60 and now - self._last_adjust > 60 and self.limit < self.ceiling:
                self.limit = min(self.limit * 1.03, self.ceiling)
                self._last_adjust = now

    def _used(self, now: float) -> float:
        self._events = [e for e in self._events if now - e[0] < self.window]
        return sum(e[1] for e in self._events)

    async def reserve(self, tokens: int) -> list[float]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            t0 = time.monotonic()
            while True:
                now = time.monotonic()
                if now < self._cooldown_until:
                    await asyncio.sleep(min(self._cooldown_until - now, 0.5))
                    continue
                est = None
                if self._hdr is not None and now - self._hdr[0] < 30:  # account-wide bucket as last reported, plus refill
                    est = min(self.account_limit, self._hdr[1] + self.account_limit / 60.0 * (now - self._hdr[0])) - self._since_hdr
                if (self._used(now) + tokens <= self.limit and (est is None or est >= tokens + self.margin)) or not self._events:
                    self._since_hdr += tokens
                    ev = [now, float(tokens)]
                    self._events.append(ev)
                    self.waited_s += now - t0
                    return ev
                await asyncio.sleep(0.5)

    @staticmethod
    def true_up(ev: list[float], actual: int) -> None:
        ev[1] = float(actual)


class CompatChat(OpenAIChat):
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        auth_scheme: str = "Bearer",
        sampling: Sampling | None = None,
        prices: Prices | None = None,
        seed: int | None = None,
        vllm_extensions: bool = False,
        timeout: float = 150.0,
        max_retries: int = 10,
        gate: TokenGate | None = None,
        extra_headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: Mapping[str, Any] | None = None,
        retain_reasoning: bool = True,
    ) -> None:
        headers = dict(extra_headers or {})
        if api_key:
            headers["Authorization"] = f"{auth_scheme} {api_key}"
        super().__init__(
            base_url, None, model, headers, timeout=timeout, vllm_extensions=vllm_extensions, transport=transport
        )
        self.model = model
        self.sampling = sampling or Sampling()
        self.prices = prices or Prices()
        self.seed = seed
        self.extra_body = dict(extra_body or {})  # e.g. {"reasoning_effort": "high"} for self-hosted Spark Open
        self.max_retries = max_retries
        self.gate = gate
        self.clock: Any = None  # RealClock; rate-limit waits are excluded from the loop's budget
        self._last_in = 2000
        self._last_cost = 3500  # tokens the gate is charged for the previous call of this rollout
        self.gate_uncached_only = False
        self.turn_log: list[dict[str, Any]] = []
        self.n429 = 0
        self.n_retries = 0
        self._seen = 0
        #  capture(call) (opt-in, default off) receives {prompt_token_ids, token_ids, finish, n_tool_calls, n_elided} per successful
        #  call (needs `return_token_ids` in extra_body). retain_reasoning resends each earlier assistant turn's reasoning_content so
        #  the context equals the rendered training sequence (the SFT template renders it on every assistant turn). Default ON since
        #  2026-10-27, matching production scout (owner decision); pass retain_reasoning=False to reproduce results produced before it.
        self.capture: Any = None
        self.retain_reasoning = retain_reasoning
        #  message keys under which retained reasoning is resent. vLLM 0.32.0 (renderers/chat_utils.py) reads only `reasoning` and drops
        #  `reasoning_content`, so a run against such a server sets ("reasoning_content", "reasoning"); the default is the pre-existing behaviour.
        self.reasoning_keys: tuple[str, ...] = ("reasoning_content",)
        self._last_messages: list[dict[str, Any]] | None = None
        self._msg_log: list[dict[str, Any]] = []

    # --- usage ------------------------------------------------------------------------------
    def totals(self) -> dict[str, int]:
        us = [t["usage"] for t in self.turn_log if t.get("usage")]
        return {k: sum(u[k] for u in us) for k in ("in", "out", "cached", "reasoning")}

    def cost_usd(self) -> float:
        t = self.totals()
        return self.prices.cost(t["in"], t["out"], t["cached"])

    # --- transcript -------------------------------------------------------------------------
    def _snapshot(self, messages: Any) -> None:
        if not isinstance(messages, list):
            return
        self._last_messages = messages
        for m in messages[self._seen :]:
            self._msg_log.append(copy.deepcopy(dict(m)))
        self._seen = len(messages)

    def transcript(self) -> list[dict[str, Any]]:
        """Full messages (tool calls and un-elided results), reasoning attached to assistant turns."""
        if self._last_messages is not None:
            for m in self._last_messages[self._seen :]:
                self._msg_log.append(copy.deepcopy(dict(m)))
            self._seen = len(self._last_messages)
        out = copy.deepcopy(self._msg_log)
        reasoning = [t.get("reasoning_content") for t in self.turn_log if t.get("ok")]
        it = iter(reasoning)
        for m in out:
            if m.get("role") == "assistant":
                r = next(it, None)
                if r:
                    m["reasoning_content"] = r
        return out

    def _excl(self, seconds: float) -> None:
        if self.clock is not None and seconds > 0:
            self.clock.exclude(seconds)

    # --- the call ---------------------------------------------------------------------------
    async def complete(self, req: ChatRequest) -> ChatTurn:
        s = self.sampling
        req = dataclasses.replace(
            req,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            presence_penalty=s.presence_penalty,
            max_tokens=s.max_tokens,
        )
        self._snapshot(req.messages)
        body = self._body(req)
        if self.retain_reasoning:
            rs = iter([t.get("reasoning_content") for t in self.turn_log if t.get("ok")])
            for m in body["messages"]:
                if m.get("role") == "assistant":
                    r = next(rs, None)
                    if r:
                        for key in self.reasoning_keys:
                            m[key] = r
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.extra_body)
        entry: dict[str, Any] = {"ok": False, "retries": 0, "statuses": []}
        t0 = time.monotonic()
        res: httpx.Response | None = None
        ev: list[float] | None = None
        for attempt in range(self.max_retries + 1):
            if self.gate is not None:
                if ev is not None:
                    TokenGate.true_up(ev, 0)  # a failed attempt consumed nothing we can see
                tg = time.monotonic()
                ev = await self.gate.reserve((self._last_cost if self.gate_uncached_only else self._last_in) + 1500)
                self._excl(time.monotonic() - tg)
            try:
                async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                    res = await client.post(self._url, headers=self._headers, json=body)
                status = res.status_code
            except httpx.TimeoutException:
                res, status = None, "timeout"
            except httpx.HTTPError:
                res, status = None, "conn"
            entry["statuses"].append(status)
            if self.gate is not None and hasattr(self.gate, "observe"):
                self.gate.observe(status, res.headers if res is not None else None)
            if status == 200:
                break
            retryable = status in ("timeout", "conn", 429, 500, 502, 503, 504)
            if status == 429:
                self.n429 += 1
            if not retryable or attempt == self.max_retries:
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
            self.turn_log.append(entry)
            raise _scout_client._status_error(res)
        turn = _scout_client._parse_turn(res)
        if not self.retain_reasoning:  # the loop must not hold (and budget for) reasoning that is not sent
            turn = dataclasses.replace(turn, reasoning=None)
        data = res.json()
        if self.capture is not None:
            ch0 = data["choices"][0]
            lp_content = (ch0.get("logprobs") or {}).get("content")  # only when the request sets logprobs (GRPO rollouts): behaviour-policy logprobs
            self.capture({"prompt_token_ids": data.get("prompt_token_ids"), "token_ids": ch0.get("token_ids"),
                          **({"logprobs": [float(x["logprob"]) for x in lp_content]} if lp_content else {}),
                          "finish": ch0.get("finish_reason"), "n_tool_calls": len(ch0["message"].get("tool_calls") or []), "tools": body.get("tools"),
                          # scout's loop rewrites OLD tool results to "[elided: ..." stubs when the prompt nears its budget;
                          # after that the prompt is no longer an extension of the earlier ones (see wayfinder/opd/rollout.py)
                          "n_elided": sum(1 for m in req.messages if m.get("role") == "tool" and str(m.get("content", "")).startswith("[elided: "))})
        usage = data.get("usage") or {}
        self._last_in = int(usage.get("prompt_tokens") or self._last_in)
        if self.gate is not None and ev is not None:
            pin = int(usage.get("prompt_tokens") or 0)
            cch = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            charged = (pin - cch if self.gate_uncached_only else pin) + int(usage.get("completion_tokens") or 0)
            self._last_cost = charged
            TokenGate.true_up(ev, charged)
        details = usage.get("prompt_tokens_details") or {}
        cdetails = usage.get("completion_tokens_details") or {}
        msg = data["choices"][0]["message"]
        entry.update(
            ok=True,
            usage={
                "in": int(usage.get("prompt_tokens") or 0),
                "out": int(usage.get("completion_tokens") or 0),
                "cached": int(details.get("cached_tokens") or 0),
                "reasoning": int(cdetails.get("reasoning_tokens") or 0),
            },
            finish=turn.finish,
            reasoning_content=msg.get("reasoning_content") or msg.get("reasoning"),
            ratelimit={k: v for k, v in res.headers.items() if k.startswith("x-ratelimit")},
        )
        self.turn_log.append(entry)
        return turn
