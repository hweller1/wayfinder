"""Teacher scoring of student chains with Spark Open (vLLM `/v1/completions`, token-id prompt, `prompt_logprobs=20`).

The teacher scores EXACTLY the student's token sequence, except for the system block, which is re-rendered with Spark's own chat
template (its system block differs from Glimmer's: knowledge cutoff, recipients, reasoning-effort text), the way the teacher saw it in
its own rollouts. Every token after the first `<|start|>user` (user turn, tool results, and the student's sampled assistant
tokens) is byte for byte what the student produced, so no re-tokenisation can shift an assistant token.
Only assistant positions are kept: row j of the stored top-20 describes token j of the student sequence (teacher position
j + shift). Added-token ids are NOT dropped here; the trainer's loss masks them (wayfinder/opd/loss.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from wayfinder.opd.data import K, parse_prompt_logprobs, save_chain, spans_to_positions
from wayfinder.opd.spans import USER, spark_head_text

TEACHER_KEY = "wf-spark-local"
TEACHER_MODEL = "spark"


class HeadCache:
    def __init__(self, spark_tok: Any, reasoning_effort: str = "high") -> None:
        self.tok = spark_tok
        self.effort = reasoning_effort
        self._c: dict[str, list[int]] = {}

    def ids(self, system: str, tools: list[dict[str, Any]] | None) -> list[int]:
        key = hashlib.sha1(json.dumps([system, tools], sort_keys=True).encode()).hexdigest()
        if key not in self._c:
            text = spark_head_text(self.tok, system, tools, self.effort)
            self._c[key] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self._c[key]


def teacher_ids(student_ids: list[int], head: list[int]) -> tuple[list[int], int]:
    """(teacher ids, shift): teacher_index = student_index + shift."""
    k = next(i for i in range(len(student_ids) - 1) if student_ids[i] == 200022 and student_ids[i + 1] == USER)
    return head + student_ids[k:], len(head) - k


def build_request(tids: list[int]) -> dict[str, Any]:
    return {"model": TEACHER_MODEL, "prompt": tids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": K}


async def post_score(client: Any, url: str, tids: list[int], timeout: float = 1800.0, retries: int = 3) -> tuple[list[Any], dict[str, Any]]:
    """POST one chain; returns (prompt_logprobs list, timing/size info). Raises on persistent failure."""
    import orjson

    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = await client.post(url + "/v1/completions", content=orjson.dumps(build_request(tids)),
                                  headers={"Authorization": f"Bearer {TEACHER_KEY}", "Content-Type": "application/json"}, timeout=timeout)
            t1 = time.time()
            if r.status_code != 200:
                raise RuntimeError(f"http {r.status_code}: {r.text[:300]}")
            nbytes = len(r.content)
            data = orjson.loads(r.content)
            pl = data["choices"][0]["prompt_logprobs"]
            if not pl or len(pl) != len(tids):
                raise RuntimeError(f"prompt_logprobs has {len(pl or [])} entries for {len(tids)} tokens")
            return pl, {"request_s": round(t1 - t0, 2), "parse_s": round(time.time() - t1, 2), "resp_mb": round(nbytes / 1e6, 1), "attempts": attempt + 1}
        except Exception as e:  # noqa: BLE001
            last = e
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"scoring failed after {retries + 1} attempts: {last!r}"[:400])


def store_chain(out_dir: Path, cid: str, chain: dict[str, Any], tids_shift: int, pl: list[Any], meta: dict[str, Any]) -> dict[str, Any]:
    pos = spans_to_positions(chain["assistant_spans"])
    ids, lp, missing = parse_prompt_logprobs(pl, pos, tids_shift, K)
    # drop trailing/embedded positions with no teacher data from the stored arrays' count (kept in `n_missing`)
    save_chain(out_dir / f"{cid}.npz", chain["input_ids"], chain["assistant_spans"], pos, ids, lp, {**meta, "id": cid, "shift": tids_shift})
    top1 = np.exp(lp[:, 0])
    return {"id": cid, "n_tokens": len(chain["input_ids"]), "n_positions": int(len(pos)), "n_missing": int(missing),
            "teacher_top1_p_mean": float(top1[ids[:, 0] >= 0].mean()) if (ids[:, 0] >= 0).any() else None, **meta}
