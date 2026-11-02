"""Token-level structure of ATEM transcripts (Glimmer chat template): assistant spans, and the teacher's prompt head.

Span semantics are exactly wayfinder/sft/convert.py `assistant_char_spans`, applied to token ids: a segment starts at
`<|start|>assistant`; the model generates everything after those two tokens (the routing header ` to=<tool>` included)
up to and including the `<|eom|>`/`<|eot|>` that ends the segment; a segment that follows an `<|eom|>` of the same turn is
generated from its own `<|start|>`.
"""

from __future__ import annotations

from typing import Any

# Glimmer / Spark shared ids (data/opd_shared_vocab.json `structural_ids`).
START, MESSAGE, EOM, EOT, BOS = 200022, 200023, 200007, 200008, 200000
ASSISTANT = 140680
USER = 1556


def assistant_spans_from_ids(ids: list[int], start: int = START, assistant: int = ASSISTANT, eom: int = EOM, eot: int = EOT) -> list[list[int]]:
    """[[a, b), ...] token spans the model generated. An unterminated last segment (generation cut by max_tokens) runs to the end."""
    spans: list[list[int]] = []
    n = len(ids)
    i = 0
    prev_eom = False  # the previous token run was an assistant segment ended by <|eom|>, and nothing else came in between
    while i < n:
        if ids[i] == start and i + 1 < n and ids[i + 1] == assistant:
            j = i + 2
            while j < n and ids[j] != eom and ids[j] != eot:
                j += 1
            end = min(j + 1, n)
            spans.append([i if prev_eom else i + 2, end])
            prev_eom = j < n and ids[j] == eom
            i = end
        else:
            if ids[i] == start:
                prev_eom = False
            elif prev_eom:
                prev_eom = False
            i += 1
    return spans


def last_terminated_end(ids: list[int]) -> int:
    """Index just past the last assistant segment that ends in <|eom|>/<|eot|> (0 if none)."""
    end = 0
    for a, b in assistant_spans_from_ids(ids):
        if ids[b - 1] in (EOM, EOT):
            end = b
    return end


def build_chains(calls: list[dict[str, Any]], max_len: int = 32768, strict: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn one rollout's per-call records into training sequences ("chains").

    `calls[i]` = {prompt_token_ids, token_ids, finish, n_elided} as vLLM returned them for model call i. Scoring every call
    separately would re-prefill the whole prefix on the teacher for each of ~25 calls, so consecutive calls whose prompt
    EXTENDS the previous call's sequence are scored once, as one sequence:

        prompt_0 + completion_0 + (tool results + header of prompt_1) + completion_1 + ...

    Loss spans are the SAMPLED completions only (the ids the student generated, plus `<|eot|>` when the server stopped on it and
    left it out of token_ids). Assistant turns that only appear re-rendered inside a prompt are context: they are never trained on,
    so every sampled token is trained exactly once, in the chain where it was sampled.
    A chain ends where the next prompt is not an extension (strict: the next prompt's first len(seq) tokens differ from seq, which
    covers scout eliding old tool results, and re-tokenisation of a sampled turn). With strict=False a re-render difference of the
    previous completion is tolerated as long as the history before it is unchanged (the tail is then cut from the next prompt after
    its last terminated assistant segment); the sequence keeps the sampled form of that turn, so the context of later turns
    differs slightly from what the student saw. A chain is cut at its last sampled-segment end <= max_len tokens (causal, so
    earlier tokens' teacher scores are unaffected).

    Returns (chains, diag). chain = {input_ids, assistant_spans, call_start, call_end, n_calls, truncated, n_tokens, assistant_tokens}."""
    diag: dict[str, Any] = {"spans_v": 2, "n_calls": len(calls), "chain_breaks": 0, "eot_appended": 0, "no_terminator": 0, "lenient_mismatch": 0,
                            "dropped_chains": 0, "n_elided_max": 0}
    chains: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    nan = float("nan")
    with_lp = any(c.get("logprobs") for c in calls)  # GRPO: behaviour-policy logprob of every sampled token (NaN elsewhere)

    def close() -> None:
        nonlocal cur
        if cur is None:
            return
        seq = cur["seq"]
        spans = [s for s in cur["spans"] if s[1] <= max_len and seq[s[1] - 1] in (EOM, EOT)]
        truncated = len(seq) > (spans[-1][1] if spans else 0) or len(spans) < len(cur["spans"])
        if not spans:
            diag["dropped_chains"] += 1
        else:
            cut = spans[-1][1]
            ch = {"input_ids": seq[:cut], "assistant_spans": [list(s) for s in spans], "call_start": cur["call_start"], "call_end": cur["call_end"],
                  "n_calls": cur["call_end"] - cur["call_start"] + 1, "truncated": truncated, "n_tokens": cut,
                  "assistant_tokens": sum(b - a for a, b in spans)}
            if with_lp:
                ch["logprobs"] = cur["lp"][:cut]
            chains.append(ch)
        cur = None

    for i, c in enumerate(calls):
        P = list(c["prompt_token_ids"] or [])
        C = list(c["token_ids"] or [])
        diag["n_elided_max"] = max(diag["n_elided_max"], int(c.get("n_elided") or 0))
        if len(P) < 3 or P[-2:] != [START, ASSISTANT] or not C:
            close()
            diag["chain_breaks"] += 1
            continue
        if C[-1] not in (EOM, EOT):
            if c.get("finish") == "stop":
                C.append(EOT)  # the server stopped on <|eot|> and kept it out of token_ids
                diag["eot_appended"] += 1
            else:
                diag["no_terminator"] += 1  # cut by max_tokens: the turn is incomplete and cannot be scored as a full segment
        lpC = list(c.get("logprobs") or [])
        if with_lp:  # aligned with the sampled ids; an appended <|eot|> and any misaligned call carry NaN (treated as on-policy)
            n_samp = len(c["token_ids"] or [])
            lpC = lpC if len(lpC) == n_samp else [nan] * n_samp
            lpC = lpC + [nan] * (len(C) - len(lpC))
        tail: list[int] = []
        extended = False
        if cur is not None:
            seq = cur["seq"]
            if P[: len(seq)] == seq and len(P) > len(seq):
                tail = P[len(seq):]
                extended = True
            elif not strict and cur["prompt"] and P[: len(cur["prompt"])] == cur["prompt"] and seq[-1] in (EOM, EOT):
                e = last_terminated_end(P[:-2])
                if e > len(cur["prompt"]) and P[e : e + 1] == [START]:
                    tail = P[e:]
                    extended = True
                    diag["lenient_mismatch"] += 1
            if not extended:
                close()
                diag["chain_breaks"] += 1
        if cur is None:
            cur = {"seq": P + C, "prompt": P, "call_start": i, "call_end": i, "spans": [[len(P), len(P) + len(C)]]}
            if with_lp:
                cur["lp"] = [nan] * len(P) + lpC
        else:
            n0 = len(cur["seq"]) + len(tail)
            cur["seq"] = cur["seq"] + tail + C
            if with_lp:
                cur["lp"] = cur["lp"] + [nan] * len(tail) + lpC
            cur["prompt"] = P
            cur["call_end"] = i
            cur["spans"].append([n0, n0 + len(C)])
        if C[-1] not in (EOM, EOT):
            close()
    close()
    diag["n_chains"] = len(chains)
    diag["chain_tokens"] = sum(ch["n_tokens"] for ch in chains)
    diag["assistant_tokens"] = sum(ch["assistant_tokens"] for ch in chains)
    diag["sampled_tokens"] = sum(len(c.get("token_ids") or []) for c in calls)
    return chains, diag


def merge_turn_spans(spans: list[list[int]]) -> list[list[int]]:
    """Join contiguous segments of one turn (eom -> start continuation) into one [a, b) per assistant message."""
    out: list[list[int]] = []
    for a, b in spans:
        if out and out[-1][1] == a:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def first_user_index(ids: list[int], user_id: int, start: int = START) -> int:
    for i in range(len(ids) - 1):
        if ids[i] == start and ids[i + 1] == user_id:
            return i
    raise ValueError("no <|start|>user in sequence")


def spark_head_text(spark_tok: Any, system: str, tools: list[dict[str, Any]] | None, reasoning_effort: str = "high") -> str:
    """The system block exactly as Spark's own template renders it (the format the teacher was trained and served with),
    cut before the first user turn."""
    text = spark_tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": "x"}],
        tools=tools or None, tokenize=False, add_generation_prompt=False, reasoning_effort=reasoning_effort,
    )
    k = text.index("<|start|>user")
    return text[:k]


def teacher_sequence(student_ids: list[int], spark_tok: Any, system: str, tools: list[dict[str, Any]] | None, user_id: int,
                     reasoning_effort: str = "high") -> tuple[list[int], int]:
    """Replace the student's system block (Glimmer template) by Spark's, keep every later token (user, tool results and
    the student's own sampled assistant ids) byte for byte. Returns (teacher_ids, shift) with teacher_index = student_index + shift."""
    head = spark_tok(spark_head_text(spark_tok, system, tools, reasoning_effort), add_special_tokens=False)["input_ids"]
    k = first_user_index(student_ids, user_id)
    return head + student_ids[k:], len(head) - k
