"""Display-only model names (renamed 2026-11-01). Internal ids, row keys, file paths and CLI values keep their old
spelling (`v3`, `qwen`, `replica`, `wayfinder-v3`); only text printed for people goes through `disp`.

    from display_names import disp, print   # print shadows the builtin and relabels every string it prints
"""
from __future__ import annotations

import builtins
import re

GLIMMER = "wayfinder-glimmer"
QWEN = "wayfinder-qwen"
QWEN_SEMUP = "wayfinder-qwen-semup"
_BEFORE = r"(?<![\w/.\-])"
_QWEN_LABEL = [QWEN]


def set_qwen_label(variant: str) -> None:
    """The semantic-upweight variant (`--variant upweight`) is shown as wayfinder-qwen-semup in tables."""
    _QWEN_LABEL[0] = QWEN_SEMUP if variant == "upweight" else QWEN


_RULES = [
    (re.compile(r"SFT v3\b"), GLIMMER),
    (re.compile(_BEFORE + r"v3r_(grep|sem)"), GLIMMER + r" retain \1"),
    (re.compile(_BEFORE + r"v3_(?=grep|sem|minus|retain)"), GLIMMER + " "),
    (re.compile(r"(?<=minus_)v3(?=_)"), GLIMMER),
    (re.compile(_BEFORE + r"v3(?=[ ,:;)|\]'\"}]|$)"), GLIMMER),
    (re.compile(_BEFORE + r"qwen_(?=grep|sem|minus)"), lambda m: _QWEN_LABEL[0] + " "),
    (re.compile(r"Qwen3\.6-35B-A3B SFT replica"), QWEN),
    (re.compile(r"Qwen3\.6-35B-A3B SFT\b(?! replica)"), QWEN),
]


def disp(text: str) -> str:
    for rx, rep in _RULES:
        text = rx.sub(rep, text)
    return text


def print(*args, **kwargs):  # noqa: A001 - deliberate shadow
    builtins.print(*(disp(a) if isinstance(a, str) else a for a in args), **kwargs)
