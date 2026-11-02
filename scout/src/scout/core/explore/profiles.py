"""Explore profiles (design §1 core.explore, §9 Parity): `product` and `parity`.

`parity` reproduces benchmark arm g: the neutral prompt (byte copy of `sophisticated`, which arm g ran), the
benchmark `declaration` chunker, benchmark tool names and caps, a text terminal and no `report` tool.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a bundled prompt as UTF-8 bytes decoded verbatim (no newline translation)."""
    return (_PROMPTS / f"{name}.md").read_bytes().decode("utf-8")


@dataclass(frozen=True, slots=True)
class ExploreProfile:
    name: str
    prompt_name: str
    chunker: Literal["tree-sitter", "declaration"]
    tools: tuple[str, ...]
    terminal: Literal["report", "text"]
    thinking: bool
    max_turns: int
    caps: dict[str, int]
    # Prompt-size guard for the loop (task 1.11a). Parity: the benchmark's 65,536 context minus headroom.
    context_tokens: int = 60_000

    @property
    def prompt(self) -> str:
        return load_prompt(self.prompt_name)


PROFILES: dict[str, ExploreProfile] = {
    "product": ExploreProfile(
        name="product",
        prompt_name="product",
        chunker="tree-sitter",
        tools=("semantic_search", "grep", "glob", "read_file", "report"),
        terminal="report",
        thinking=True,
        max_turns=40,
        caps={
            "grep_files": 200,
            "grep_matches_per_file": 20,
            "glob_paths": 1000,
            "read_lines": 400,
            "semantic_k": 10,
        },
        context_tokens=60_000,
    ),
    # Benchmark caps: grep 200 files / 20 matches per file, glob 1000 paths, search k=5 (delivery.yaml),
    # snippet cap 512 tokens, turn_limit 63 (limits.yaml); read_file had no line cap there.
    "parity": ExploreProfile(
        name="parity",
        prompt_name="neutral",
        chunker="declaration",
        tools=("semantic_search", "grep", "glob", "read_file"),
        terminal="text",
        thinking=True,
        max_turns=63,
        caps={
            "grep_files": 200,
            "grep_matches_per_file": 20,
            "glob_paths": 1000,
            "semantic_k": 5,
            "snippet_max_tokens": 512,
        },
        context_tokens=60_000,
    ),
}
