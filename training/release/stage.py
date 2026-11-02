"""Pure helpers for the Hugging Face release (stdlib only, no Modal, no torch). Tested by tests/test_release_stage.py.

publish.py imports this inside the Modal container and on the laptop, so the checks that stop a bad upload are
the same code in both places.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ATTRIBUTION = "Built with Muse Spark Open"
TODO_MARK = "TODO(final)"

# Files copied from the base checkpoint into the release. README, LICENSE and USAGE_POLICY are replaced by ours,
# everything else (config, tokenizer, chat template, index) is the base's, unchanged.
BASE_SKIP = {"README.md", "LICENSE", "USAGE_POLICY.md", ".gitattributes", ".download_complete"}
BASE_SKIP_PREFIX = (".cache", ".eval_results")

# Files that must exist in the staged folder before any upload.
REQUIRED = (
    "README.md",
    "LICENSE",
    "USAGE_POLICY.md",
    "LICENSE-MUSE-SPARK-OPEN.md",
    "USE_POLICY-MUSE-SPARK-OPEN.md",
    "NOTICE",
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)

# Anything outside this set in the staged folder blocks the upload (keeps adapters, logs, keys and data out).
ALLOWED_EXACT = {
    *REQUIRED,
    "generation_config.json",
    "processor_config.json",
    "wayfinder_release.json",
    "scout_tool_specs.json",
    "scout_product_prompt.md",
}
ALLOWED_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

LICENSE_FILES = {  # staged name -> vendored name in release/licenses
    "LICENSE": "LICENSE",
    "USAGE_POLICY.md": "USAGE_POLICY.md",
    "LICENSE-MUSE-SPARK-OPEN.md": "LICENSE-MUSE-SPARK-OPEN.md",
    "USE_POLICY-MUSE-SPARK-OPEN.md": "USE_POLICY-MUSE-SPARK-OPEN.md",
}

NOTICE = """Wayfinder Glimmer 30B

This model is a derivative of three works:

1. Muse Glimmer 30B (Meta Platforms, Inc.), licensed under the Apache License 2.0. See LICENSE. Meta's Glimmer
   Usage Policy is in USAGE_POLICY.md.
2. Muse Spark Open 816B-A42B (Meta Platforms, Inc.), under the Muse Model License (version "Muse Spark Open").
   The model was fine-tuned on outputs of Muse Spark Open, so it is a Derivative as that licence defines the
   term. See LICENSE-MUSE-SPARK-OPEN.md, which includes the Muse Spark Open Usage Policy as Exhibit A, and
   USE_POLICY-MUSE-SPARK-OPEN.md.
3. Agents-A1-35B (InternScience), licensed under the Apache License 2.0. A small number of its search
   transcripts were used as additional fine-tuning examples.

Built with Muse Spark Open.
"""


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def render_card(template: str, *, repo_id: str, variant: str) -> str:
    out = template.replace("{{REPO_ID}}", repo_id).replace("{{VARIANT}}", variant)
    left = re.findall(r"\{\{[A-Z_]+\}\}", out)
    if left:
        raise ValueError(f"unresolved card placeholders: {sorted(set(left))}")
    return out


def count_todos(card: str) -> int:
    return card.count(TODO_MARK)


def public_refusal(card: str, public: bool) -> str | None:
    """Reason to refuse making the repo public, or None. Used by publish.py both on the laptop and in the container."""
    todos = count_todos(card)
    if public and todos:
        return f"--public refused: {todos} {TODO_MARK} markers remain in the model card"
    return None


def check_card(card: str) -> list[str]:
    """Hard errors for the card, regardless of visibility. TODO markers are not an error here (see count_todos)."""
    errs: list[str] = []
    if not card.startswith("---\n"):
        errs.append("card must start with YAML front matter")
    else:
        end = card.find("\n---\n", 4)
        if end < 0:
            errs.append("YAML front matter is not closed")
        else:
            front = card[4:end]
            for key in ("license:", "base_model:", "library_name:"):
                if key not in front:
                    errs.append(f"front matter lacks {key}")
            body = card[end + 5 :]
            head = "\n".join(body.strip().splitlines()[:12])
            if ATTRIBUTION not in head:
                errs.append(f'"{ATTRIBUTION}" must appear in the first lines of the card body')
    if ATTRIBUTION not in card:
        errs.append(f'card lacks "{ATTRIBUTION}"')
    for name in ("LICENSE-MUSE-SPARK-OPEN.md", "USE_POLICY-MUSE-SPARK-OPEN.md", "USAGE_POLICY.md"):
        if name not in card:
            errs.append(f"card does not reference {name}")
    if "—" in card:
        errs.append("card contains an em-dash")
    return errs


def check_staged(folder: Path) -> list[str]:
    """Hard errors for a staged release folder: missing required files, unexpected files, wrong attribution."""
    errs: list[str] = []
    names = {p.name for p in folder.iterdir()}
    for n in REQUIRED:
        if n not in names:
            errs.append(f"missing {n}")
    if not any(ALLOWED_RE.match(n) for n in names):
        errs.append("no model-XXXXX-of-XXXXX.safetensors shards")
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).as_posix()
        if p.is_dir():
            errs.append(f"unexpected directory {rel}")
        elif p.name not in ALLOWED_EXACT and not ALLOWED_RE.match(p.name):
            errs.append(f"unexpected file {rel}")
    notice = folder / "NOTICE"
    if notice.exists() and ATTRIBUTION not in notice.read_text():
        errs.append("NOTICE lacks the attribution line")
    return errs


def base_files_to_copy(names: list[str]) -> list[str]:
    """Non-weight files of the base checkpoint that carry over unchanged."""
    keep = []
    for n in names:
        if n in BASE_SKIP or n.startswith(BASE_SKIP_PREFIX) or n.endswith(".safetensors"):
            continue
        keep.append(n)
    return sorted(keep)


def lora_module_name(adapter_key: str) -> tuple[str, str]:
    """('base_model.model.<mod>.lora_A.weight') -> ('<mod>.weight', 'A')."""
    if not adapter_key.startswith("base_model.model."):
        raise ValueError(f"unexpected adapter key {adapter_key!r}")
    name = adapter_key[len("base_model.model.") :]
    mod, ab = name.rsplit(".lora_", 1)
    if ab not in ("A.weight", "B.weight"):
        raise ValueError(f"unexpected adapter key {adapter_key!r}")
    return mod + ".weight", ab[0]
