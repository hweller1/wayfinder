"""Modal CPU job: convert the Spark Open teacher rollouts on the volume into SFT v1 training data.

    modal run wayfinder/sft/convert_teacher_modal.py                 # -> wayfinder-data:/sft/teacher_v1/train-*.jsonl + summary.json
    modal run wayfinder/sft/convert_teacher_modal.py --out sft/x     # other output dir (relative to the volume root)

Reads /teacher/spark_v1/*.jsonl and /teacher/spark_v1_baseten/*.jsonl (files starting with `_` are progress/failure
logs and are skipped), the Glimmer tokenizer from wayfinder-weights, and the 33 strict all-gold A1 examples from
/sft/a1_converted/train-*.jsonl. Conversion logic and filters live in wayfinder/sft/convert_teacher.py. The run
replaces the output directory's train-*.jsonl so a re-run never mixes generations of the data.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
TOKENIZER_DIR = "/weights/Muse-Glimmer-30B"
TEACHER_DIRS = ["/data/teacher/spark_v1", "/data/teacher/spark_v1_baseten"]
SHARD = 100  # examples per output file

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.21.0", "tokenizers", "jinja2", "huggingface_hub")
    .env({"PYTHONPATH": "/root", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(REPO / "serve" / "scout_tool_specs.json", "/srv/scout_tool_specs.json", copy=False)
    .add_local_file(REPO / "data" / "splits.json", "/srv/splits.json", copy=False)
    .add_local_file(REPO / "artifacts" / "sft" / "e2e_excluded_tasks.json", "/srv/e2e_excluded_tasks.json", copy=False)
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
app = modal.App("wayfinder-sftconv-teacher", image=image)
data = modal.Volume.from_name("wayfinder-data")
weights = modal.Volume.from_name("wayfinder-weights")


@app.function(volumes={"/data": data, "/weights": weights}, timeout=30 * 60, cpu=4, memory=16384)
def convert(out_dir: str, include_a1: bool = True, max_tokens: int = 32768, min_recall: float | None = None,
            exclude_e2e: bool = False, teacher_dirs: list[str] | None = None) -> dict:
    from collections import Counter

    from transformers import AutoTokenizer

    from wayfinder.sft.convert_teacher import a1_examples, convert_rows

    t0 = time.time()
    data.reload()
    splits = json.load(open("/srv/splits.json"))
    tools = json.load(open("/srv/scout_tool_specs.json"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    rows, bad, files = [], 0, 0
    for d in (teacher_dirs or TEACHER_DIRS):
        for f in sorted(glob.glob(f"{d}/*.jsonl")):
            if os.path.basename(f).startswith("_"):
                continue
            files += 1
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1  # a row being appended while we read
    excl = {t["task_id"] for t in json.load(open("/srv/e2e_excluded_tasks.json"))["tasks"]} if exclude_e2e else set()
    t_ex, stats = convert_rows(rows, splits, tok, tools, max_tokens, min_recall, excl)
    a1: list[dict] = []
    if include_a1:
        raw = []
        for f in sorted(glob.glob("/data/sft/a1_converted/train-*.jsonl")):
            raw += [json.loads(line) for line in open(f) if line.strip()]
        a1 = a1_examples(raw, tools, excl)
    allx = t_ex + a1
    out = Path("/data") / out_dir.strip("/")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("train-*.jsonl"):
        old.unlink()
    for i in range(0, len(allx), SHARD):
        with open(out / f"train-{i // SHARD:03d}.jsonl", "w") as fh:
            for e in allx[i : i + SHARD]:
                fh.write(json.dumps(e) + "\n")
    summary = {
        "generated_unix": int(time.time()), "teacher_dirs": teacher_dirs or TEACHER_DIRS, "teacher_files": files, "unparseable_rows": bad, **stats,
        "teacher_examples": len(t_ex), "a1_strict_all_gold_examples": len(a1), "examples_total": len(allx),
        "tokens_total": sum(e["n_tokens"] for e in allx),
        "assistant_tokens_total_all": sum(sum(b - a for a, b in e["assistant_spans"]) for e in allx),
        "semantic_examples_total": sum(1 for e in allx if e["meta"].get("semantic")),
        "by_source": dict(Counter(e["meta"]["source"] for e in allx)),
        "e2e_excluded_task_ids": sorted(excl), "tasks_unique": len({e["meta"]["task_id"] for e in allx}), "repos_unique": len({e["meta"]["repo"] for e in allx}),
        "semantic_share": round(sum(1 for e in allx if e["meta"].get("semantic")) / max(1, len(allx)), 4),
        "max_tokens_filter": max_tokens, "min_recall": min_recall, "max_example_tokens": max((e["n_tokens"] for e in allx), default=0), "out_dir": str(out), "secs": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    data.commit()
    return summary


@app.local_entrypoint()
def main(out: str = "sft/teacher_v1", no_a1: bool = False, max_tokens: int = 32768, min_recall: float = -1.0,
         exclude_e2e: bool = False, summary: str = "teacher_v1_summary.json", teacher_dirs: str = "") -> None:
    """teacher_dirs: comma-separated volume paths (e.g. /data/teacher/spark_v1,/data/teacher/spark_v1_baseten,/data/teacher/spark_v2); default = v1 dirs."""
    dirs = [d for d in teacher_dirs.split(",") if d] or None
    s = convert.remote(out, not no_a1, max_tokens, None if min_recall < 0 else min_recall, exclude_e2e, dirs)
    d = REPO / "artifacts" / "sft"
    d.mkdir(parents=True, exist_ok=True)
    (d / summary).write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
