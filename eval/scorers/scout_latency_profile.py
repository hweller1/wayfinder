"""Latency decomposition of cc_scout_spec vs cc_explore_spec (cc50 full run). CPU only, read-only on inputs.

    python3 scripts/scout_latency_profile.py            # prints the numbers used in docs/scout_latency_profile.md
    python3 scripts/scout_latency_profile.py --fig      # also writes docs/figures/s11b_latency.png
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

WF = Path(__file__).resolve().parents[1]
VS = WF.parent / "voyage-scout"
FULL = VS / "artifacts/eval/cc50/full"
load = lambda p: [json.loads(x) for x in open(p)]  # noqa: E731
S = load(FULL / "rows_cc_scout_spec_s0.jsonl")
X = load(FULL / "rows_cc_explore_spec_s0.jsonl")
pct = lambda v: (float(np.percentile(v, 50)), float(np.percentile(v, 90)))  # noqa: E731

# Estimate 1: Sonnet subagent time = a + b * tool calls, fit on the explore arm (no MCP, one Sonnet layer).
b, a = np.polyfit([r["subagent_tool_calls"] for r in X], [r["explore_s"] for r in X], 1)

def parent_split(r):  # estimate: residual split 45/55 pre/post (output-token argument, see doc)
    res = r["wall_s"] - r["explore_s"]
    return 0.45 * res, 0.55 * res

comp = {"scout": {}, "explore": {}}
sc = [r for r in S if r["explore_s"]]
cols = {k: [] for k in ("pre", "post", "sonnet", "gw", "tools", "v3model", "wall")}
for r in sc:
    pre, post = parent_split(r)
    n_other = len(r["sub_tool_seq"]) - 1
    sonnet = a + b * n_other
    call = r["explore_s"] - sonnet
    gw, tools = 1.4, 3.9  # estimates: gateway RT (smoke first_request_s), v3 tool time (heldout wall-model_latency median)
    for k, v in zip(cols, (pre, post, sonnet, gw, tools, call - gw - tools, r["wall_s"])):
        cols[k].append(v)
comp["scout"] = {k: pct(v) for k, v in cols.items()}
cx = {"pre": [], "post": [], "sub": [], "wall": []}
for r in X:
    pre, post = parent_split(r)
    for k, v in zip(cx, (pre, post, r["explore_s"], r["wall_s"])):
        cx[k].append(v)
comp["explore"] = {k: pct(v) for k, v in cx.items()}
if __name__ == "__main__":
    print("explore-arm fit: sonnet_s = %.2f + %.2f * calls" % (a, b))
    for arm, d in comp.items():
        print(arm)
        for k, (m, p) in d.items():
            print("  %-8s med %6.1f  p90 %6.1f" % (k, m, p))
    # turn-budget what-if from direct v3 sem runs (heldout hard v1)
    rows = []
    for f in glob.glob(str(WF / "artifacts/heldout_hard/shards/v3_s*/sft_v3_sem.jsonl")):
        rows += load(f)
    for cap in (4, 6, 8):
        fr = [sum(t["latency_s"] for t in r["turn_log"][cap:]) / sum(t["latency_s"] for t in r["turn_log"]) for r in rows]
        print("cap", cap, "mean share of model latency after cap", round(float(np.mean(fr)), 3))
    if "--fig" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        BG, INK, MUTED, GRID = "#001E2B", "#FFFFFF", "#C1C7C6", "#1F3B49"
        plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                             "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID,
                             "font.size": 15, "font.family": "DejaVu Sans"})
        s, e = comp["scout"], comp["explore"]
        for tag, idx in (("median", 0), ("p90", 1)):
            pass
        segs = [  # (label, color, explore value, scout value) at the median
            ("Parent turns (before + after)", "#7E8C91", e["pre"][0] + e["post"][0], s["pre"][0] + s["post"][0]),
            ("Sonnet subagent turns", "#5C9DF0", e["sub"][0], s["sonnet"][0]),
            ("Gateway + tool time (est.)", "#EDA100", 0, s["gw"][0] + s["tools"][0]),
            ("wayfinder-glimmer model turns inside explore (est.)", "#00ED64", 0, s["v3model"][0]),
        ]
        fig, ax = plt.subplots(figsize=(12, 5.2))
        for yi, (name, col) in enumerate((("cc_explore_spec", 0), ("cc_scout_spec", 1))):
            left = 0
            for lab, c, ev, sv in segs:
                v = (ev, sv)[col]
                ax.barh(yi, v, left=left, color=c, edgecolor=BG, height=0.55, label=lab if yi == 0 else None)
                if v > 9:
                    ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", color=BG, fontsize=15, fontweight="bold")
                left += v
            ax.text(left + 2, yi, f"{left:.0f} s", va="center", color=INK, fontsize=16)
        ax.set_yticks([0, 1], ["cc_explore_spec", "cc_scout_spec"])
        ax.set_xlabel("median seconds per run (component medians; they sum to about the wall median)")
        ax.set_xlim(0, 200)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("Where 164 s goes: wayfinder-glimmer's turn loop inside one explore call", loc="left", color=INK, fontsize=18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, labelcolor=INK, fontsize=13)
        fig.tight_layout()
        fig.savefig(WF / "docs/figures/s11b_latency.png", dpi=150)
