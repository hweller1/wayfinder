"""Scout architecture diagram (dark theme), 1600x1000 px. Run: python3 docs/figures/proto_arch_fig.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "s_proto_arch.png"
BG, INK, MUTED, EDGE = "#001E2B", "#FFFFFF", "#C1C7C6", "#1F3B49"
GREEN, AMBER, BLUE = "#00ED64", "#EDA100", "#5C9DF0"
fig = plt.figure(figsize=(16, 10), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="#06293A", ec=color, lw=3))
    ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="center", fontsize=23, fontweight="bold", color=color)


def item(x, y, w, h, lines, color, fs=16.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc=BG, ec=EDGE, lw=2))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)


def arrow(p, q, color, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=28, lw=3, color=color, shrinkA=0, shrinkB=0))


def label(x, y, s, color):
    ax.text(x, y, s, ha="center", va="center", fontsize=16, color=color, fontweight="bold")


# Box 1: laptop
box(2, 17, 42, 76, "Developer laptop", GREEN)
item(5, 70, 36, 14, ["Muse Code / Claude Code", "+ Scout plugin"], GREEN, 15)
item(5, 52, 36, 14, ["Retrieval spec", "(.scout/spec.toml)"], GREEN, 15)
item(5, 35, 36, 13, ["Local grep / read"], GREEN, 15)
item(5, 20, 36, 11.5, ["Merkle sync", "(changed files only)"], GREEN, 15)
# Box 2: hosted
box(68, 17, 38, 76, "One GCP VM (A100)", AMBER)
item(71, 56, 32, 28, ["Scout gateway", "(Caddy HTTPS)", "one scout key,", "tenant isolation"], AMBER)
item(71, 22, 32, 28, ["wayfinder-glimmer", "(vLLM, localhost)"], AMBER, 15)
arrow((87, 56), (87, 50), AMBER, both=True)
# Box 3: stores
box(128, 17, 30, 76, "Stores", BLUE)
item(130.5, 56, 25, 28, ["turbopuffer", "encrypted paths", "+ line ranges only"], BLUE, 14.5)
item(130.5, 22, 25, 28, ["Voyage", "voyage-code-4", "(embeddings)"], BLUE, 14.5)
arrow((44, 78), (71, 78), GREEN); label(57.5, 82.5, "explore", GREEN)
arrow((44, 62), (71, 62), GREEN); label(57.5, 66.5, "chunks (sync)", GREEN)
arrow((103, 72), (130.5, 72), BLUE, both=True); label(117, 76.5, "vectors", BLUE)
arrow((103, 56), (130.5, 40), BLUE); label(120, 53, "embed", BLUE)
ax.text(80, 8, "Code is chunked on the laptop. Chunk text is embedded in transit and never stored; the index holds vectors and encrypted paths.",
        ha="center", va="center", fontsize=17, color=MUTED)
fig.savefig(OUT, dpi=100, facecolor=BG)
print(OUT)
