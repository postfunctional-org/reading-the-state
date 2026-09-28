"""One clustered bar chart: every answerer, three tasks. Nothing else."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report" / "bars.png"

rep = json.loads((ROOT / "report.json").read_text())
TASKS = [("textworld", "TextWorld", "#B8892F"),
         ("roguelike", "Roguelike", "#5E92D2"),
         ("minesweeper", "Minesweeper", "#C56B52")]
MODELS = ["Jev", "OpenJev", "CLM", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Laya", "Haiku-4.5"]

GROUND, INK, MUTED, RULE = "#0B0D12", "#E9E6DF", "#8B8880", "#252932"
plt.rcParams.update({"figure.facecolor": GROUND, "savefig.facecolor": GROUND,
                     "axes.facecolor": GROUND, "font.family": "Noto Sans"})

fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
x = np.arange(len(MODELS))
w = 0.26

for i, (env, label, col) in enumerate(TASKS):
    vals = [rep["answerers"][m][env]["rate"] * 100 for m in MODELS]
    ax.bar(x + (i - 1) * w, vals, w, color=col, label=label, zorder=3)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 1.4, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                color=MUTED, family="Noto Sans Mono", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11, color=INK)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=9.5, color=MUTED,
                   family="Noto Sans Mono")
ax.set_ylabel("optimal picks", fontsize=10.5, color=MUTED,
              labelpad=10)
ax.grid(axis="y", color=RULE, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.tick_params(length=0)

leg = ax.legend(frameon=False, fontsize=10.5, loc="upper right", ncol=3)
for t in leg.get_texts():
    t.set_color(INK)

fig.tight_layout()
fig.savefig(OUT, facecolor=GROUND)
print(OUT)
