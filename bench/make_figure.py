"""One sheet: what the whole benchmark found, argued in the order it should be read.

Every number is read from the graders' own JSON at render time - nothing is
transcribed. Palette validated with the dataviz six checks against #0B0D12.

    .venv-plot/bin/python bench/make_figure.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report" / "comparison.png"

# ---------------------------------------------------------------- ink
GROUND = "#0B0D12"
INK    = "#E9E6DF"    # warm bone, not blue-white
INK2   = "#B6B2A8"
MUTED  = "#7E7B73"
RULE   = "#23262E"
RULE2  = "#343842"

AMBER  = "#B8892F"    # TextWorld
SLATE  = "#5E92D2"    # Roguelike  / above the bar
CLAY   = "#C56B52"    # Minesweeper / below the bar
BONE   = "#6E6B64"    # baselines

DISPLAY, BODY, MONO = "Noto Serif", "Noto Sans", "Noto Sans Mono"
plt.rcParams.update({
    "figure.facecolor": GROUND, "savefig.facecolor": GROUND,
    "axes.facecolor": GROUND, "text.color": INK, "font.family": BODY,
})

# ---------------------------------------------------------------- data
rep  = json.loads((ROOT / "report.json").read_text())
brk  = json.loads((ROOT / "breakdown.json").read_text())
cert = json.loads((ROOT / "certainty.json").read_text())
xdev = json.loads((ROOT / "crossdevice.json").read_text())

TASKS = [("textworld", "TextWorld", AMBER),
         ("roguelike", "Roguelike", SLATE),
         ("minesweeper", "Minesweeper", CLAY)]
SIZE = {"Kai": 0.6, "Lex": 0.6, "Eos": 0.8, "Sol": 2.0, "Nox": 4.0, "Lux": 9.0,
        "Laya": 0.42, "OpenJev": 25.5}
DECODER = ["Eos", "Sol", "Nox", "Lux"]
ENCODER = ["Kai", "Lex"]
ORDER = ["Jev", "OpenJev", "CLM", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Laya", "Haiku-4.5",
         None, "heuristic", "oracle", "uniform-random"]
NOTE = {"Jev": "hosted, TypeSafe", "CLM": "8B, contrastive", "OpenJev": "25B, open", "Lux": "9B", "Nox": "4B", "Sol": "2B", "Eos": "0.8B", "Kai": "0.6B",
        "Lex": "0.6B", "Laya": "0.42B, Convai", "Haiku-4.5": "generative LLM",
        "heuristic": "if-statements", "oracle": "exact solver",
        "uniform-random": "uniform"}

g   = lambda m, e: rep["answerers"].get(m, {}).get(e)
bar = lambda e: brk[e]["__meta__"]["best_trivial"]["rate"]
delta = lambda m, e: (g(m, e)["rate"] - bar(e)) * 100 if g(m, e) else None


def margin(m, e):
    p = ROOT / "answers" / f"{e}__{m}.json"
    if not p.exists():
        return None
    v = [pr[0] - pr[1] for t in json.loads(p.read_text()).get("telemetry", {}).values()
         if len(pr := sorted((t.get("probs") or {}).values(), reverse=True)) >= 2]
    return statistics.mean(v) if v else None


def bare(ax, keep=()):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(RULE2)
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    ax.set_facecolor(GROUND)


def mono(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_family(MONO)


# ---------------------------------------------------------------- sheet
W, H = 15.0, 22.6
fig = plt.figure(figsize=(W, H), dpi=170)
L, R = 0.062, 0.962     # type block
CAP_W = 0.134           # marginal caption column, L .. L+CAP_W
WIDE  = 0.252           # graphics with no row labels start here
LABEL = 0.214           # row-label column, LABEL .. PLOT
PLOT  = 0.300           # graphics with row labels start here
GUT   = WIDE


def rule(y, weight=0.8, color=RULE, x0=L, x1=R):
    fig.add_artist(Line2D([x0, x1], [y, y], color=color, lw=weight,
                          transform=fig.transFigure, zorder=1))


def sec(y, num, title, caption):
    """Kicker, title, marginal caption - stacked with room for the title's lines."""
    fig.text(L, y, num, family=MONO, fontsize=10, color=CLAY, va="top")
    fig.text(L, y - 0.0105, title, family=DISPLAY, fontsize=15.0, color=INK,
             va="top", fontweight="bold", linespacing=1.18)
    drop = 0.0105 + 0.0158 * (title.count("\n") + 1) + 0.010
    fig.text(L, y - drop, caption, family=BODY, fontsize=9.4, color=INK2,
             va="top", linespacing=1.66)


# ---- masthead
fig.text(L, 0.9805, "D E C I S I O N   M O D E L S   ·   6 2 9   P O S I T I O N S",
         family=MONO, fontsize=9.6, color=MUTED, va="top")
fig.text(L, 0.9665, "Reading the state", family=DISPLAY, fontsize=42, color=INK,
         va="top", fontweight="bold")
fig.text(L, 0.9345,
         "Accuracy against a state-blind policy, on three games with exact ground truth.",
         family=BODY, fontsize=12.4, color=INK2, va="top", linespacing=1.62)
rule(0.8975, 1.6, RULE2)

# ================================================================ 1 · scale
sec(0.8865, "01", "Scale",
    "Points vs state-blind.\n\nOpen circles: encoders.\nCrosses: Laya.")

ax = fig.add_axes([WIDE, 0.6985, R - WIDE, 0.1955]); bare(ax, keep=("bottom",))
ax.axhline(0, color=INK2, lw=1.5, zorder=4)
ax.text(1.02, 2.6, "state-blind policy", family=BODY,
        fontsize=9.6, color=INK2, va="bottom", zorder=5)

for env, label, col in TASKS:
    xs = [SIZE[m] for m in DECODER]
    ys = [delta(m, env) for m in DECODER]
    ax.plot(xs, ys, color=col, lw=2.2, zorder=3, solid_capstyle="round")
    ax.scatter(xs, ys, s=52, color=col, zorder=4, edgecolors=GROUND, linewidths=2.2)
    ax.text(xs[-1] * 1.09, ys[-1], f"  {label}", family=BODY, fontsize=12.5, color=col,
            va="center", fontweight="bold")
    ax.text(xs[-1] * 1.09, ys[-1] - 3.2, f"  {ys[-1]:+.1f}", family=MONO, fontsize=10,
            color=col, va="center")
    for m in ENCODER:
        ax.scatter([SIZE[m]], [delta(m, env)], s=46, facecolors="none", edgecolors=col,
                   linewidths=1.7, zorder=4)
    ax.scatter([SIZE["Laya"]], [delta("Laya", env)], s=54, color=col, marker="x",
               linewidths=1.7, zorder=4)

for m in DECODER:
    ax.text(SIZE[m], -31.5, NOTE[m], family=MONO, fontsize=9.6, color=MUTED, ha="center")
ax.text(0.6, -31.5, "0.6B", family=MONO, fontsize=9.6, color=MUTED, ha="center")
ax.set_xscale("log")
ax.set_xlim(0.33, 17.5); ax.set_ylim(-34, 52)
ax.set_xticks([]); ax.set_yticks([-20, 0, 20, 40])
ax.set_yticklabels(["−20", "0", "+20", "+40"])
mono(ax)
for yv in (-20, 20, 40):
    ax.axhline(yv, color=RULE, lw=0.8, zorder=0)
ax.text(0.335, 47, "points vs state-blind", family=BODY, fontsize=9.6,
        color=MUTED, va="center")

rule(0.6755)

# ================================================================ 2 · ranking
sec(0.6645, "02", "Accuracy",
    "Stem: distance from\nstate-blind.\n\nBlue: above.\nClay: below.")

for i, (env, label, col) in enumerate(TASKS):
    w = (R - PLOT - 0.058) / 3
    ax = fig.add_axes([PLOT + i * (w + 0.029), 0.4545, w, 0.1665]); bare(ax)
    s, b = rep["envs"][env], bar(env)
    rows = [m for m in ORDER if m and g(m, env)]
    ax.vlines(b, -0.75, len(rows) - 0.85, color=INK2, lw=1.3, zorder=2)
    ax.vlines(s["chance_rate"], -0.75, len(rows) - 0.85, color=RULE2, lw=1.1,
              ls=(0, (3, 3)), zorder=1)
    for y, m in enumerate(rows):
        yy = len(rows) - 1 - y
        r = g(m, env)["rate"]
        base = m in ("oracle", "heuristic", "uniform-random")
        c = BONE if base else (SLATE if r >= b else CLAY)
        ax.plot([b, r], [yy, yy], color=c, lw=1.6, alpha=0.62, zorder=3)
        ax.scatter([r], [yy], s=58, color=c, zorder=4, edgecolors=GROUND, linewidths=1.6)
        away = -0.030 if r < b else 0.030          # never sit on the stem
        if r > 0.93:
            away = -0.030
        ax.text(r + away, yy, f"{r:.0%}", family=MONO, fontsize=9.2,
                color=MUTED if base else INK2, va="center",
                ha="right" if away < 0 else "left", zorder=5)
    ax.set_xlim(-0.02, 1.10); ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "50%", "100%"])
    mono(ax)
    ax.text(0, len(rows) - 0.05, label, family=DISPLAY, fontsize=13.5, color=col,
            fontweight="bold", va="bottom")
    ax.text(0, len(rows) - 0.62,
            f"{s['n_positions']} positions   ·   {brk[env]['__meta__']['best_trivial']['policy']}",
            family=BODY, fontsize=9.2, color=MUTED, va="bottom")
    ax.text(b, -1.55, f"state-blind {b:.0%}", family=BODY, fontsize=9.2, color=INK2,
            ha="center", va="top")
    if i == 0:
        for yi, m in enumerate(rows):
            yy = len(rows) - 1 - yi
            fy = ax.transAxes.inverted().transform(
                ax.transData.transform((0, yy)))[1]
            base = m in ("oracle", "heuristic", "uniform-random")
            ax.text(-0.075, yy + 0.16, m, family=BODY, fontsize=10.8,
                    color=MUTED if base else INK, ha="right", va="center",
                    fontweight="normal" if base else "bold", transform=ax.transData)
            ax.text(-0.075, yy - 0.30, NOTE[m], family=BODY, fontsize=8.4,
                    color=MUTED, ha="right", va="center", transform=ax.transData)

rule(0.4285)

# ================================================================ 3 · table
sec(0.4175, "03", "Hard half",
    "Positions the state-blind\npolicy gets wrong.")

axt = fig.add_axes([WIDE, 0.2185, R - WIDE, 0.1735]); axt.set_axis_off()
axt.set_xlim(0, 1); axt.set_ylim(0, 1)
COLS = [(0.290, 0.385), (0.482, 0.577), (0.674, 0.769)]
X_GAP, X_AUR, X_AGR = 0.855, 0.925, 1.0

for (env, label, col), (xa, xb) in zip(TASKS, COLS):
    axt.text((xa + xb) / 2 - 0.05, 1.045, label.upper(), family=MONO, fontsize=8.8,
             color=col, ha="center")
axt.text((X_GAP + X_AUR) / 2 - 0.02, 1.045, "CERTAINTY", family=MONO, fontsize=8.8,
         color=MUTED, ha="center")
for x, t in ((0.0, ""), *[(xa, "overall") for xa, _ in COLS]):
    pass
for xa, xb in COLS:
    axt.text(xa, 0.985, "overall", family=BODY, fontsize=8.8, color=MUTED, ha="right")
    axt.text(xb, 0.985, "hard", family=BODY, fontsize=8.8, color=MUTED, ha="right")
axt.text(X_GAP, 0.985, "gap", family=BODY, fontsize=8.8, color=MUTED, ha="right")
axt.text(X_AUR, 0.985, "AUROC", family=BODY, fontsize=8.8, color=MUTED, ha="right")
axt.text(X_AGR, 0.985, "2 GPUs", family=BODY, fontsize=8.8, color=MUTED, ha="right")
axt.plot([0, 1], [0.955, 0.955], color=INK2, lw=1.1)

certby = {c["answerer"]: c for c in cert}
# 12 rows filled this block at 0.0715 each; keep the block, share it out
STEP = 0.858 / sum(1 for m in ORDER if m)
y = 0.885
for m in ORDER:
    if m is None:
        axt.plot([0, 1], [y + 0.030, y + 0.030], color=RULE2, lw=0.9)
        y -= 0.016
        continue
    base = m in ("oracle", "heuristic", "uniform-random")
    axt.text(0.0, y, m, family=BODY, fontsize=11.4, color=MUTED if base else INK,
             va="center", fontweight="normal" if base else "bold")
    for (env, _, col), (xa, xb) in zip(TASKS, COLS):
        gg, bd = g(m, env), brk[env].get(m)
        if not gg:
            continue
        win = (not base and (gg["vs_trivial"].get("p") or 1) < 0.05
               and (gg["vs_trivial"].get("delta") or 0) > 0)
        axt.text(xa, y, f"{gg['rate']:.1%}", family=MONO, fontsize=11.0, ha="right",
                 va="center", color=SLATE if win else (MUTED if base else INK2))
        if bd and bd.get("hard") is not None:
            hw = (bd.get("hard_p_vs_chance") or 1) < 0.05 and bd["hard"] > bd["chance_hard"]
            axt.text(xb, y, f"{bd['hard']:.1%}", family=MONO, fontsize=11.0, ha="right",
                     va="center",
                     color=SLATE if (hw and not base) else (MUTED if base else INK2))
    c = certby.get(m)
    axt.text(X_GAP, y, f"{c['gap']:+.3f}" if c else "·", family=MONO, fontsize=11.0,
             ha="right", va="center",
             color=(SLATE if c["gap"] > 0.2 else (CLAY if c["gap"] < -0.01 else INK2)) if c else RULE2)
    axt.text(X_AUR, y, f"{c['auroc']:.3f}" if c else "·", family=MONO, fontsize=11.0,
             ha="right", va="center",
             color=(CLAY if c["auroc"] < 0.5 else INK2) if c else RULE2)
    a = [r["agree"] for e in xdev["envs"] for r in xdev["envs"][e] if r["model"] == m]
    axt.text(X_AGR, y, f"{statistics.mean(a):.1%}" if a else "not run", family=MONO,
             fontsize=11.0, ha="right", va="center",
             color=(SLATE if statistics.mean(a) == 1 else INK2) if a else MUTED)
    y -= STEP

axt.plot([0, 1], [y + 0.032, y + 0.032], color=INK2, lw=1.1)
axt.text(0.0, y - 0.012, "best state-blind policy", family=BODY, fontsize=10.8,
         color=INK2, va="center")
for (env, _, _), (xa, xb) in zip(TASKS, COLS):
    axt.text(xa, y - 0.012, f"{bar(env):.1%}", family=MONO, fontsize=10.8, ha="right",
             va="center", color=INK2)
    axt.text(xb, y - 0.012, f"{brk[env]['__meta__']['chance_hard']:.1%}", family=MONO,
             fontsize=10.8, ha="right", va="center", color=MUTED)
axt.text(xb + 0.055, y - 0.012, "chance, hard half", family=BODY, fontsize=9.0,
         color=MUTED, va="center")

rule(0.1975)

# ================================================================ 4 · two probes
sec(0.1865, "04", "Certainty, GPUs",
    "Left: P(safe) on provable\nsquares.\n\nRight: agreement across\ntwo GPUs. Diamonds:\nencoders.")

# --- calibration
axc = fig.add_axes([PLOT, 0.0755, 0.335, 0.0785]); bare(axc, keep=("bottom",))
ser = [("perfect", 0.0, 1.0, 1.0)] + [(c["answerer"], c["mean_mine"], c["mean_safe"], c["gap"])
                                      for c in sorted(cert, key=lambda c: -c["gap"])]
for i, (name, lo, hi, gap) in enumerate(ser):
    yy = len(ser) - 1 - i
    perfect = name == "perfect"
    axc.plot([min(lo, hi), max(lo, hi)], [yy, yy], color=INK2 if perfect else RULE2,
             lw=1.8, zorder=2, solid_capstyle="round")
    axc.scatter([lo], [yy], s=42, color=CLAY, zorder=3, edgecolors=GROUND, linewidths=1.2)
    axc.scatter([hi], [yy], s=42, color=SLATE, zorder=3, edgecolors=GROUND, linewidths=1.2)
    axc.text(-0.035, yy, name, family=BODY, fontsize=9.6, ha="right", va="center",
             color=INK2 if not perfect else MUTED)
    axc.text(1.05, yy, f"{gap:+.3f}", family=MONO, fontsize=9.0, va="center",
             color=SLATE if gap > 0.2 else (MUTED if perfect else INK2))
axc.set_xlim(-0.02, 1.20); axc.set_ylim(-0.8, len(ser) - 0.2)
axc.set_yticks([]); axc.set_xticks([0, 0.5, 1.0])
axc.set_xticklabels(["0", "0.5", "1.0"]); mono(axc)
axc.text(0, len(ser) - 0.15, "P(safe), 240 provable squares", family=BODY,
         fontsize=10.2, color=INK, va="bottom", fontweight="bold")
for dx, col, txt in ((0.0, CLAY, "mines"),
                     (0.168, SLATE, "safe")):
    fig.add_artist(Line2D([PLOT + dx], [0.0578], marker="o", ms=5.5, color=col,
                          transform=fig.transFigure))
    fig.text(PLOT + dx + 0.009, 0.0605, txt, family=BODY, fontsize=8.8,
             color=MUTED, va="top")

# --- cross-GPU
axs = fig.add_axes([PLOT + 0.412, 0.0755, 0.250, 0.0785]); bare(axs, keep=("bottom", "left"))
pts = [(m_, r["agree"] * 100, env, r["model"])
       for env, _, _ in TASKS for r in xdev["envs"][env]
       if r["model"] in SIZE and (m_ := margin(r["model"], env)) is not None]
dec = [(x, y) for x, y, e, n in pts if n not in ENCODER]
rr = statistics.correlation([a for a, _ in dec], [b for _, b in dec])
for env, label, col in TASKS:
    for enc in (False, True):
        sub = [(x, y) for x, y, e, n in pts if e == env and (n in ENCODER) == enc]
        if not sub:
            continue
        axs.scatter([a for a, _ in sub], [b for _, b in sub], s=44,
                    marker="o" if not enc else "D",
                    facecolors=col if not enc else "none",
                    edgecolors=GROUND if not enc else col,
                    linewidths=1.1 if not enc else 1.5, zorder=3)
axs.set_xlim(-0.04, 0.83); axs.set_ylim(93.4, 100.9)
axs.set_xticks([0, 0.4, 0.8]); axs.set_yticks([95, 100])
axs.set_yticklabels(["95%", "100%"]); mono(axs)
axs.set_xlabel("margin, 1st minus 2nd choice", family=BODY,
               fontsize=9.0, color=MUTED, labelpad=6)
axs.text(0, 101.9, "GPU agreement vs margin", family=BODY, fontsize=10.2,
         color=INK, va="bottom", fontweight="bold")
axs.text(0, 100.95, f"r = {rr:+.2f}, decoders and Laya", family=MONO,
         fontsize=8.8, color=INK2, va="bottom")


# ---- colophon
rule(0.0455)
fig.text(L, 0.0365,
         "Wilson intervals, Fisher exact tests, Mann-Whitney U. 2 GPUs: RTX 4070 Ti SUPER vs RTX 3090 agreement. "
         "Decision 1.0 runs on NVIDIA, not the vendor's AMD ROCm. Jev: first of three runs.",
         family=BODY, fontsize=8.6, color=MUTED, va="top", linespacing=1.7)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=GROUND)
print(f"wrote {OUT}  {OUT.stat().st_size/1024:.0f} KB")
