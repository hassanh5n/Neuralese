"""Fig 7: the model reads the residual (Exp 23, coupled ablation, GSM8K items 50-149).

Reads  results/exp23/exp23_residual_ablation_items50-149.jsonl
Writes figures/fig7.pdf, figures/fig7.png
Run from anywhere:  python figures/make_fig7.py

Every number in the figure and the caption has a claim-ledger row and is asserted here:
  median first divergence (C23.4), which arm diverges first (C23.5), sign test (C23.6, recomputed here as an
  exact binomial test), soft > rand on matched steps (C23.8), median per-problem KL ratio and CI, median
  matched steps (C23.15).

Draft caption:
  The model reads the residual. Three arms run in lockstep with shared noise on 100 GSM8K problems:
  sampling (top token only), Soft Thinking (full blend) and a random-residual control (the residual
  moved to a random token). (a) Share of problems on which each arm has so far picked the same
  token as sampling at every thinking step. Soft Thinking leaves sampling sooner: median step 35.5
  vs 66; it leaves first on 59 problems, the random residual on 24, both at once on 17 (sign test
  p = 0.0002). (b) Per problem, how far Soft Thinking moves next-token predictions away from
  sampling's, divided by how far the random residual does (KL divergence, averaged over the steps
  before either arm leaves sampling; median 31 steps). Soft Thinking moves them more on 94 of 100
  problems; the median problem is 3.9 times (marker: median and 95% bootstrap CI over problems,
  [3.1, 5.0]). Items 50-149, not used in any other experiment; seed 0.
"""
from math import comb
from statistics import mean, median

import numpy as np

from data import boot_stat, rows
from style import COLORS, FULL, LABELS, TEX, plain_ticks, plt, save

e23 = rows("exp23/exp23_residual_ablation_items50-149.jsonl")
assert len(e23) == 100
fs = np.array([r["flip"]["soft"] for r in e23])
fr = np.array([r["flip"]["rand"] for r in e23])
assert (median(fs), median(fr)) == (35.5, 66.0)  # C23.4
n_soft, n_rand, n_same = int((fs < fr).sum()), int((fs > fr).sum()), int((fs == fr).sum())
assert (n_soft, n_rand, n_same) == (59, 24, 17)  # C23.5
n = n_soft + n_rand
p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(min(n_soft, n_rand) + 1)) / 2**n)
assert f"{p_sign:.4f}" == "0.0002"  # C23.6
m = np.minimum(fs, fr)
ks = np.array([mean(r["kl"]["soft"][:k]) for r, k in zip(e23, m)])
kr = np.array([mean(r["kl"]["rand"][:k]) for r, k in zip(e23, m)])
q = ks / kr
assert (q > 1).sum() == 94 and (q < 1).sum() == 6  # C23.8
lo, hi = boot_stat([dict(item=r["item"], q=x) for r, x in zip(e23, q)], lambda rs: median(x["q"] for x in rs))
q_med = float(np.median(q))
assert (round(q_med, 1), round(lo, 1), round(hi, 1), median(m)) == (3.9, 3.1, 5.0, 31)  # C23.15

DARK = "#333333"
SMALL = 6.5
times = "$\\times$"

fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))

# (a) share of problems still matching sampling, by thinking step
ax = axes[0]
steps = np.unique(np.concatenate([[1], fs, fr, [600]]))
for key, f in (("soft", fs), ("rand", fr)):
    still = [(f > s).mean() for s in steps]  # flip = first step that differs; before it the arm matches sampling
    ax.step(steps, still, where="post", color=COLORS[key], lw=1.4, label=LABELS[key], zorder=3)
    med = median(f)
    ax.plot([med, med], [0, 0.5], color=COLORS[key], lw=0.7, ls=(0, (2, 1.5)), zorder=2)
    left = key == "soft"  # numbers sit just above the axis, outside the two dashed lines
    ax.text(med * (0.94 if left else 1.06), 0.015, f"{med:g}", color=COLORS[key], fontsize=SMALL,
            ha="right" if left else "left", va="bottom")
ax.axhline(0.5, color=COLORS["guide"], lw=0.6, ls=(0, (2, 1.5)), zorder=1)
ax.text(590, 0.515, "median", color=COLORS["guide"], fontsize=SMALL, ha="right", va="bottom")

# paired split: which arm leaves sampling first on each problem
x0, x1, y, dy = 0.035, 0.36, 0.445, 0.078
ax.text(x0, y, "first to leave sampling", transform=ax.transAxes, fontsize=SMALL, color=DARK, va="top")
for label, k, c in (("Soft Thinking", n_soft, COLORS["soft"]), ("random residual", n_rand, COLORS["rand"]),
                    ("same step", n_same, COLORS["sampling"])):
    y -= dy
    ax.text(x0, y, label, transform=ax.transAxes, fontsize=SMALL, color=c, va="top")
    ax.text(x1, y, f"{k}", transform=ax.transAxes, fontsize=SMALL, color=c, va="top", ha="right")
ax.text(x0, y - dy * 1.15, f"sign test $p = {p_sign:.4f}$", transform=ax.transAxes, fontsize=SMALL,
        color=DARK, va="top")

ax.set_xscale("log")
ax.set_xlim(2, 600)
ax.set_ylim(0, 1.02)
ax.set_xticks([3, 10, 30, 100, 300])
ax.set_xticklabels(["3", "10", "30", "100", "300"])
ax.minorticks_off()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlabel("thinking step")
ax.set_ylabel("share of problems\nstill matching sampling")
ax.set_title("(a) Soft Thinking leaves sampling sooner", loc="left")
ax.legend(loc="upper right", borderaxespad=0.3)
plain_ticks(ax)

# (b) per-problem ratio of how far predictions move, Soft Thinking / random residual
ax = axes[1]
edges = 2.0 ** np.arange(-2, 9)
h, _ = np.histogram(q, bins=edges)
for (a, b), k in zip(zip(edges, edges[1:]), h):
    ax.bar(np.sqrt(a * b), k, width=(b - a) * 0.9, color=COLORS["soft" if a >= 1 else "rand"], zorder=2)
ax.axvline(1, color=DARK, lw=0.8, zorder=3)
ax.text(1.12, 34.5, "equal", ha="left", va="top", fontsize=SMALL, color=DARK)

# median and its 95% bootstrap CI, drawn above the bars
ym = 32.6
ax.plot([lo, hi], [ym, ym], color=DARK, lw=0.9, solid_capstyle="butt", zorder=4)
for v in (lo, hi):
    ax.plot([v, v], [ym - 0.7, ym + 0.7], color=DARK, lw=0.9, zorder=4)
ax.plot([q_med], [ym], "o", ms=3, color=DARK, zorder=5)
ax.text(hi * 1.15, ym, f"median {q_med:.1f}{times} [{lo:.1f}, {hi:.1f}]", ha="left", va="center",
        fontsize=SMALL, color=DARK)
ax.text(250, 26.5, f"per problem: KL from sampling,\nmean over matched steps\n(median {median(m):g} steps)",
        ha="right", va="top", fontsize=SMALL, color=COLORS["guide"], linespacing=1.2)

# counts on each side of 1x, next to their bars
ax.text(0.5, 5.6, f"{(q < 1).sum()}\nproblems", ha="center", va="bottom", fontsize=SMALL, color=COLORS["rand"],
        linespacing=1.1)
ax.text(45, 11, f"{(q > 1).sum()}\nproblems", ha="center", va="bottom", fontsize=SMALL, color=COLORS["soft"],
        linespacing=1.1)

ax.set_xscale("log", base=2)
ax.set_xlim(0.25, 256)
ticks = [0.25, 1, 4, 16, 64, 256]
ax.set_xticks(ticks)
ax.set_xticklabels([f"${t:g}\\times$" for t in ticks])
ax.minorticks_off()
ax.set_ylim(0, 35)
ax.set_yticks([0, 10, 20, 30])
ax.set_xlabel("KL ratio: Soft Thinking $\\div$ random residual")
ax.set_ylabel("problems")
ax.set_title("(b) Soft Thinking moves predictions further", loc="left")
plain_ticks(ax)
save(fig, "fig7")
