"""Fig 4: runner-up readback, per fork and by position in the forced text (Exp 21, items 10-49).

Reads  results/exp21/step5_kv_carryover_items10-49.jsonl
Writes figures/fig4.pdf, figures/fig4.png
Run from anywhere:  python figures/make_fig4.py

Every number in the caption has a claim-ledger row and is asserted here:
  fork-level readback with the runner-up's text, mean and CI (C21.7-8), with the own text (C21.9, CI C21.25);
  runner-up's text CI > 0 in every position bin, number vs number and word vs word (C21.26);
  own text CI includes 0 in every bin, same kinds (C21.27).

Draft caption:
  The runner-up is stored and can be read back, but stays dormant on the model's own path. At
  forks where the model followed the chosen token, we force a 64-token continuation into the cache
  built from the 50/50 blend and into a control cache (chosen token blended with an unrelated
  token), and score the next-token predictions from 0 (as if the runner-up had been written) to 1
  (as if the chosen token had). Readback = control minus blend; above 0 means the runner-up is read
  back from the cache. (a) Per fork: with the runner-up's text, +0.44 (number vs number), +0.11
  (number vs word) and +0.33 (word vs word); with the model's own text, -0.01 to +0.06, with every
  CI including 0. (b) With the runner-up's text, the readback stays above 0 at every position up
  to 64 tokens for number vs number and word vs word. (c) With the model's own text, it stays at 0.
  Positions are binned on a log scale; Figure A1 shows them unbinned. Bars and bands: 95% bootstrap CIs over problems.
  n = 96, 34 and 139 forks.
"""
import numpy as np
from matplotlib.lines import Line2D

from data import POS_BINS, boot_mean, followed_chosen, group, readback, rows, unique
from style import BAND_ALPHA, COLORS, FULL, LABELS, plain_ticks, plt, save

KINDS = ("nn", "nw", "ww")
LEDGER = {  # kind: (n C21.7-8, runner-up text mean C21.7, CI C21.8, own text mean C21.9, CI C21.25)
    "nn": (96, 0.44, (0.35, 0.53), -0.01, (-0.06, 0.04)),
    "nw": (34, 0.11, (0.03, 0.19), 0.06, (-0.0, 0.19)),
    "ww": (139, 0.33, (0.27, 0.40), 0.02, (-0.02, 0.07)),
}


def ci_of(rs, vals):  # bootstrap over problems for a 1-d list of per-fork values
    lo, hi = boot_mean(rs, np.asarray(vals, float)[:, None])
    return float(lo[0]), float(hi[0])


ca = unique(rows("exp21/step5_kv_carryover_items10-49.jsonl"))
fork, by_pos = {}, {}
for g in KINDS:
    c = [r for r in ca if group(r) == g and followed_chosen(r) and readback(r, "B") == readback(r, "B")]
    n, mb, cib, ma, cia = LEDGER[g]
    assert len(c) == n, (g, len(c))
    for src, m_led, ci_led in (("B", mb, cib), ("A", ma, cia)):
        v = [readback(r, src) for r in c]
        m, ci = float(np.mean(v)), ci_of(c, v)
        assert round(m, 2) == m_led and tuple(round(x, 2) for x in ci) == ci_led, (g, src, m, ci)
        fork[g, src] = (m, ci)
        pts = []
        for lo, hi in POS_BINS:
            fr = [(r, readback(r, src, lo, hi)) for r in c]
            fr = [(r, x) for r, x in fr if x == x]
            pts.append((float(np.mean([x for _, x in fr])), *ci_of([r for r, _ in fr], [x for _, x in fr])))
        by_pos[g, src] = np.array(pts)  # rows: bins; cols: mean, lo, hi
for g in ("nn", "ww"):
    assert (by_pos[g, "B"][:, 1] > 0).all(), g  # C21.26
    assert ((by_pos[g, "A"][:, 1] <= 0) & (by_pos[g, "A"][:, 2] >= 0)).all(), g  # C21.27

fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), gridspec_kw={"width_ratios": [1.3, 1, 1]})
for ax in axes:
    ax.axhline(0, color=COLORS["guide"], lw=0.6, zorder=1)

# (a) per fork
ax = axes[0]
for i, g in enumerate(KINDS):
    for src, dx, face in (("B", -0.14, COLORS[g]), ("A", 0.14, "white")):
        m, (lo, hi) = fork[g, src]
        ax.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.6, mfc=face, mec=COLORS[g], mew=1,
                    color=COLORS[g], elinewidth=1, capsize=1.8, capthick=0.8, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels(["number vs\nnumber", "number vs\nword", "word vs\nword"], fontsize=6.5)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.12, 0.72)
ax.set_ylabel("readback (control $-$ blend)")
ax.set_title("(a) per fork, positions 1--64", loc="left")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.6, mfc="#555555", mec="#555555", label="runner-up's text"),
                   Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec="#555555", label="model's own text")],
          loc="upper right", handletextpad=0.2, borderaxespad=0.1)

# (b), (c) by position; each bin drawn at its geometric centre on a log axis
x = np.array([np.sqrt(a * b) for a, b in POS_BINS])
for ax, src, title in ((axes[1], "B", "(b) runner-up's text"), (axes[2], "A", "(c) model's own text")):
    for g in KINDS:
        p = by_pos[g, src]
        ax.fill_between(x, p[:, 1], p[:, 2], color=COLORS[g], alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(x, p[:, 0], color=COLORS[g], marker="o", ms=2.4, lw=1.1, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 64)
    ax.set_ylim(-0.32, 1.17)
    ax.set_xlabel("position in forced text")
    ax.set_title(title, loc="left")
axes[1].set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
axes[2].sharey(axes[1])
axes[2].tick_params(labelleft=False)
for ax in axes:
    plain_ticks(ax)

axes[2].legend(handles=[Line2D([], [], color=COLORS[g], marker="o", ms=2.4, lw=1.1, label=LABELS[g])
                        for g in KINDS], loc="upper right", borderaxespad=0.1, handlelength=1.3)
save(fig, "fig4")
