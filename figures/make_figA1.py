"""Fig A1 (appendix): runner-up readback at each forced-text position 1-16, without binning (Exp 21).

Reads  results/exp21/step5_kv_carryover_items10-49.jsonl
Writes figures/figA1.pdf, figures/figA1.png
Same forks and score as Figure 4; shows that its log-spaced bins do not drive the result.
Caption claim asserted here: runner-up's text means above 0 at every position 1-16, num-num and word-word (C21.38).

Draft caption:
  Readback at each position of the forced text, without the binning of Figure 4b-c. Top: mean
  readback (control minus blend) with 95% bootstrap CIs over problems, for the runner-up's text
  (left) and the model's own text (right). Bottom: number of forks with a defined score at that
  position (positions where the chosen token and the runner-up barely differ are skipped). With
  the runner-up's text, the mean stays above 0 at every position for number vs number and word vs
  word; with the model's own text it stays around 0. Single positions hold few forks, so their CIs
  are wide; this is why Figure 4 pools positions into log-spaced bins.
"""
import numpy as np

from data import boot_mean, followed_chosen, group, readback, rows, unique
from style import BAND_ALPHA, COLORS, FULL, LABELS, plain_ticks, plt, save

KINDS, POS = ("nn", "nw", "ww"), np.arange(1, 17)
ca = unique(rows("exp21/step5_kv_carryover_items10-49.jsonl"))
fig = plt.figure(figsize=(FULL, 2.7))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
bot = [fig.add_subplot(gs[1, 0], sharex=top[0]), fig.add_subplot(gs[1, 1], sharex=top[1])]
for col, (src, title) in enumerate((("B", "(a) runner-up's text"), ("A", "(b) model's own text"))):
    for g in KINDS:
        c = [r for r in ca if group(r) == g and followed_chosen(r) and readback(r, "B") == readback(r, "B")]
        m, lo, hi, n = [], [], [], []
        for p in POS:
            fr = [(r, readback(r, src, p, p)) for r in c]
            fr = [(r, x) for r, x in fr if x == x]
            a, b = boot_mean([r for r, _ in fr], np.array([[x] for _, x in fr]))
            m.append(np.mean([x for _, x in fr])), lo.append(a[0]), hi.append(b[0]), n.append(len(fr))
        top[col].fill_between(POS, lo, hi, color=COLORS[g], alpha=BAND_ALPHA, lw=0, zorder=2)
        top[col].plot(POS, m, color=COLORS[g], marker="o", ms=2.2, lw=1.0, label=LABELS[g], zorder=3)
        if src == "B" and g != "nw":
            assert min(m) > 0, g  # C21.38
        bot[col].plot(POS, n, color=COLORS[g], marker="o", ms=1.8, lw=0.9)
    top[col].axhline(0, color=COLORS["guide"], lw=0.6, zorder=1)
    top[col].set_title(title, loc="left")
    top[col].tick_params(labelbottom=False)
    bot[col].set_xticks([1, 4, 8, 12, 16])
    bot[col].set_xlim(0.5, 16.5)
    bot[col].set_ylim(0, 150)
    bot[col].set_yticks([0, 50, 100, 150])
    bot[col].set_xlabel("position in forced text")
top[0].set_ylim(-0.6, 1.6)
top[1].sharey(top[0])
top[1].tick_params(labelleft=False)
bot[1].tick_params(labelleft=False)
top[0].set_ylabel("readback")
bot[0].set_ylabel("forks")
top[1].legend(loc="upper right", borderaxespad=0.1)
for ax in top + bot:
    plain_ticks(ax)
save(fig, "figA1")
