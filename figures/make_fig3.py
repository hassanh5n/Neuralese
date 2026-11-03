"""Fig 3: snap by layer, one line per fork kind, with 95% CI bands (Exp 21, items 10-49).

Reads  results/exp21/step4_alpha_sweep_items10-49.jsonl
Writes figures/fig3.pdf, figures/fig3.png
Run from anywhere:  python figures/make_fig3.py

Every number in the caption has a claim-ledger row and is asserted here:
  n per kind (C21.3), snap at layers 7/14/21/final (C21.4), final-layer CI (C21.5), layer 0 exact (C21.2),
  number vs number below word vs word at layers 1-12 and above from 13 (C21.23),
  number vs word highest at every layer (C21.24).

Draft caption:
  Blends collapse progressively. At each fork we feed a blend of the chosen token and the
  runner-up and measure how sharply each layer resolves it (snap: 0 = carried linearly,
  1 = clean switch). Snap is 0 at the input by construction (layer 0 is the blend itself) and
  reaches 0.98 (number vs word), 0.84
  (number vs number) and 0.77 (word vs word) at the output. Number vs word snaps hardest at
  every layer; number vs number starts slowest and passes word vs word at layer 13. Bands:
  95% bootstrap CIs over problems. 670 unique forks from 40 GSM8K problems (items 10-49).
"""
import numpy as np

from data import boot_mean, group, rows, snap, unique
from style import BAND_ALPHA, COLORS, HALF, LABELS, plain_ticks, plt, save

LAYERS = np.arange(29)  # 0 = input embedding, 28 = final (normed) hidden state; index 29 (logits) not plotted
LEDGER = {  # kind: (n forks C21.3, snap at L7/14/21/final C21.4, final CI C21.5)
    "nn": (238, (0.37, 0.59, 0.76, 0.84), (0.83, 0.86)),
    "nw": (56, (0.70, 0.80, 0.90, 0.98), (0.95, 1.00)),
    "ww": (376, (0.44, 0.54, 0.65, 0.77), (0.75, 0.79)),
}

sw = unique(rows("exp21/step4_alpha_sweep_items10-49.jsonl"))
mean, lo, hi = {}, {}, {}
for g, (n, snaps, ci) in LEDGER.items():
    s = [r for r in sw if group(r) == g]
    S = np.array([[snap(r, L) for L in LAYERS] for r in s])
    mean[g], (lo[g], hi[g]) = S.mean(0), boot_mean(s, S)
    assert len(s) == n, (g, len(s))
    assert tuple(round(float(mean[g][L]), 2) for L in (7, 14, 21, 28)) == snaps, g
    assert (round(float(lo[g][28]), 2), round(float(hi[g][28]), 2)) == ci, g

L1 = LAYERS[1:]
assert list(L1[mean["nn"][1:] < mean["ww"][1:]]) == list(range(1, 13))  # C21.23
assert list(L1[hi["nn"][1:] < lo["ww"][1:]]) == list(range(1, 10))
assert list(L1[lo["nn"][1:] > hi["ww"][1:]]) == list(range(13, 29))
assert (lo["nw"][1:] > np.maximum(hi["nn"][1:], hi["ww"][1:])).all()  # C21.24

fig, ax = plt.subplots(figsize=(HALF, 2.1))
ax.axhline(1, color=COLORS["guide"], lw=0.6, ls=(0, (1, 1.5)), zorder=1)
ax.text(0.4, 1.012, "clean switch", color=COLORS["guide"], fontsize=6.5, va="bottom")
for g in ("nw", "nn", "ww"):  # legend order = top to bottom at the output
    ax.fill_between(LAYERS, lo[g], hi[g], color=COLORS[g], alpha=BAND_ALPHA, lw=0, zorder=2)
    ax.plot(LAYERS, mean[g], color=COLORS[g], label=f"{LABELS[g]} ($n = {LEDGER[g][0]}$)", zorder=3)

ax.set_xlim(0, 28)
ax.set_ylim(0, 1.07)
ax.set_xticks([0, 7, 14, 21, 28])
ax.set_xticklabels(["0\ninput", "7", "14", "21", "28\noutput"])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
plain_ticks(ax)
ax.set_xlabel("layer")
ax.set_ylabel("snap (1 = clean switch)")
ax.legend(loc="lower right", borderaxespad=0.2)
save(fig, "fig3")
