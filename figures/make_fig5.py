"""Fig 5: where the runner-up is stored. Share of the blend-control gap closed by KV patching (Exp 21, items 10-49).

Reads  results/exp21/step5_kv_patch_items10-49.jsonl
Writes figures/fig5.pdf, figures/fig5.png
Run from anywhere:  python figures/make_fig5.py

Every number in the caption has a claim-ledger row and is asserted here:
  layers 0-7 / 12-27 share (C21.10-11), per-block shares (C21.15-18), single layers 0/7/13 (C21.19),
  n forks and problems (C21.28a), problem-level CIs of layers 0/7/13 above 0 (C21.28b),
  single layers 1 and 2 (C21.29), every other single layer at most 0.05 (C21.30).
  Robustness numbers live in the appendix tables (verify_robustness.py, C21.32-C21.40).

Draft caption:
  The runner-up is stored mostly in the early layers. At each fork (50/50 blend; forks where the
  model followed the chosen token) we swap the keys and values of the fork position between the
  blend cache and the control cache at a set of layers, and report the pooled share of the
  blend-control readback gap this closes (summed over forks). Remove (filled): put control K/V into
  the blend cache (are these layers needed?). Insert (open): put blend K/V into the control cache
  (are they enough?). (a, b) Layers 0-7 close 77% / 71% (number vs number, remove / insert) and
  77% / 91% (word vs word) of the gap; layers 12-27 close 12% / 26% and 8% / 14%. (c) Number vs
  number, single layers: the early effect sits in layers 0-2 (remove 0.38, 0.21, 0.20). Outside
  layers 0-3, only layers 7 (0.13 / 0.12) and 13 (0.08 / 0.13) stand out, with CIs excluding 0;
  every other layer closes at most 0.05. Note the different y-axis in (c). The single-layer
  analysis was not pre-registered; Table A2 shows these layers survive a Bonferroni correction
  over all 56 single-layer tests. Table A1 shows the shares hold when any one problem, or forks
  with a small gap, are dropped; Figure A2 shows the per-fork spread, which is wide for layer 0.
  Error bars: 95% bootstrap CIs
  over problems. 96 forks from 35 problems (number vs number) and 139 forks from 40 problems
  (word vs word). Number vs word is not shown: its gap (+0.11) is too small to localise
  (Table A3).
"""
import numpy as np
from matplotlib.patches import Patch

from data import BLOCKS, boot_stat, group, patch_rows, share_closed
from style import COLORS, FULL, plain_ticks, plt, save

EARLY, LATE = ["L0-3", "L4-7"], ["L12-15", "L16-19", "L20-23", "L24-27"]
LEDGER = {  # C21.10 early, C21.11 late, C21.15/17 first blocks (remove, insert); C21.28a (forks, problems)
    "nn": {"early": (0.77, 0.71), "late": (0.12, 0.26), "blocks": ((0.58, 0.55), (0.19, 0.16), (0.06, 0.09), (0.12, 0.17)),
           "rest_max": 0.04, "n": (96, 35)},
    "ww": {"early": (0.77, 0.91), "late": (0.08, 0.14), "blocks": ((0.66, 0.75), (0.10, 0.16)), "rest_max": 0.07, "n": (139, 40)},
}
SINGLE = {"L0": (0.38, 0.16), "L7": (0.13, 0.12), "L13": (0.08, 0.13)}  # C21.19, num-num
SINGLE_EARLY = {"L1": (0.21, 0.07), "L2": (0.20, 0.05)}  # C21.29
NAMES = ("remove", "insert")

pk = patch_rows()
data = {}
for g, led in LEDGER.items():
    rs = [r for r in pk if group(r) == g]
    assert (len(rs), len({r["item"] for r in rs})) == led["n"], g
    for n_i, name in enumerate(NAMES):
        assert round(share_closed(rs, name, EARLY), 2) == led["early"][n_i], (g, name)
        assert round(share_closed(rs, name, LATE), 2) == led["late"][n_i], (g, name)
    keys = BLOCKS + ([f"L{i}" for i in range(28)] if g == "nn" else [])
    for k in keys:
        for name in NAMES:
            m = share_closed(rs, name, [k])
            data[g, k, name] = (m, *boot_stat(rs, lambda x, k=k, name=name: share_closed(x, name, [k])))
    nb = len(led["blocks"])
    assert tuple((round(data[g, k, "remove"][0], 2), round(data[g, k, "insert"][0], 2)) for k in BLOCKS[:nb]) == led["blocks"], g
    assert all(abs(round(data[g, k, nm][0], 2)) <= led["rest_max"] for k in BLOCKS[nb:] for nm in NAMES), g  # C21.16 / C21.18
for k, v in SINGLE.items():
    assert (round(data["nn", k, "remove"][0], 2), round(data["nn", k, "insert"][0], 2)) == v, k
    assert all(round(data["nn", k, nm][1], 2) > 0 for nm in NAMES), k  # C21.28b
for k, v in SINGLE_EARLY.items():
    assert (round(data["nn", k, "remove"][0], 2), round(data["nn", k, "insert"][0], 2)) == v, k
assert all(abs(round(data["nn", f"L{i}", nm][0], 2)) <= 0.05 for i in range(28) if i not in (0, 1, 2, 3, 7, 13) for nm in NAMES)  # C21.30

fig = plt.figure(figsize=(FULL, 3.6))
gs = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]


def bars(ax, g, keys, width):
    x = np.arange(len(keys))
    for dx, name in ((-width / 2, "remove"), (width / 2, "insert")):
        m, lo, hi = np.array([data[g, k, name] for k in keys]).T
        ax.bar(x + dx, m, width, color=COLORS[g] if name == "remove" else "white", edgecolor=COLORS[g], lw=0.7, zorder=2)
        ax.errorbar(x + dx, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#333333", elinewidth=0.55,
                    capsize=1 if width > 0.3 else 0, capthick=0.5, zorder=3)
    ax.axhline(0, color=COLORS["guide"], lw=0.6, zorder=1)
    return x


for ax, g, title in ((axes[0], "nn", "(a) number vs number, blocks"), (axes[1], "ww", "(b) word vs word, blocks")):
    x = bars(ax, g, BLOCKS, 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([b[1:].replace("-", "--") for b in BLOCKS])
    ax.set_xlim(-0.6, len(BLOCKS) - 0.4)
    ax.set_xlabel("layers")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("share of the gap closed")
axes[1].sharey(axes[0])
axes[1].tick_params(labelleft=False)
axes[0].set_ylim(-0.08, 0.92)
axes[0].set_yticks([0, 0.25, 0.5, 0.75])

ax = axes[2]
layers = [f"L{i}" for i in range(28)]
bars(ax, "nn", layers, 0.38)
ax.set_xticks(range(28))
ax.set_xlim(-0.7, 27.7)
ax.set_ylim(-0.08, 0.55)
ax.set_yticks([0, 0.25, 0.5])
ax.set_ylabel("share of the gap closed")
ax.set_xlabel("layer")
ax.set_title("(c) number vs number, single layers", loc="left")
for ax_ in axes:
    plain_ticks(ax_)
axes[0].legend(handles=[Patch(facecolor="#666666", edgecolor="#666666", lw=0.7, label="remove (needed?)"),
                        Patch(facecolor="white", edgecolor="#666666", lw=0.7, label="insert (enough?)")],
               loc="upper right", handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
save(fig, "fig5")
