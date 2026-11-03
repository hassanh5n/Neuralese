"""Fig A2 (appendix): per-fork patching shares behind the pooled bars of Figure 5 (Exp 21).

Reads  results/exp21/step5_kv_patch_items10-49.jsonl
Writes figures/figA2.pdf, figures/figA2.png
Numbers quoted in the caption: C21.32b (forks with gap >= 0.1), C21.34 (layer 0 pooled vs median),
C21.39 (layer 0 remove quartiles), asserted here.

Draft caption:
  Per-fork shares behind the pooled bars of Figure 5, for layer 0 and layers 0-3. Each dot is one
  fork with a blend-control gap of at least 0.1 (78 number vs number forks, 108 word vs word);
  filled = remove, open = insert. Black bar: median per fork. Diamond: pooled share, as plotted in
  Figure 5. For number vs number, layer 0 is uneven across forks: half of them have a remove
  share below 0.12 and a quarter are above 0.68, so the pooled 0.38 sits well above the median
  fork (0.12; insert: 0.16 vs 0.02). For layers 0-3 the median fork is close to the pooled share.
  Dots outside the axis range are drawn at its edge as triangles.
"""
import numpy as np

from data import group, patch_rows, share_closed
from style import COLORS, FULL, plain_ticks, plt, save

KEYS = [("L0", "remove"), ("L0", "insert"), ("L0-3", "remove"), ("L0-3", "insert")]
XLAB = ["layer 0\nremove", "layer 0\ninsert", "layers 0--3\nremove", "layers 0--3\ninsert"]
YLIM = (-0.5, 1.5)
gap = lambda r: r["c_ctrl"] - r["c_blend"]
gain = lambda r, nm, k: r["remove"][k] - r["c_blend"] if nm == "remove" else r["c_ctrl"] - r["insert"][k]

pk = patch_rows()
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
rng = np.random.default_rng(0)  # jitter only
for ax, g, n_big, title in ((axes[0], "nn", 78, "(a) number vs number"), (axes[1], "ww", 108, "(b) word vs word")):
    rs = [r for r in pk if group(r) == g]
    big = [r for r in rs if gap(r) >= 0.1]
    assert len(big) == n_big  # C21.32b
    for x, (k, nm) in enumerate(KEYS):
        v = np.array([gain(r, nm, k) / gap(r) for r in big])
        md, pooled = np.median(v), share_closed(rs, nm, [k])
        if g == "nn" and k == "L0":  # C21.34
            assert (round(pooled, 2), round(md, 2)) == {"remove": (0.38, 0.12), "insert": (0.16, 0.02)}[nm]
            if nm == "remove":  # C21.39
                assert tuple(round(q, 2) for q in np.percentile(v, [25, 50, 75])) == (0.01, 0.12, 0.68)
        jx = x + rng.uniform(-0.18, 0.18, len(v))
        inside = (v >= YLIM[0]) & (v <= YLIM[1])
        face = COLORS[g] if nm == "remove" else "white"
        ax.scatter(jx[inside], v[inside], s=6, facecolors=face, edgecolors=COLORS[g], linewidths=0.5, alpha=0.8, zorder=2)
        for edge, mk in ((YLIM[1], "^"), (YLIM[0], "v")):
            out = v > edge if mk == "^" else v < edge
            ax.scatter(jx[out], np.full(out.sum(), edge - 0.03 if mk == "^" else edge + 0.03), s=9, marker=mk,
                       facecolors=face, edgecolors=COLORS[g], linewidths=0.5, zorder=2)
        ax.plot([x - 0.26, x + 0.26], [md, md], color="black", lw=1.2, zorder=4)
        ax.scatter([x + 0.33], [pooled], marker="D", s=14, facecolors="white", edgecolors="black", linewidths=0.8, zorder=4)
    ax.axhline(0, color=COLORS["guide"], lw=0.6, zorder=1)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.55, len(KEYS) - 0.35)
    ax.set_title(title, loc="left")
axes[0].set_ylim(*YLIM)
axes[0].set_ylabel("share of the gap closed (per fork)")
for ax in axes:
    plain_ticks(ax)
from matplotlib.lines import Line2D  # noqa: E402
fig.legend(handles=[Line2D([], [], color="black", lw=1.2, label="median fork"),
                        Line2D([], [], ls="", marker="D", ms=3.5, mfc="white", mec="black", label="pooled (Figure 5)")],
           loc="outside lower center", ncol=2)
save(fig, "figA2")
