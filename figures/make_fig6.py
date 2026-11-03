"""Fig 6: what noise actually feeds at forks, and whether Gumbel's winner is a sample (Exp 22, items 10-29).

Reads  results/exp22/exp22_mixtures_items10-29.jsonl
Writes figures/fig6.pdf, figures/fig6.png
Run from anywhere:  python figures/make_fig6.py

Every number in the caption has a claim-ledger row and is asserted here:
  fork rate, fed top >= 0.9 at forks, median fed top / second weight (C22.1-3), vanilla = pre-noise (C22.5),
  forks per arm (C22.7), flip rate minus sampling rate (C22.8), largest Gumbel bin deviation (C22.9),
  Dirichlet below sampling in every bin (C22.10), fork shares by fed top weight band (C22.11),
  median residual (C22.12).

Draft caption:
  Noise turns real forks into lopsided blends, and Gumbel's winner is a sample. A fork is a
  thinking step where the pre-noise runner-up holds at least 25% of the weight (9.2% of steps for
  vanilla, 13.0% for Gumbel, 13.3% for Dirichlet). (a) Share of forks by the top weight of the
  blend actually fed. Vanilla feeds the pre-noise weights unchanged, so it shows what the model
  wanted: every fork is balanced (top weight at most 0.75, by the fork rule). Gumbel feeds a near
  one-token blend (top weight at least 0.9) at 47% of forks and Dirichlet at 37%. The median fed
  blend has a residual (weight off the top token) of 0.12 for Gumbel and 0.17 for Dirichlet,
  against 0.44 before noise. (b) How often noise makes a token other than the pre-noise top one
  the winner, minus how often ordinary sampling would (1 minus the pre-noise top weight), with
  forks binned by the sampling rate. Gumbel matches sampling: 0.00 [-0.01, 0.02] overall and
  within 0.04 in every bin with at least 100 forks. Dirichlet flips less often than sampling in
  every bin, -0.03 [-0.04, -0.02] overall (dashed line). Vanilla never flips (not shown). Error
  bars: 95% bootstrap CIs over problems. 3,774 (vanilla), 5,761 (Gumbel) and 4,978 (Dirichlet)
  forks from 20 GSM8K problems; each arm is its own run, so the forks differ.
"""
import numpy as np

from data import boot_stat, rows
from style import COLORS, FULL, LABELS, plain_ticks, plt, save

ARMS = [(None, "vanilla"), ("gumbel", "gumbel"), ("dirichlet", "dirichlet")]
LEDGER = {  # fork rate, fed top >= 0.9 at forks, median (top, second) C22.1-3; forks C22.7
    "vanilla": (0.092, 0.00, (0.56, 0.33), 3774), "gumbel": (0.130, 0.47, (0.88, 0.10), 5761),
    "dirichlet": (0.133, 0.37, (0.83, 0.16), 4978)}
EDGES = [0.25 + 0.05 * j for j in range(11)]  # sampling-rate bins, as verify_claims C22.9-10
NEAR_ONE = 0.9

e22 = rows("exp22/exp22_mixtures_items10-29.jsonl")
forks = {}
for nz, key in ARMS:
    rs = [r for r in e22 if r["noise"] == nz]
    fk = [{"item": r["item"], "pre1": r["pre1"][i], "fed1": r["fed1"][i], "fed2": r["fed2"][i], "flip": r["flip"][i],
           "e": 1 - r["pre1"][i]} for r in rs for i in range(len(r["pre1"])) if r["pre2"][i] >= 0.25]
    rate, near, med, n = LEDGER[key]
    assert len(fk) == n, key
    assert abs(len(fk) / sum(len(r["pre1"]) for r in rs) - rate) <= 0.0005, key
    assert round(np.mean([f["fed1"] >= NEAR_ONE for f in fk]), 2) == near, key
    assert (round(float(np.median([f["fed1"] for f in fk])), 2), round(float(np.median([f["fed2"] for f in fk])), 2)) == med, key
    if nz is None:
        assert all(r["fed1"] == r["pre1"] for r in rs) and not any(f["flip"] for f in fk)  # C22.5
    forks[key] = fk
assert len({f["item"] for k in forks for f in forks[k]}) == 20  # C22.7

dflip = lambda fs: np.mean([f["flip"] for f in fs]) - np.mean([f["e"] for f in fs])
for key, (m, ci) in (("gumbel", (0.00, (-0.01, 0.02))), ("dirichlet", (-0.03, (-0.04, -0.02)))):  # C22.8
    assert round(dflip(forks[key]), 2) + 0.0 == m and tuple(round(x, 2) for x in boot_stat(forks[key], dflip)) == ci, key


def binned(fs):
    out = []
    for j, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
        b = [f for f in fs if lo <= f["e"] < hi or (j == 9 and f["e"] == hi)]
        out.append((np.mean([f["e"] for f in b]), dflip(b), *boot_stat(b, dflip), len(b)))
    return np.array(out)  # cols: x, observed - sampling, its 95% CI lo, hi, n forks


B = {k: binned(forks[k]) for k in ("gumbel", "dirichlet")}
g = B["gumbel"]
assert round(np.abs(g[g[:, 4] >= 100, 1]).max(), 2) == 0.04  # C22.9
assert (B["dirichlet"][:, 1] < 0).all()  # C22.10

BANDS = [("balanced\n(top $\\leq$ 0.75)", lambda t: t <= 0.75), ("lopsided\n(0.75--0.9)", lambda t: 0.75 < t < 0.9),
         ("near one token\n(top $\\geq$ 0.9)", lambda t: t >= NEAR_ONE)]
SHARES = {"vanilla": (100, 0, 0), "gumbel": (31, 22, 47), "dirichlet": (38, 25, 37)}  # C22.11, % of forks
RESID = {"vanilla": 0.44, "gumbel": 0.12, "dirichlet": 0.17}  # C22.12, median 1 - fed top
for key in SHARES:
    t = [f["fed1"] for f in forks[key]]
    assert tuple(round(100 * np.mean([c(x) for x in t])) for _, c in BANDS) == SHARES[key], key
    assert round(float(np.median([1 - x for x in t])), 2) == RESID[key], key

fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35), gridspec_kw={"width_ratios": [1, 1]})

# (a) share of forks per band of the fed top weight
ax = axes[0]
order = ("vanilla", "gumbel", "dirichlet")
w = 0.26
for j, key in enumerate(order):
    x = np.arange(len(BANDS)) + (j - 1) * w
    ax.bar(x, SHARES[key], w * 0.92, color=COLORS[key], label=LABELS[key], zorder=2)
    for xi, v in zip(x, SHARES[key]):
        ax.text(xi, v + 1.5, f"{v}\\%", ha="center", va="bottom", fontsize=6.3, color="#333333")
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels([b for b, _ in BANDS])
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("share of forks (\\%)")
ax.set_title("(a) top weight of the fed blend at forks", loc="left")
ax.legend(loc="upper right", borderaxespad=0.1)

# (b) observed minus sampling flip rate, per bin
ax = axes[1]
ax.axhline(0, color=COLORS["sampling"], lw=1.0, ls=(0, (3, 2)), zorder=1, label="same as sampling")
for key, dx, mk in (("gumbel", -0.006, "o"), ("dirichlet", 0.006, "s")):
    b = B[key]
    d, lo, hi = b[:, 1], b[:, 2], b[:, 3]
    ax.errorbar(b[:, 0] + dx, d, yerr=[d - lo, hi - d], fmt=mk, ms=2.8, color=COLORS[key], elinewidth=0.8,
                capsize=1.2, capthick=0.6, label=LABELS[key], zorder=3)
ax.axhline(dflip(forks["dirichlet"]), color=COLORS["dirichlet"], lw=0.7, ls=(0, (1.5, 1.5)), zorder=1,
           label="Dirichlet, all forks")
ax.set_xlim(0.245, 0.755)
ax.set_ylim(-0.15, 0.24)
ax.set_xticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15])
ax.set_xlabel("flip rate under sampling\n(1 $-$ pre-noise top weight)")
ax.set_ylabel("observed $-$ sampling flip rate")
ax.set_title("(b) is the winner a sample?", loc="left")
h, l = ax.get_legend_handles_labels()
ax.legend([h[i] for i in (2, 3, 0, 1)], [l[i] for i in (2, 3, 0, 1)], loc="upper left", ncol=2, borderaxespad=0.1,
          columnspacing=1.0)
plain_ticks(axes[0])
plain_ticks(axes[1])
plain_ticks(axes[1], "x")
save(fig, "fig6")
