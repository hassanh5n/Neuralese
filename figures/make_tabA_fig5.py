"""Appendix tables for Fig 5 (Exp 21 KV patching, items 10-49): robustness of the shares.

Reads  results/exp21/step5_kv_patch_items10-49.jsonl
Writes figures/tabA1.tex  robustness: pooled share, forks with gap >= 0.1, leave-one-problem-out range, per-fork median
       figures/tabA2.tex  number vs number single layers: 95% CI and Bonferroni-corrected lower bound (56 tests)
       figures/tabA3.tex  number vs word blocks (gap too small to localise), shown for completeness
Every cell is recomputed independently by verify_robustness.py, which compares it with these files.
Run from anywhere:  python figures/make_tabA_fig5.py
"""
import random
from statistics import median

import numpy as np

from data import BLOCKS, boot_stat, group, patch_rows, share_closed
from style import FIG_DIR

MIN_GAP = 0.1  # forks with a smaller blend-control gap give unstable per-fork shares
N_TESTS = 28 * 2  # single layers x (remove, insert)
BONF_DRAWS, BONF_SEED = 20000, 1
ROWS_A1 = {"nn": ["L0-3", "L4-7", "L8-11", "L12-15", "L0", "L1", "L2", "L7", "L13"], "ww": ["L0-3", "L4-7", "L8-11", "L12-15"]}
SINGLES_A2 = ["L0", "L1", "L2", "L7", "L13"]
NAMES = ("remove", "insert")
TITLE = {"nn": "Number vs number", "ww": "Word vs word", "nw": "Number vs word"}

gap = lambda r: r["c_ctrl"] - r["c_blend"]
gain = lambda r, name, k: r["remove"][k] - r["c_blend"] if name == "remove" else r["c_ctrl"] - r["insert"][k]


def num(x):  # 2 decimals, LaTeX minus
    s = f"{x:.2f}"
    if s == "-0.00":
        s = "0.00"
    return "$-$" + s[1:] if s.startswith("-") else s


def label(k):
    return ("layers " if "-" in k else "layer ") + k[1:].replace("-", "--")


pk = patch_rows()
kinds = {g: [r for r in pk if group(r) == g] for g in ("nn", "nw", "ww")}


def robustness(rs, name, k):
    items = list(dict.fromkeys(r["item"] for r in rs))
    big = [r for r in rs if gap(r) >= MIN_GAP]
    loo = [share_closed([r for r in rs if r["item"] != i], name, [k]) for i in items]
    return (share_closed(rs, name, [k]), share_closed(big, name, [k]), min(loo), max(loo),
            median(gain(r, name, k) / gap(r) for r in big))


def bonferroni_lower(rs, keys, n=BONF_DRAWS, seed=BONF_SEED):
    """Lower bound of the (1 - 0.05/N_TESTS) two-sided percentile interval, resampling problems."""
    items = list(dict.fromkeys(r["item"] for r in rs))
    pos = {i: j for j, i in enumerate(items)}
    G = np.zeros(len(items))
    S = {(k, nm): np.zeros(len(items)) for k in keys for nm in NAMES}
    for r in rs:
        G[pos[r["item"]]] += gap(r)
        for k in keys:
            for nm in NAMES:
                S[k, nm][pos[r["item"]]] += gain(r, nm, k)
    rng = random.Random(seed)
    draws = np.array([[pos[i] for i in rng.choices(items, k=len(items))] for _ in range(n)])
    den = G[draws].sum(1)
    idx = int(0.05 / N_TESTS / 2 * n)
    return {key: np.sort(v[draws].sum(1) / den)[idx] for key, v in S.items()}


def table(colspec, header, body, caption, lab):
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{lab}}}",
                      rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular}", r"\end{table}", ""])


# Table A1
body = []
for g, keys in ROWS_A1.items():
    rs = kinds[g]
    nbig = sum(gap(r) >= MIN_GAP for r in rs)
    if body:
        body.append(r"\addlinespace")
    body.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{TITLE[g]}: {len(rs)} forks, {nbig} with gap $\geq$ {MIN_GAP}}}}} \\")
    for k in keys:
        cells = []
        for nm in NAMES:
            p, t, lo, hi, md = robustness(rs, nm, k)
            cells += [num(p), num(t), f"{num(lo)}--{num(hi)}", num(md)]
        body.append(f"{label(k)} & " + " & ".join(cells) + r" \\")
hdr = [r" & \multicolumn{4}{c}{Remove} & \multicolumn{4}{c}{Insert} \\",
       r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
       r"Patched & Pooled & Gap $\geq$ 0.1 & Leave one out & Median & Pooled & Gap $\geq$ 0.1 & Leave one out & Median \\"]
cap = (r"Robustness of the patching shares in Figure~5. Pooled: share of the summed blend--control gap closed "
       r"(as plotted). Gap $\geq$ 0.1: the same, keeping only forks whose gap is at least 0.1. Leave one out: range of "
       r"the pooled share when each problem is dropped in turn. Median: median of per-fork shares over forks with gap "
       r"$\geq$ 0.1. Exp~21, GSM8K items 10--49, 50/50 blends, forks where the model followed the chosen token.")
(FIG_DIR / "tabA1.tex").write_text(table("l" + "c" * 8, hdr, body, cap, "tab:fig5-robust"))

# Table A2
rs = kinds["nn"]
bl = bonferroni_lower(rs, SINGLES_A2)
body = []
for k in SINGLES_A2:
    cells = []
    for nm in NAMES:
        lo, hi = boot_stat(rs, lambda x, k=k, nm=nm: share_closed(x, nm, [k]))
        cells += [num(share_closed(rs, nm, [k])), f"[{num(lo)}, {num(hi)}]", num(bl[k, nm])]
    body.append(f"{label(k)} & " + " & ".join(cells) + r" \\")
hdr = [r" & \multicolumn{3}{c}{Remove} & \multicolumn{3}{c}{Insert} \\",
       r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
       r"Patched & Share & 95\% CI & Corrected lower & Share & 95\% CI & Corrected lower \\"]
cap = (r"Number vs number single layers that stand out in Figure~5c. 95\% CI: bootstrap over problems "
       rf"(2,000 resamples). Corrected lower: lower bound of the Bonferroni-corrected interval for all {N_TESTS} "
       rf"single-layer tests (28 layers $\times$ remove/insert; {BONF_DRAWS:,} resamples). Every bound is above 0. "
       r"The single-layer analysis was not pre-registered.")
(FIG_DIR / "tabA2.tex").write_text(table("l" + "c" * 6, hdr, body, cap, "tab:fig5-singles"))

# Table A3
rs = kinds["nw"]
body = []
for k in BLOCKS:
    cells = []
    for nm in NAMES:
        lo, hi = boot_stat(rs, lambda x, k=k, nm=nm: share_closed(x, nm, [k]))
        cells += [num(share_closed(rs, nm, [k])), f"[{num(lo)}, {num(hi)}]"]
    body.append(f"{label(k)} & " + " & ".join(cells) + r" \\")
hdr = [r" & \multicolumn{2}{c}{Remove} & \multicolumn{2}{c}{Insert} \\",
       r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}", r"Patched & Share & 95\% CI & Share & 95\% CI \\"]
mg = np.mean([gap(r) for r in rs])
cap = (rf"Number vs word forks ({len(rs)} forks from {len({r['item'] for r in rs})} problems), not shown in "
       rf"Figure~5. The mean blend--control gap is only {mg:.2f}, so the shares are unstable (wide CIs, values above 1 "
       r"and below 0) and cannot localise the runner-up. 95\% CI: bootstrap over problems.")
(FIG_DIR / "tabA3.tex").write_text(table("lcccc", hdr, body, cap, "tab:fig5-nw"))
print("wrote figures/tabA1.tex, tabA2.tex, tabA3.tex")
