"""Table 1: accuracy, paired difference from sampling, and McNemar tests (accuracy comparison and coupled ablation).

Reads  results/exp16/gsm8k_eval.jsonl, results/exp23/exp23_residual_ablation_items50-149.jsonl
Writes figures/tab1.tex
Every cell is recomputed independently by verify_robustness.py, which compares it with this file.
Ledger rows: C16.1-3 (accuracy), C16.11 (Wilson CI, vanilla), C16.15 (problem-clustered CIs and the Gumbel - sampling
difference), C16.16 (vanilla - sampling), C16.12-13 (McNemar vs sampling, seed 0), C16.2, C16.8, C16.5, C23.1-3
(accuracy, McNemar), C23.9 (Wilson CIs), C23.17 (paired differences).
Run from anywhere:  python figures/make_tab1.py
"""
import random
from math import comb, sqrt
from statistics import mean

from data import rows
from style import FIG_DIR


def wilson(k, n, z=1.959964):  # Wilson score 95% CI for a proportion
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_p(a, b):  # exact two-sided McNemar (binomial on discordant pairs)
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar(x, y):  # x, y: {problem: correct}; returns (x only, y only, p)
    a, b = sum(x[i] and not y[i] for i in y), sum(y[i] and not x[i] for i in y)
    return a, b, sign_p(a, b)


def newcombe(x, y):  # paired difference acc(x) - acc(y), Newcombe (1998) hybrid score 95% CI
    a = sum(x[i] and y[i] for i in y)
    b, c = sum(x[i] and not y[i] for i in y), sum(y[i] and not x[i] for i in y)
    d, n = len(y) - a - b - c, len(y)
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = wilson(a + b, n), wilson(a + c, n)
    num, den = a * d - b * c, sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = max(num - n / 2, 0) if num > 0 else num
    phi = num / den if den else 0.0
    lo = p1 - p2 - sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = p1 - p2 + sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return p1 - p2, lo, hi


def cluster_ci(per_item, n=2000):  # 95% percentile bootstrap over problems (as verify_claims.boot)
    rng, items = random.Random(0), list(per_item)
    v = sorted(mean(per_item[i] for i in rng.choices(items, k=len(items))) for _ in range(n))
    return round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2)


def f2(x):
    return f"{x:.2f}".replace("-0.00", "0.00")


def ci(est, lo, hi, sign=False):
    s = (lambda v: ("+" if v > 0.004 else "") + f2(v)) if sign else f2
    return f"{s(est)} [{s(lo)}, {s(hi)}]".replace("-", "$-$")


# Accuracy comparison: items 0-29. Sampling and Gumbel at seeds 0 and 1 (runs clustered by problem), vanilla at seed 0.
e16 = rows("exp16/gsm8k_eval.jsonl")
arm16 = lambda nz: [r for r in e16 if r["noise"] == nz]
s0 = lambda nz: {r["item"]: r["correct"] for r in e16 if r["noise"] == nz and r["seed"] == 0}
per = lambda nz: {i: mean(r["correct"] for r in arm16(nz) if r["item"] == i) for i in range(30)}
rows16 = []
for name, nz in (("Sampling", "discrete"), ("Vanilla Soft Thinking", None), ("Gumbel Soft Thinking", "gumbel")):
    a = arm16(nz)
    k, n = sum(r["correct"] for r in a), len(a)
    lo, hi = wilson(k, n) if nz is None else cluster_ci(per(nz))
    cells = [str(n), str(k), ci(k / n, lo, hi)]
    if nz == "discrete":
        cells += [""] * 4
    else:
        if nz is None:
            diff = ci(*newcombe(s0(None), s0("discrete")), sign=True)
        else:
            dd = {i: per("gumbel")[i] - per("discrete")[i] for i in range(30)}
            diff = ci(mean(dd.values()), *cluster_ci(dd), sign=True)
        m = mcnemar(s0(nz), s0("discrete"))
        cells += [diff, str(m[0]), str(m[1]), f2(m[2])]
    rows16.append((name, cells))
van = arm16(None)
cold, noans = sum(r["stop_reason"] == "cold_stop" for r in van), sum(not r["answered"] for r in van)
vg = mcnemar(s0(None), s0("gumbel"))

# Coupled ablation: items 50-149, seed 0, three arms in lockstep with shared noise.
e23 = rows("exp23/exp23_residual_ablation_items50-149.jsonl")
c23 = lambda a: {r["item"]: r[a]["correct"] for r in e23}
rows23 = []
for name, a in (("Sampling", "hard"), ("Gumbel Soft Thinking", "soft"), ("Random-residual control", "rand")):
    k = sum(c23(a).values())
    cells = [str(len(e23)), str(k), ci(k / len(e23), *wilson(k, len(e23)))]
    if a == "hard":
        cells += [""] * 4
    else:
        m = mcnemar(c23(a), c23("hard"))
        cells += [ci(*newcombe(c23(a), c23("hard")), sign=True), str(m[0]), str(m[1]), f2(m[2])]
    rows23.append((name, cells))

# ledger values; stop if anything moved
assert [c for _, c in rows16] == [["60", "52", "0.87 [0.75, 0.97]", "", "", "", ""],
                                  ["30", "22", "0.73 [0.56, 0.86]", "$-$0.17 [$-$0.33, $-$0.01]", "0", "5", "0.06"],
                                  ["60", "52", "0.87 [0.73, 0.97]", "0.00 [$-$0.10, +0.10]", "1", "2", "1.00"]], rows16
assert [c for _, c in rows23] == [["100", "86", "0.86 [0.78, 0.91]", "", "", "", ""],
                                  ["100", "88", "0.88 [0.80, 0.93]", "+0.02 [$-$0.05, +0.09]", "6", "4", "0.75"],
                                  ["100", "82", "0.82 [0.73, 0.88]", "$-$0.04 [$-$0.11, +0.03]", "3", "7", "0.34"]], rows23
assert all(r["stop_reason"] == "cold_stop" for r in van if not r["answered"])  # C16.8
assert (cold, noans, vg[:2], f"{vg[2]:.2f}") == (14, 5, (1, 5), "0.22")  # C16.2, C16.5

body = [r"\multicolumn{8}{@{}l}{\emph{Accuracy comparison: GSM8K items 0--29}} \\"]
body += [f"{name} & " + " & ".join(c) + r" \\" for name, c in rows16]
body += [r"\addlinespace", r"\multicolumn{8}{@{}l}{\emph{Coupled ablation: GSM8K items 50--149, arms in lockstep}} \\"]
body += [f"{name} & " + " & ".join(c) + r" \\" for name, c in rows23]
hdr = [r" & & & & & \multicolumn{3}{c}{McNemar vs sampling} \\",
       r"\cmidrule(lr){6-8}",
       r"Arm & Runs & Correct & Accuracy [95\% CI] & Difference vs sampling [95\% CI] & Arm only & Sampling only & $p$ \\"]
cap = (r"Neither Gumbel Soft Thinking nor the random-residual control differs detectably from sampling in accuracy, "
       r"and the paired intervals still allow differences of several points either way. "
       r"Accuracy comparison: sampling and Gumbel Soft Thinking at seeds 0 and 1 on the same 30 problems, so their "
       r"60 runs are clustered by problem and their CIs are 95\% bootstrap intervals over problems (per seed: sampling "
       r"27 and 25 of 30, Gumbel 26 and 26); vanilla Soft Thinking has one seed and a Wilson interval. "
       r"Its difference from sampling is paired on seed 0 (Newcombe interval); Gumbel's is the mean per-problem "
       r"difference over both seeds, bootstrapped over problems. McNemar uses the seed 0 runs, the only seed vanilla has. "
       rf"Vanilla hit Cold Stop on {cold} of 30 runs, and all {noans} runs without an answer ended there; against Gumbel "
       rf"(seed 0) it is {vg[0]} vs {vg[1]}, $p = {vg[2]:.2f}$. "
       r"Coupled ablation: seed 0; the three arms share their noise, and the random-residual control moves the residual "
       r"to a random token. Accuracy has Wilson intervals and the paired difference a Newcombe hybrid-score interval. "
       r"Its problems are not used in any other experiment. McNemar: exact two-sided test on the problems where exactly "
       r"one of the pair is correct.")
tex = "\n".join([r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{cap}}}", r"\label{tab:accuracy}",
                 r"\begin{tabular}{@{}lrrccccc@{}}", r"\toprule", *hdr, r"\midrule", *body, r"\bottomrule",
                 r"\end{tabular}", r"\end{table*}", ""])
(FIG_DIR / "tab1.tex").write_text(tex)
print("wrote figures/tab1.tex")
