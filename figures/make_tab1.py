"""Table 1: accuracy and McNemar tests (Exp 16 and Exp 23).

Reads  results/exp16/gsm8k_eval.jsonl, results/exp23/exp23_residual_ablation_items50-149.jsonl
Writes figures/tab1.tex
Every cell is recomputed independently by verify_robustness.py, which compares it with this file.
Ledger rows: C16.1-3 (accuracy), C16.11 (Wilson CIs), C16.12-13 (McNemar vs sampling, seed 0), C16.2 (Cold Stops,
no answer), C16.8, C16.5 (vanilla vs Gumbel), C23.1-3 (accuracy, McNemar), C23.9 (Wilson CIs).
Run from anywhere:  python figures/make_tab1.py
"""
from math import comb, sqrt

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


def acc_cells(k, n):
    lo, hi = wilson(k, n)
    return [str(n), str(k), f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"]


# Exp 16: items 0-29. Sampling and Gumbel at seeds 0 and 1, vanilla at seed 0. McNemar on seed 0 pairs.
e16 = rows("exp16/gsm8k_eval.jsonl")
arm16 = lambda nz: [r for r in e16 if r["noise"] == nz]
s0 = lambda nz: {r["item"]: r["correct"] for r in e16 if r["noise"] == nz and r["seed"] == 0}
rows16 = []
for name, nz in (("Sampling", "discrete"), ("Vanilla Soft Thinking", None), ("Gumbel Soft Thinking", "gumbel")):
    a = arm16(nz)
    cells = acc_cells(sum(r["correct"] for r in a), len(a))
    cells += [""] * 3 if nz == "discrete" else [str(v) if i < 2 else f"{v:.2f}" for i, v in enumerate(mcnemar(s0(nz), s0("discrete")))]
    rows16.append((name, cells))
van = arm16(None)
cold, noans = sum(r["stop_reason"] == "cold_stop" for r in van), sum(not r["answered"] for r in van)
vg = mcnemar(s0(None), s0("gumbel"))

# Exp 23: items 50-149, seed 0, three arms in lockstep with shared noise.
e23 = rows("exp23/exp23_residual_ablation_items50-149.jsonl")
c23 = lambda a: {r["item"]: r[a]["correct"] for r in e23}
rows23 = []
for name, a in (("Sampling", "hard"), ("Gumbel Soft Thinking", "soft"), ("Random-residual control", "rand")):
    cells = acc_cells(sum(c23(a).values()), len(e23))
    cells += [""] * 3 if a == "hard" else [str(v) if i < 2 else f"{v:.2f}" for i, v in enumerate(mcnemar(c23(a), c23("hard")))]
    rows23.append((name, cells))

# ledger values (C16.1-3, C16.11-13, C23.1-3, C23.9); stop if anything moved
assert [c for _, c in rows16] == [["60", "52", "0.87 [0.76, 0.93]", "", "", ""],
                                  ["30", "22", "0.73 [0.56, 0.86]", "0", "5", "0.06"],
                                  ["60", "52", "0.87 [0.76, 0.93]", "1", "2", "1.00"]]
assert [c for _, c in rows23] == [["100", "86", "0.86 [0.78, 0.91]", "", "", ""],
                                  ["100", "88", "0.88 [0.80, 0.93]", "6", "4", "0.75"],
                                  ["100", "82", "0.82 [0.73, 0.88]", "3", "7", "0.34"]]
assert all(r["stop_reason"] == "cold_stop" for r in van if not r["answered"])  # C16.8
assert (cold, noans, vg[:2], f"{vg[2]:.2f}") == (14, 5, (1, 5), "0.22")  # C16.2, C16.5

body = [r"\multicolumn{7}{l}{\emph{Exp~16: GSM8K items 0--29}} \\"]
body += [f"{name} & " + " & ".join(c) + r" \\" for name, c in rows16]
body += [r"\addlinespace", r"\multicolumn{7}{l}{\emph{Exp~23: GSM8K items 50--149, arms in lockstep}} \\"]
body += [f"{name} & " + " & ".join(c) + r" \\" for name, c in rows23]
hdr = [r" & & & & \multicolumn{3}{c}{McNemar vs sampling} \\",
       r"\cmidrule(lr){5-7}",
       r"Arm & Runs & Correct & Accuracy [95\% CI] & Arm only & Sampling only & $p$ \\"]
cap = (r"Neither Gumbel Soft Thinking nor the random-residual control differs detectably from sampling in accuracy; "
       r"vanilla Soft Thinking trails it. Accuracy with Wilson 95\% CIs. "
       r"McNemar: exact two-sided test on the problems where exactly one of the pair is correct (arm only / sampling "
       r"only). Exp~16: sampling and Gumbel Soft Thinking at seeds 0 and 1, vanilla Soft Thinking at seed 0; McNemar "
       r"uses the seed 0 runs, the only seed vanilla has. Vanilla hit Cold Stop on "
       rf"{cold} of 30 runs, and all {noans} runs without an answer ended there; against Gumbel (seed 0) it is {vg[0]} vs {vg[1]}, "
       rf"$p = {vg[2]:.2f}$. Exp~23: seed 0; the three arms share their noise, and the random-residual control moves "
       r"the residual to a random token. Problems in Exp~23 are not used in any other experiment.")
tex = "\n".join([r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{cap}}}", r"\label{tab:accuracy}",
                 r"\begin{tabular}{lrrcccc}", r"\toprule", *hdr, r"\midrule", *body, r"\bottomrule",
                 r"\end{tabular}", r"\end{table}", ""])
(FIG_DIR / "tab1.tex").write_text(tex)
print("wrote figures/tab1.tex")
