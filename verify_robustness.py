"""Robustness checks for Fig 5 (Exp 21 KV patching): recompute every cell of the appendix tables and the numbers
quoted from them in the text, independently of figures/, and compare. Also checks every cell of Table 1 (accuracy and
McNemar, Exp 16 and Exp 23; added 2026-10-26).

Run from the repo root:  python verify_robustness.py      (CPU only, standard library, ~20 s)
Claim ids (C21.32-C21.40) match rows of the ledger in docs/neuralese-paper-log.md.
Checks figures/tab1.tex, tabA1.tex, tabA2.tex, tabA3.tex cell by cell; exits non-zero on any mismatch.
"""
import json
import random
import re
import sys
from statistics import median

R, F = "results/", "figures/"
BLOCKS = ["L0-3", "L4-7", "L8-11", "L12-15", "L16-19", "L20-23", "L24-27"]
NAMES = ("remove", "insert")
fails = []


def check(cid, expected, got):
    ok = expected == got
    print(f"{'ok  ' if ok else 'FAIL'} {cid:8s} expected {expected}  got {got}")
    if not ok:
        fails.append(cid)


def fmt(x):
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


def group(r):
    num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
    return "nn" if all(num) else "nw" if any(num) else "ww"


def share(rs, name, k):
    g = sum((r["remove"][k] - r["c_blend"]) if name == "remove" else (r["c_ctrl"] - r["insert"][k]) for r in rs)
    return g / sum(r["c_ctrl"] - r["c_blend"] for r in rs)


def by_item(rs):
    out = {}
    for r in rs:
        out.setdefault(r["item"], []).append(r)
    return out


def ci95(rs, name, k, n=2000):  # as verify_claims.boot: Random(0), items in first-seen order
    by = by_item(rs)
    rng, items = random.Random(0), list(by)
    v = sorted(share([r for i in rng.choices(items, k=len(items)) for r in by[i]], name, k) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def bonf_lower(rs, keys, n=20000, seed=1, tests=56):
    by = by_item(rs)
    items = list(by)
    den = {i: sum(r["c_ctrl"] - r["c_blend"] for r in by[i]) for i in items}
    num = {(k, nm): {i: sum((r["remove"][k] - r["c_blend"]) if nm == "remove" else (r["c_ctrl"] - r["insert"][k])
                            for r in by[i]) for i in items} for k in keys for nm in NAMES}
    rng = random.Random(seed)
    reps = {key: [] for key in num}
    for _ in range(n):
        pick = rng.choices(items, k=len(items))
        d = sum(den[i] for i in pick)
        for key, t in num.items():
            reps[key].append(sum(t[i] for i in pick) / d)
    idx = int(0.05 / tests / 2 * n)
    return {key: sorted(v)[idx] for key, v in reps.items()}


def tex_cells(path):  # numbers in data rows, in reading order
    out = []
    for line in open(F + path, encoding="utf-8"):
        if "&" in line and line.rstrip().endswith("\\\\") and not re.search(r"Patched|multicolumn", line):
            out += [m.replace("$-$", "-") for m in re.findall(r"(?:\$-\$)?\d+\.\d+", line)]
    return out


pk, seen = [], set()
for line in open(R + "exp21/step5_kv_patch_items10-49.jsonl"):
    if line.strip():
        r = json.loads(line)
        key = (r["item"], r["step"], r["A"], r["B"])
        if "remove" in r and key not in seen:
            seen.add(key)
            pk.append(r)
kind = {g: [r for r in pk if group(r) == g] for g in ("nn", "nw", "ww")}
gap = lambda r: r["c_ctrl"] - r["c_blend"]

# ---- Table A1: pooled, gap >= 0.1, leave one problem out, per-fork median ----
A1 = {"nn": ["L0-3", "L4-7", "L8-11", "L12-15", "L0", "L1", "L2", "L7", "L13"], "ww": ["L0-3", "L4-7", "L8-11", "L12-15"]}
cells, heads, shift, stats = [], [], 0.0, {}
for g, keys in A1.items():
    rs = kind[g]
    big = [r for r in rs if gap(r) >= 0.1]
    heads.append((len(rs), len(big)))
    for k in keys:
        for nm in NAMES:
            p = share(rs, nm, k)
            loo = [share([r for r in rs if r["item"] != i], nm, k) for i in by_item(rs)]
            md = median(((r["remove"][k] - r["c_blend"]) if nm == "remove" else (r["c_ctrl"] - r["insert"][k])) / gap(r)
                        for r in big)
            stats[g, k, nm] = (p, md)
            shift = max(shift, max(abs(x - p) for x in loo))
            cells += [fmt(p), fmt(share(big, nm, k)), fmt(min(loo)), fmt(max(loo)), fmt(md)]
            trim_shift = max(globals().get("trim_shift", 0.0), abs(share(big, nm, k) - p))
check("C21.32a", cells, tex_cells("tabA1.tex"))
txt = open(F + "tabA1.tex", encoding="utf-8").read()
check("C21.32b", [(96, 78), (139, 108)], heads)
check("C21.32c", True, all(f"{a} forks, {b} with gap" in txt for a, b in heads))
check("C21.33", "0.04", fmt(shift))
check("C21.40", "0.03", fmt(trim_shift))  # largest change of a Table A1 pooled share when forks with gap < 0.1 are dropped  # largest leave-one-problem-out change of any pooled share in Table A1
check("C21.34", (("0.38", "0.12"), ("0.16", "0.02")),  # num-num layer 0: pooled vs per-fork median, remove / insert
      tuple((fmt(stats["nn", "L0", nm][0]), fmt(stats["nn", "L0", nm][1])) for nm in NAMES))

# ---- Table A2: num-num single layers, 95% CI and Bonferroni lower bound ----
S2 = ["L0", "L1", "L2", "L7", "L13"]
bl = bonf_lower(kind["nn"], S2)
cells = []
for k in S2:
    for nm in NAMES:
        lo, hi = ci95(kind["nn"], nm, k)
        cells += [fmt(share(kind["nn"], nm, k)), fmt(lo), fmt(hi), fmt(bl[k, nm])]
check("C21.35a", cells, tex_cells("tabA2.tex"))
check("C21.35b", True, all(v > 0 for v in bl.values()))  # every corrected lower bound above 0
check("C21.35c", ("0.01", "L1/L2 insert"), (fmt(min(bl.values())), "L1/L2 insert" if min(bl, key=bl.get)[0] in ("L1", "L2")
                                              and min(bl, key=bl.get)[1] == "insert" else str(min(bl, key=bl.get))))

# ---- Table A3: num-word blocks ----
cells = []
for k in BLOCKS:
    for nm in NAMES:
        lo, hi = ci95(kind["nw"], nm, k)
        cells += [fmt(share(kind["nw"], nm, k)), fmt(lo), fmt(hi)]
check("C21.36a", cells, tex_cells("tabA3.tex"))
mg = sum(gap(r) for r in kind["nw"]) / len(kind["nw"])
check("C21.36b", (34, 21, "0.11"), (len(kind["nw"]), len({r["item"] for r in kind["nw"]}), fmt(mg)))
check("C21.36c", True, f"({len(kind['nw'])} forks from 21 problems)" in open(F + "tabA3.tex").read() and f"only {fmt(mg)}" in open(F + "tabA3.tex").read())

# ---- concentration of the gap (text) ----
conc = []
for g in ("nn", "ww"):
    gs = sorted(gap(r) for r in kind[g])
    conc.append((fmt(sum(gs[-10:]) / sum(gs)), sum(x <= 0 for x in gs)))
check("C21.37", [("0.27", 6), ("0.24", 19)], conc)  # top-10 forks' share of total gap; forks with gap <= 0

# ---- Fig A2: spread of the num-num layer 0 remove share per fork (gap >= 0.1) ----
from statistics import quantiles  # noqa: E402
v = [(r["remove"]["L0"] - r["c_blend"]) / gap(r) for r in kind["nn"] if gap(r) >= 0.1]
check("C21.39", ("0.01", "0.12", "0.68"), tuple(fmt(q) for q in quantiles(v, n=4, method="inclusive")))

# ---- Fig A1: unbinned readback, positions 1-16: runner-up's text means above 0 (num-num, word-word) ----
ca, seen = [], set()
for line in open(R + "exp21/step5_kv_carryover_items10-49.jsonl"):
    if line.strip():
        r = json.loads(line)
        key = (r["item"], r["step"], r["A"], r["B"])
        if key not in seen:
            seen.add(key)
            ca.append(r)


def lb(r, src, lo, hi):
    x = r["cont"][src]
    d = [k - b for b, k in zip(x["c"][lo:hi + 1], x["c_ctrl"][lo:hi + 1]) if b is not None]
    return sum(d) / len(d) if d else None


pos_ok = True
for g in ("nn", "ww"):
    c = [r for r in ca if group(r) == g and r["cont"]["A"]["top_is_A"][0] and not r["cont"]["A"]["top_is_B"][0]
         and lb(r, "B", 1, 64) is not None]
    for p in range(1, 17):
        xs = [lb(r, "B", p, p) for r in c]
        xs = [x for x in xs if x is not None]
        pos_ok &= sum(xs) / len(xs) > 0
check("C21.38", True, pos_ok)

# ---- Table 1: accuracy (Wilson 95% CI) and McNemar vs sampling, Exp 16 and Exp 23 ----
from math import comb, sqrt  # noqa: E402


def wilson_ci(k, n, z=1.959964):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcn(x, y):  # x only, y only, exact two-sided p
    a, b = sum(x[i] and not y[i] for i in y), sum(y[i] and not x[i] for i in y)
    return [str(a), str(b), f"{min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)):.2f}"]


def acc(ok, lohi=None):
    lo, hi = lohi or wilson_ci(sum(ok), len(ok))
    return [str(len(ok)), str(sum(ok)), f"{sum(ok) / len(ok):.2f} [{lo:.2f}, {hi:.2f}]"]


def pdiff(x, y):  # Newcombe (1998) hybrid score CI for a paired difference, written out independently
    n = len(y)
    a = sum(x[i] and y[i] for i in y); b = sum(x[i] and not y[i] for i in y); c = sum(y[i] and not x[i] for i in y)
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n); l2, u2 = wilson_ci(a + c, n)
    ad_bc = a * d - b * c
    if ad_bc > 0:
        ad_bc = max(ad_bc - n / 2, 0)
    phi = ad_bc / sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (p1 - p2, p1 - p2 - sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2),
            p1 - p2 + sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))


def pboot(vals, n=2000):  # problem bootstrap, same draws as verify_claims.boot
    rng = random.Random(0)
    v = sorted(sum(vals[i] for i in rng.choices(range(len(vals)), k=len(vals))) / len(vals) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def fmt(est, lo, hi):
    g = lambda v: "0.00" if abs(v) < 0.005 else ("+" if v > 0 else "$-$") + f"{abs(v):.2f}"
    return f"{g(est)} [{g(lo)}, {g(hi)}]"


e16 = [json.loads(l) for l in open(R + "exp16/gsm8k_eval.jsonl") if l.strip()]
e23 = [json.loads(l) for l in open(R + "exp23/exp23_residual_ablation_items50-149.jsonl") if l.strip()]
seed0 = lambda nz: {r["item"]: r["correct"] for r in e16 if r["noise"] == nz and r["seed"] == 0}
pp = lambda nz: [sum(r["correct"] for r in e16 if r["noise"] == nz and r["item"] == i) / 2 for i in range(30)]
want = [acc([r["correct"] for r in e16 if r["noise"] == "discrete"], pboot(pp("discrete"))),
        acc([r["correct"] for r in e16 if r["noise"] is None]) + [fmt(*pdiff(seed0(None), seed0("discrete")))] + mcn(seed0(None), seed0("discrete")),
        acc([r["correct"] for r in e16 if r["noise"] == "gumbel"], pboot(pp("gumbel")))
        + [fmt(0.0, *pboot([g - q for g, q in zip(pp("gumbel"), pp("discrete"))]))] + mcn(seed0("gumbel"), seed0("discrete"))]
arm23 = lambda a: {r["item"]: r[a]["correct"] for r in e23}
for a in ("hard", "soft", "rand"):
    want.append(acc(list(arm23(a).values())) + ([] if a == "hard" else [fmt(*pdiff(arm23(a), arm23("hard")))] + mcn(arm23(a), arm23("hard"))))
got = []
for line in open(F + "tab1.tex", encoding="utf-8"):
    if "&" in line and line.rstrip().endswith("\\\\") and not re.search(r"Arm &|multicolumn", line):
        got.append([c.strip() for c in line.rstrip().rstrip("\\").split("&")[1:] if c.strip()])
check("Tab1a", want, got)
van = [r for r in e16 if r["noise"] is None]
vg = mcn(seed0(None), seed0("gumbel"))
cap = open(F + "tab1.tex", encoding="utf-8").read()
check("Tab1b", True, (f"Cold Stop on {sum(r['stop_reason'] == 'cold_stop' for r in van)} of {len(van)} runs" in cap
                      and f"all {sum(not r['answered'] for r in van)} runs without an answer ended there" in cap
                      and all(r["stop_reason"] == "cold_stop" for r in van if not r["answered"])
                      and f"it is {vg[0]} vs {vg[1]}, $p = {vg[2]}$" in cap))

print(f"\n{'ALL ROBUSTNESS CHECKS VERIFIED' if not fails else 'MISMATCHES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
