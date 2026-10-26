"""Recompute every number in docs/neuralese-paper-log.md (claim ledger) from results/ and check it.

Run from the repo root:  python verify_claims.py      (CPU only, standard library, ~5 s)
Each claim id (e.g. C21.3) matches a row of the ledger. Exits non-zero if any value differs.
"""
import json
import random
import sys
from math import comb
from statistics import mean, median
from math import exp, log, sqrt

R = "results/"
ALPHAS = [i / 10 for i in range(11)]
STEP_SNAP = sum(abs((a > 0.5) - a) for a in ALPHAS) / len(ALPHAS)  # perfect step at 0.5 scores 1
EARLY, LATE = ["L0-3", "L4-7"], ["L12-15", "L16-19", "L20-23", "L24-27"]
fails = []


def check(cid, expected, got, tol=0.005):
    ok = got == expected if isinstance(expected, (str, tuple, list)) else abs(got - expected) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {cid:7s} expected {expected}  got {got if not isinstance(got, float) else round(got, 4)}")
    if not ok:
        fails.append(cid)


def rows(path):
    return [json.loads(l) for l in open(R + path) if l.strip()]


def unique(rs):
    seen, out = set(), []
    for r in rs:
        k = (r["item"], r["step"], r["A"], r["B"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def sign_p(a, b):  # exact two-sided binomial (McNemar / sign test)
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(rs, stat, n=2000):  # 95% CI resampling problems (items), as neuralese_stats.py
    by = {}
    for r in rs:
        by.setdefault(r["item"], []).append(r)
    rng, items = random.Random(0), list(by)
    v = sorted(x for x in (stat([r for i in rng.choices(items, k=len(items)) for r in by[i]]) for _ in range(n)) if x == x)
    return round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2)


def group(r):
    num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
    return "nn" if all(num) else "nw" if any(num) else "ww"


def wilson(k, n, z=1.959964):  # Wilson score 95% CI for a proportion
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)


# ---------------- Exp 16: GSM8K accuracy, items 0-29 ----------------
e16 = rows("exp16/gsm8k_eval.jsonl")
arm = lambda n: [r for r in e16 if r["noise"] == n]
for cid, n, cor, tot, noans, cold in (("C16.1", "discrete", 52, 60, 2, 0), ("C16.2", None, 22, 30, 5, 14), ("C16.3", "gumbel", 52, 60, 2, 0)):
    a = arm(n)
    check(cid + "a", (cor, tot), (sum(r["correct"] for r in a), len(a)))
    check(cid + "b", noans, sum(not r["answered"] for r in a), 0)
    check(cid + "c", cold, sum(r["stop_reason"] == "cold_stop" for r in a), 0)
check("C16.4", 141, sum(r["answered"] for r in e16), 0)
by = lambda n, s: {r["item"]: r["correct"] for r in e16 if r["noise"] == n and r["seed"] == s}
v, g0 = by(None, 0), by("gumbel", 0)
check("C16.5", 0.219, sign_p(sum(g0[i] and not v[i] for i in v), sum(v[i] and not g0[i] for i in v)), 0.001)
gu = {i: sum(r["correct"] for r in arm("gumbel") if r["item"] == i) for i in range(30)}
di = {i: sum(r["correct"] for r in arm("discrete") if r["item"] == i) for i in range(30)}
check("C16.6", [12, 20, 21, 23], [i for i in range(30) if gu[i] != di[i]])
van = arm(None)
check("C16.7", 7, sum(r["correct"] and r["stop_reason"] == "cold_stop" for r in van), 0)  # loops, then recovers
na = sorted(r["item"] for r in van if not r["answered"])
check("C16.8", ([8, 19, 21, 22, 28], True), (na, all(r["stop_reason"] == "cold_stop" for r in van if not r["answered"])))
check("C16.9", [8, 19, 22, 28], [i for i in na if all(r["correct"] for r in e16 if r["noise"] is not None and r["item"] == i)])
check("C16.10", [3, 7], [i for i in range(30) if not any(r["correct"] for r in e16 if r["item"] == i)])
check("C16.11", ((0.76, 0.93), (0.56, 0.86)), (wilson(52, 60), wilson(22, 30)))  # discrete = Gumbel 52/60; vanilla 22/30

# ---------------- Exp 21: replication of the fork measurements, items 10-49 ----------------
sw = rows("exp21/step4_alpha_sweep_items10-49.jsonl")
check("C21.1", (694, 670), (len(sw), len(unique(sw))))
sw, ca, pa = unique(sw), unique(rows("exp21/step5_kv_carryover_items10-49.jsonl")), unique(rows("exp21/step5_kv_patch_items10-49.jsonl"))
snap = lambda r, L=-2: sum(abs(c - a) for c, a in zip(r["c"][L], ALPHAS)) / len(ALPHAS) / STEP_SNAP
check("C21.2", 0.0, max(abs(c - a) for r in sw for c, a in zip(r["c"][0], ALPHAS)), 1e-5)


def leftover(r, src):
    x = r["cont"][src]
    d = [k - b for b, k in zip(x["c"][1:], x["c_ctrl"][1:]) if b is not None]
    return mean(d) if d else float("nan")


def frac(rs, name, sets):
    gained = sum(r["remove"][k] - r["c_blend"] if name == "remove" else r["c_ctrl"] - r["insert"][k] for r in rs for k in sets)
    return gained / sum(r["c_ctrl"] - r["c_blend"] for r in rs)


expect = {  # group: n forks, snap L7/L14/L21/final, final CI, follow, B-text (est, CI), A-text, early rem/ins, late rem/ins
    "nn": (238, (0.37, 0.59, 0.76, 0.84), (0.83, 0.86), 0.95, (0.44, (0.35, 0.53)), -0.01, (0.77, 0.71), (0.12, 0.26)),
    "nw": (56, (0.70, 0.80, 0.90, 0.98), (0.95, 1.00), 0.98, (0.11, (0.03, 0.19)), 0.06, None, None),
    "ww": (376, (0.44, 0.54, 0.65, 0.77), (0.75, 0.79), 0.88, (0.33, (0.27, 0.40)), 0.02, (0.77, 0.91), (0.08, 0.14)),
}
for g, (n, snaps, sci, fol, (lb, lbci), la, early, late) in expect.items():
    s = [r for r in sw if group(r) == g]
    check(f"C21.3{g}", n, len(s), 0)
    check(f"C21.4{g}", snaps, tuple(round(mean(snap(r, L) for r in s), 2) for L in (7, 14, 21, 28)))
    check(f"C21.5{g}", sci, boot(s, lambda rs: mean([snap(r) for r in rs])))
    fl = lambda rs: mean([r["next_top"][5] in (r["next_top"][0], r["next_top"][10]) for r in rs if r["next_top"][0] != r["next_top"][10]])
    check(f"C21.6{g}", fol, fl(s))
    c = [r for r in ca if group(r) == g and r["cont"]["A"]["top_is_A"][0] and not r["cont"]["A"]["top_is_B"][0]]
    c = [dict(r, lB=leftover(r, "B"), lA=leftover(r, "A")) for r in c]
    c = [r for r in c if r["lB"] == r["lB"]]
    check(f"C21.7{g}", lb, mean(r["lB"] for r in c))
    check(f"C21.8{g}", lbci, boot(c, lambda rs: mean([r["lB"] for r in rs])))
    check(f"C21.9{g}", la, mean(r["lA"] for r in c))
    if early:
        p = [r for r in pa if group(r) == g and "remove" in r]
        check(f"C21.10{g}", early, (round(frac(p, "remove", EARLY), 2), round(frac(p, "insert", EARLY), 2)))
        check(f"C21.11{g}", late, (round(frac(p, "remove", LATE), 2), round(frac(p, "insert", LATE), 2)))
nn = [r for r in sw if group(r) == "nn"]
nt = [r["next_top"] for r in nn]
check("C21.12", (96, 80, 10, 52), (sum(t[5] == t[10] != t[0] for t in nt), sum(t[5] == t[0] != t[10] for t in nt),
                                   sum(t[5] not in (t[0], t[10]) and t[0] != t[10] for t in nt), sum(t[0] == t[10] for t in nt)))
check("C21.13", 0.28, mean(r["off"][28][5] for r in nn), 0.01)
check("C21.14", (0.85, 0.84), tuple(round(mean(snap(r) for r in nn if r["noise"] == nz), 2) for nz in (None, "gumbel")))
# Per-block / per-layer shares with fork-level CIs, reproducing neuralese_kv_patch.summarize exactly
# (its dedup, row order, one Random(0) shared across groups in the order num-num, num-word, word-word).
pk, seen = [], set()
for r in rows("exp21/step5_kv_patch_items10-49.jsonl"):
    if "remove" in r and (r["item"], r["step"], r["A"], r["B"]) not in seen:
        seen.add((r["item"], r["step"], r["A"], r["B"]))
        pk.append(r)
BLOCKS = ["L0-3", "L4-7", "L8-11", "L12-15", "L16-19", "L20-23", "L24-27"]
frac1 = lambda rs, name, k: frac(rs, name, [k])
rng_patch, fl_ci = random.Random(0), {}
for g in ("nn", "nw", "ww"):
    rs = [r for r in pk if group(r) == g]
    boots = [rng_patch.choices(rs, k=len(rs)) for _ in range(2000)]
    for k in BLOCKS + ["L0", "L7", "L13"]:
        for name in ("remove", "insert"):
            b = sorted(frac1(x, name, k) for x in boots)
            fl_ci[g, k, name] = (round(frac1(rs, name, k), 2), round(b[49], 2), round(b[1949], 2))
blk = lambda g, ks: tuple((fl_ci[g, k, "remove"][0], fl_ci[g, k, "insert"][0]) for k in ks)
check("C21.15", ((0.58, 0.55), (0.19, 0.16), (0.06, 0.09), (0.12, 0.17)), blk("nn", BLOCKS[:4]))
check("C21.16", True, all(abs(v) <= 0.04 for pair in blk("nn", BLOCKS[4:]) for v in pair))  # num-num L16-27 <= 0.04 each
check("C21.17", ((0.66, 0.75), (0.10, 0.16)), blk("ww", BLOCKS[:2]))
check("C21.18", True, all(abs(v) <= 0.07 for pair in blk("ww", BLOCKS[2:]) for v in pair))  # word-word rest <= 0.07
check("C21.19", ((0.38, 0.16), (0.13, 0.12), (0.08, 0.13)), blk("nn", ["L0", "L7", "L13"]))
check("C21.20", True, all(fl_ci["nn", k, n][1] > 0 for k in ("L0", "L7", "L13") for n in ("remove", "insert")))  # CIs exclude 0
nnp = [r for r in pk if group(r) == "nn"]
check("C21.21", (0.95, 1.06), tuple(round(sum(frac1(nnp, n, k) for k in BLOCKS), 2) for n in ("remove", "insert")))
# P6 margin: min over (num-num, word-word) x (remove, insert) of the problem-level CI lower bound of early - late
p6 = [boot([r for r in pa if group(r) == g and "remove" in r], lambda rs, n=n: frac(rs, n, EARLY) - frac(rs, n, LATE))[0]
      for g in ("nn", "ww") for n in ("remove", "insert")]
check("C21.22", 0.31, min(p6))

# ---------------- Exp 22: what blends noise feeds, items 10-29 ----------------
e22 = rows("exp22/exp22_mixtures_items10-29.jsonl")
for cid, nz, frate, allp, forkp, med, flip in (("C22.1", None, 0.092, 0.78, 0.00, (0.56, 0.33), 0.00),
                                               ("C22.2", "gumbel", 0.130, 0.85, 0.47, (0.88, 0.10), 0.45),
                                               ("C22.3", "dirichlet", 0.133, 0.83, 0.37, (0.83, 0.16), 0.42)):
    rs = [r for r in e22 if r["noise"] == nz]
    fk = [(r, i) for r in rs for i in range(len(r["pre1"])) if r["pre2"][i] >= 0.25]
    check(cid + "a", frate, len(fk) / sum(len(r["pre1"]) for r in rs), 0.0005)
    check(cid + "b", allp, sum(x >= 0.9 for r in rs for x in r["fed1"]) / sum(len(r["fed1"]) for r in rs))
    check(cid + "c", forkp, mean(r["fed1"][i] >= 0.9 for r, i in fk))
    check(cid + "d", med, (round(median(r["fed1"][i] for r, i in fk), 2), round(median(r["fed2"][i] for r, i in fk), 2)))
    check(cid + "e", flip, mean(r["flip"][i] for r, i in fk))
    if nz == "gumbel":
        check("C22.4", 0.45, 1 - mean(r["pre1"][i] for r, i in fk))  # Gumbel-max: flip rate = 1 - mean top weight
    if nz is None:
        check("C22.5", True, all(r["fed1"] == r["pre1"] and r["fed2"] == r["pre2"] for r in rs))

# Amendment closed form (two-candidate Gumbel, tau = 0.5): P(fed top >= 0.9) = 1 - sig(ln3 - d) + sig(-ln3 - d), d = ln(w1/w2)
sig = lambda x: 1 / (1 + exp(-x))
cform = lambda w1, w2: 1 - sig(0.5 * log(9) - log(w1 / w2)) + sig(-0.5 * log(9) - log(w1 / w2))
check("C22.6", (0.50, 0.60), (round(cform(0.5, 0.5), 2), round(cform(0.75, 0.25), 2)))

# ---------------- Exp 23: coupled residual ablation, items 50-149 ----------------
e23 = rows("exp23/exp23_residual_ablation_items50-149.jsonl")
check("C23.1", (100, 86, 88, 82), (len(e23), *(sum(r[a]["correct"] for r in e23) for a in ("hard", "soft", "rand"))))
for cid, a, only, honly, p, same in (("C23.2", "soft", 6, 4, 0.754, 84), ("C23.3", "rand", 3, 7, 0.344, 84)):
    x = sum(r[a]["correct"] and not r["hard"]["correct"] for r in e23)
    y = sum(r["hard"]["correct"] and not r[a]["correct"] for r in e23)
    check(cid + "a", (only, honly), (x, y))
    check(cid + "b", p, sign_p(x, y), 0.001)
    check(cid + "c", same, sum(r[a]["answer_num"] == r["hard"]["answer_num"] for r in e23), 0)
check("C23.4", (35.5, 66.0), (median(r["flip"]["soft"] for r in e23), median(r["flip"]["rand"] for r in e23)))
f = [(r["flip"]["soft"], r["flip"]["rand"]) for r in e23]
first = (sum(s < q for s, q in f), sum(s > q for s, q in f), sum(s == q for s, q in f))
check("C23.5", (59, 24, 17), first)
check("C23.6", 0.0002, sign_p(first[0], first[1]), 0.0001)
ms = [mean(r["kl"]["soft"][:min(r["flip"]["soft"], r["flip"]["rand"])]) for r in e23]
mr = [mean(r["kl"]["rand"][:min(r["flip"]["soft"], r["flip"]["rand"])]) for r in e23]
check("C23.7", (0.0077, 0.0014), (round(mean(ms), 4), round(mean(mr), 4)))
check("C23.8", 94, sum(s > q for s, q in zip(ms, mr)), 0)
check("C23.9", ((0.80, 0.93), (0.78, 0.91), (0.73, 0.88)), tuple(wilson(k, 100) for k in (88, 86, 82)))  # soft / hard / rand
# Embedding-norm check (neuralese_exp23_norms.py, Kaggle T4): is `rand` a fair control?
nj = json.load(open(R + "exp23/exp23_embedding_norms.json"))
pop, drw, rea = nj["population_real_token_rows"], nj["exp23_rand_draws"], nj["reasoning_tokens_sampled_cot"]
check("C23.10", (1.195, 1.195, 1.196, 1.202), (round(pop["median"], 3), round(drw["median"], 3),
                                              round(drw["usage_weighted_median"], 3), round(rea["median"], 3)))
check("C23.11", (0.994, 0.961), (round(drw["median"] / rea["median"], 3), round(drw["mean"] / rea["mean"], 3)))
check("C23.12", (0.897, 1.058), (round(drw["p10"], 3), round(rea["p10"], 3)))
check("C23.13", (8192, 9168, 5), (drw["n"], rea["n"], len(rea["think_steps_per_item"])), 0)

print(f"\n{'ALL CLAIMS VERIFIED' if not fails else 'MISMATCHES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
