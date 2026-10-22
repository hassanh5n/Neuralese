"""Recompute every number in docs/neuralese-paper-log.md (claim ledger) from results/ and check it.

Run from the repo root:  python verify_claims.py      (CPU only, standard library, ~5 s)
Each claim id (e.g. C21.3) matches a row of the ledger. Exits non-zero if any value differs.
"""
import json
import random
import sys
from math import comb
from statistics import mean, median

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

print(f"\n{'ALL CLAIMS VERIFIED' if not fails else 'MISMATCHES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
