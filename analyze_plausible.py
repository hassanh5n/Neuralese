"""Pre-registered analysis of the plausible-token control (docs/exp24-plausible-control-prereg.md).

Usage:  python analyze_plausible.py results/exp24/exp24_plausible_items50-149_kl_only.jsonl
Standard library only. Same statistics as verify_claims.py: matched-step KL, median per-problem ratio,
95% percentile bootstrap over problems (2,000 resamples, seed 0), exact two-sided sign test.
"""
import json
import random
import sys
from math import comb
from statistics import mean, median


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def boot_median(vals, n=2000):
    rng = random.Random(0)
    v = sorted(median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def matched(r, a, b, keep=None):
    """Mean KL of arms a and b over steps before either leaves sampling (optionally only steps where keep(t))."""
    fa, fb = r["flip"][a], r["flip"][b]
    end = min(x for x in (fa, fb, len(r["kl"][a]), len(r["kl"][b])) if x is not None)
    steps = [t for t in range(end) if keep is None or keep(r, t)]
    if not steps:
        return None
    return mean(r["kl"][a][t] for t in steps), mean(r["kl"][b][t] for t in steps)


def compare(rows, a, b, keep=None, label=""):
    pairs = [m for m in (matched(r, a, b, keep) for r in rows) if m and m[1] > 0]
    ratios = [x / y for x, y in pairs]
    more = sum(x > y for x, y in pairs)
    lo, hi = boot_median(ratios)
    print(f"{label}{a} vs {b}: {len(pairs)} problems; median KL ratio {median(ratios):.2f} [{lo:.2f}, {hi:.2f}]; "
          f"{a} moves predictions more on {more}/{len(pairs)} (sign test p = {sign_p(more, len(pairs) - more):.2g})")
    return median(ratios), lo, hi


def order(rows, a, b):
    f = [(r["flip"][a], r["flip"][b]) for r in rows if r["flip"][a] is not None and r["flip"][b] is not None]
    first = (sum(x < y for x, y in f), sum(x > y for x, y in f), sum(x == y for x, y in f))
    print(f"first to leave sampling, {a} / {b} / same step: {first[0]} / {first[1]} / {first[2]} "
          f"(sign test p = {sign_p(first[0], first[1]):.2g})")


rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{len(rows)} problems, mode {rows[0]['mode']}")
rep = [r["reproduces_exp23_flips"] for r in rows if "reproduces_exp23_flips" in r]
if rep:
    print(f"reproduce the coupled ablation's soft and rand first-divergence steps: {sum(rep)}/{len(rep)}")
print("\nPRIMARY (H1): real residual vs plausible token, matched steps")
m, lo, hi = compare(rows, "soft", "plaus")
print("\nSECONDARY")
compare(rows, "plaus", "rand", label="(H2) ")
compare(rows, "soft", "rand", label="(replication) ")
ok = lambda r, t: t < len(r["stats"]["q_c"]) and r["stats"]["q_c"][t] >= 0.05
compare(rows, "soft", "plaus", keep=ok, label="(H3, steps where the plausible token has >= 0.05) ")
order(rows, "soft", "plaus")
if rows[0]["mode"] == "full":
    for a in ("soft", "rand", "plaus"):
        x = sum(r[a]["correct"] and not r["hard"]["correct"] for r in rows)
        y = sum(r["hard"]["correct"] and not r[a]["correct"] for r in rows)
        print(f"accuracy {a} {sum(r[a]['correct'] for r in rows)} vs hard {sum(r['hard']['correct'] for r in rows)}; "
              f"{a} only {x}, hard only {y}, McNemar p = {sign_p(x, y):.3f} (descriptive)")
verdict = ("H1 supported: the model reads the specific candidate, not just any plausible token" if lo > 1 else
           "H1 not supported: the real residual does not move predictions detectably more than a plausible token")
print(f"\nVERDICT (pre-registered rule: CI lower bound of the soft/plaus median ratio > 1): {verdict}")
