"""Loaders and statistics shared by the figure scripts. Same definitions as verify_claims.py."""
import json
import random
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHAS = [i / 10 for i in range(11)]
STEP_SNAP = sum(abs((a > 0.5) - a) for a in ALPHAS) / len(ALPHAS)  # perfect step at 0.5 scores 1


def rows(path):
    with open(RESULTS / path) as f:
        return [json.loads(l) for l in f if l.strip()]


def unique(rs):  # drop forks seen in both the vanilla and the Gumbel trace
    seen, out = set(), []
    for r in rs:
        k = (r["item"], r["step"], r["A"], r["B"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def group(r):  # fork kind: nn / nw / ww
    num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
    return "nn" if all(num) else "nw" if any(num) else "ww"


def snap(r, layer):  # rescaled snap (2026-10-15): 0 = carried linearly, 1 = clean switch
    return sum(abs(c - a) for c, a in zip(r["c"][layer], ALPHAS)) / len(ALPHAS) / STEP_SNAP


def boot_mean(rs, values, n=2000):
    """95% CI of the mean of `values` (forks x k), resampling items. Same draws as verify_claims.boot().

    Returns (lo, hi), each an array of length k. Unrounded.
    """
    values = np.asarray(values, dtype=float)
    items = list(dict.fromkeys(r["item"] for r in rs))
    pos = {i: j for j, i in enumerate(items)}
    sums = np.zeros((len(items), values.shape[1]))
    counts = np.zeros(len(items))
    for r, v in zip(rs, values):
        sums[pos[r["item"]]] += v
        counts[pos[r["item"]]] += 1
    rng = random.Random(0)
    draws = np.array([[pos[i] for i in rng.choices(items, k=len(items))] for _ in range(n)])
    means = np.sort(sums[draws].sum(1) / counts[draws].sum(1)[:, None], axis=0)
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---- Exp 21 KV carry-over (step5_kv_carryover) ----
POS_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64)]  # forced-text positions, log-spaced


def followed_chosen(r):  # forks where the blend's next token followed the chosen token A (the runner-up lost)
    return r["cont"]["A"]["top_is_A"][0] and not r["cont"]["A"]["top_is_B"][0]


def readback(r, src, lo=1, hi=64):
    """Mean of control - blend over forced positions lo..hi of text src ("B" = runner-up's, "A" = own).

    Positions where A and B barely differ are None in the raw file and skipped. Returns nan if none left.
    """
    x = r["cont"][src]
    d = [k - b for b, k in zip(x["c"][lo:hi + 1], x["c_ctrl"][lo:hi + 1]) if b is not None]
    return sum(d) / len(d) if d else float("nan")


# ---- Exp 21 KV patching (step5_kv_patch) ----
BLOCKS = ["L0-3", "L4-7", "L8-11", "L12-15", "L16-19", "L20-23", "L24-27"]


def patch_rows():  # forks with patching results, deduplicated as neuralese_kv_patch.summarize
    out, seen = [], set()
    for r in rows("exp21/step5_kv_patch_items10-49.jsonl"):
        k = (r["item"], r["step"], r["A"], r["B"])
        if "remove" in r and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def share_closed(rs, name, keys):
    """Share of the blend-control gap closed by patching layers `keys` ("remove" or "insert"), summed over forks."""
    gained = sum(r["remove"][k] - r["c_blend"] if name == "remove" else r["c_ctrl"] - r["insert"][k] for r in rs for k in keys)
    return gained / sum(r["c_ctrl"] - r["c_blend"] for r in rs)


def boot_stat(rs, stat, n=2000):
    """95% CI of stat(rows), resampling problems. Same draws and percentiles as verify_claims.boot(). Unrounded."""
    by = {}
    for r in rs:
        by.setdefault(r["item"], []).append(r)
    rng, items = random.Random(0), list(by)
    v = sorted(x for x in (stat([r for i in rng.choices(items, k=len(items)) for r in by[i]]) for _ in range(n)) if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
