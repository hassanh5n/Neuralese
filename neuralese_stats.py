"""Part A statistics: headline numbers of Exp 17 / 18 / 20 with problem-level bootstrap CIs.

Forks from the same GSM8K problem are not independent, so the CIs resample problems, not forks.
Reads every step4_alpha_sweep*.jsonl, step5_kv_carryover*.jsonl and step5_kv_patch*.jsonl found under
/kaggle (inputs and working dir) or the current folder; a fork seen twice (same item, step, A, B) counts once.
Reports items 10-49 (the replication, checked against the predictions written in the log before running)
and items 0-49 pooled, if the items 0-9 files are attached too.

Kaggle: after the three runs,  %run neuralese_stats.py   (CPU only, no model)
"""
import glob
import json
import os
import random

ROOT = "/kaggle" if os.path.isdir("/kaggle") else "."
N_BOOT = 2000
EARLY, LATE = ["L0-3", "L4-7"], ["L12-15", "L16-19", "L20-23", "L24-27"]


def load(prefix):
    rows, seen = [], set()
    for path in sorted(glob.glob(f"{ROOT}/**/{prefix}*.jsonl", recursive=True)):
        for r in map(json.loads, open(path)):
            if (r["item"], r["step"], r["A"], r["B"]) not in seen:
                seen.add((r["item"], r["step"], r["A"], r["B"]))
                rows.append(r)
    print(f"{prefix}: {len(rows)} unique forks from {len(glob.glob(f'{ROOT}/**/{prefix}*.jsonl', recursive=True))} file(s)")
    return rows


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def group(r):
    num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
    return "num-num" if all(num) else "num-word" if any(num) else "word-word"


def ci(rows, stat):
    """(estimate, low, high): 95% bootstrap CI resampling problems (all forks of a drawn problem come along)."""
    by = {}
    for r in rows:
        by.setdefault(r["item"], []).append(r)
    rng, items = random.Random(0), list(by)
    boot = sorted(x for x in (stat([r for i in rng.choices(items, k=len(items)) for r in by[i]])
                              for _ in range(N_BOOT)) if x == x)  # x == x drops nan
    return stat(rows), boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]


def leftover(r, src):
    """Per-fork mean of control - blend over positions 1-64 with src's text forced (Exp 18)."""
    x = r["cont"][src]
    return mean([k - b for b, k in zip(x["c"][1:], x["c_ctrl"][1:]) if b is not None])


def frac(rs, name, sets):
    """Fraction of the blend-vs-control gap closed by patching the given blocks together (Exp 20)."""
    gained = sum(r["remove"][k] - r["c_blend"] if name == "remove" else r["c_ctrl"] - r["insert"][k]
                 for r in rs for k in sets)
    return gained / sum(r["c_ctrl"] - r["c_blend"] for r in rs) if rs else float("nan")


def report(label, sweep, carry, patch):
    print(f"\n===== {label} =====")
    fmt = "{:+.2f} [{:+.2f}, {:+.2f}]".format
    res = {}
    for g in ("num-num", "num-word", "word-word"):
        s = [r for r in sweep if group(r) == g]
        c = [r for r in carry if group(r) == g and r["cont"]["A"]["top_is_A"][0] and not r["cont"]["A"]["top_is_B"][0]]
        c = [dict(r, lB=leftover(r, "B"), lA=leftover(r, "A")) for r in c]
        c = [r for r in c if r["lB"] == r["lB"]]
        p = [r for r in patch if group(r) == g and "remove" in r]
        res[g] = {
            "snap": ci(s, lambda rs: mean([r["snap"][-2] for r in rs])),
            # 50/50 input: next token == pure A's or pure B's (next_top[10] / next_top[0]); skip forks where those agree
            "follow": ci(s, lambda rs: mean([r["next_top"][5] in (r["next_top"][0], r["next_top"][10])
                                             for r in rs if r["next_top"][0] != r["next_top"][10]])),
            "lB": ci(c, lambda rs: mean([r["lB"] for r in rs])),
            "lA": ci(c, lambda rs: mean([r["lA"] for r in rs])),
            **{f"{n}_{w}": ci(p, lambda rs, n=n, sets=sets: frac(rs, n, sets))
               for n in ("remove", "insert") for w, sets in (("early", EARLY), ("late", LATE))},
            **{f"{n}_diff": ci(p, lambda rs, n=n: frac(rs, n, EARLY) - frac(rs, n, LATE)) for n in ("remove", "insert")}}
        print(f"\n{g}: Exp 17 {len(s)} forks / {len({r['item'] for r in s})} problems; "
              f"Exp 18 {len(c)} followed-A forks; Exp 20 {len(p)} followed-A forks")
        for k, name in (("snap", "Exp 17 final-layer snap"), ("follow", "Exp 17 50/50 next token follows A or B"),
                        ("lB", "Exp 18 control - blend, B's text"), ("lA", "Exp 18 control - blend, A's text"),
                        ("remove_early", "Exp 20 remove, layers 0-7"), ("insert_early", "Exp 20 insert, layers 0-7"),
                        ("remove_late", "Exp 20 remove, layers 12-27"), ("insert_late", "Exp 20 insert, layers 12-27"),
                        ("remove_diff", "Exp 20 remove, 0-7 minus 12-27"), ("insert_diff", "Exp 20 insert, 0-7 minus 12-27")):
            print(f"  {name:38s} {fmt(*res[g][k])}")
    return res


def check(res):
    """Predictions from the log (written before the run), on items 10-49."""
    nn, ww = res["num-num"], res["word-word"]
    preds = [("P1 num-num final snap >= 0.6", nn["snap"][0] >= 0.6),
             ("P2 num-num 50/50 follows A or B >= 85%", nn["follow"][0] >= 0.85),
             ("P3 B's text: control - blend CI above 0 (num-num, word-word)", nn["lB"][1] > 0 and ww["lB"][1] > 0),
             ("P4 A's text: |control - blend| < 0.1 (num-num, word-word)", abs(nn["lA"][0]) < 0.1 and abs(ww["lA"][0]) < 0.1),
             ("P5 layers 0-7 close >= 50% of gap (remove, insert; num-num, word-word)",
              all(g[k][0] >= 0.5 for g in (nn, ww) for k in ("remove_early", "insert_early"))),
             ("P6 layers 0-7 minus 12-27 CI above 0 (remove, insert; num-num, word-word)",
              all(g[k][1] > 0 for g in (nn, ww) for k in ("remove_diff", "insert_diff")))]
    print("\n===== Predictions (items 10-49) =====")
    for name, ok in preds:
        print(f"  {'met    ' if ok else 'NOT met'}  {name}")


def main():
    assert ci([{"item": 0, "x": 1.0}, {"item": 1, "x": 3.0}], lambda rs: mean([r["x"] for r in rs]))[0] == 2.0
    data = [load(p) for p in ("step4_alpha_sweep", "step5_kv_carryover", "step5_kv_patch")]
    check(report("items 10-49 (replication)", *[[r for r in d if r["item"] >= 10] for d in data]))
    if any(r["item"] < 10 for d in data for r in d):
        report("items 0-49 pooled", *data)


if __name__ == "__main__":
    main()
