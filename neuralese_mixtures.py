"""Exp 22: what blends does noise actually feed during real runs?

Exp 17-21 fed exact 50/50 blends (the most "superposed" case). This replays ordinary runs (same RNG use as
neuralese_r1.run, so the same traces as Exp 16/17) for vanilla, Gumbel and Dirichlet, and records at every
think step the pre-noise weights (what the model wanted) and the fed weights (what noise turned them into).
  fed top >= 0.9 = the step is effectively one token (like sampling)
  fork = pre-noise runner-up >= 25% (same rule as Exp 17)
  flip = the fed winner is not the pre-noise top candidate
Checks: vanilla fed == pre-noise at every step (run stops otherwise); Gumbel's winner is mathematically a sample
from the pre-noise weights (Gumbel-max), so its flip rate at forks should match 1 - mean pre-noise top weight.

Kaggle: load the code as for Part A, then  %run neuralese_mixtures.py   (~1.5-2 h)
"""
import json
import statistics
import urllib.request

import neuralese_r1
from neuralese_eval import DATA_URL, SUFFIX
from neuralese_r1 import OUT_DIR, load, make_inputs, run
from neuralese_stats import ci, mean
from neuralese_sweep import FORK_MIN_SECOND

ITEMS = range(10, 30)
CONDITIONS = [(None, 0), ("gumbel", 0), ("dirichlet", 0)]
OUT = f"{OUT_DIR}/exp22_mixtures_items10-29.jsonl"
PRE = []  # pre-noise weights of every concept_token call in the current run
_concept_token = neuralese_r1.concept_token


def spy(embed, logits, noise):
    PRE.append(_concept_token(embed, logits, None)[1][0].tolist())  # pre-noise call uses no RNG
    return _concept_token(embed, logits, noise)


def summarize(rows):
    fmt = "{:.2f} [{:.2f}, {:.2f}]".format
    res = {}
    for noise in dict.fromkeys(r["noise"] for r in rows):
        steps = [{"item": r["item"], "pre1": a, "pre2": b, "fed1": c, "fed2": d, "flip": e} for r in rows
                 if r["noise"] == noise for a, b, c, d, e in zip(r["pre1"], r["pre2"], r["fed1"], r["fed2"], r["flip"])]
        forks = [s for s in steps if s["pre2"] >= FORK_MIN_SECOND]
        res[noise] = {"all": ci(steps, lambda ss: mean([s["fed1"] >= 0.9 for s in ss])),
                      "fork": ci(forks, lambda ss: mean([s["fed1"] >= 0.9 for s in ss])),
                      "flip": mean([s["flip"] for s in forks]), "expected_flip": 1 - mean([s["pre1"] for s in forks])}
        print(f"\n{noise}: {len(steps)} think steps, {len(forks)} forks ({len(forks) / len(steps):.1%} of steps)")
        print(f"  all steps: fed top >= 0.9          {fmt(*res[noise]['all'])}")
        print(f"  forks:     fed top >= 0.9          {fmt(*res[noise]['fork'])}")
        print(f"  forks:     median fed top / runner-up  {statistics.median(s['fed1'] for s in forks):.2f} / "
              f"{statistics.median(s['fed2'] for s in forks):.2f}   (pre-noise: "
              f"{statistics.median(s['pre1'] for s in forks):.2f} / {statistics.median(s['pre2'] for s in forks):.2f})")
        print(f"  forks:     winner flipped by noise {res[noise]['flip']:.2f}  (sampling would flip {res[noise]['expected_flip']:.2f})")
    if "gumbel" in res:
        g = res["gumbel"]
        print(f"\nCHECK Gumbel flip rate matches sampling (|diff| < 0.1): {'ok' if abs(g['flip'] - g['expected_flip']) < 0.1 else 'FAILED'}")
        print("Predictions (written in the log before running):")
        for name, ok in (("P1 Gumbel: fed top >= 0.9 at >= 70% of forks", g["fork"][0] >= 0.7),
                         ("P2 Dirichlet: fed top >= 0.9 at >= 70% of forks", res.get("dirichlet", {"fork": [0]})["fork"][0] >= 0.7),
                         ("P3 Gumbel: fed top >= 0.9 at >= 90% of all steps", g["all"][0] >= 0.9)):
            print(f"  {'met    ' if ok else 'NOT met'}  {name}")


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    tok, model = load()
    neuralese_r1.concept_token = spy  # run() looks concept_token up in neuralese_r1 at call time
    rows = []
    with open(OUT, "w") as f:
        for i in ITEMS:
            inputs = make_inputs(tok, json.loads(lines[i])["question"] + SUFFIX)
            for noise, seed in CONDITIONS:
                PRE.clear()
                rec, _ = run(model, tok, inputs, noise, seed)
                assert len(PRE) == len(rec["steps"]), "STOP: spy out of step with run()"
                fed = [s["w"] for s in rec["steps"]]
                if noise is None:
                    assert all(abs(max(w) - p[0]) < 1e-3 for w, p in zip(fed, PRE)), "STOP: vanilla fed != pre-noise"
                r = {"item": i, "noise": noise, "seed": seed, "stop_reason": rec["stop_reason"],
                     "pre1": [round(p[0], 4) for p in PRE], "pre2": [round(p[1], 4) for p in PRE],
                     "fed1": [max(w) for w in fed], "fed2": [sorted(w)[-2] for w in fed],
                     "flip": [w.index(max(w)) != 0 for w in fed]}
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                k = [j for j, b in enumerate(r["pre2"]) if b >= FORK_MIN_SECOND]
                print(f"item {i} {str(noise):9s} {rec['stop_reason']:12s} {len(fed):5d} steps, {len(k):4d} forks, "
                      f"fed top>=0.9 at forks {mean([r['fed1'][j] >= 0.9 for j in k]):.2f}", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
