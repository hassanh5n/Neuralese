"""Step 3: ProsQA accuracy - discrete CoT vs vanilla Soft Thinking vs Gumbel.

Colab: same first three cells as neuralese_r1.py, then
  %run neuralese_eval.py
Resumable: after a disconnect, re-run the cell and finished runs are skipped.
"""
import json
import os
import re
import time
import urllib.request

from neuralese_r1 import OUT_DIR, load, make_inputs, run

DATA_URL = "https://raw.githubusercontent.com/facebookresearch/coconut/main/data/prosqa_test.json"
N_ITEMS = 30  # first 30 test items, fixed before looking at results
CONDITIONS = [("discrete", 0), ("discrete", 1), (None, 0), ("gumbel", 0), ("gumbel", 1)]
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
OUT = f"{OUT_DIR}/prosqa_eval.jsonl"


def score(answer, target, neg):
    """Correct = last \\boxed{} names the target concept and not the distractor."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", answer)
    if not boxed:
        return False
    has = lambda word: re.search(rf"\b{word}\b", boxed[-1]) is not None
    return has(target) and not has(neg)


def summarize():
    rows = [json.loads(line) for line in open(OUT)]
    print(f"\n{len(rows)} runs in {OUT}")
    for noise in dict.fromkeys(r["noise"] for r in rows):
        rs = [r for r in rows if r["noise"] == noise]
        print(f"  {str(noise):9s} acc={sum(r['correct'] for r in rs)}/{len(rs)}  "
              f"budget_hits={sum(r['stop_reason'] == 'budget' for r in rs)}  "
              f"cold_stops={sum(r['stop_reason'] == 'cold_stop' for r in rs)}  "
              f"mean_think={sum(r['think_steps'] for r in rs) / len(rs):.0f}")


def main():
    assert score(r"so \boxed{\text{sterpus}}", "sterpus", "hilpus")
    assert not score(r"\boxed{hilpus}", "sterpus", "hilpus") and not score("sterpus", "sterpus", "hilpus")

    items = json.load(urllib.request.urlopen(DATA_URL))[:N_ITEMS]
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r["item"], r["noise"], r["seed"]) for r in map(json.loads, open(OUT))}

    tok, model = load()
    for i, item in enumerate(items):
        inputs = make_inputs(tok, item["question"] + SUFFIX)
        target, neg = (item["idx_to_symbol"][item[k]] for k in ("target", "neg_target"))
        for noise, seed in CONDITIONS:
            if (i, noise, seed) in done:
                continue
            t = time.time()
            # ponytail: per-step traces not saved here (~1 MB/run); regenerate any run from (item, noise, seed)
            r, _ = run(model, tok, inputs, noise, seed)
            row = {"item": i, "noise": noise, "seed": seed, "target": target, "neg": neg,
                   "correct": score(r["answer"], target, neg), "stop_reason": r["stop_reason"],
                   "think_steps": r["think_steps"], "answer": r["answer"], "secs": round(time.time() - t)}
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"item {i:2d} {str(noise):9s} s={seed}: correct={row['correct']} "
                  f"{row['stop_reason']} {row['think_steps']} steps {row['secs']}s")
    summarize()


if __name__ == "__main__":
    main()
