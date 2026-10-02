"""Step 3: GSM8K accuracy - discrete CoT vs vanilla Soft Thinking vs Gumbel.

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

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
N_ITEMS = 30  # first 30 test items, fixed before looking at results
CONDITIONS = [("discrete", 0), ("discrete", 1), (None, 0), ("gumbel", 0), ("gumbel", 1)]
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
OUT = f"{OUT_DIR}/gsm8k_eval.jsonl"


def boxed(text):
    """Contents of the last \\boxed{...}, brace-matched; None if there is none."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, start = 0, i + 6
    for k in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[start + 1:k]
    return text[start + 1:]  # answer was cut off before the closing brace


def score(answer, gold):
    """True/False = boxed answer right/wrong; None = no boxed answer at all."""
    b = boxed(answer)
    if b is None:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", b.replace(",", ""))
    return bool(nums) and abs(float(nums[-1]) - float(gold)) < 1e-6


def summarize():
    rows = [json.loads(line) for line in open(OUT)]
    print(f"\n{len(rows)} runs in {OUT}")
    for noise in dict.fromkeys(r["noise"] for r in rows):
        rs = [r for r in rows if r["noise"] == noise]
        ok, answered = sum(r["correct"] for r in rs), sum(r["answered"] for r in rs)
        print(f"  {str(noise):9s} correct={ok} wrong={answered - ok} no_answer={len(rs) - answered} of {len(rs)}  "
              f"budget_hits={sum(r['stop_reason'] == 'budget' for r in rs)}  "
              f"cold_stops={sum(r['stop_reason'] == 'cold_stop' for r in rs)}  "
              f"mean_think={sum(r['think_steps'] for r in rs) / len(rs):.0f}  "
              f"mean_secs={sum(r['secs'] for r in rs) / len(rs):.0f}")


def main():
    assert score(r"so \boxed{\$1,250.00}", "1250") is True
    assert score(r"\boxed{\text{18 dollars}}", "18") is True and score(r"\boxed{17}", "18") is False
    assert score("the answer is 18", "18") is None and boxed(r"\boxed{\frac{1}{2}} end") == r"\frac{1}{2}"

    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    items = [json.loads(line) for line in lines[:N_ITEMS]]
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r["item"], r["noise"], r["seed"]) for r in map(json.loads, open(OUT))}

    tok, model = load()
    for i, item in enumerate(items):
        inputs = make_inputs(tok, item["question"] + SUFFIX)
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        for noise, seed in CONDITIONS:
            if (i, noise, seed) in done:
                continue
            t = time.time()
            # ponytail: per-step traces not saved here (~1 MB/run); regenerate any run from (item, noise, seed)
            r, _ = run(model, tok, inputs, noise, seed)
            s = score(r["answer"], gold)
            row = {"item": i, "noise": noise, "seed": seed, "gold": gold, "correct": s is True,
                   "answered": s is not None, "stop_reason": r["stop_reason"],
                   "think_steps": r["think_steps"], "answer": r["answer"], "secs": round(time.time() - t)}
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"item {i:2d} {str(noise):9s} s={seed}: correct={row['correct']} answered={row['answered']} "
                  f"{row['stop_reason']} {row['think_steps']} steps {row['secs']}s")
    summarize()


if __name__ == "__main__":
    main()
