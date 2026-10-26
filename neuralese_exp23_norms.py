"""Exp 23 control check: is `rand` a fair control? Embedding norms of the random rows vs real reasoning tokens.

The rand arm moves the residual onto a random real-token row, drawn uniformly from ids < len(tok) by
neuralese_residual_ablation.keyed_noise, keyed only by (SEED, step). If those rows had much smaller norms than
the tokens the model really feeds, `rand` would inject less signal and the "soft moves predictions ~5x more than
rand" result (C23.7-8) would be inflated. Three numbers, saved to exp23_embedding_norms.json:
  population   - norms of every row rand can draw from (ids 0..len(tok)-1); exact, uses no RNG
  exp23_draws  - the exact random ids Exp 23 used, regenerated with keyed_noise on the GPU (steps 0..MAX_THINK-1);
                 also weighted by how many items' rand arm reached each step, if the Exp 23 jsonl is attached
  reasoning    - chosen tokens of ordinary sampled CoT (= Exp 23's hard arm) on the first N_REF Exp 23 items
Also records the software versions and model revision for the paper's reproducibility section.

Kaggle (GPU T4): load the code as for neuralese_eval.py, then  %run neuralese_exp23_norms.py   (~5-15 min)
"""
import glob
import json
import os
import platform
import statistics
import urllib.request

import torch
import transformers

from neuralese_eval import DATA_URL, SUFFIX
from neuralese_r1 import MAX_THINK, MODEL, OUT_DIR, load, make_inputs, run
from neuralese_residual_ablation import ITEMS, keyed_noise

N_REF = 5  # reasoning reference: sampled CoT (seed 0) on items 50..54
OUT = f"{OUT_DIR}/exp23_embedding_norms.json"


def summary(x):
    x = sorted(float(v) for v in x)
    q = lambda f: x[min(len(x) - 1, int(f * len(x)))]
    return {"n": len(x), "median": round(statistics.median(x), 4), "p10": round(q(0.10), 4),
            "p90": round(q(0.90), 4), "mean": round(statistics.mean(x), 4)}


def weighted_median(values, weights):
    pairs, half, acc = sorted(zip(values, weights)), sum(weights) / 2, 0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return round(float(v), 4)


@torch.no_grad()
def main():
    tok, model = load()
    embed = model.get_input_embeddings().weight
    norms = embed.norm(dim=-1).cpu()
    vocab, n_tok = embed.shape[0], len(tok)
    assert n_tok <= vocab

    # 1. Population rand draws from (uniform over real-token ids).
    population = summary(norms[:n_tok].tolist())

    # 2. The exact ids Exp 23 drew: same function, same device, same call order as lockstep().
    rid = [keyed_noise(step, vocab, n_tok, embed.device)[1] for step in range(MAX_THINK)]
    assert all(0 <= r < n_tok for r in rid)
    draws = summary(norms[rid].tolist())
    found = sorted(glob.glob("/kaggle/**/exp23_residual_ablation_items*.jsonl", recursive=True)
                   + glob.glob("exp23_residual_ablation_items*.jsonl"))
    if found:  # weight each step by the number of items whose rand arm was still thinking at that step
        steps = [json.loads(l)["rand"]["think_steps"] for l in open(found[0]) if l.strip()]
        wts = [sum(s > t for s in steps) for t in range(MAX_THINK)]
        draws["usage_weighted_median"] = weighted_median(norms[rid].tolist(), wts)
        draws["usage_weights_from"] = os.path.basename(found[0])

    # 3. Real reasoning tokens: ordinary sampled CoT, the token chosen at every think step.
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    chosen, per_item = [], {}
    for i in list(ITEMS)[:N_REF]:
        rec, _ = run(model, tok, make_inputs(tok, json.loads(lines[i])["question"] + SUFFIX), "discrete", 0)
        ids = [s["ids"][s["w"].index(max(s["w"]))] for s in rec["steps"]]
        chosen += ids
        per_item[i] = len(ids)
        print(f"item {i}: {len(ids)} think steps ({rec['stop_reason']})", flush=True)
    reasoning = summary(norms[chosen].tolist())
    reasoning["think_steps_per_item"] = per_item

    result = {
        "population_real_token_rows": population,
        "exp23_rand_draws": draws,
        "reasoning_tokens_sampled_cot": reasoning,
        "ratio_median_draws_to_reasoning": round(draws["median"] / reasoning["median"], 3),
        "ratio_median_population_to_reasoning": round(population["median"] / reasoning["median"], 3),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0), "model": MODEL,
                     "model_revision": getattr(model.config, "_commit_hash", None),
                     "vocab_rows": vocab, "real_tokens": n_tok},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
