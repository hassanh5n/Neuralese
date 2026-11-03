"""Plausible-token control for the coupled residual ablation (follow-up to neuralese_residual_ablation.py).

Question: the real residual of Gumbel Soft Thinking moves next-token predictions ~3.9x more than the same residual
moved onto a uniformly random vocabulary token. Is that because the model reads *this* candidate, or only because a
context-plausible token moves predictions more than an off-context one?

Four arms in lockstep with the SAME noise as the coupled ablation (keyed by seed, step and token id):
  hard   = one-hot of the blend's top token                 (ordinary sampling)
  soft   = the normal Gumbel blend                           (Gumbel Soft Thinking; the real residual)
  rand   = top token at its blend weight, residual on a uniformly random real token     (as before)
  plaus  = top token at its blend weight, residual on the most probable token that is neither the top token nor
           the token that holds the largest share of the real residual (the context-plausible "next best" token)
Same residual weight in rand and plaus as in soft. While an arm has produced the same tokens as hard, differences
in its next-token distribution come from where the residual goes.

MODE = "kl_only" (default, cheap): think phase only, stops each item once every arm has left sampling. Gives the
           pre-registered primary endpoint (KL over matched steps). About 1-2 h for 100 items on a T4.
MODE = "full": as the coupled ablation (every arm finishes and answers), adds accuracy. About 12-14 h for 100 items
           on a T4, so it needs two Kaggle sessions (the run resumes: finished items are skipped).

Kaggle: same cells as the coupled ablation, then  %run neuralese_plausible_ablation.py
If exp23_residual_ablation_items50-149.jsonl is in the working folder, the script also reports how many items
reproduce the saved first-divergence steps of soft and rand (a reproducibility check, not a gate).
"""
import json
import os
import urllib.request

import torch

from neuralese_eval import DATA_URL, SUFFIX, boxed, score
from neuralese_r1 import (COLD_STOP_ENTROPY, COLD_STOP_LEN, GUMBEL_TAU, MAX_THINK, OUT_DIR, TEMPERATURE,
                          concept_token, load, make_inputs)
from neuralese_residual_ablation import answer, keyed_noise

MODE = os.environ.get("PLAUS_MODE", "kl_only")  # "kl_only" or "full"
ITEMS = range(50, 150)  # the coupled ablation's items, so results pair with it problem by problem
ARMS = ("hard", "soft", "rand", "plaus")  # arms[0] must be the reference ("hard")
OUT = f"{OUT_DIR}/exp24_plausible_items{ITEMS.start}-{ITEMS.stop - 1}_{MODE}.jsonl"
EXP23 = f"{OUT_DIR}/exp23_residual_ablation_items50-149.jsonl"


def feed_for(arm, embed, w, ids, g, rand_id, probs):
    """Embedding an arm feeds next, its top token, and per-step stats.
    w, ids: pre-noise filtered top-k (1, k); probs: full next-token distribution at the sampling temperature."""
    y = torch.softmax((w.clamp_min(1e-12).log() + g[ids]) / GUMBEL_TAU, dim=-1)  # same noise as the coupled ablation
    order = y[0].argsort(descending=True)
    j = order[0].item()
    tid = ids[0, j].item()
    p = y[0, j]
    r = ids[0, order[1]].item() if ids.shape[1] > 1 else None  # token holding the largest share of the residual
    if arm == "soft":
        return (embed[ids] * y.unsqueeze(-1)).sum(1), tid, {"resid": round(1 - p.item(), 6), "q_r": round(probs[r].item(), 6) if r is not None else 0.0}
    if arm == "hard":
        return embed[tid][None], tid, {}
    if arm == "rand":
        other = rand_id
    else:  # plaus: most probable token that is neither the top token nor the real residual's main token
        top = torch.topk(probs, 3).indices.tolist()
        other = next(t for t in top if t != tid and t != r)
    stats = {"q_c": round(probs[other].item(), 6)} if arm == "plaus" else {}
    return (p * embed[tid] + (1 - p) * embed[other])[None], tid, stats


@torch.no_grad()
def lockstep(model, tok, inputs, arms, max_steps=MAX_THINK, kl_only=False):
    embed = model.get_input_embeddings().weight
    stop = {tok.convert_tokens_to_ids("</think>"), tok.eos_token_id}
    st = [{"arm": a, "past": None, "mask": inputs["attention_mask"], "feed": {"input_ids": inputs["input_ids"]},
           "low": 0, "ids": [], "reason": "budget", "done": False} for a in arms]
    kl = {i: [] for i in range(1, len(arms))}
    flip = {i: None for i in range(1, len(arms))}
    stats = {"resid": [], "q_r": [], "q_c": []}
    for step in range(max_steps):
        g, rand_id = keyed_noise(step, embed.shape[0], len(tok), embed.device)
        logp = {}
        for i, s in enumerate(st):
            if s["done"]:
                continue
            out = model(**s["feed"], attention_mask=s["mask"], past_key_values=s["past"], use_cache=True)
            s["past"], logits = out.past_key_values, out.logits[:, -1, :]
            logp[i] = torch.log_softmax(logits[0] / TEMPERATURE, -1)
            _, w, ids, h = concept_token(embed, logits, None)
            x, tid, extra = feed_for(s["arm"], embed, w, ids, g, rand_id, logp[i].exp())
            if any(flip[k] is None for k in flip):  # log per-step stats while some arm is still matched
                for key, v in extra.items():
                    stats[key].append(v)
            s["ids"].append(tid)
            s["low"] = s["low"] + 1 if h < COLD_STOP_ENTROPY else 0
            if tid in stop:
                s["done"], s["reason"] = True, "eos_in_think" if tid == tok.eos_token_id else "think_end"
            elif s["low"] >= COLD_STOP_LEN:
                s["done"], s["reason"] = True, "cold_stop"
            else:
                s["feed"] = {"inputs_embeds": x.unsqueeze(1)}
                s["mask"] = torch.cat([s["mask"], s["mask"].new_ones(1, 1)], dim=1)
        for i in kl:
            if flip[i] is None and 0 in logp and i in logp:
                kl[i].append(round((logp[i].exp() * (logp[i] - logp[0])).sum().item(), 6))
                if st[i]["ids"][-1] != st[0]["ids"][-1]:
                    flip[i] = step
        if all(s["done"] for s in st):
            break
        if kl_only and (all(flip[i] is not None for i in flip) or st[0]["done"]):
            break
    return st, kl, flip, stats


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    tok, model = load()
    done = {json.loads(l)["item"] for l in open(OUT)} if os.path.exists(OUT) else set()
    old = {r["item"]: r for r in map(json.loads, open(EXP23))} if os.path.exists(EXP23) else {}

    st, kl, flip, _ = lockstep(model, tok, make_inputs(tok, json.loads(lines[ITEMS[0]])["question"] + SUFFIX),
                               ("hard", "hard"), max_steps=64)
    assert flip[1] is None and max(kl[1]) < 1e-6, f"STOP: two hard arms differ (flip {flip[1]}, max KL {max(kl[1])})"
    print(f"DETERMINISM GATE: pass   MODE = {MODE}   output {OUT}")

    for i in ITEMS:
        if i in done:
            continue
        item = json.loads(lines[i])
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        st, kl, flip, stats = lockstep(model, tok, make_inputs(tok, item["question"] + SUFFIX), ARMS,
                                       kl_only=(MODE == "kl_only"))
        row = {"item": i, "gold": gold, "mode": MODE, "kl": {ARMS[k]: v for k, v in kl.items()},
               "flip": {ARMS[k]: v for k, v in flip.items()}, "stats": stats}
        if i in old:
            row["reproduces_exp23_flips"] = (row["flip"]["soft"], row["flip"]["rand"]) == (old[i]["flip"]["soft"], old[i]["flip"]["rand"])
        if MODE == "full":
            for s in st:
                text = answer(model, tok, s)
                sc = score(text, gold)
                row[s["arm"]] = {"correct": sc is True, "answered": sc is not None, "reason": s["reason"],
                                 "think_steps": len(s["ids"]), "answer_num": boxed(text), "answer_tail": text[-200:]}
        for s in st:
            s["past"] = None
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"item {i}: first flip " + " ".join(f"{a}={row['flip'][a]}" for a in ARMS[1:])
              + (f"  reproduces coupled ablation: {row['reproduces_exp23_flips']}" if i in old else ""), flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    rep = [r["reproduces_exp23_flips"] for r in rows if "reproduces_exp23_flips" in r]
    print(f"\n{len(rows)} items written to {OUT}")
    if rep:
        print(f"reproduce the saved soft and rand first-divergence steps: {sum(rep)}/{len(rep)}")
    print("Analyse with:  python analyze_plausible.py " + OUT)


if __name__ == "__main__":
    main()
