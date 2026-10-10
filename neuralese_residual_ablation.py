"""Exp 23: is the residual of Gumbel Soft Thinking load-bearing? (coupled residual ablation)

Gumbel-max: the top token of a Gumbel-softmax blend is exactly a sample from the pre-noise weights, so
Gumbel Soft Thinking = ordinary sampling + a residual (the soft mass left on the other candidates).
Three arms run in lockstep with the SAME noise, keyed by (seed, step, token id) so pairing survives rank swaps:
  soft = the normal Gumbel blend                     (what Soft Thinking feeds)
  hard = one-hot of the blend's top token            (exactly ordinary sampling, same token choices)
  rand = top token at its blend weight, residual moved onto a random vocabulary token (disturbance control)
While an arm has produced the same tokens as `hard`, every difference in their next-token distributions comes
from the residual alone: per-step KL(arm || hard) up to the first token flip (primary measure).
Each arm then finishes on its own (Cold Stop on pre-noise entropy in all arms, same budget) -> boxed answer;
paired McNemar on correctness (secondary). soft vs hard is also the matched Gumbel-vs-sampling comparison.
Gate: two 'hard' arms must produce identical tokens with zero KL for 64 steps (run stops otherwise).

Kaggle: load the code as for neuralese_eval.py, then  %run neuralese_residual_ablation.py
(3 forward passes per step, ~5-6 min per item, ~9-10 h for 100 items; resumable: finished items are skipped)
"""
import json
import os
import urllib.request
from math import comb

import torch

from neuralese_eval import DATA_URL, SUFFIX, boxed, score
from neuralese_r1 import (COLD_STOP_ENTROPY, COLD_STOP_LEN, GUMBEL_TAU, MAX_ANSWER, MAX_THINK, OUT_DIR,
                          TEMPERATURE, concept_token, load, make_inputs)

ITEMS = range(50, 150)  # GSM8K test items never used before (Exp 16-22 used 0-49)
SEED = 0
ARMS = ("hard", "soft", "rand")  # arms[0] must be the reference ("hard")
OUT = f"{OUT_DIR}/exp23_residual_ablation_items{ITEMS.start}-{ITEMS.stop - 1}.jsonl"


def keyed_noise(step, vocab, n_tok, device):
    """Gumbel noise over the embedding rows + a random real-token id (< n_tok, skips untrained padding rows),
    identical in every arm at this step."""
    g = torch.Generator(device=device).manual_seed(SEED * 1_000_003 + step)
    u = torch.rand(vocab, generator=g, device=device).clamp_min(1e-20)
    return -torch.log(-torch.log(u)), torch.randint(n_tok, (1,), generator=g, device=device).item()


def feed_for(arm, embed, w, ids, g, rand_id):
    """The embedding an arm feeds next, and its chosen (top) token. w, ids: pre-noise top-k (1, k)."""
    y = torch.softmax((w.clamp_min(1e-12).log() + g[ids]) / GUMBEL_TAU, dim=-1)  # = F.gumbel_softmax, noise by token id
    j = y[0].argmax().item()
    tid = ids[0, j].item()
    if arm == "soft":
        return (embed[ids] * y.unsqueeze(-1)).sum(1), tid
    if arm == "hard":
        return embed[tid][None], tid
    p = y[0, j]
    return (p * embed[tid] + (1 - p) * embed[rand_id])[None], tid


@torch.no_grad()
def lockstep(model, tok, inputs, arms, max_steps=MAX_THINK):
    """Think phase for all arms in lockstep. Returns per-arm state, KL lists vs arms[0], first-flip steps."""
    embed = model.get_input_embeddings().weight
    stop = {tok.convert_tokens_to_ids("</think>"), tok.eos_token_id}
    st = [{"arm": a, "past": None, "mask": inputs["attention_mask"], "feed": {"input_ids": inputs["input_ids"]},
           "low": 0, "ids": [], "reason": "budget", "done": False} for a in arms]
    kl = {i: [] for i in range(1, len(arms))}
    flip = {i: None for i in range(1, len(arms))}
    for step in range(max_steps):
        g, rand_id = keyed_noise(step, embed.shape[0], len(tok), embed.device)
        logp = {}
        for i, s in enumerate(st):
            if s["done"]:
                continue
            out = model(**s["feed"], attention_mask=s["mask"], past_key_values=s["past"], use_cache=True)
            s["past"], logits = out.past_key_values, out.logits[:, -1, :]
            logp[i] = torch.log_softmax(logits[0] / TEMPERATURE, -1)
            _, w, ids, h = concept_token(embed, logits, None)  # pre-noise candidates and entropy; no RNG
            x, tid = feed_for(s["arm"], embed, w, ids, g, rand_id)
            s["ids"].append(tid)
            s["low"] = s["low"] + 1 if h < COLD_STOP_ENTROPY else 0
            if tid in stop:
                s["done"], s["reason"] = True, "eos_in_think" if tid == tok.eos_token_id else "think_end"
            elif s["low"] >= COLD_STOP_LEN:
                s["done"], s["reason"] = True, "cold_stop"
            else:
                s["feed"] = {"inputs_embeds": x.unsqueeze(1)}
                s["mask"] = torch.cat([s["mask"], s["mask"].new_ones(1, 1)], dim=1)
        for i in kl:  # same token history so far -> difference comes from the residual alone
            if flip[i] is None and 0 in logp and i in logp:
                kl[i].append(round((logp[i].exp() * (logp[i] - logp[0])).sum().item(), 6))
                if st[i]["ids"][-1] != st[0]["ids"][-1]:
                    flip[i] = step
        if all(s["done"] for s in st):
            break
    return st, kl, flip


@torch.no_grad()
def answer(model, tok, s):
    """Feed a real </think>, then greedy answer (as neuralese_r1.run)."""
    eos, ids = tok.eos_token_id, torch.tensor([[tok.convert_tokens_to_ids("</think>")]], device=s["mask"].device)
    mask, past, out_ids = s["mask"], s["past"], []
    for _ in range(MAX_ANSWER):
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
        out = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
        past, ids = out.past_key_values, out.logits[:, -1:, :].argmax(-1)
        if ids.item() == eos:
            break
        out_ids.append(ids.item())
    return tok.decode(out_ids)


def mcnemar(rows, a, b):
    """Exact two-sided McNemar on correctness: (a right & b wrong, b right & a wrong, p)."""
    x = sum(r[a]["correct"] and not r[b]["correct"] for r in rows)
    y = sum(r[b]["correct"] and not r[a]["correct"] for r in rows)
    n = x + y
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def summarize(rows):
    print(f"\n{len(rows)} items")
    for a in ARMS:
        print(f"  {a:4s} correct {sum(r[a]['correct'] for r in rows)}  answered {sum(r[a]['answered'] for r in rows)}  "
              f"cold_stops {sum(r[a]['reason'] == 'cold_stop' for r in rows)}  budget {sum(r[a]['reason'] == 'budget' for r in rows)}")
    for a in ARMS[1:]:
        per_item = [sum(r["kl"][a]) / len(r["kl"][a]) for r in rows if r["kl"][a]]
        flips = sorted(r["flip"][a] for r in rows if r["flip"][a] is not None)
        x, y, p = mcnemar(rows, a, "hard")
        same = sum(r[a]["answer_num"] == r["hard"]["answer_num"] for r in rows)
        print(f"  {a} vs hard: mean per-item KL before flip {sum(per_item) / max(len(per_item), 1):.2e}; "
              f"first flip median step {flips[len(flips) // 2] if flips else None} ({len(flips)}/{len(rows)} items flip); "
              f"same final answer {same}/{len(rows)}; McNemar {a}-only {x}, hard-only {y}, p = {p:.3f}")
    paired = [(sum(r["kl"]["soft"]) / len(r["kl"]["soft"]), sum(r["kl"]["rand"]) / len(r["kl"]["rand"]))
              for r in rows if r["kl"]["soft"] and r["kl"]["rand"]]
    print(f"  items where soft drifts less than rand: {sum(s < q for s, q in paired)}/{len(paired)}")


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    tok, model = load()
    done = {json.loads(l)["item"] for l in open(OUT)} if os.path.exists(OUT) else set()

    # Gate: the noise keying is deterministic -> two 'hard' arms must agree exactly.
    st, kl, flip = lockstep(model, tok, make_inputs(tok, json.loads(lines[ITEMS[0]])["question"] + SUFFIX),
                            ("hard", "hard"), max_steps=64)
    assert flip[1] is None and max(kl[1]) < 1e-6, f"STOP: two hard arms differ (flip {flip[1]}, max KL {max(kl[1])})"
    print("DETERMINISM GATE: pass")

    for i in ITEMS:
        if i in done:
            continue
        item = json.loads(lines[i])
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        st, kl, flip = lockstep(model, tok, make_inputs(tok, item["question"] + SUFFIX), ARMS)
        row = {"item": i, "gold": gold, "kl": {ARMS[k]: v for k, v in kl.items()},
               "flip": {ARMS[k]: v for k, v in flip.items()}}
        for s in st:
            text = answer(model, tok, s)
            sc = score(text, gold)
            row[s["arm"]] = {"correct": sc is True, "answered": sc is not None, "reason": s["reason"],
                             "think_steps": len(s["ids"]), "answer_num": boxed(text), "answer_tail": text[-200:]}
            s["past"] = None  # free the cache
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"item {i}: " + "  ".join(f"{a}={row[a]['correct']}({row[a]['reason']},{row[a]['think_steps']})" for a in ARMS)
              + f"  first flip soft={row['flip']['soft']} rand={row['flip']['rand']}", flush=True)
    summarize([json.loads(l) for l in open(OUT)])


if __name__ == "__main__":
    main()
