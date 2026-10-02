"""Step 4: alpha-sweep at real fork points - does the model carry a blend, or snap to one token?

At a fork (top-2 pre-noise candidates A, B), feed alpha*emb(A) + (1-alpha)*emb(B) for
alpha = 0, 0.1, ..., 1 from the same cached prefix, and project every layer's hidden state
(plus final logits) onto the line from the pure-B state (alpha=0) to the pure-A state (alpha=1):
  c(alpha)   = where on that line the blended state sits (0 = like B, 1 = like A)
  off(alpha) = distance off the line, relative to the A-B gap (a "third thing")
  snap       = mean|c - alpha| / 0.25  ->  0 = carries the blend linearly, 1 = step function

Colab or Kaggle: load the code as for neuralese_eval.py, then  %run neuralese_sweep.py
"""
import copy
import json
import urllib.request

import torch

from neuralese_eval import DATA_URL, SUFFIX
from neuralese_r1 import MAX_THINK, OUT_DIR, concept_token, load, make_inputs

N_ITEMS = 10  # GSM8K items 0-9 (same traces as Exp 16, same seeds)
CONDITIONS = [(None, 0), ("gumbel", 0)]
FORKS_PER_RUN = 5  # first 5 forks in step order, fixed in advance (no cherry-picking)
FORK_MIN_SECOND = 0.25  # fork = runner-up candidate holds >= 25% of pre-noise weight
ALPHAS = [i / 10 for i in range(11)]
OUT = f"{OUT_DIR}/step4_alpha_sweep.jsonl"


@torch.no_grad()
def sweep(model, embed, past, mask, a_id, b_id):
    """Per layer: c(alpha) and off(alpha) for blends of tokens a_id / b_id fed after `past`."""
    mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
    states, next_top = [], []
    for a in ALPHAS:
        x = (a * embed[a_id] + (1 - a) * embed[b_id]).view(1, 1, -1)
        # ponytail: deepcopy per alpha (~100 MB at 2k tokens); batch the 11 branches if this gets slow
        out = model(inputs_embeds=x, attention_mask=mask, past_key_values=copy.deepcopy(past),
                    use_cache=True, output_hidden_states=True)
        states.append([h[0, -1] for h in out.hidden_states] + [out.logits[0, -1]])
        next_top.append(out.logits[0, -1].argmax().item())

    c, off = [], []
    for layer in range(len(states[0])):
        h0, d = states[0][layer], states[-1][layer] - states[0][layer]
        dd = d.dot(d).clamp_min(1e-12)
        proj = [(s[layer] - h0).dot(d) / dd for s in states]
        c.append([p.item() for p in proj])
        off.append([((s[layer] - h0 - p * d).norm() / dd.sqrt()).item() for s, p in zip(states, proj)])
    # Layer 0 is the input blend itself: must sit exactly at c = alpha, on the line.
    assert all(abs(ci - a) < 1e-3 for ci, a in zip(c[0], ALPHAS)) and max(off[0]) < 1e-3, "sweep misaligned"
    snap = [sum(abs(ci - a) for ci, a in zip(cl, ALPHAS)) / len(ALPHAS) / 0.25 for cl in c]
    return {"c": c, "off": off, "snap": snap, "next_top": next_top}


@torch.no_grad()
def forks_in_run(model, tok, inputs, noise, seed):
    """Replays the eval run (same RNG use as neuralese_r1.run) and sweeps its first forks."""
    torch.manual_seed(seed)
    embed = model.get_input_embeddings().weight
    stop = {tok.convert_tokens_to_ids("</think>"), tok.eos_token_id}
    mask, past, feed, rows = inputs["attention_mask"], None, {"input_ids": inputs["input_ids"]}, []
    for step in range(MAX_THINK):
        out = model(**feed, attention_mask=mask, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        _, w0, ids0, _ = concept_token(embed, logits, None)  # pre-noise candidates; uses no RNG
        if w0[0, 1] >= FORK_MIN_SECOND:
            a_id, b_id = ids0[0, 0].item(), ids0[0, 1].item()
            pair = [tok.decode([a_id]), tok.decode([b_id])]
            rows.append({"step": step, "A": pair[0], "B": pair[1], "wA": round(w0[0, 0].item(), 3),
                         "wB": round(w0[0, 1].item(), 3),
                         "kind": "digit" if any(ch.isdigit() for ch in "".join(pair)) else "other",
                         **sweep(model, embed, past, mask, a_id, b_id)})
            rows[-1]["next_top"] = [tok.decode([t]) for t in rows[-1]["next_top"]]
            if len(rows) == FORKS_PER_RUN:
                break
        # ponytail: no Cold Stop here; a vanilla loop could repeat forks, fine at 5 forks per run
        blend, w, ids, _ = concept_token(embed, logits, noise)
        if ids[0, w[0].argmax()].item() in stop:
            break
        feed = {"inputs_embeds": blend.unsqueeze(1)}
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
    return rows


def summarize(rows):
    print(f"\n{len(rows)} forks. Mean snap per layer (0 = carries blend linearly, 1 = snaps); "
          f"last two columns = final hidden, logits")
    for kind in ("digit", "other"):
        for noise in dict.fromkeys(r["noise"] for r in rows):
            rs = [r for r in rows if r["kind"] == kind and r["noise"] == noise]
            if rs:
                mean = [sum(r["snap"][i] for r in rs) / len(rs) for i in range(len(rs[0]["snap"]))]
                print(f"  {kind:5s} {str(noise):6s} n={len(rs):2d}: " + " ".join(f"{m:.2f}" for m in mean))


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    items = [json.loads(line) for line in lines[:N_ITEMS]]
    tok, model = load()
    rows = []
    with open(OUT, "w") as f:
        for i, item in enumerate(items):
            inputs = make_inputs(tok, item["question"] + SUFFIX)
            for noise, seed in CONDITIONS:
                for r in forks_in_run(model, tok, inputs, noise, seed):
                    r.update(item=i, noise=noise, seed=seed)
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    print(f"item {i} {str(noise):6s} step {r['step']:4d} {r['A']!r}/{r['B']!r} "
                          f"({r['wA']}/{r['wB']}) snap: L14={r['snap'][14]:.2f} final={r['snap'][-2]:.2f} "
                          f"logits={r['snap'][-1]:.2f}  next@0.5={r['next_top'][5]!r}")
    summarize(rows)


if __name__ == "__main__":
    main()
