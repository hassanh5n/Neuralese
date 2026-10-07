"""Exp 20 (Step 5 mechanism): which layers' KV cache holds the losing candidate B?

Exp 18: when the blend followed A, it reads B's continuation like B, but the A+C control does not ->
B is stored in the cache. Later tokens see the fork position only through its keys/values, one set per
layer. So swap the fork position's K/V between blend and control, per layer L and per block of 4 layers:
  remove: blend cache, layer-L fork K/V taken from the control -> does the blend stop reading B's text like B?
  insert: control cache, layer-L fork K/V taken from the blend -> does the control start reading it like B?
Score: c on the B -> A line (1 = like pure A, 0 = like pure B) at positions 1-64 of B's forced greedy
continuation, as in Exp 18. Fraction of the gap closed, with c_gap = c_ctrl - c_blend:
  remove = (c_patch - c_blend) / c_gap        insert = (c_ctrl - c_patch) / c_gap
Check: patching all layers must reproduce the donor exactly (the run stops otherwise).
Only forks where the blend followed A are patched; the others are logged with followed_A = False.
Same replay and forks as Exp 17/18 (sweep swapped out; the measurement uses no RNG).

Kaggle: load the code as for neuralese_eval.py, then  %run neuralese_kv_patch.py
"""
import copy
import json
import random
import urllib.request

import torch

import neuralese_sweep
from neuralese_eval import DATA_URL, SUFFIX
from neuralese_kv_carryover import LAST, MIN_TV, N_CONT, spy_concept_token, step
from neuralese_r1 import OUT_DIR, TEMPERATURE, load, make_inputs
from neuralese_sweep import CONDITIONS, ITEMS, TAG, forks_in_run

N_LAYERS = 28
SETS = {f"L{l}": [l] for l in range(N_LAYERS)} | {f"L{l}-{l + 3}": list(range(l, l + 4)) for l in range(0, N_LAYERS, 4)}
CHECK_TOL = 1e-3  # all-layer patch == donor up to fp32 noise; a mis-wired patch is off by ~0.1-1
OUT = f"{OUT_DIR}/step5_kv_patch_{TAG}.jsonl"


def swap(cache, donor, layers):
    """Copy of `cache` with the last (fork) position's keys/values at `layers` taken from `donor`."""
    cache = copy.deepcopy(cache)
    for l in layers:
        cache.layers[l].keys[..., -1, :] = donor.layers[l].keys[..., -1, :]
        cache.layers[l].values[..., -1, :] = donor.layers[l].values[..., -1, :]
    return cache


@torch.no_grad()
def patch(model, embed, past, mask, a_id, b_id):
    """Replaces neuralese_sweep.sweep at each fork."""
    assert LAST["ids"][0] == a_id and LAST["ids"][1] == b_id, "STOP: spy out of step with the fork"
    c_id = LAST["ids"][2].item()
    fork = {"A": embed[a_id], "B": embed[b_id], "blend": (embed[a_id] + embed[b_id]) / 2,
            "control": (embed[a_id] + embed[c_id]) / 2}
    start = {k: step(model, copy.deepcopy(past), mask, inputs_embeds=x.view(1, 1, -1)) for k, x in fork.items()}
    top = {k: lg[-1].argmax().item() for k, (lg, _, _) in start.items()}
    row = {"next_top": [top["blend"]], "C": c_id, "followed_A": top["blend"] == top["A"] != top["B"]}
    if not row["followed_A"]:
        return row
    assert len(past.layers) == N_LAYERS, "STOP: unexpected layer count"

    lg, p, m = start["B"]
    p, ids = copy.deepcopy(p), []  # B's own greedy continuation
    for _ in range(N_CONT):
        ids.append(lg[-1].argmax().item())
        lg, p, m = step(model, p, m, input_ids=torch.tensor([[ids[-1]]], device=m.device))
    text, m = torch.tensor([ids], device=mask.device), start["A"][2]

    def probs(cache):  # next-token distributions at positions 1..N_CONT with B's text forced; consumes cache
        return torch.softmax(step(model, cache, m, input_ids=text)[0] / TEMPERATURE, dim=-1)

    P = {k: probs(copy.deepcopy(cache)) for k, (_, cache, _) in start.items()}
    d = P["A"] - P["B"]
    keep = 0.5 * d.abs().sum(-1) >= MIN_TV  # skip positions where pure A and B barely differ
    if not keep.any():
        return {**row, "n_pos": 0}
    dd = (d * d).sum(-1).clamp_min(1e-12)

    def c(p):
        return (((p - P["B"]) * d).sum(-1) / dd)[keep]

    blend, ctrl = start["blend"][1], start["control"][1]
    cb, cc = c(P["blend"]), c(P["control"])
    for name, base, donor, ref in (("remove", blend, ctrl, cc), ("insert", ctrl, blend, cb)):
        assert (c(probs(swap(base, donor, range(N_LAYERS)))) - ref).abs().max() < CHECK_TOL, \
            f"STOP: {name} with all layers != donor"
    return {**row, "n_pos": int(keep.sum()), "c_blend": round(cb.mean().item(), 4), "c_ctrl": round(cc.mean().item(), 4),
            "remove": {k: round(c(probs(swap(blend, ctrl, ls))).mean().item(), 4) for k, ls in SETS.items()},
            "insert": {k: round(c(probs(swap(ctrl, blend, ls))).mean().item(), 4) for k, ls in SETS.items()}}


def group(r):
    num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
    return "num-num" if all(num) else "num-word" if any(num) else "word-word"


def frac(rs, name, k):
    """Fraction of the blend-vs-control gap closed by patching set k (ratio of sums over forks)."""
    gained = sum(r["remove"][k] - r["c_blend"] if name == "remove" else r["c_ctrl"] - r["insert"][k] for r in rs)
    return gained / sum(r["c_ctrl"] - r["c_blend"] for r in rs)


def summarize(rows):
    seen, uniq = set(), []
    for r in rows:  # vanilla and Gumbel traces can share a prefix: count each fork once
        if "remove" in r and (r["item"], r["step"], r["A"], r["B"]) not in seen:
            seen.add((r["item"], r["step"], r["A"], r["B"]))
            uniq.append(r)
    rng = random.Random(0)
    for g in ("num-num", "num-word", "word-word"):
        rs = [r for r in uniq if group(r) == g]
        if len(rs) < 2:
            continue
        gap = sum(r["c_ctrl"] - r["c_blend"] for r in rs) / len(rs)
        print(f"\n{g}: {len(rs)} forks (blend followed A), mean gap control - blend = {gap:+.2f}")
        print("  fraction of gap closed, 95% bootstrap CI over forks:   remove (needed?)   |   insert (enough?)")
        boots = [rng.choices(rs, k=len(rs)) for _ in range(2000)]
        for k in SETS:
            cells = []
            for name in ("remove", "insert"):
                b = sorted(frac(s, name, k) for s in boots)
                cells.append(f"{frac(rs, name, k):+.2f} [{b[49]:+.2f}, {b[1949]:+.2f}]")
            print(f"  {k:7s} " + "   |   ".join(cells))


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    items = {i: json.loads(lines[i]) for i in ITEMS}
    tok, model = load()
    neuralese_sweep.sweep = patch
    neuralese_sweep.concept_token = spy_concept_token  # forks_in_run calls sweep() by name -> same forks as Exp 17
    rows = []
    with open(OUT, "w") as f:
        for i, item in items.items():
            inputs = make_inputs(tok, item["question"] + SUFFIX)
            for noise, seed in CONDITIONS:
                for r in forks_in_run(model, tok, inputs, noise, seed):
                    r["C"] = tok.decode([r["C"]])
                    r.update(item=i, noise=noise, seed=seed)
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    msg = "skipped (blend did not follow A)" if not r["followed_A"] else "no positions" \
                        if not r["n_pos"] else (f"gap {r['c_ctrl'] - r['c_blend']:+.2f}  biggest remove: "
                                                f"{max(r['remove'], key=r['remove'].get)}  biggest insert: "
                                                f"{min(r['insert'], key=r['insert'].get)}")
                    print(f"item {i} {str(noise):6s} step {r['step']:4d} {r['A']!r}/{r['B']!r} C={r['C']!r}  {msg}")
    summarize(rows)


if __name__ == "__main__":
    main()
