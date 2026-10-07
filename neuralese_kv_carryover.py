"""Step 5 / Test 1: does a blend's leftover in the KV cache steer later tokens?

At each Exp 17 fork (same replay, same forks), three branches start from the same cache:
pure A, pure B and the 50/50 blend. The same text is then forced into all three - first A's own
greedy continuation, then B's - and at every later position the blend's next-token distribution
(temperature 0.6) is placed on the line from pure B to pure A:
  c = (p_blend - p_B).(p_A - p_B) / |p_A - p_B|^2      1 = reads like pure A, 0 = like pure B
c at the candidate the blend followed and staying there = the losing candidate's leftover is ignored;
c staying in between = information from the losing candidate carries forward through the KV cache.
Forcing identical text removes the butterfly effect (free-running texts drift apart anyway).
Positions where pure A and pure B already predict nearly the same thing are skipped (c = None).

Control: a 4th branch feeds A blended 50/50 with C = the 3rd-ranked pre-noise candidate at the fork
(a token that fits the context, unlike the unrelated control of Exp 18 v2). It is as "weakened" as the
blend but holds no B. If the blend reads B's text like B but the control does not, B's own information
carried forward; if both do, it is only a weakened A.
Summary statistics count each fork once: vanilla and Gumbel traces can share a prefix, so a fork with
the same item, step and candidates is kept only the first time (the jsonl keeps every row).

Kaggle: load the code as for neuralese_eval.py, then  %run neuralese_kv_carryover.py
"""
import copy
import json
import random
import urllib.request

import torch

import neuralese_sweep
from neuralese_eval import DATA_URL, SUFFIX
from neuralese_r1 import OUT_DIR, TEMPERATURE, concept_token, load, make_inputs
from neuralese_sweep import CONDITIONS, ITEMS, TAG, forks_in_run

N_CONT = 64  # forced continuation length after the fork token
MIN_TV = 0.05  # skip positions where p_A and p_B differ by < 5% probability mass (total variation)
CHECK_TOL = 1e-2  # ponytail: batched vs token-by-token fp32 drift is ~1e-4; a misaligned feed is >> 1
BUCKETS = [(0, 0), (1, 8), (9, N_CONT)]  # position 0 = token right after the fork (Exp 17's next_top)
OUT = f"{OUT_DIR}/step5_kv_carryover_{TAG}.jsonl"
LAST = {}  # pre-noise candidates of the current step, recorded by spy_concept_token


def spy_concept_token(embed, logits, noise):
    """Wraps concept_token inside forks_in_run to see the fork's 3rd candidate; changes nothing."""
    out = concept_token(embed, logits, noise)
    if noise is None:  # forks_in_run's pre-noise call, made just before sweep() at a fork
        LAST["ids"], LAST["w"] = out[2][0], out[1][0]
    return out


def step(model, past, mask, **inp):
    """Feed input_ids or inputs_embeds after `past`; returns (logits per fed position, cache, mask)."""
    n = next(iter(inp.values())).shape[1]
    mask = torch.cat([mask, mask.new_ones(1, n)], dim=1)
    out = model(**inp, attention_mask=mask, past_key_values=past, use_cache=True)
    return out.logits[0], out.past_key_values, mask


@torch.no_grad()
def carryover(model, embed, past, mask, a_id, b_id):
    """Replaces neuralese_sweep.sweep at each fork. Uses no RNG, so the replay stays identical."""
    assert LAST["ids"][0] == a_id and LAST["ids"][1] == b_id, "STOP: spy out of step with the fork"
    c_id, w_c = LAST["ids"][2].item(), round(LAST["w"][2].item(), 3)
    fork = {"A": embed[a_id], "B": embed[b_id], "blend": (embed[a_id] + embed[b_id]) / 2,
            "control": (embed[a_id] + embed[c_id]) / 2}
    start = {}  # branch -> (prediction right after the fork token, cache, mask)
    for name, x in fork.items():
        lg, p, m = step(model, copy.deepcopy(past), mask, inputs_embeds=x.view(1, 1, -1))
        start[name] = (lg[-1], p, m)

    cont = {}
    for src in ("A", "B"):
        lg, p, m = start[src]
        p, ids, own = copy.deepcopy(p), [], [lg]  # src's own greedy continuation
        for _ in range(N_CONT):
            ids.append(own[-1].argmax().item())
            lg, p, m = step(model, p, m, input_ids=torch.tensor([[ids[-1]]], device=m.device))
            own.append(lg[-1])

        probs = {}
        for name, (lg0, p0, m0) in start.items():
            lg, _, _ = step(model, copy.deepcopy(p0), m0, input_ids=torch.tensor([ids], device=m0.device))
            logits = torch.cat([lg0[None], lg])  # N_CONT + 1 predictions
            if name == src:  # check: forcing the text in one pass == generating it token by token
                assert (logits - torch.stack(own)).abs().max() < CHECK_TOL, "STOP: forced feed != generation"
            probs[name] = torch.softmax(logits / TEMPERATURE, dim=-1)

        d = probs["A"] - probs["B"]
        tv = 0.5 * d.abs().sum(-1)
        dd = (d * d).sum(-1).clamp_min(1e-12)

        def line(p):  # position on the B -> A line; None where A and B barely differ
            c = ((p - probs["B"]) * d).sum(-1) / dd
            return [round(ci, 3) if ti >= MIN_TV else None for ci, ti in zip(c.tolist(), tv.tolist())]

        top = {k: v.argmax(-1) for k, v in probs.items()}
        cont[src] = {"ids": ids, "tv": [round(x, 3) for x in tv.tolist()],
                     "c": line(probs["blend"]), "c_ctrl": line(probs["control"]),
                     "top_is_A": (top["blend"] == top["A"]).tolist(), "top_is_B": (top["blend"] == top["B"]).tolist()}
    return {"cont": cont, "next_top": [start["blend"][0].argmax().item()], "C": c_id, "wC": w_c}


def summarize(rows):
    seen, unique = set(), []
    for r in rows:
        if (r["item"], r["step"], r["A"], r["B"]) not in seen:
            seen.add((r["item"], r["step"], r["A"], r["B"]))
            unique.append(r)
    print(f"\n{len(rows) - len(unique)} duplicate forks (vanilla/Gumbel shared prefix) dropped from the summary")
    rows = unique

    def group(r):
        num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
        return "num-num" if all(num) else "num-word" if any(num) else "word-word"

    def follows(r):  # which candidate the blend's next token followed (position 0)
        a, b = r["cont"]["A"]["top_is_A"][0], r["cont"]["A"]["top_is_B"][0]
        return "A" if a and not b else "B" if b and not a else "tie/new"

    def mean(v):
        return sum(v) / len(v)

    print(f"\n{len(rows)} forks. Mean c, blend/control (1 = reads like pure A, 0 = like pure B) per position bucket")
    for g in ("num-num", "num-word", "word-word"):
        for f in ("A", "B", "tie/new"):
            rs = [r for r in rows if group(r) == g and follows(r) == f]
            for src in ("A", "B") if rs else ():
                cells = []
                for lo, hi in BUCKETS:
                    x = [r["cont"][src] for r in rs]
                    v = [c for r in x for c in r["c"][lo:hi + 1] if c is not None]
                    vc = [c for r in x for c in r["c_ctrl"][lo:hi + 1] if c is not None]
                    cells.append(f"pos {lo}-{hi}: {mean(v):.2f}/{mean(vc):.2f} (n={len(v)})" if v else f"pos {lo}-{hi}: -")
                print(f"  {g:9s} blend followed {f:7s} n={len(rs):2d}  {src}'s text forced | " + "  ".join(cells))

    # Forks where the blend followed A (B lost), B's text forced, pos 1-64. The fork is the unit (positions
    # within a fork are not independent): per-fork mean of control - blend, bootstrap 95% CI over forks.
    print("\nB-leftover test (blend followed A, B's text forced): control - blend > 0 = B's own info carries forward")
    rng = random.Random(0)
    for g in ("num-num", "num-word", "word-word"):
        diffs = []
        for r in rows:
            x = r["cont"]["B"]
            pairs = [(b, k) for b, k in zip(x["c"][1:], x["c_ctrl"][1:]) if b is not None]
            if group(r) == g and follows(r) == "A" and pairs:
                diffs.append(mean([k - b for b, k in pairs]))
        if len(diffs) > 1:
            boot = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
            print(f"  {g:9s} n={len(diffs):2d} forks  mean {mean(diffs):+.2f}  95% CI [{boot[249]:+.2f}, {boot[9749]:+.2f}]")


def main():
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    items = {i: json.loads(lines[i]) for i in ITEMS}
    tok, model = load()
    neuralese_sweep.sweep = carryover
    neuralese_sweep.concept_token = spy_concept_token  # forks_in_run calls sweep() by name -> same forks as Exp 17
    rows = []
    with open(OUT, "w") as f:
        for i, item in items.items():
            inputs = make_inputs(tok, item["question"] + SUFFIX)
            for noise, seed in CONDITIONS:
                for r in forks_in_run(model, tok, inputs, noise, seed):
                    for c in r["cont"].values():
                        c["text"] = tok.decode(c.pop("ids"))
                    r["C"] = tok.decode([r["C"]])
                    r.update(item=i, noise=noise, seed=seed)
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    m = {(s, k): [c for c in r["cont"][s][k][1:] if c is not None]
                         for s in ("A", "B") for k in ("c", "c_ctrl")}
                    print(f"item {i} {str(noise):6s} step {r['step']:4d} {r['A']!r}/{r['B']!r} C={r['C']!r}({r['wC']}) "
                          f"next@blend={r['next_top'][0]!r}  mean c after fork, blend/control: " +
                          "  ".join(f"{s}-text={sum(m[s, 'c']) / len(m[s, 'c']):.2f}/"
                                    f"{sum(m[s, 'c_ctrl']) / len(m[s, 'c_ctrl']):.2f}" if m[s, "c"] else f"{s}-text=-"
                                    for s in ("A", "B")))
    summarize(rows)


if __name__ == "__main__":
    main()
