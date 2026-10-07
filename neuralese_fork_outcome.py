"""Step 5 / Test 2: does the losing candidate stored in the KV cache ever change the final answer?

At each number fork of Exp 17/18 (same replay, same forks, duplicates skipped), copies continue from
the same cache with Gumbel Soft Thinking up to the boxed answer (mirrors neuralese_r1.run):
  pure A, pure B, blend = A+B (50/50), control = A+C (C = 3rd-ranked pre-noise candidate, as Exp 18 v3).
All copies use the same noise seed, so any difference comes only from what was fed at the fork.
Pure A and B run first; blend and control run only when A and B reach different answers - otherwise
the fork cannot tell "follows A" from "follows B" (saves ~half the GPU time).
Question: does the blend end on B's answer more often than the control does? (control = how often a
small nudge alone flips the answer, i.e. the butterfly effect)

Kaggle: load the code as for neuralese_eval.py, then  %run neuralese_fork_outcome.py   (~3-4 h)
"""
import copy
import json
import re
import time
import urllib.request
from math import comb

import torch

import neuralese_sweep
from neuralese_eval import DATA_URL, SUFFIX, boxed
from neuralese_kv_carryover import LAST, spy_concept_token
from neuralese_r1 import (COLD_STOP_ENTROPY, COLD_STOP_LEN, MAX_ANSWER, MAX_THINK, OUT_DIR, concept_token,
                          load, make_inputs)
from neuralese_sweep import CONDITIONS, ITEMS, TAG, forks_in_run

NOISE, SEED = "gumbel", 0  # same seed for every copy at every fork
OUT = f"{OUT_DIR}/step5_fork_outcome_{TAG}.jsonl"
STATE = {"seen": set()}  # current item / tokenizer / prompt length, set in main()


def answer_num(text):
    """Last number inside the last \\boxed{}; None if no boxed number (same parsing as neuralese_eval.score)."""
    b = boxed(text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", b.replace(",", "")) if b else []
    return float(nums[-1]) if nums else None


@torch.no_grad()
def finish(model, tok, embed, past, mask, x, steps_left):
    """From a fork: feed x, Gumbel Soft Thinking until </think>/EOS, Cold Stop or budget; then greedy answer."""
    think_end, eos = tok.convert_tokens_to_ids("</think>"), tok.eos_token_id
    past, feed, low, reason, n = copy.deepcopy(past), {"inputs_embeds": x.view(1, 1, -1)}, 0, "budget", 0
    # ponytail: Cold Stop counter restarts at the fork (replay's count is not passed in); matters only in loops
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):  # leaves the replay's RNG untouched
        torch.manual_seed(SEED)
        for n in range(steps_left):
            mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            out = model(**feed, attention_mask=mask, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            blend, w, ids, h = concept_token(embed, logits, NOISE)
            low = low + 1 if h < COLD_STOP_ENTROPY else 0
            top = ids[0, w[0].argmax()].item()
            if top in (think_end, eos):
                reason = "think_end" if top == think_end else "eos_in_think"
                break
            if low >= COLD_STOP_LEN:
                reason = "cold_stop"
                break
            feed = {"inputs_embeds": blend.unsqueeze(1)}

    ids, answer = torch.tensor([[think_end]], device=mask.device), []
    for _ in range(MAX_ANSWER):
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
        out = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
        past, ids = out.past_key_values, out.logits[:, -1:, :].argmax(-1)
        if ids.item() == eos:
            break
        answer.append(ids.item())
    text = tok.decode(answer)
    return {"stop": reason, "think_steps": n + 1, "num": answer_num(text), "answer_tail": text[-300:]}


def outcome(model, embed, past, mask, a_id, b_id):
    """Replaces neuralese_sweep.sweep at each fork; non-number and duplicate forks are skipped."""
    tok = STATE["tok"]
    key = (STATE["item"], mask.shape[1], a_id, b_id)  # vanilla/Gumbel traces can share a prefix
    if not any(ch.isdigit() for ch in tok.decode([a_id]) + tok.decode([b_id])) or key in STATE["seen"]:
        return {"next_top": [], "skip": True}
    STATE["seen"].add(key)
    assert LAST["ids"][0] == a_id and LAST["ids"][1] == b_id, "STOP: spy out of step with the fork"
    c_id = LAST["ids"][2].item()
    feeds = {"A": embed[a_id], "B": embed[b_id], "blend": (embed[a_id] + embed[b_id]) / 2,
             "control": (embed[a_id] + embed[c_id]) / 2}
    steps_left = MAX_THINK - (mask.shape[1] - STATE["prompt_len"])
    t = time.time()
    res = {k: finish(model, tok, embed, past, mask, feeds[k], steps_left) for k in ("A", "B")}
    informative = res["A"]["num"] != res["B"]["num"]
    if informative:
        res.update({k: finish(model, tok, embed, past, mask, feeds[k], steps_left) for k in ("blend", "control")})
    for r in res.values():
        r["correct"] = r["num"] is not None and abs(r["num"] - STATE["gold"]) < 1e-6
    return {"next_top": [], "branches": res, "informative": informative, "C": tok.decode([c_id]),
            "secs": round(time.time() - t)}


def summarize(rows):
    def group(r):
        num = [any(ch.isdigit() for ch in s) for s in (r["A"], r["B"])]
        return "num-num" if all(num) else "num-word"

    inf = [r for r in rows if r["informative"]]
    print(f"\n{len(rows)} number forks; {len(inf)} informative (pure A and pure B reach different answers): "
          + ", ".join(f"{g} {sum(group(r) == g for r in inf)}/{sum(group(r) == g for r in rows)}"
                      for g in ("num-num", "num-word")))
    for name in ("blend", "control"):
        a = sum(r["branches"][name]["num"] == r["branches"]["A"]["num"] for r in inf)
        b = sum(r["branches"][name]["num"] == r["branches"]["B"]["num"] for r in inf)
        print(f"  {name:7s} ends on A's answer {a}, B's answer {b}, other {len(inf) - a - b}")
    # Paired: forks where only the blend lands on B vs only the control does; one-sided exact sign test.
    k = sum(r["branches"]["blend"]["num"] == r["branches"]["B"]["num"] != r["branches"]["control"]["num"] for r in inf)
    m = sum(r["branches"]["control"]["num"] == r["branches"]["B"]["num"] != r["branches"]["blend"]["num"] for r in inf)
    p = sum(comb(k + m, i) for i in range(k, k + m + 1)) / 2 ** (k + m) if k + m else float("nan")
    print(f"  only blend -> B: {k}, only control -> B: {m}, one-sided sign test p = {p:.3f}")
    for name in ("A", "B", "blend", "control"):
        rs = [r for r in rows if name in r["branches"]]
        print(f"  accuracy {name:7s} {sum(r['branches'][name]['correct'] for r in rs)}/{len(rs)}")


def main():
    assert answer_num(r"so \boxed{\$1,250.00}") == 1250 and answer_num("no box 18") is None
    lines = urllib.request.urlopen(DATA_URL).read().decode().splitlines()
    items = {i: json.loads(lines[i]) for i in ITEMS}
    tok, model = load()
    STATE["tok"] = tok
    neuralese_sweep.sweep = outcome  # forks_in_run calls sweep() by name -> same forks as Exp 17/18
    neuralese_sweep.concept_token = spy_concept_token  # exposes the 3rd candidate, changes nothing
    rows = []
    with open(OUT, "w") as f:
        for i, item in items.items():
            inputs = make_inputs(tok, item["question"] + SUFFIX)
            STATE.update(item=i, prompt_len=inputs["input_ids"].shape[1],
                         gold=float(item["answer"].split("####")[-1].strip().replace(",", "")))
            for noise, seed in CONDITIONS:
                for r in forks_in_run(model, tok, inputs, noise, seed):
                    if r.get("skip"):
                        continue
                    r.update(item=i, noise=noise, seed=seed)
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()  # partial results survive if the run is stopped
                    nums = {k: v["num"] for k, v in r["branches"].items()}
                    print(f"item {i} {str(noise):6s} step {r['step']:4d} {r['A']!r}/{r['B']!r} C={r['C']!r} "
                          f"answers {nums} gold={STATE['gold']:g} {r['secs']}s", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
