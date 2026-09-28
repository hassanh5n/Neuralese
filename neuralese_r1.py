import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuralese_harness_v2 import entropy, generate_baseline, sanity_generate

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPT = (
    "You have a 3-liter jug and a 5-liter jug, both empty, and unlimited water. "
    "You need to end up with exactly 4 liters in one of the jugs. "
    "What is the first thing you do?"
)
OUT_DIR = "/content/drive/MyDrive/neuralese_runs"

# Soft Thinking repo defaults (UCSB-AI/Soft-Thinking README run command).
TEMPERATURE, MAX_TOPK, TOP_P, MIN_P = 0.6, 10, 0.95, 0.001
COLD_STOP_ENTROPY, COLD_STOP_LEN = 0.01, 256
GUMBEL_TAU, DIRICHLET_GAMMA = 0.5, 1.0
MAX_THINK, MAX_ANSWER = 4096, 512
LENS_TOL = 1e-3  # ponytail: GPU fp32 GEMM drift is ~1e-5..1e-4; the wrong convention is ~10
CONDITIONS = [(None, 0), ("gumbel", 0), ("gumbel", 1), ("gumbel", 2)]


def concept_token(embed, logits, noise):
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    w, ids = torch.topk(probs, MAX_TOPK, dim=-1)
    w = w * (((w.cumsum(-1) - w) <= TOP_P) & (w >= MIN_P * w[:, :1]))
    w = w / w.sum(-1, keepdim=True)
    # ponytail: Cold Stop watches pre-noise entropy (model confidence); check vs repo if stops look off
    h = entropy(w[0]).item()
    if noise == "gumbel":
        w = F.gumbel_softmax(w.clamp_min(1e-12).log(), tau=GUMBEL_TAU, dim=-1)
    elif noise == "dirichlet":
        w = torch.distributions.Dirichlet((w * DIRICHLET_GAMMA).clamp_min(1e-6)).sample()
    return (embed[ids] * w.unsqueeze(-1)).sum(1), w, ids, h


@torch.no_grad()
def run(model, tok, inputs, noise, seed, gate_steps=()):
    """Soft phase inside <think> until </think>/EOS, Cold Stop, or budget; then greedy answer."""
    torch.manual_seed(seed)
    embed = model.get_input_embeddings().weight
    think_end, eos = tok.convert_tokens_to_ids("</think>"), tok.eos_token_id
    mask, past, feed = inputs["attention_mask"], None, {"input_ids": inputs["input_ids"]}
    steps, gate, low, reason = [], [], 0, "budget"

    for step in range(MAX_THINK):
        out = model(**feed, attention_mask=mask, past_key_values=past, use_cache=True,
                    output_hidden_states=step in gate_steps)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        if step in gate_steps:
            gate.append((out.hidden_states[-1][:, -1, :], logits))
        blend, w, ids, h = concept_token(embed, logits, noise)
        top = ids[0, w[0].argmax()].item()
        steps.append({"ids": ids[0].tolist(), "w": [round(x, 4) for x in w[0].tolist()],
                      "H": round(h, 4), "raw_argmax": logits.argmax().item()})
        low = low + 1 if h < COLD_STOP_ENTROPY else 0
        if top in (think_end, eos):
            reason = "think_end" if top == think_end else "eos_in_think"
            break
        if low >= COLD_STOP_LEN:
            reason = "cold_stop"
            break
        feed = {"inputs_embeds": blend.unsqueeze(1)}
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)

    # Every exit feeds a real </think>, then plain greedy decoding for the answer.
    ids, answer = torch.tensor([[think_end]], device=mask.device), []
    for _ in range(MAX_ANSWER):
        mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
        out = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
        past, ids = out.past_key_values, out.logits[:, -1:, :].argmax(-1)
        if ids.item() == eos:
            break
        answer.append(ids.item())

    return {"noise": noise, "seed": seed, "stop_reason": reason, "think_steps": len(steps),
            "answer": tok.decode(answer), "steps": steps}, gate


@torch.no_grad()
def lens_gate(model, gate):
    err = {name: max((model.lm_head(f(h)) - lg).abs().max().item() for h, lg in gate)
           for name, f in (("already_normalized", lambda x: x), ("pre_norm", model.model.norm))}
    top1 = all(model.lm_head(h).argmax() == lg.argmax() for h, lg in gate)
    print(f"LENS GATE: {err}, top1 all match={top1}")
    assert err["already_normalized"] < LENS_TOL and err["pre_norm"] > 1 and top1, \
        "STOP: final hidden state does not reconstruct live logits"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda").eval()
    assert len(tok.encode("</think>", add_special_tokens=False)) == 1, "</think> is not one token"

    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                   add_generation_prompt=True, tokenize=False)
    if not text.rstrip().endswith("<think>"):  # older template versions omit it
        text += "<think>\n"
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    # Gate 1: hand-rolled greedy loop == model.generate().
    expected = sanity_generate(tok, model, inputs, max_new_tokens=32)
    got = tok.decode(generate_baseline(model, inputs["input_ids"], inputs["attention_mask"], 32,
                                       {tok.eos_token_id})[0], skip_special_tokens=True)
    assert got.strip() == expected.strip(), "STOP: manual greedy != generate()"
    print("GREEDY GATE: pass")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    config = {k: globals()[k] for k in ("MODEL", "PROMPT", "TEMPERATURE", "MAX_TOPK", "TOP_P", "MIN_P",
                                        "COLD_STOP_ENTROPY", "COLD_STOP_LEN", "GUMBEL_TAU",
                                        "DIRICHLET_GAMMA", "MAX_THINK", "MAX_ANSWER")}
    for noise, seed in CONDITIONS:
        t = time.time()
        record, gate = run(model, tok, inputs, noise, seed, gate_steps=range(8) if noise is None else ())
        if noise is None:
            lens_gate(model, gate)  # Gate 2
            assert all(s["ids"][s["w"].index(max(s["w"]))] == s["raw_argmax"] for s in record["steps"]), \
                "STOP: vanilla mixture top-1 != raw argmax"
        path = f"{OUT_DIR}/{stamp}_{noise}_{seed}.json"
        with open(path, "w") as f:
            json.dump({"config": config, **record}, f)
        print(f"{noise} seed={seed}: {record['stop_reason']} after {record['think_steps']} steps, "
              f"{time.time() - t:.0f}s -> {path}\n  answer: {record['answer'][-200:]!r}")


if __name__ == "__main__":
    main()