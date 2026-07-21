import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
#PROMPT = "43 * 34 = ?"
#PROMPT = "A train leaves at 3pm going 60mph. Another leaves the same station at 4pm going 90mph in the same direction. What time does the second train catch the first?"
PROMPT = "You have a 3-liter jug and a 5-liter jug, both empty, and unlimited water. You need to end up with exactly 4 liters in one of the jugs. What is the first thing you do?"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(4)
    print(model.generation_config)
    return tok, model

def sanity_generate(tok, model, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    return tok.decode(out[0], skip_special_tokens=True), inputs

def generate_baseline(model, input_ids, attention_mask, max_new_tokens=64, eos_ids=None):
    generated = input_ids
    cur_mask = attention_mask
    past = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=generated, attention_mask=cur_mask, use_cache=True)
            else:
                out = model(input_ids=generated[:, -1:], attention_mask=cur_mask,
                             past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        if eos_ids is not None and next_id.item() in eos_ids:
            break
    return generated


def soft_thinking_step(embed_matrix, logits, top_k=15, top_p=0.95):
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
    sorted_probs, order = torch.sort(topk_probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumulative - sorted_probs) <= top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    kept_idx = torch.gather(topk_idx, -1, order)
    step_embeds = embed_matrix[kept_idx]
    blended = (step_embeds * sorted_probs.unsqueeze(-1)).sum(dim=1) 
    return blended, sorted_probs, kept_idx

def generate_soft_then_hard(model, tok, input_ids, attention_mask,
                             soft_steps=40, hard_steps=40,
                             top_k=15, top_p=0.95, eos_ids=None):
    embed_matrix = model.get_input_embeddings().weight
    past = None
    cur_mask = attention_mask
    step_records = []

    # prefill + soft phase
    cur_input_ids = input_ids
    cur_embeds = None
    for _ in range(soft_steps):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=cur_input_ids, attention_mask=cur_mask, use_cache=True)
            else:
                out = model(inputs_embeds=cur_embeds, attention_mask=cur_mask,
                            past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        blended, weights, idx = soft_thinking_step(embed_matrix, logits, top_k, top_p)

        step_records.append({"weights": weights.detach().clone(),
                            "idx": idx.detach().clone(),
                            "blended": blended.detach().clone()})
        print(f"  soft step {len(step_records)-1}: top_token={tok.decode([idx[0,0].item()])!r}  weight={weights[0,0].item():.3f}")
        cur_embeds = blended.unsqueeze(1) 
        cur_mask = torch.cat([cur_mask, torch.ones((cur_mask.shape[0], 1))], dim=1)

        if idx[0, 0].item() in eos_ids and weights[0, 0].item() > 0.5:
            break

    # bridging step: feed last blended embed, get first hard logits
    with torch.no_grad():
        out = model(inputs_embeds=cur_embeds, attention_mask=cur_mask,
                     past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    hard_generated = next_id
    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)

    # hard phase: normal discrete decoding for the actual answer
    for _ in range(hard_steps):
        with torch.no_grad():
            out = model(input_ids=next_id, attention_mask=cur_mask,
                         past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        hard_generated = torch.cat([hard_generated, next_id], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        if eos_ids is not None and next_id.item() in eos_ids:
            break

    return hard_generated, step_records


def main():
    print(f"Loading {MODEL_NAME} ...")
    tok, model = load_model()

    print("\n--- Step 3: sanity generate() ---")
    sanity_text, inputs = sanity_generate(tok, model, PROMPT)
    print(sanity_text)

    print("\n--- Step 4: manual baseline loop (must match Step 3 exactly) ---")
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    baseline_ids = generate_baseline(model, inputs["input_ids"], inputs["attention_mask"],
                                      max_new_tokens=64, eos_ids=eos_ids)
    baseline_text = tok.decode(baseline_ids[0], skip_special_tokens=True)
    print(baseline_text)

    match = baseline_text.strip() == sanity_text.strip()
    print(f"\nCHECKPOINT 1 — baseline matches .generate(): {match}")
    if not match:
        print("STOP: fix the manual loop before continuing (see attention_mask / cache handling).")
        return

    print("\n--- Steps 5-6: soft-thinking generation ---")
    hard_ids, records = generate_soft_then_hard(model, tok, inputs["input_ids"], inputs["attention_mask"],
                                             soft_steps=40, hard_steps=40, eos_ids=eos_ids)
    print(tok.decode(hard_ids[0], skip_special_tokens=True))

    print("\n--- Step 7: diagnostics — is the soft phase actually blending? ---")
    for i, rec in enumerate(records):
        top_w = rec["weights"][0].max().item()
        top_tok = tok.decode([rec["idx"][0, 0].item()])
        print(f"step {i:2d}  top_weight={top_w:.3f}  top_token={top_tok!r}")


if __name__ == "__main__":
    main()