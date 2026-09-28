"""Validated Soft Thinking harness for Qwen2.5-1.5B.

This intentionally keeps the earlier harness untouched.  It adds two hard
requirements before interpreting a run: final-layer lens reconstruction must
match the model's live logits, and activation experiments overwrite a decoder
layer output rather than merely replacing the next input embedding.
"""

import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = (
    "You have a 3-liter jug and a 5-liter jug, both empty, and unlimited water. "
    "You need to end up with exactly 4 liters in one of the jugs. "
    "What is the first thing you do?"
)

TOP_K = 15
TOP_P = 0.95
SOFT_STEPS = 100
HARD_STEPS = 60
SANITY_NEW_TOKENS = 4
SMOKE_SOFT_STEPS = 2
SMOKE_HARD_STEPS = 2
SMOKE_TEST_ONLY = False  # set True to run a quick smoke test and exit
PROGRESS_EVERY = 5
GUMBEL_TAU = 0.5
DIRICHLET_GAMMA = 1.0
SEEDS = (0, 1, 2, 42, 123)
RANDOM_STEPS_PER_RUN = 12
LENS_GATE_STEPS = (0, 4, 11, 21, 63, 71, 94, 99)
# CPU-friendly default: run the two remaining Gumbel seeds with visible
# progress. The earlier full 10-condition sweep is still impractical here.
RUN_CONDITIONS = [("gumbel", 2), ("gumbel", 123)]

# Activation patching is deliberately opt-in.  Pick a target only after the
# fresh 1.5B trace has passed the lens gate.  The integer is the soft-mixture
# step; its embedding is consumed on the following forward pass.
ACTIVATION_PATCH_TARGET = None  # e.g. ("gumbel", 42, 80)
ACTIVATION_LAYER = -1  # final decoder layer; -1 adapts to either Qwen size
CONTROL_TOKEN = "1"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(4)
    return tok, model


def make_inputs(tok, prompt=PROMPT):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tok(text, return_tensors="pt")


def eos_ids(tok):
    return {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}


def generate_baseline(model, input_ids, attention_mask, max_new_tokens=64, stop_ids=None):
    generated, mask, past = input_ids, attention_mask, None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=generated, attention_mask=mask, use_cache=True)
            else:
                out = model(
                    input_ids=generated[:, -1:],
                    attention_mask=mask,
                    past_key_values=past,
                    use_cache=True,
                )
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
        if stop_ids and next_id.item() in stop_ids:
            break
    return generated


def sanity_generate(tok, model, inputs, max_new_tokens=SANITY_NEW_TOKENS):
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def soft_thinking_step(embed_matrix, logits, noise=None):
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, k=TOP_K, dim=-1)
    keep = (torch.cumsum(weights, dim=-1) - weights) <= TOP_P
    weights = weights * keep
    weights = weights / weights.sum(dim=-1, keepdim=True)

    if noise == "gumbel":
        weights = F.gumbel_softmax(
            torch.log(weights.clamp_min(1e-12)), tau=GUMBEL_TAU, hard=False, dim=-1
        )
    elif noise == "dirichlet":
        weights = torch.distributions.Dirichlet(
            (weights * DIRICHLET_GAMMA).clamp_min(1e-6)
        ).sample()
    elif noise is not None:
        raise ValueError("noise must be None, 'gumbel', or 'dirichlet'")

    blend = (embed_matrix[ids] * weights.unsqueeze(-1)).sum(dim=1)
    return blend, weights, ids


def entropy(weights):
    weights = weights.clamp_min(1e-12)
    return -(weights * weights.log()).sum(dim=-1)


def validate_final_lens(model, records):
    """Hard gate: return the validated convention or raise RuntimeError."""
    checked = [record for record in records if record["live_logits"] is not None]
    if not checked:
        raise RuntimeError("No records available for lens validation.")

    scores = {}
    for convention in ("already_normalized", "pre_norm"):
        checks = []
        for record in checked:
            hidden = record["final_hidden"]
            if convention == "pre_norm":
                hidden = model.model.norm(hidden)
            reconstructed = model.lm_head(hidden)
            live_logits = record["live_logits"]
            checks.append(
                {
                    "max_abs_error": (reconstructed - live_logits).abs().max().item(),
                    # The identical fp32 lm_head computation may differ by about
                    # 1e-5 when run separately on CPU because GEMM reduction
                    # order changes with the input shape. The competing
                    # pre-norm path is orders of magnitude farther away.
                    "allclose": torch.allclose(reconstructed, live_logits, rtol=1e-4, atol=2e-5),
                    "top1": reconstructed.argmax(dim=-1).item() == record["raw_argmax_id"],
                }
            )
        scores[convention] = {
            "max_abs_error": max(check["max_abs_error"] for check in checks),
            "allclose_steps": sum(check["allclose"] for check in checks),
            "top1_matches": sum(check["top1"] for check in checks),
        }

    print("\nFINAL-LAYER LENS VALIDATION")
    for convention, score in scores.items():
        print(
            f"  {convention}: max_abs_error={score['max_abs_error']:.3g}, "
            f"allclose={score['allclose_steps']}/{len(checked)}, "
            f"top1={score['top1_matches']}/{len(checked)}"
        )

    valid = [
        convention
        for convention, score in scores.items()
        if score["allclose_steps"] == len(checked)
        and score["top1_matches"] == len(checked)
    ]
    if len(valid) != 1:
        raise RuntimeError(
            "STOP: final-layer lens does not uniquely reconstruct live logits. "
            "Do not interpret lens-vs-mixture results until this is fixed."
        )
    print(f"  PASS: final hidden state is {valid[0]}.\n")
    return valid[0]


def final_lens_logits(model, hidden, convention):
    if convention == "pre_norm":
        hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


def final_lens_top(model, tok, record, convention, top_n=5):
    logits = final_lens_logits(model, record["final_hidden"], convention)
    probs = torch.softmax(logits[0], dim=-1)
    values, ids = torch.topk(probs, top_n)
    return [(tok.decode([token.item()]), token.item(), value.item()) for value, token in zip(values, ids)]


class LayerIntervention:
    """Captures or overwrites one decoder layer's final-position activation."""

    def __init__(self, donor=None):
        self.donor = donor
        self.captured = None

    def __call__(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.captured = hidden[:, -1, :].detach().clone()
        if self.donor is None:
            return output

        patched = hidden.clone()
        patched[:, -1, :] = self.donor.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched


def _layer_index(model, index):
    return index if index >= 0 else len(model.model.layers) + index


def generate_soft_then_hard(
    model,
    tok,
    inputs,
    *,
    noise=None,
    seed=0,
    soft_steps=SOFT_STEPS,
    hard_steps=HARD_STEPS,
    progress_label=None,
    input_override=None,
    capture_at=None,
    activation_patch=None,
    collect_lens_gate=False,
):
    """Generate one fixed-budget trace.

    `input_override=(mixture_step, token_id)` makes a donor run by replacing
    the blend created at that step.  The replacement is consumed on forward
    pass `mixture_step + 1`.

    `capture_at=(forward_step, layer)` captures that decoder layer output.
    `activation_patch=(forward_step, layer, donor_activation)` is the genuine
    causal intervention: it overwrites a layer output during the target
    forward pass while retaining the original blended input.
    """
    torch.manual_seed(seed)
    embed_matrix = model.get_input_embeddings().weight
    stop_ids = eos_ids(tok)
    past, mask = None, inputs["attention_mask"]
    cur_ids, cur_embeds = inputs["input_ids"], None
    records, eos_fire_step, captured = [], None, None

    for step in range(soft_steps):
        intervention = None
        if capture_at and step == capture_at[0]:
            intervention = LayerIntervention()
            layer = _layer_index(model, capture_at[1])
        elif activation_patch and step == activation_patch[0]:
            intervention = LayerIntervention(activation_patch[2])
            layer = _layer_index(model, activation_patch[1])
        else:
            layer = None

        handle = None
        if intervention is not None:
            handle = model.model.layers[layer].register_forward_hook(intervention)
        try:
            with torch.no_grad():
                if past is None:
                    out = model(
                        input_ids=cur_ids,
                        attention_mask=mask,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                else:
                    out = model(
                        inputs_embeds=cur_embeds,
                        attention_mask=mask,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
        finally:
            if handle is not None:
                handle.remove()
        if intervention is not None:
            captured = intervention.captured

        logits = out.logits[:, -1, :]
        past = out.past_key_values
        final_hidden = out.hidden_states[-1][:, -1, :].detach().clone()
        raw_argmax_id = logits.argmax(dim=-1).item()
        blend, weights, ids = soft_thinking_step(embed_matrix, logits, noise=noise)
        top_pos = weights[0].argmax().item()
        top_id = ids[0, top_pos].item()

        if eos_fire_step is None and top_id in stop_ids:
            eos_fire_step = step
        is_void = eos_fire_step is not None and step > eos_fire_step
        records.append(
            {
                "weights": weights.detach().clone(),
                "ids": ids.detach().clone(),
                "entropy": entropy(weights[0]).item(),
                "void": is_void,
                "raw_argmax_id": raw_argmax_id,
                "final_hidden": final_hidden,
                "live_logits": logits.detach().clone() if collect_lens_gate and step in LENS_GATE_STEPS else None,
            }
        )

        if input_override and step == input_override[0]:
            cur_embeds = embed_matrix[input_override[1]].view(1, 1, -1)
        else:
            cur_embeds = blend.unsqueeze(1)
        mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)], dim=1)

        if progress_label and ((step + 1) % PROGRESS_EVERY == 0 or step + 1 == soft_steps):
            values, positions = torch.topk(weights[0], k=min(3, weights.shape[1]))
            mixture = ", ".join(
                f"{tok.decode([ids[0, pos].item()])!r}:{value.item():.3f}"
                for value, pos in zip(values, positions)
            )
            print(
                f"  {progress_label}: soft step {step + 1}/{soft_steps} | "
                f"H={entropy(weights[0]).item():.3f} | {mixture}",
                flush=True,
            )

    with torch.no_grad():
        out = model(
            inputs_embeds=cur_embeds,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = next_id
    mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)

    for step in range(hard_steps):
        with torch.no_grad():
            out = model(
                input_ids=next_id,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
        if next_id.item() in stop_ids:
            break

        if progress_label and ((step + 1) % PROGRESS_EVERY == 0 or step + 1 == hard_steps):
            print(
                f"  {progress_label}: hard step {step + 1}/{hard_steps} | "
                f"token={tok.decode([next_id.item()])!r}",
                flush=True,
            )

    return tok.decode(generated[0], skip_special_tokens=True), records, eos_fire_step, captured


def random_nonvoid_steps(records, n=RANDOM_STEPS_PER_RUN, seed=20260918):
    valid = [index for index, record in enumerate(records) if not record["void"]]
    return sorted(random.Random(seed).sample(valid, min(n, len(valid))))


def report_random_lens_sample(model, tok, records, convention, *, label):
    steps = random_nonvoid_steps(records)
    print(f"\n{label}: preregistered random steps {steps}")
    disagreements = 0
    for step in steps:
        record = records[step]
        mix_pos = record["weights"][0].argmax().item()
        mix_id = record["ids"][0, mix_pos].item()
        lens = final_lens_top(model, tok, record, convention, top_n=1)[0]
        same = mix_id == lens[1]
        disagreements += not same
        print(
            f"  step {step:2d}: mixture={tok.decode([mix_id])!r} "
            f"({record['weights'][0, mix_pos].item():.3f}), "
            f"lens={lens[0]!r} ({lens[2]:.3f}) {'=' if same else '!='}"
        )
    print(f"  disagreement: {disagreements}/{len(steps)}")


def _single_token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one tokenizer token: {ids}")
    return ids[0]


def run_activation_patch_suite(model, tok, inputs, convention, noise, seed, mixture_step):
    """Run true layer-output patching after a validated fresh 1.5B trace."""
    base_text, base_records, _, _ = generate_soft_then_hard(
        model, tok, inputs, noise=noise, seed=seed
    )
    record = base_records[mixture_step]
    if record["void"]:
        raise ValueError("Selected mixture step is post-EOS and cannot be patched.")

    mix_pos = record["weights"][0].argmax().item()
    mix_id = record["ids"][0, mix_pos].item()
    lens_id = final_lens_top(model, tok, record, convention, top_n=1)[0][1]
    control_id = _single_token_id(tok, CONTROL_TOKEN)
    forward_step = mixture_step + 1
    layer = _layer_index(model, ACTIVATION_LAYER)

    print(
        f"\nACTIVATION PATCH: mixture step {mixture_step}; forward step {forward_step}; "
        f"decoder layer {layer}"
    )
    print(f"  baseline: {base_text!r}")

    for label, token_id in (("lens", lens_id), ("mixture", mix_id), ("control", control_id)):
        # First obtain a donor activation from an otherwise identical run that
        # changes only the selected incoming embedding.
        _, _, _, donor = generate_soft_then_hard(
            model,
            tok,
            inputs,
            noise=noise,
            seed=seed,
            input_override=(mixture_step, token_id),
            capture_at=(forward_step, ACTIVATION_LAYER),
        )
        if donor is None:
            raise RuntimeError("Donor activation was not captured.")

        # Then preserve the original blend and overwrite only the selected
        # decoder-layer activation during the matching forward pass.
        patched_text, _, _, _ = generate_soft_then_hard(
            model,
            tok,
            inputs,
            noise=noise,
            seed=seed,
            activation_patch=(forward_step, ACTIVATION_LAYER, donor),
        )
        token = tok.decode([token_id])
        print(f"  {label:8s} donor={token!r}: changed={patched_text != base_text}")
        print(f"    {patched_text!r}")


def main():
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tok, model = load_model()
    inputs = make_inputs(tok)

    print(f"Running {SANITY_NEW_TOKENS}-token greedy sanity check ...", flush=True)
    expected = sanity_generate(tok, model, inputs)
    baseline = tok.decode(
        generate_baseline(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            max_new_tokens=SANITY_NEW_TOKENS,
            stop_ids=eos_ids(tok),
        )[0],
        skip_special_tokens=True,
    )
    if baseline.strip() != expected.strip():
        raise RuntimeError("STOP: manual greedy baseline does not match model.generate().")
    print("CHECKPOINT 1 PASS: manual greedy decode matches model.generate().")

    if SMOKE_TEST_ONLY:
        print(
            f"Running {SMOKE_SOFT_STEPS}-step Soft Thinking smoke test ...",
            flush=True,
        )
        generate_soft_then_hard(
            model,
            tok,
            inputs,
            noise="gumbel",
            seed=42,
            soft_steps=SMOKE_SOFT_STEPS,
            hard_steps=SMOKE_HARD_STEPS,
            progress_label="Smoke test",
        )
        print(
            "SMOKE TEST PASS. Set SMOKE_TEST_ONLY = False to run the full study.",
            flush=True,
        )
        return

    # The first 1.5B vanilla trace is the lens gate.  No result is reported
    # as a logit-lens finding until this validates exactly.
    _, vanilla_records, _, _ = generate_soft_then_hard(
        model,
        tok,
        inputs,
        noise=None,
        seed=0,
        collect_lens_gate=True,
        progress_label="Vanilla lens gate",
    )
    convention = validate_final_lens(model, vanilla_records)
    report_random_lens_sample(model, tok, vanilla_records, convention, label="VANILLA")

    # Pre-specified random samples avoid selecting only visually interesting
    # entropy dips. Run one listed condition at a time on CPU.
    for noise, seed in RUN_CONDITIONS:
        _, records, eos_step, _ = generate_soft_then_hard(
            model,
            tok,
            inputs,
            noise=noise,
            seed=seed,
            progress_label=f"{noise} seed={seed}",
        )
        print(f"{noise}, seed={seed}: eos_fire_step={eos_step}")
        report_random_lens_sample(
            model,
            tok,
            records,
            convention,
            label=f"{noise.upper()} seed={seed}",
        )

    if ACTIVATION_PATCH_TARGET is not None:
        noise, seed, mixture_step = ACTIVATION_PATCH_TARGET
        run_activation_patch_suite(model, tok, inputs, convention, noise, seed, mixture_step)


if __name__ == "__main__":
    main()
