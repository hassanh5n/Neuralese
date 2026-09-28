"""Fast follow-up for the Gumbel seed-123 5-vs-3 candidate.

Runs only the first 12 soft steps. Step 9 here is the tenth printed soft step
from the 100-step trace, so it reproduces the observed 5/3 mixture exactly.
"""

import torch

import neuralese_harness_v2 as h


SOFT_STEPS = 12
TARGET_STEP = 9
NOISE = "gumbel"
SEED = 123


def show_top5(tok, mixture_weights, ids, final_hidden, live_logits, convention, model):
    weights, positions = torch.topk(mixture_weights[0], k=5)
    mixture = [
        (tok.decode([ids[0, pos].item()]), weight.item())
        for weight, pos in zip(weights, positions)
    ]
    lens_logits = h.final_lens_logits(model, final_hidden, convention)
    if not torch.allclose(lens_logits, live_logits, rtol=1e-4, atol=2e-5):
        raise RuntimeError("STOP: target lens no longer reconstructs the live logits.")
    lens_probs = torch.softmax(lens_logits[0], dim=-1)
    lens_weights, lens_ids = torch.topk(lens_probs, k=5)

    print(
        f"\nTarget: mixture from step {TARGET_STEP} "
        f"(printed step {TARGET_STEP + 1}), consumed at forward step {TARGET_STEP + 1}"
    )
    print(f"Entropy: {h.entropy(mixture_weights[0]).item():.3f}")
    print("Mixture:")
    for token, weight in mixture:
        print(f"  {token!r}: {weight:.3f}")
    print("Final-layer lens AFTER consuming that mixture:")
    for probability, token_id in zip(lens_weights, lens_ids):
        token = tok.decode([token_id.item()])
        print(f"  {token!r}: {probability:.3f}")


def run_time_aligned_target(model, tok, inputs):
    """Return the state after the model consumes mixture TARGET_STEP."""
    torch.manual_seed(SEED)
    embed_matrix = model.get_input_embeddings().weight
    past = None
    mask = inputs["attention_mask"]
    cur_ids = inputs["input_ids"]
    cur_embeds = None
    incoming = None

    for step in range(SOFT_STEPS):
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

        if (step + 1) % 5 == 0 or step == TARGET_STEP + 1:
            print(f"  Gumbel seed=123: forward step {step + 1}/{SOFT_STEPS}", flush=True)

        final_hidden = out.hidden_states[-1][:, -1, :].detach().clone()
        if incoming is not None and incoming["source_step"] == TARGET_STEP:
            return incoming, final_hidden, out.logits[:, -1, :].detach().clone()

        past = out.past_key_values
        blend, weights, ids = h.soft_thinking_step(embed_matrix, out.logits[:, -1, :], noise=NOISE)
        incoming = {"source_step": step, "weights": weights, "ids": ids}
        cur_embeds = blend.unsqueeze(1)
        mask = torch.cat(
            [mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)],
            dim=1,
        )

    raise RuntimeError("The target mixture was not consumed within the requested trace.")


def main():
    print(f"Loading {h.MODEL_NAME} ...", flush=True)
    tok, model = h.load_model()
    inputs = h.make_inputs(tok)

    # Validate the final-layer convention against all 12 vanilla positions.
    # This is the guard that prevents the old double-normalization mistake.
    h.LENS_GATE_STEPS = tuple(range(SOFT_STEPS))
    print("Running 12-step vanilla validation ...", flush=True)
    _, vanilla_records, _, _ = h.generate_soft_then_hard(
        model,
        tok,
        inputs,
        noise=None,
        seed=0,
        soft_steps=SOFT_STEPS,
        hard_steps=0,
        collect_lens_gate=True,
        progress_label="Vanilla",
    )
    convention = h.validate_final_lens(model, vanilla_records)

    print("Running time-aligned 12-step Gumbel seed-123 target trace ...", flush=True)
    incoming, final_hidden, live_logits = run_time_aligned_target(model, tok, inputs)
    show_top5(
        tok,
        incoming["weights"],
        incoming["ids"],
        final_hidden,
        live_logits,
        convention,
        model,
    )


if __name__ == "__main__":
    main()
