import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
#PROMPT = "43 * 34 = ?"
#PROMPT = "A train leaves at 3pm going 60mph. Another leaves the same station at 4pm going 90mph in the same direction. What time does the second train catch the first?"
PROMPT = "You have a 3-liter jug and a 5-liter jug, both empty, and unlimited water. You need to end up with exactly 4 liters in one of the jugs. What is the first thing you do?"

# --- noise settings (Step 1, added 2026-08-24) ---
# NOISE_MODE: None = vanilla (old behavior, unchanged) / "gumbel" / "dirichlet"
NOISE_MODE = "gumbel"
GUMBEL_TAU = 0.5
DIRICHLET_GAMMA = 1.0
SEED = 42

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


def soft_thinking_step(embed_matrix, logits, top_k=15, top_p=0.95,
                        noise=None, gumbel_tau=0.5, dirichlet_gamma=1.0):
    probs = torch.softmax(logits, dim=-1)
    # topk already returns values sorted descending, so no secondary sort needed.
    sorted_probs, kept_idx = torch.topk(probs, k=top_k, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumulative - sorted_probs) <= top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # --- noise injection: fixes the Greedy Pitfall (Wu et al., arXiv:2508.03440) ---
    # kept_idx (which tokens) stays fixed. Only the weights change below.
    # So after this, "sorted_probs" is no longer strictly sorted high-to-low.
    if noise == "gumbel":
        # y = softmax((log(pi) + gumbel_noise) / tau)
        log_probs = torch.log(sorted_probs.clamp_min(1e-12))
        sorted_probs = F.gumbel_softmax(log_probs, tau=gumbel_tau, hard=False, dim=-1)
    elif noise == "dirichlet":
        # sample a fresh mixture from Dir(gamma * pi)
        concentration = (sorted_probs * dirichlet_gamma).clamp_min(1e-6)
        sorted_probs = torch.distributions.Dirichlet(concentration).sample()
    elif noise is not None:
        raise ValueError(f"unknown noise mode: {noise!r} (use None / 'gumbel' / 'dirichlet')")

    step_embeds = embed_matrix[kept_idx]
    blended = (step_embeds * sorted_probs.unsqueeze(-1)).sum(dim=1)
    return blended, sorted_probs, kept_idx


def compute_entropy(weights):
    """H(p) = -sum(p_i * log(p_i)), natural log, over the last dim.
    Call on the same (post-noise, renormalized) weights already being logged per step."""
    p = weights.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


# ======================================================================
# Next Step 4 (2026-08-29): logit lens + hidden-state capture
# ======================================================================

def logit_lens(model, hidden_vector, tok, top_n=5, apply_norm=True):
    """
    Projects a hidden state through the model's final norm + unembedding,
    early -- the same operation the model applies for real only at the very
    last layer. Doing it on an earlier/intermediate hidden state gives a
    *guess* at what the model would predict if it stopped right there.

    hidden_vector: [hidden_dim] or [1, hidden_dim]
    apply_norm: whether to run model.model.norm first. See
    check_lens_checkpoint() below -- whether this is needed for the LAST
    captured layer specifically depends on the transformers version, so it
    must be checked empirically, not assumed. Earlier (non-final) layers
    always need apply_norm=True, since they were never normed by the model.
    """
    if hidden_vector.dim() == 1:
        hidden_vector = hidden_vector.unsqueeze(0)
    with torch.no_grad():
        h = model.model.norm(hidden_vector) if apply_norm else hidden_vector
        logits = model.lm_head(h)
        probs = torch.softmax(logits[0], dim=-1)
    top_probs, top_idx = torch.topk(probs, k=top_n)
    return [(tok.decode([i.item()]), p.item()) for i, p in zip(top_idx, top_probs)]


def check_lens_checkpoint(model, tok, hidden_states_last_layer, actual_argmax_id):
    """
    CHECKPOINT 2 (same role as the byte-for-byte .generate() checkpoint
    earlier in this file). Confirms whether the last entry of a captured
    hidden_states tuple is pre- or post-final-norm for THIS model /
    transformers version, by checking which setting reproduces the model's
    own real top-1 choice at that step exactly. Run once -- this is a fixed
    property of the library/model, not something that varies per experiment.
    """
    actual_token = tok.decode([actual_argmax_id])
    with_norm = logit_lens(model, hidden_states_last_layer, tok, top_n=1, apply_norm=True)
    without_norm = logit_lens(model, hidden_states_last_layer, tok, top_n=1, apply_norm=False)

    print(f"  actual argmax:      {actual_token!r}")
    print(f"  lens WITH norm:     {with_norm[0][0]!r} (p={with_norm[0][1]:.3f})")
    print(f"  lens WITHOUT norm:  {without_norm[0][0]!r} (p={without_norm[0][1]:.3f})")

    if with_norm[0][0] == actual_token:
        print("  -> hidden_states[-1] is PRE-norm. Use apply_norm=True everywhere.")
        return True
    elif without_norm[0][0] == actual_token:
        print("  -> hidden_states[-1] is POST-norm already. apply_norm=False for the")
        print("     LAST layer only -- every earlier layer still needs apply_norm=True.")
        return False
    else:
        print("  -> NEITHER matches. STOP: don't trust lens output until this is resolved.")
        return None


def generate_soft_then_hard(model, tok, input_ids, attention_mask,
                             soft_steps=40, hard_steps=40,
                             top_k=15, top_p=0.95, eos_ids=None,
                             noise=None, gumbel_tau=0.5, dirichlet_gamma=1.0,
                             capture_hidden=True):
    embed_matrix = model.get_input_embeddings().weight
    past = None
    cur_mask = attention_mask
    step_records = []
    eos_fire_step = None  # first soft-step index where EOS becomes the top candidate

    cur_input_ids = input_ids
    cur_embeds = None
    for step_idx in range(soft_steps):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=cur_input_ids, attention_mask=cur_mask,
                             use_cache=True, output_hidden_states=capture_hidden)
            else:
                out = model(inputs_embeds=cur_embeds, attention_mask=cur_mask,
                            past_key_values=past, use_cache=True,
                            output_hidden_states=capture_hidden)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        # True greedy pick at this step, independent of noise/mixture.
        # Needed as ground truth for the logit-lens checkpoint below.
        raw_argmax_id = torch.argmax(logits, dim=-1).item()

        blended, weights, idx = soft_thinking_step(
            embed_matrix, logits, top_k, top_p,
            noise=noise, gumbel_tau=gumbel_tau, dirichlet_gamma=dirichlet_gamma,
        )

        top_weight, top_pos = weights[0].max(dim=-1)
        top_token_id = idx[0, top_pos].item()
        entropy = compute_entropy(weights[0]).item()

        # EOS/frozen-budget fix (2026-08-27): record the FIRST step where EOS
        # is top-1, at any weight -- no >0.5 gate, since a shallow EOS lead
        # under noise is still a real signal the model wants to stop.
        if eos_fire_step is None and eos_ids and top_token_id in eos_ids:
            eos_fire_step = step_idx
        is_void = eos_fire_step is not None and step_idx > eos_fire_step

        # Last-token hidden state at every layer (embeddings + each decoder
        # layer). With the KV cache, seq_len is already 1 from step 1 onward,
        # so [:, -1, :] only actually matters at step 0 (the full-prompt pass).
        step_hidden = None
        if capture_hidden:
            step_hidden = tuple(h[:, -1, :].detach().clone() for h in out.hidden_states)

        step_records.append({
            "weights": weights.detach().clone(),
            "idx": idx.detach().clone(),
            "blended": blended.detach().clone(),
            "entropy": entropy,
            "void": is_void,
            "raw_argmax_id": raw_argmax_id,
            "hidden_states": step_hidden,
        })

        flag = "  [VOID - post-EOS drift]" if is_void else ""
        print(f"  soft step {step_idx:2d}: top_token={tok.decode([top_token_id])!r}  "
              f"weight={top_weight.item():.3f}  entropy={entropy:.3f}{flag}")

        cur_embeds = blended.unsqueeze(1)
        cur_mask = torch.cat([cur_mask, torch.ones((cur_mask.shape[0], 1))], dim=1)

        # No early break: soft_steps must stay frozen at the same value across
        # every seed in the sweep so traces stay aligned. Steps after
        # eos_fire_step are still real forward passes -- just tagged void.

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

    return hard_generated, step_records, eos_fire_step


# Priority candidate steps from Exp 7 / 7b / 8 (see Next Step 4 in the log).
# Keyed by (noise_mode, seed).
PRIORITY_STEPS = {
    ("dirichlet", 1):   [55, 69, 88, 93],
    ("dirichlet", 123): [28, 47, 53, 76, 89, 97],
    ("dirichlet", 2):   list(range(20, 26)),   # sustained 20-25 cluster
    ("gumbel", 1):      [93],
    ("gumbel", 42):     [55],
    ("gumbel", 2):      [80],   # added — flagged in Exp 7b, missing from original list
    ("gumbel", 123):    [62],   # added — flagged in Exp 7b, missing from original list
}


def report_priority_steps(model, tok, records, noise_mode, seed, apply_norm):
    """Prints the mixture (already known) side-by-side with the logit lens
    (new) for each flagged step -- first real look at whether they agree."""
    steps = PRIORITY_STEPS.get((noise_mode, seed), [])
    if not steps:
        print(f"No flagged priority steps for ({noise_mode!r}, seed={seed}). "
              f"See Next Step 4 in the log for the full list.")
        return

    print(f"\n--- lens vs. mixture, ({noise_mode}, seed={seed}) ---")
    for s in steps:
        if s >= len(records):
            print(f"step {s}: out of range (soft_steps too short), skipping")
            continue
        rec = records[s]
        if rec["void"]:
            print(f"step {s}: VOID (post-EOS), skipping")
            continue

        top_w = rec["weights"][0].max().item()
        top_tok = tok.decode([rec["idx"][0, rec["weights"][0].argmax().item()].item()])
        mixture = [(tok.decode([rec["idx"][0, i].item()]), round(rec["weights"][0, i].item(), 3))
                   for i in range(rec["idx"].shape[1]) if rec["weights"][0, i].item() > 0.01]
        lens_out = logit_lens(model, rec["hidden_states"][-1], tok, top_n=5, apply_norm=apply_norm)

        print(f"\nstep {s}  mixture top-1={top_tok!r} weight={top_w:.3f} entropy={rec['entropy']:.3f}")
        print(f"  mixture: {mixture}")
        print(f"  lens:    {[(t, round(p, 3)) for t, p in lens_out]}")

def run_priority_sweep(model, tok, input_ids, attention_mask, eos_ids, apply_norm):
    """Loops every (noise_mode, seed) condition in PRIORITY_STEPS, runs
    generate_soft_then_hard once each, and prints the lens-vs-mixture report.
    Baseline checkpoint should already have passed before calling this."""
    conditions = sorted(PRIORITY_STEPS.keys())
    for noise_mode, seed in conditions:
        print(f"\n{'='*60}\nCondition: noise={noise_mode!r}, seed={seed}\n{'='*60}")
        torch.manual_seed(seed)
        _, records, eos_fire_step = generate_soft_then_hard(
            model, tok, input_ids, attention_mask,
            soft_steps=100, hard_steps=40, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU, dirichlet_gamma=DIRICHLET_GAMMA,
        )
        if eos_fire_step is not None:
            print(f"EOS fired at step {eos_fire_step}")
        report_priority_steps(model, tok, records, noise_mode, seed, apply_norm=apply_norm)


# ======================================================================
# Exp 12A (2026-09-13): vanilla baseline + unbiased (random) step sampling
# ======================================================================
#
# WHY THIS EXISTS:
#   Exp 9 only ever compared lens vs. mixture under NOISE (gumbel/dirichlet).
#   The paper's claim is "noise does not restore superposition" -- but that
#   claim has no same-pipeline counterfactual: nobody ran the noise-FREE
#   (vanilla) condition through the exact same lens comparison. This closes
#   that gap.
#
#   Separately, Exp 9's 18 steps were all hand-picked *because* they already
#   looked like dips -- that's a selection-biased sample. A step that was
#   flagged for having unusually high entropy is more likely to also show
#   lens/mixture disagreement, regardless of noise. This function also draws
#   a random (non-cherry-picked) sample of steps from vanilla and from two
#   reference noise conditions, so we have an unbiased disagreement-rate
#   baseline to compare Exp 9's 7/18 (39%) disagreement rate against.

import random as _random


def select_top_entropy_steps(records, n=8):
    """The N highest-entropy non-void steps in a run -- a condition's own
    'best case' for a genuine fork, used for the vanilla condition since it
    has no pre-flagged PRIORITY_STEPS list of its own."""
    scored = [(i, r["entropy"]) for i, r in enumerate(records) if not r["void"]]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:n]]


def select_random_steps(records, n=8, rng_seed=1234):
    """N random non-void steps, independent of entropy -- the unbiased
    sample used to check whether Exp 9's flagged-step disagreement rate is
    actually elevated relative to disagreement everywhere."""
    valid = [i for i, r in enumerate(records) if not r["void"]]
    rng = _random.Random(rng_seed)
    return sorted(rng.sample(valid, min(n, len(valid))))


def report_lens_vs_mixture_at_steps(model, tok, records, steps, apply_norm, label=""):
    """Generalized version of report_priority_steps: takes an explicit list
    of step indices instead of looking them up in PRIORITY_STEPS, and also
    tallies an agree/disagree count so runs can be compared numerically."""
    print(f"\n--- lens vs. mixture{f' ({label})' if label else ''} ---")
    agree, disagree = 0, 0
    for s in steps:
        if s >= len(records):
            print(f"step {s}: out of range, skipping")
            continue
        rec = records[s]
        if rec["void"]:
            print(f"step {s}: VOID (post-EOS), skipping")
            continue

        top_w = rec["weights"][0].max().item()
        top_tok = tok.decode([rec["idx"][0, rec["weights"][0].argmax().item()].item()])
        mixture = [(tok.decode([rec["idx"][0, i].item()]), round(rec["weights"][0, i].item(), 3))
                   for i in range(rec["idx"].shape[1]) if rec["weights"][0, i].item() > 0.01]
        lens_out = logit_lens(model, rec["hidden_states"][-1], tok, top_n=5, apply_norm=apply_norm)
        lens_top_tok, _ = lens_out[0]

        same = (lens_top_tok.strip() == top_tok.strip())
        agree += int(same)
        disagree += int(not same)

        print(f"\nstep {s}  mixture top-1={top_tok!r} weight={top_w:.3f} entropy={rec['entropy']:.3f}")
        print(f"  mixture: {mixture}")
        print(f"  lens:    {[(t, round(p, 3)) for t, p in lens_out]}   "
              f"{'agree' if same else 'DISAGREE'}")

    total = agree + disagree
    if total:
        print(f"\n  summary: {agree}/{total} agree, {disagree}/{total} disagree "
              f"({100 * disagree / total:.0f}% disagreement rate)")
    return agree, disagree


def run_exp12a_vanilla_baseline_and_unbiased_sampling(model, tok, input_ids, attention_mask,
                                                        eos_ids, apply_norm):
    """
    Exp 12A: runs the vanilla (noise=None) condition through the exact same
    lens-comparison pipeline as Exp 9, then adds a random-step (unbiased)
    sample for vanilla plus two reference noise conditions, to get a
    baseline disagreement rate that isn't cherry-picked.
    """
    print("\n" + "=" * 75)
    print("EXP 12A: VANILLA BASELINE + UNBIASED (RANDOM) STEP SAMPLING")
    print("=" * 75)

    results = {}

    # --- Vanilla condition. Seed is inert with noise=None (no RNG is ever
    # called), so there is exactly one vanilla trace, full stop. ---
    print("\n--- Running VANILLA (noise=None) ---")
    torch.manual_seed(0)  # inert here, set only for hygiene
    _, vanilla_records, vanilla_eos = generate_soft_then_hard(
        model, tok, input_ids, attention_mask,
        soft_steps=100, hard_steps=40, eos_ids=eos_ids,
        noise=None, capture_hidden=True,
    )
    if vanilla_eos is not None:
        print(f"EOS fired at step {vanilla_eos}")

    vanilla_top_entropy_steps = select_top_entropy_steps(vanilla_records, n=8)
    vanilla_random_steps = select_random_steps(vanilla_records, n=8)

    print(f"\nVanilla's own top-8 highest-entropy steps (its 'best case' for a fork): "
          f"{vanilla_top_entropy_steps}")
    a, d = report_lens_vs_mixture_at_steps(
        model, tok, vanilla_records, vanilla_top_entropy_steps, apply_norm,
        label="VANILLA, top-entropy steps")
    results[("vanilla", "top_entropy")] = (a, d)

    print(f"\nVanilla, unbiased random-8 steps: {vanilla_random_steps}")
    a, d = report_lens_vs_mixture_at_steps(
        model, tok, vanilla_records, vanilla_random_steps, apply_norm,
        label="VANILLA, random steps")
    results[("vanilla", "random")] = (a, d)

    # --- Reference noised conditions: random-step (unbiased) sampling only.
    # Their PRIORITY_STEPS numbers already exist from Exp 9 -- this adds the
    # unbiased comparison point for the same two runs. ---
    for noise_mode, seed in [("gumbel", 42), ("dirichlet", 1)]:
        print(f"\n--- Running {noise_mode.upper()} seed={seed} for random-step sampling ---")
        torch.manual_seed(seed)
        _, records, eos_fire = generate_soft_then_hard(
            model, tok, input_ids, attention_mask,
            soft_steps=100, hard_steps=40, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU, dirichlet_gamma=DIRICHLET_GAMMA,
            capture_hidden=True,
        )
        if eos_fire is not None:
            print(f"EOS fired at step {eos_fire}")
        random_steps = select_random_steps(records, n=8)
        print(f"\n{noise_mode} seed={seed}, unbiased random-8 steps: {random_steps}")
        a, d = report_lens_vs_mixture_at_steps(
            model, tok, records, random_steps, apply_norm,
            label=f"{noise_mode.upper()} seed={seed}, random steps")
        results[(noise_mode, seed, "random")] = (a, d)

    print("\n" + "=" * 75)
    print("EXP 12A SUMMARY (disagreement rates)")
    print("=" * 75)
    for key, (a, d) in results.items():
        total = a + d
        rate = (100 * d / total) if total else float("nan")
        print(f"  {key}: {d}/{total} disagree ({rate:.0f}%)")
    print("\n  Compare these rates against Exp 9's flagged-step rate: 7/18 disagree (39%).")

    return results


# ======================================================================
# Next Step 6 (2026-09-11): Causal patching
# ======================================================================
#
# WHAT THIS DOES:
#   1. Run the model normally with noise -> get the ORIGINAL output
#   2. Run the model AGAIN with the same noise/seed, but at ONE specific
#      step, swap the blended embedding for a single discrete token's
#      embedding (e.g., replace the blend with just '2' instead of the
#      noisy mix that favored '4')
#   3. Compare the two outputs:
#      - If the output CHANGES -> that step's input actually mattered
#        (it's "causally important")
#      - If the output STAYS THE SAME -> that step was decorative
#        (the answer came from somewhere else)
#
# PRIORITY TARGETS (from Exp 9 lens-vs-mixture disagreements):
#   1. Gumbel seed=2, step 80: mixture='4'(wrong), lens='2'(correct!)
#   2. Dir seed=123, step 89: mixture='5', lens='3' (high-confidence flip)
#   3. Dir seed=1, step 93: mixture='4', lens='3' (clean disagreement)
#   4. Gumbel seed=1, step 93: mixture='5', lens='4' (Gumbel-side flip)

# The 4 priority targets for causal patching, derived from Exp 9.
# Format: (noise_mode, seed, step_idx, mixture_top1_token, lens_top1_token, description)
PATCH_TARGETS = [
    ("gumbel",    2,  80, "4", "2", "mixture='4'(WRONG), lens='2'(CORRECT) — does patching fix the answer?"),
    ("dirichlet", 123, 89, "5", "3", "mixture='5', lens='3' — high-confidence lens flip"),
    ("dirichlet", 1,  93, "4", "3", "mixture='4', lens='3' — clean top-1 disagreement"),
    ("gumbel",    1,  93, "5", "4", "mixture='5', lens='4' — Gumbel-side flip"),
]


def generate_soft_with_patch(model, tok, input_ids, attention_mask,
                              soft_steps=100, hard_steps=40,
                              top_k=15, top_p=0.95, eos_ids=None,
                              noise=None, gumbel_tau=0.5, dirichlet_gamma=1.0,
                              patch_step=None, patch_token_str=None):
    """
    Same as generate_soft_then_hard, but at exactly one step (patch_step),
    replaces the blended embedding with the discrete embedding of patch_token_str.

    This is the causal intervention: everything before patch_step is identical
    to the unpatched run (same seed = same noise draws), everything after
    diverges because the model received a different input at that step.

    Returns: (hard_phase_text, was_patched_step_reached)
    """
    embed_matrix = model.get_input_embeddings().weight

    # Look up the patch token's embedding once
    patch_token_id = None
    if patch_step is not None and patch_token_str is not None:
        patch_token_id = tok.encode(patch_token_str, add_special_tokens=False)
        if len(patch_token_id) != 1:
            raise ValueError(f"patch_token_str={patch_token_str!r} encodes to "
                             f"{len(patch_token_id)} tokens, need exactly 1")
        patch_token_id = patch_token_id[0]

    past = None
    cur_mask = attention_mask
    eos_fire_step = None
    patched = False

    cur_input_ids = input_ids
    cur_embeds = None
    for step_idx in range(soft_steps):
        with torch.no_grad():
            if past is None:
                out = model(input_ids=cur_input_ids, attention_mask=cur_mask,
                             use_cache=True)
            else:
                out = model(inputs_embeds=cur_embeds, attention_mask=cur_mask,
                            past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        blended, weights, idx = soft_thinking_step(
            embed_matrix, logits, top_k, top_p,
            noise=noise, gumbel_tau=gumbel_tau, dirichlet_gamma=dirichlet_gamma,
        )

        _, top_pos = weights[0].max(dim=-1)
        top_token_id = idx[0, top_pos].item()

        if eos_fire_step is None and eos_ids and top_token_id in eos_ids:
            eos_fire_step = step_idx

        # === THE PATCH: at exactly this step, swap the blend for a discrete token ===
        if step_idx == patch_step and patch_token_id is not None:
            cur_embeds = embed_matrix[patch_token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
            patched = True
        else:
            cur_embeds = blended.unsqueeze(1)

        cur_mask = torch.cat([cur_mask, torch.ones((cur_mask.shape[0], 1))], dim=1)

    # bridging step
    with torch.no_grad():
        out = model(inputs_embeds=cur_embeds, attention_mask=cur_mask,
                     past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    hard_generated = next_id
    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)

    # hard phase
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

    hard_text = tok.decode(hard_generated[0], skip_special_tokens=True)
    return hard_text, patched


def run_causal_patches(model, tok, input_ids, attention_mask, eos_ids):
    """
    For each priority target from Exp 9:
      1. Run UNPATCHED (normal noisy soft thinking) -> get baseline output
      2. Run PATCHED with the LENS top-1 token -> see if output changes
      3. Run PATCHED with the MIXTURE top-1 token (as discrete) -> control
      4. Print all three side by side

    If the lens-patched output differs from unpatched, the blend content at
    that step is causally load-bearing (changing it changes the answer).
    """
    print("\n" + "=" * 70)
    print("CAUSAL PATCHING EXPERIMENTS (Exp 10)")
    print("=" * 70)

    for noise_mode, seed, step_idx, mix_tok, lens_tok, desc in PATCH_TARGETS:
        print(f"\n{'─' * 70}")
        print(f"Target: ({noise_mode}, seed={seed}, step={step_idx})")
        print(f"  {desc}")
        print(f"  Will compare: unpatched vs. patch-with-'{lens_tok}' "
              f"vs. patch-with-'{mix_tok}'")
        print(f"{'─' * 70}")

        # --- Run 1: UNPATCHED (baseline) ---
        print(f"\n  [1/3] Running UNPATCHED ({noise_mode}, seed={seed})...")
        torch.manual_seed(seed)
        unpatched_text, _ = generate_soft_with_patch(
            model, tok, input_ids, attention_mask,
            soft_steps=100, hard_steps=60, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU,
            dirichlet_gamma=DIRICHLET_GAMMA,
            patch_step=None, patch_token_str=None,
        )

        # --- Run 2: PATCHED with LENS top-1 token ---
        print(f"  [2/3] Running PATCHED with lens token '{lens_tok}' "
              f"at step {step_idx}...")
        torch.manual_seed(seed)
        lens_patched_text, did_patch = generate_soft_with_patch(
            model, tok, input_ids, attention_mask,
            soft_steps=100, hard_steps=60, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU,
            dirichlet_gamma=DIRICHLET_GAMMA,
            patch_step=step_idx, patch_token_str=lens_tok,
        )
        assert did_patch, f"Patch step {step_idx} was never reached!"

        # --- Run 3: PATCHED with MIXTURE top-1 token (discrete control) ---
        print(f"  [3/3] Running PATCHED with mixture token '{mix_tok}' "
              f"at step {step_idx}...")
        torch.manual_seed(seed)
        mix_patched_text, did_patch = generate_soft_with_patch(
            model, tok, input_ids, attention_mask,
            soft_steps=100, hard_steps=60, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU,
            dirichlet_gamma=DIRICHLET_GAMMA,
            patch_step=step_idx, patch_token_str=mix_tok,
        )
        assert did_patch, f"Patch step {step_idx} was never reached!"

        # --- Compare ---
        print(f"\n  === RESULTS ===")
        print(f"  UNPATCHED output (hard phase):")
        print(f"    {unpatched_text!r}")
        print(f"\n  PATCHED with LENS '{lens_tok}' "
              f"(what the model internally preferred):")
        print(f"    {lens_patched_text!r}")
        print(f"\n  PATCHED with MIXTURE '{mix_tok}' "
              f"(what noise pushed to top):")
        print(f"    {mix_patched_text!r}")

        # Did the patch change anything?
        lens_changed = (lens_patched_text != unpatched_text)
        mix_changed = (mix_patched_text != unpatched_text)
        lens_vs_mix = (lens_patched_text != mix_patched_text)

        print(f"\n  --- Verdict ---")
        if lens_changed:
            print(f"  LENS patch CHANGED the output "
                  f"-> step {step_idx} input is CAUSALLY LOAD-BEARING")
        else:
            print(f"  LENS patch did NOT change output "
                  f"-> step {step_idx} input is DECORATIVE (model ignores it)")

        if mix_changed:
            print(f"  MIXTURE patch also changed output "
                  f"-> even discrete mixture-top1 differs from the blend")
        else:
            print(f"  MIXTURE patch same as unpatched "
                  f"-> discrete mixture-top1 matches blend behavior")

        if lens_vs_mix:
            print(f"  LENS patch != MIXTURE patch "
                  f"-> the CHOICE of token at this step MATTERS for "
                  f"downstream text")
        else:
            print(f"  LENS patch = MIXTURE patch "
                  f"-> both patches produce same text (step matters, "
                  f"but not which token)")


# ======================================================================
# Exp 12B (2026-09-13): control-token arm for causal patching
# ======================================================================
#
# WHY THIS EXISTS:
#   Exp 10 only compared "patch with lens token" vs. "patch with mixture
#   token" vs. unpatched. But feeding ANY different discrete token at a
#   generation step will generically change the continuation -- so "lens
#   patch changed the output" doesn't, on its own, show the lens found
#   something special. It could just mean the position is sensitive to
#   input changes at all.
#
#   This adds a CONTROL arm: patch with a token that had near-zero weight
#   under BOTH the mixture AND the lens at that step. If the control patch
#   changes the output about as much as the lens patch does (and produces
#   similarly different text), that undercuts the lens's special status.
#   If only the lens patch reliably moves the output while the control
#   patch does little/nothing, that's much stronger evidence the lens
#   found something real.
#
#   Control token chosen: "1" -- checked against Exp 9's own tables and
#   confirmed to carry near-zero weight in both the mixture and the lens
#   distribution at all four PATCH_TARGETS steps.
CONTROL_TOKEN = "1"


# ======================================================================
# Exp 12C (2026-09-14): diagnose the vanilla lens-anomaly from Exp 12A
# ======================================================================
#
# Under noise=None, top-k -> top-p -> renormalize CANNOT reorder candidates
# (topk is already sorted; the top-p keep-mask only zeroes a trailing
# suffix, position 0's cumulative-before-it is always 0 <= top_p so it's
# never dropped; renormalizing divides everything by the same constant).
# So mixture top-1 MUST always equal raw_argmax_id for every vanilla step --
# 0% disagreement is the only mathematically possible outcome. Exp 12A found
# 25-38% instead. This reruns vanilla and prints raw_argmax_id (stored per
# step already) next to mixture-top-1 and lens-top-1, to tell apart:
#   (a) mixture always == raw_argmax (as it must) but lens sometimes != raw
#       -> the LENS reconstruction has a precision/near-tie issue, not a
#          real finding about vanilla's internal representations
#   (b) mixture itself != raw_argmax on some step -> a real bug elsewhere
#       (would contradict the math above -- worth a hard stop if seen)
def diagnose_vanilla_lens_anomaly(model, tok, input_ids, attention_mask, eos_ids):
    print("\n" + "=" * 75)
    print("EXP 12C: DIAGNOSING THE VANILLA LENS-ANOMALY FROM 12A")
    print("=" * 75)

    torch.manual_seed(0)
    _, records, _ = generate_soft_then_hard(
        model, tok, input_ids, attention_mask,
        soft_steps=100, hard_steps=40, eos_ids=eos_ids,
        noise=None, capture_hidden=True,
    )

    steps = sorted(set(select_top_entropy_steps(records, n=8) + select_random_steps(records, n=8)))
    print(f"\nChecking steps: {steps}\n")

    mix_vs_raw_mismatches = 0
    lens_vs_raw_mismatches = 0
    for s in steps:
        rec = records[s]
        if rec["void"]:
            print(f"step {s}: VOID, skipping")
            continue
        mix_top = tok.decode([rec["idx"][0, rec["weights"][0].argmax().item()].item()])
        raw_top = tok.decode([rec["raw_argmax_id"]])
        lens_top, lens_p = logit_lens(model, rec["hidden_states"][-1], tok, top_n=1, apply_norm=True)[0]

        mix_flag = mix_top.strip() != raw_top.strip()
        lens_flag = lens_top.strip() != raw_top.strip()
        mix_vs_raw_mismatches += int(mix_flag)
        lens_vs_raw_mismatches += int(lens_flag)

        note = ""
        if mix_flag:
            note += "  <-- mixture != raw (SHOULD BE IMPOSSIBLE under noise=None -- real bug if seen)"
        if lens_flag:
            note += "  <-- lens != raw (precision/near-tie artifact in the lens reconstruction)"

        print(f"step {s:3d}: raw_argmax={raw_top!r:12s} mixture_top1={mix_top!r:12s} "
              f"lens_top1={lens_top!r:12s} (lens_p={lens_p:.3f}){note}")

    print(f"\nmixture != raw_argmax:  {mix_vs_raw_mismatches} (must be 0 -- anything else is a real bug)")
    print(f"lens != raw_argmax:     {lens_vs_raw_mismatches} (this is the anomaly from 12A -- "
          f"nonzero here, zero above, points to a lens precision artifact at near-ties)")


def run_causal_patches_with_control(model, tok, input_ids, attention_mask, eos_ids,
                                     control_token=CONTROL_TOKEN):
    """
    Exp 12B: re-runs Exp 10's 4 patch targets with a 4th arm added.
    Four runs per target: unpatched, patch-with-lens-token,
    patch-with-mixture-token, patch-with-control-token.
    """
    print("\n" + "=" * 70)
    print("EXP 12B: CAUSAL PATCHING WITH CONTROL ARM")
    print("=" * 70)

    for noise_mode, seed, step_idx, mix_tok, lens_tok, desc in PATCH_TARGETS:
        print(f"\n{'─' * 70}")
        print(f"Target: ({noise_mode}, seed={seed}, step={step_idx})")
        print(f"  {desc}")
        print(f"  Comparing: unpatched vs. lens='{lens_tok}' vs. mixture='{mix_tok}' "
              f"vs. control='{control_token}'")
        print(f"{'─' * 70}")

        runs = {}
        for label, patch_tok in [
            ("unpatched", None),
            ("lens", lens_tok),
            ("mixture", mix_tok),
            ("control", control_token),
        ]:
            print(f"\n  Running [{label}] ...")
            torch.manual_seed(seed)
            text, did_patch = generate_soft_with_patch(
                model, tok, input_ids, attention_mask,
                soft_steps=100, hard_steps=60, eos_ids=eos_ids,
                noise=noise_mode, gumbel_tau=GUMBEL_TAU,
                dirichlet_gamma=DIRICHLET_GAMMA,
                patch_step=(step_idx if patch_tok is not None else None),
                patch_token_str=patch_tok,
            )
            if patch_tok is not None:
                assert did_patch, f"Patch step {step_idx} was never reached!"
            runs[label] = text

        print(f"\n  === RESULTS ===")
        for label in ["unpatched", "lens", "mixture", "control"]:
            print(f"  [{label:9s}] {runs[label]!r}")

        lens_changed = runs["lens"] != runs["unpatched"]
        mix_changed = runs["mixture"] != runs["unpatched"]
        control_changed = runs["control"] != runs["unpatched"]
        lens_vs_control = runs["lens"] != runs["control"]

        print(f"\n  --- Verdict ---")
        print(f"  lens patch changed output from unpatched:    {lens_changed}")
        print(f"  mixture patch changed output from unpatched: {mix_changed}")
        print(f"  control patch changed output from unpatched: {control_changed}")
        print(f"  lens patch != control patch:                 {lens_vs_control}")

        if lens_changed and not control_changed:
            print("  -> STRONG: only the lens token steers the output; a generic "
                  "override at this position does nothing. This step really is "
                  "sensitive to the SPECIFIC token, not just 'any change here'.")
        elif lens_changed and control_changed and lens_vs_control:
            print("  -> WEAKER: both lens and control patches change the output, "
                  "but produce DIFFERENT text -- the position is generically "
                  "sensitive to input changes, but token identity still matters "
                  "somewhat. The lens's specialness is not established by this "
                  "alone; would need many more targets to say more.")
        elif lens_changed and control_changed and not lens_vs_control:
            print("  -> WEAK: lens and control patches produce the SAME output. "
                  "This suggests 'lens patch changed the output' in Exp 10 was "
                  "just 'any different token here changes the output' -- not "
                  "evidence the lens specifically found something meaningful.")
        else:
            print("  -> lens patch did not change output at all here -- step is "
                  "decorative regardless of what the control shows.")


# ======================================================================
# Exp 11 (2026-09-11): The Final Superposition Hunt
# Part A: Layer-by-Layer Depth Probing (Layers 6, 12, 18, 24)
# Part B: Abstract Multi-Concept / Creative Fusion Superposition
# ======================================================================

def probe_layer_depth(model, tok, hidden_states_tuple, target_layers=(6, 12, 18, 24), top_n=4):
    """
    Decodes the internal representation at multiple network depths using the logit lens.
    Qwen2.5-0.5B has 24 decoder layers. hidden_states_tuple indexing:
      - Index 0:  token embedding (before any decoder layer)
      - Index 1:  output of decoder layer 1
      - Index 24: output of decoder layer 24 (final, pre-norm)
    Target layers to probe:
      - Layer 6:  Early semantic abstraction
      - Layer 12: Mid-network reasoning (most likely home for superposition)
      - Layer 18: Late consolidation
      - Layer 24: Final layer right before output (same as logit-lens used in Exp 9)
    """
    layer_readouts = {}
    for layer_idx in target_layers:
        h = hidden_states_tuple[layer_idx]
        decoded = logit_lens(model, h, tok, top_n=top_n, apply_norm=True)
        layer_readouts[layer_idx] = decoded
    return layer_readouts


def run_exp11_depth_probing(model, tok, input_ids, attention_mask, eos_ids):
    """
    Part A: Test whether superposition exists in INTERMEDIATE layers (e.g. Layer 12)
    before collapsing at Layer 24.
    Tested on our key disagreement points from Exp 9/10:
      - (gumbel, seed=2, step 80): mixture favored '4', lens favored '2'
      - (dirichlet, seed=123, step 89): mixture favored '5', lens favored '3'
    """
    print("\n" + "=" * 75)
    print("EXP 11A: LAYER-BY-LAYER DEPTH PROBING (Does superposition exist mid-network?)")
    print("=" * 75)

    test_cases = [
        ("gumbel", 2, 80, "4", "2"),
        ("dirichlet", 123, 89, "5", "3"),
        ("gumbel", 1, 93, "5", "4"),
    ]

    for noise_mode, seed, step_idx, mix_tok, final_lens_tok in test_cases:
        print(f"\n{'─' * 75}")
        print(f"Condition: ({noise_mode}, seed={seed}) at Step {step_idx}")
        print(f"  Mixture top-1: {mix_tok!r} | Final Layer 24 lens top-1: {final_lens_tok!r}")
        print(f"{'─' * 75}")

        torch.manual_seed(seed)
        _, records, _ = generate_soft_then_hard(
            model, tok, input_ids, attention_mask,
            soft_steps=step_idx + 1, hard_steps=5, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU, dirichlet_gamma=DIRICHLET_GAMMA,
            capture_hidden=True,
        )

        rec = records[step_idx]
        # Build the mixture display list (threshold 2% to cut noise)
        mixture = [(tok.decode([rec["idx"][0, i].item()]), round(rec["weights"][0, i].item(), 3))
                   for i in range(rec["idx"].shape[1]) if rec["weights"][0, i].item() > 0.02]

        print(f"\n  Input Mixture (top candidates, weight > 2%): {mixture[:5]}")
        print(f"  Probing depth progression across layers:")

        depth_results = probe_layer_depth(model, tok, rec["hidden_states"], target_layers=(6, 12, 18, 24), top_n=3)
        for layer_num, top_cands in depth_results.items():
            cand_str = ", ".join([f"{t!r} ({p:.3f})" for t, p in top_cands])
            print(f"    Layer {layer_num:2d}: {cand_str}")

        # Scientific diagnostic
        mid_tokens = [t.strip() for t, _ in depth_results[12]]
        has_both = (mix_tok.strip() in mid_tokens and final_lens_tok.strip() in mid_tokens)
        if has_both:
            print(f"  >>> SUPERPOSITION DETECTED IN MID-LAYERS! Both {mix_tok!r} and {final_lens_tok!r} active at Layer 12.")
        else:
            print(f"  >>> No mid-layer superposition: Layer 12 already committed or dominated by a single path.")


ABSTRACT_PROMPT = "Describe an entity that is simultaneously a delicate blooming flower and a lethal razor-sharp blade."

def run_exp11_abstract_concept_hunt(model, tok, eos_ids):
    """
    Part B: Conceptual Superposition on an Abstract Multi-Concept Blend.
    Instead of rigid arithmetic numbers, we test whether the model can sustain
    simultaneous representation of two opposing conceptual domains:
      - Domain 1: Floral / Delicate ('petal', 'bloom', 'rose', 'fragile', 'flower')
      - Domain 2: Lethal / Weapon ('blade', 'steel', 'edge', 'razor', 'sharp')
    """
    print("\n" + "=" * 75)
    print("EXP 11B: ABSTRACT MULTI-CONCEPT BLEND (Can concepts co-exist in superposition?)")
    print(f"Prompt: {ABSTRACT_PROMPT!r}")
    print("=" * 75)

    msgs = [{"role": "user", "content": ABSTRACT_PROMPT}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt")

    for noise_mode, seed in [("gumbel", 42), ("dirichlet", 1)]:
        print(f"\n{'─' * 75}")
        print(f"Testing Abstract Concept Blend: ({noise_mode}, seed={seed})")
        print(f"{'─' * 75}")

        torch.manual_seed(seed)
        hard_ids, records, _ = generate_soft_then_hard(
            model, tok, inputs["input_ids"], inputs["attention_mask"],
            soft_steps=35, hard_steps=35, eos_ids=eos_ids,
            noise=noise_mode, gumbel_tau=GUMBEL_TAU, dirichlet_gamma=DIRICHLET_GAMMA,
            capture_hidden=True,
        )

        hard_text = tok.decode(hard_ids[0], skip_special_tokens=True)
        print(f"\nGenerated completion:\n  {hard_text.strip()!r}")

        # Find the 3 steps with the highest entropy (most ambiguous concept forks)
        entropies = [(i, rec["entropy"]) for i, rec in enumerate(records) if not rec["void"]]
        entropies.sort(key=lambda x: x[1], reverse=True)
        top_fork_steps = [idx for idx, _ in entropies[:3]]

        print(f"\nAnalyzing the top ambiguity / conceptual fork steps: {top_fork_steps}")
        for s in sorted(top_fork_steps):
            rec = records[s]
            mixture = [(tok.decode([rec["idx"][0, i].item()]), round(rec["weights"][0, i].item(), 3))
                       for i in range(rec["idx"].shape[1]) if rec["weights"][0, i].item() > 0.03]
            print(f"\n  [Step {s}] (Entropy = {rec['entropy']:.3f})")
            print(f"    Mixture: {mixture[:5]}")

            depth_results = probe_layer_depth(model, tok, rec["hidden_states"], target_layers=(6, 12, 18, 24), top_n=3)
            for layer_num, top_cands in depth_results.items():
                cand_str = ", ".join([f"{t!r} ({p:.3f})" for t, p in top_cands])
                print(f"      Layer {layer_num:2d}: {cand_str}")


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

    # ----------------------------------------------------------------
    # EXP 11: The Final Superposition Hunt (already logged -- comment out
    # these two lines if you just want to run Exp 12 faster)
    # ----------------------------------------------------------------
    run_exp11_depth_probing(model, tok, inputs["input_ids"], inputs["attention_mask"], eos_ids)
    run_exp11_abstract_concept_hunt(model, tok, eos_ids)

    # ----------------------------------------------------------------
    # EXP 12: closing the two gaps flagged in review
    #   12A: vanilla (noise=None) baseline run through the SAME lens
    #        pipeline as Exp 9, plus unbiased random-step sampling to
    #        check whether Exp 9's flagged-step disagreement rate (7/18,
    #        39%) is actually elevated vs. disagreement everywhere.
    #   12B: adds a control-token arm to the 4 Exp 10 causal-patch
    #        targets, to check whether "lens patch changed the output"
    #        means anything beyond "any different token here changes
    #        the output."
    # ----------------------------------------------------------------
    run_exp12a_vanilla_baseline_and_unbiased_sampling(
        model, tok, inputs["input_ids"], inputs["attention_mask"], eos_ids, apply_norm=True)
    run_causal_patches_with_control(
        model, tok, inputs["input_ids"], inputs["attention_mask"], eos_ids)
    diagnose_vanilla_lens_anomaly(
        model, tok, inputs["input_ids"], inputs["attention_mask"], eos_ids)


if __name__ == "__main__":
    main()