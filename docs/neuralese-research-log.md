# Neuralese Research Log

Running record of methodology, decisions, and experiment results. This is the raw material for the eventual paper — update it after every verified experiment, don't reconstruct from memory later.

**Research questions:**
- **(2) Superposition:** does the claim that continuous-thought vectors hold multiple search frontiers in parallel (Zhu et al.) hold on a task with rigorous classical ground truth — not just graph reachability or math benchmarks? Planned ground truth: user's own CSP exam-scheduler solver (backtracking + MRV).
- **(3) Faithfulness:** does what a logit lens shows match what causal patching says is actually driving the answer, and does this degrade as training moves from language-supervised toward RL-only?

---

## Methodology

**Why not fork `eric-ai-lab/Soft-Thinking` directly:** inspected the repo — it's a full fork of the `sglang` serving engine with custom CUDA kernels (`sgl-kernel`), built for multi-GPU tensor-parallel serving of 32B-scale models. The mixture logic is spread across `sampler.py` / `scheduler.py` / an async worker thread, and CUDA graphs + overlap scheduling would need disabling just to inspect a single step. Not CPU-feasible, and would be harder to instrument for logit-lens/causal-patching than writing our own loop. **Decision:** standalone reimplementation in plain HuggingFace `transformers`, porting only the core algorithm.

**Core algorithm (ported from sglang's `sampler.py`):** at each generation step, take the softmax probs, apply combined top-k + top-p (nucleus) filtering, renormalize the kept probabilities, then compute a weighted average of the corresponding token embedding vectors using those weights as the mixture. That blended vector is fed back in as `inputs_embeds` for the next step, instead of sampling and embedding one discrete token. Dirichlet/Gumbel noise from the original repo intentionally omitted — the repo's own comments mark these as unused in the paper's actual method.

**Harness structure (`neuralese_harness.py`):**
- `generate_baseline` — manual, hand-rolled greedy decode loop using the KV cache directly (no `inputs_embeds`), used to validate correctness against `.generate()`.
- `soft_thinking_step` — the core mixture computation described above.
- `generate_soft_then_hard` — soft phase (fixed step budget, feeds blended embeddings back in) → bridging step → hard phase (normal discrete argmax decoding for a readable final answer).

**Environment:** Windows laptop, no GPU, CPU-only. Dev model: `Qwen/Qwen2.5-0.5B-Instruct` (fast iteration; will move to `Qwen2.5-1.5B-Instruct` once methodology is settled).

**Validation checkpoint (must pass before trusting any soft-phase result):** manual baseline loop's greedy output must exactly match `model.generate(do_sample=False)`'s output. Two bugs found and fixed to get a true match:
1. `.generate()` was silently applying `model.generation_config`'s baked-in defaults (`repetition_penalty=1.1`, `temperature=0.7`, `top_k=20`, `top_p=0.8`) even with `do_sample=False`. Fixed by explicitly passing `repetition_penalty=1.0, temperature=None, top_p=None, top_k=None` to force pure greedy argmax on both sides.
2. Early-EOS-break logic in the soft loop wasn't saving `cur_embeds` before breaking, causing a `None` `inputs_embeds` crash on the bridging step. Fixed by recording the step and setting `cur_embeds` *before* the break check, not after.

Status: **Checkpoint passes** — manual baseline is byte-for-byte validated against `.generate()`.

---

## Experiment Log

### Exp 1 — Arithmetic ("43 * 34 = ?")
- **Soft steps:** 30 (before EOS-handling fix)
- **Result:** weights ~0.9–1.0 throughout the correct answer (`43 * 34 = 1462`), then the loop kept running 15+ steps past where the model tried to stop (no EOS check in soft phase yet), producing incoherent drift ("Human beings have a group of intelligent animals...").
- **Interpretation:** no real blending — the model is fully confident on memorized arithmetic, so there's nothing to blend between. The post-EOS drift is an artifact of forcing generation past the model's natural stopping point, not a finding about reasoning. **Bug identified:** soft loop needed its own EOS check (fixed after this run).

### Exp 2 — Train catch-up problem (after EOS + repetition-penalty fixes)
- **Prompt:** "A train leaves at 3pm going 60mph. Another leaves the same station at 4pm going 90mph in the same direction. What time does the second train catch the first?"
- **Soft steps:** 30
- **Result:** weights mostly 0.85–1.0. Three dips: `' time'` (0.554), `' over'` (0.499, choosing between "overtake"/"catch"/"reach"), `'.'` (0.516, sentence-end vs continue).
- **Interpretation:** this is a templated, low-ambiguity problem-setup sentence a strong instruct model has seen the shape of many times. Only word-choice-level ambiguity surfaced, no reasoning-level fork — because the prompt doesn't present one in its opening sentence.

### Exp 3 — Water-jug puzzle (genuine reasoning fork, attempt 1)
- **Prompt:** "You have a 3-liter jug and a 5-liter jug, both empty, and unlimited water. You need to end up with exactly 4 liters in one of the jugs. What is the first thing you do?"
- **Soft steps:** 40
- **Result:** weights lower and more frequently sub-0.9 than Exp 2 (e.g. `' solve'` 0.508, `' thing'` 0.525, `' you'` 0.485, `' should'` 0.601, `'Initial'` 0.504, `' State'` 0.492), but these dips occur during **preamble/setup** ("To solve this... let's analyze the situation step by step... Initial State: You have a 3-liter jug and a 5-liter jug, both empty"). The 40-step soft budget was exhausted before reaching the actual decision point — the hard phase (plain discrete decoding, no longer instrumented) is what produced "Let's consider the two possible scenarios: Scenario 1: You fill the 5...".
- **Interpretation:** **negative result on methodology, not yet on the science.** The soft-phase weight dips found so far are about *how to phrase the opening*, not *which jug to fill first* — the genuinely interesting fork wasn't captured because this model's verbose style burns the entire step budget on setup before reaching it.
- **Next fix (not yet run):** either (a) raise `soft_steps` to ~70–80 so the fork lands inside the soft phase, or (b) reduce preamble by making the prompt more direct (e.g. "Answer directly: do you fill the 3-liter jug or the 5-liter jug first?") so fewer steps are spent before the real decision.

---

## Standing observations (to revisit when drafting)

- So far, every genuine sub-0.9 dip has coincided with plausible, human-identifiable alternative continuations (synonym choice, punctuation choice, opening-phrase choice) — consistent with *some* real blending mechanism, but none of it yet at the level of "two different solution paths" that the superposition claim is actually about.
- Model verbosity/preamble length is a real confound for this experiment design — need enough soft-step budget to reach the actual fork, which varies a lot by prompt and needs to be checked per-prompt rather than assumed.
- Still pending: logit-lens decoding of the full mixture (not just top-1 weight) at each step, and causal patching to test whether a blended vector actually drives the outcome or is decorative.

## Next steps
1. Re-run water-jug puzzle with either extended `soft_steps` or a more direct prompt, to actually capture the fill-3-vs-fill-5 decision inside the soft phase.
2. Once a real fork is captured with a meaningful weight dip, add logit-lens decoding of the full top-k mixture at that step (not just the top-1 token) to see whether it looks like "3 and 5 both present" or something else entirely.
3. Add causal patching: swap the blended vector at the fork step, see if it changes which scenario the hard phase commits to.
