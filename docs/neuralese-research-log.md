# Neuralese Research Log

Running record of methodology, decisions, and experiment results. This is the raw material for the eventual paper — update it after every verified experiment, don't reconstruct from memory later.

**Research questions:**
- **(2) Superposition — active track.** Does adding noise (Gumbel-softmax / Dirichlet) to fix the Greedy Pitfall *restore genuine, causally-load-bearing superposition* in Soft Thinking, or does it just add beneficial randomness across otherwise-independent greedy rollouts? Framed (see Review Notes — 2026-08-24) as closing a specific gap: Rizvi-Martel et al. tested only vanilla (noise-free) Soft Thinking with logit lens + one KL-divergence swap check; nobody has run noise-injected Soft Thinking through logit lens *and* real causal patching (hidden-state overwrite → check if final answer flips) together.
- **(3) Faithfulness — future work, not active now.** Does what a logit lens shows match what causal patching says is actually driving the answer, and does this degrade along the SFT → RL-only training axis? Deferred until Track 2 results are in hand (see Review Notes — 2026-08-24).

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

## Review Notes — 2026-07-22 (external validation pass)

Cross-checked the methodology against the actual `Soft-Thinking` repo (now hosted as `UCSB-AI/Soft-Thinking`, formerly under `eric-ai-lab`) and the surrounding literature. Two corrections and one scope caveat came out of it. Logging them here rather than editing the entries above, so the decision trail stays visible.

**Correction — Dirichlet/Gumbel noise should not have been omitted.**

The "unused in the paper" read was true only of the *original* Soft Thinking paper (Zhang et al., arXiv:2505.15778). A direct follow-up, *LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking* (Wu et al., arXiv:2508.03440), is specifically about research question (2). Its finding: vanilla Soft Thinking collapses into single-path, greedy-like decoding — the "Greedy Pitfall" — because the model's own next-token distribution is almost always dominated by one token, leaving too little real ambiguity to blend. Their fix, which the repo has since adopted, is exactly the noise mechanism this log skipped.

Two variants, not equivalent:
- **Gumbel-Softmax** (repo default `τ = 0.5`): `y_i = softmax_i((log(π_i) + g_i) / τ)`, `g_i ~ Gumbel(0,1)`, applied to the already top-k/top-p-filtered, renormalized weights `π`. Recommended by the paper — gives controllable randomness without the tradeoff below.
- **Dirichlet resampling** (repo default `γ = 1.0`, exposed as `dirichlet_temperature` though it isn't a temperature): sample from `Dir(γ·π)`. Paper found a real limitation — low `γ` gives high randomness but individual samples still collapse near one-hot; high `γ` gives smoothness but loses randomness. Can't get both from this one.

**Implication for Exp 1–3:** the consistently-high, shallow-dip weights observed so far are consistent with the Greedy Pitfall, independently of the preamble-budget issue already identified in Exp 3. Both explanations may be stacked. Re-run planned (see Next steps).

**Scope caveat — small-model behavior may not generalize.**

The repo's own reproduction notes warn that Soft Thinking underperforms on models ≤7B, attributed to smaller hidden sizes placing the last hidden state close to unrelated embeddings, adding noise to the mixture. `Qwen2.5-0.5B`/`1.5B` (chosen here for CPU feasibility) sit well inside that range. Doesn't block the mechanistic study — it's still a valid question what *these* models do — but any write-up should scope claims to small-scale models rather than imply generalization to the 32B scale the original superposition claims were made on.

**Refinement — what actually needs a logit lens.**

The blended embedding itself isn't a target for the lens — it's already fully known by construction (`Σ weight_i · embedding[token_i]`, and `kept_idx`/`weights` already give the full readout). What genuinely needs decoding is the model's *hidden states* — the residual-stream representations it computes internally after consuming that blended input at each layer. Those aren't mixtures of known embeddings; they're transformed, and reading them requires projecting through the model's final norm + unembedding (the actual logit-lens operation). Same logic applies to causal patching for research question (3): patching the input embedding just re-tests "does a different input change the output" (trivially yes). Patching a hidden state mid-network is what actually tests whether the surface-visible mixture content is causally load-bearing or decorative.

Practical consequence: `generate_soft_then_hard` needs to additionally capture `output_hidden_states=True` per step, not just the blended vector + weights it already logs.

**Minor correction — repetition_penalty bug diagnosis.**

`temperature`/`top_k`/`top_p` are skipped by HF's own generation logic when `do_sample=False`; they weren't the actual cause of the mismatch in bug #1 above. `repetition_penalty` is a logits processor that applies regardless of sampling mode, so that part of the fix was the one doing the real work. Doesn't change the outcome — checkpoint still passes byte-for-byte — just correcting the mechanism for the record.

References: Zhang et al. 2025, *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*, arXiv:2505.15778. Wu et al. 2025, *LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking*, arXiv:2508.03440.

---

## Review Notes — 2026-08-05 (second external validation pass — literature landscape check)

A separate session did a second pass: re-verified every claim in the 2026-07-22 notes above against primary sources (all held up — Wu et al.'s "Greedy Pitfall" term, the Gumbel/Dirichlet tradeoff, the small-model caveat, and the `repetition_penalty` correction are all accurate, not paraphrased-into-existence). It then searched for whether the project's two core hypotheses had already been directly tested elsewhere. I independently re-verified the specific papers below by pulling their abstracts directly (not just trusting the prior session's summary) — all are real and accurately characterized.

**Dropped: the CSP exam-scheduler ground-truth plan.** Per instruction, this is no longer part of the project and shouldn't be referenced going forward. Practically, this also turns out not to be a loss: neither of the two open gaps identified below requires a new ground-truth task at all — both are answerable with the existing harness plus the noise parameter already queued in Next Steps.

**Track (2) has already been directly tested — at real scale, on the exact method here.**

Rizvi-Martel, Rabusseau & Mosbach, *The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models* (arXiv:2604.06374, preprint, Apr 2026). Tests exactly this project's question — whether models leverage superposition when given continuous reasoning tokens — across three regimes: training-free (Soft Thinking, this project's method), fine-tuned (Coconut), and from-scratch training. Using logit lens and entity-level probing, they find only from-scratch-trained models show real superposition; in the training-free regime (vanilla Soft Thinking) it collapses within the first few layers or isn't used at all, with models finding shortcut solutions instead. They attribute this to (i) pretraining biasing models to commit to a token in the last layers, and (ii) model capacity strongly affecting which solutions a model favors. One finding worth flagging directly: they note many soft-thinking mixtures blend semantically *unrelated* tokens (punctuation vs. words with similar logits) — i.e. syntactic uncertainty, not alternative-path exploration — which questions whether token-level superposition was ever the right thing to look for, and suggests strategy-level superposition (over entire reasoning approaches, not individual tokens) as an open direction.

**Consequence:** running Exp 1–3 as originally scoped — "does vanilla soft-thinking blend at reasoning forks, checked by manually scanning weights on Qwen 0.5B/1.5B" — is not a novel contribution as-is. It's already been answered, more rigorously, at larger scale. Treat the existing Exp 1–3 results as a useful internal replication/sanity-check of the harness, not as the paper's contribution.

**Two gaps neither paper covers — this is where the project's actual contribution now sits:**

1. **Does noise restore genuine superposition, or just accuracy?** Wu et al. (arXiv:2508.03440) show Gumbel-softmax noise fixes the *performance* regression from the Greedy Pitfall. Rizvi-Martel et al. only run their interpretability battery (logit lens / entropy / entity probing) on *vanilla* (noise-free) soft thinking. Nobody has run that same battery on the noised variant to check whether it actually holds multiple candidates jointly at a given step, versus just getting lucky across independently-randomized rollouts that each individually still collapse. This is directly buildable on top of Next Step 1 (the noise parameter) once it lands.
2. **True causal patching on soft thinking, not just correlational intervention.** Rizvi-Martel et al.'s "intervention" experiment swaps the input soft-token embedding for a discrete one and measures downstream KL divergence / cosine similarity — that tests "does changing the input change the output" (trivially yes), not whether a specific *internal hidden state* is load-bearing. The mid-layer overwrite-and-check-if-the-answer-flips test that Concepts §4.3 already describes as the real causal question is open across everything found in this search.

Combined, the project's new specific question is: **does noise-injected soft thinking produce hidden states that are both interpretability-visible as multi-candidate (logit lens) *and* causally load-bearing (patching), or does noise just add beneficial randomness across separate, still-individually-greedy rollouts?**

**Track (3) is less crowded but has one close neighbor worth reading before finalizing scope.**

Jin, Yang & Wang, *Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories* (arXiv:2607.06648, ~Jul 2026). Tracks how faithfulness evolves across saved training checkpoints for different latent-reasoning paradigms (CoT vs. NoCoT vs. CODI vs. Coconut), using a counterfactual input edit plus a noise-ablation activation patch on the latent reasoning steps. Finds the causal contribution of latent reasoning to the final answer decays across training, and that this trajectory diverges by answer format (decaying on binary-choice, rising on open-ended decoding). Methodologically this is very close to the plan here (causal patching + tracking across training), but the axis varied is *training paradigm*, not *SFT vs. RL-only* — that specific axis still looks open.

Aswal, Ferraz, Zhou & Peyrard, *Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models* (arXiv:2606.12689, Jun 2026) makes an adjacent argument on Coconut/CODI: patterns that look like evidence of internal reasoning (e.g. BFS-like frontiers) also show up in matched controls that lack the mechanism supposedly producing them, and don't always causally affect behavior — reinforcing that decodability alone (logit lens, attention) can't establish mechanism; causal tests are required. Not about Soft Thinking or the SFT/RL axis directly, but directly relevant methodological backing for why this project's patching harness (not just the lens) is the right design.

**RL-training side is now more tractable than previously assessed.** Butt, Kwiatkowski, Labiad, Kempe & Ollivier, *Soft Tokens, Hard Truths* (arXiv:2509.19170) — the first scalable method to RL-train continuous-CoT models without distilling from discrete CoT — works by injecting Gaussian noise into the continuous input embedding during rollouts to give RL exploration something to work with (conceptually the same noise-injection idea as Wu et al.'s fix for Greedy Pitfall, applied at training time instead of inference time), then training with a standard RL objective (REINFORCE-style). Reported on Llama/Qwen up to 8B, but the method itself has no special scale requirement — it's a modification to the input embedding plus a policy-gradient loss, which is plausibly implementable at small scale (a handful of RLVR steps on Qwen2.5-0.5B/1.5B against a cheaply-checkable reward, e.g. exact-match on short arithmetic/logic prompts) rather than requiring full-scale training infrastructure. This makes a real SFT-vs-RL-only comparison on this project's own model sizes more realistic than previously assessed, though still the heavier of the two tracks to build.

References added: Rizvi-Martel, Rabusseau & Mosbach 2026, *The Illusion of Superposition?*, arXiv:2604.06374. Jin, Yang & Wang 2026, *Final Checkpoints Are Not Enough*, arXiv:2607.06648. Aswal, Ferraz, Zhou & Peyrard 2026, *Observable Patterns Are Not Explanations*, arXiv:2606.12689. Zhu, Hao, Hu, Jiao, Russell & Tian 2025, *Reasoning by Superposition*, arXiv:2505.12514 (theory origin of the superposition claim; ProsQA/graph-reachability). Butt, Kwiatkowski, Labiad, Kempe & Ollivier 2025, *Soft Tokens, Hard Truths*, arXiv:2509.19170.

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

### Exp 4 — Gumbel noise (τ=0.5), 100-step budget (prior session — gap, not fully logged)
- **Status:** run before this file's noise/seeding infrastructure existed. No seed was set (PyTorch's global RNG was whatever state it happened to be in) and the full per-step weight trace wasn't preserved for this log. Flagging this explicitly per this log's own opening rule ("update after every verified experiment, don't reconstruct from memory") rather than back-filling numbers that aren't actually verified.
- **What's reliably known:** same water-jug prompt; weight dips landed closer to the fill-3-vs-fill-5 decision region than prior vanilla runs; hard-phase coherence appeared degraded relative to vanilla.
- **Why it doesn't support a conclusion:** single unseeded run (n=1), and at the time had no same-budget vanilla control — Exp 3 was still capped at 40 steps, so "noise causes this" was confounded with "more steps causes this."
- **Superseded by:** Exp 5 (same-budget vanilla control, below) and the in-progress seeded Gumbel sweep (see Harness Update and Next Steps).

### Exp 5 — Vanilla, 100-step budget (same-budget control for Exp 4)
- **Prompt:** water-jug puzzle (same as Exp 2–4).
- **Soft steps:** 100. **Noise:** None. **Seed:** 0 (inert — vanilla path makes no RNG calls; recorded for provenance only, see Harness Update).
- **Result:** reaches case-bifurcation content Exp 3's 40-step run never got to — "...Let's consider the two possible [outcomes]... Case 1: The 5-liter jug ends full...". Lowest weight in the run: step 94 `' starts'` = 0.159, inside a sustained low-confidence cluster spanning steps 93–96 (`' You'` 0.372, `' starts'` 0.159, `' to'` 0.411, `' fill'` 0.381) — four consecutive sub-0.5 weights back to back, unlike the isolated single-token dips in Exp 2/3.
- **Hard-phase output contains a logic error:** claims the 5-liter jug is filled from the 3-liter jug, then states both jugs end up at their original amounts (5L and 3L) simultaneously — violates water conservation (the 3-liter jug can't stay at 3L after being poured out).
- **Interpretation:** budget alone — no noise required — is enough to reach the case-bifurcation region; Exp 3's failure to reach it was pure budget exhaustion, confirming fix option (a) above. This also exposes the confound in Exp 4: its "dips land at decision-relevant tokens" finding was only ever compared against Exp 3's 40-step run, never a same-budget vanilla control. This run is that missing control.
- **Caveat on the steps-93–96 dip:** it's phrasing uncertainty about narrating a hypothetical Case 1 outcome ("You starts to fill...") — the model hasn't committed to an actual first action yet at this point, it's still describing a hypothetical case. Closer to reasoning-relevant than Exp 3's preamble dips, but still not the literal fork the prompt asks about.

### Exp 6 — Vanilla, budget sensitivity (100 vs. 120 steps, same seed)
- **Prompt:** water-jug puzzle. **Noise:** None. **Seed:** 0 (inert, as in Exp 5).
- **Result:** steps 0–99 are byte-identical to Exp 5, as expected — the vanilla blend is a deterministic weighted sum, so extending the budget doesn't change earlier steps. New dips appear in the extended window (100–119): `' with'` 0.337, `' water'` 0.518, `' from'` 0.431, `' Now'` 0.469, `' you'` 0.428, and `'5'` at step 118 = 0.261 — the second-lowest weight seen across both runs, and it's reporting a liter count, closer to quantitatively decision-relevant content than most earlier dips.
- **Both runs' hard-phase outputs contain the same water-conservation logic error found in Exp 5**, just worded differently after the soft→hard switch (100-step run: "the 5-liter jug has 5 liters... the 3-liter jug has 3 liters"; 120-step run: "you have 5 liters in the 5-liter jug and 3 liters in the 3-liter jug. You can then pour the 3 liters..."). Every overlapping top-1 token choice matches between the two runs — only the text *after* the soft→hard switch point differs.
- **Interpretation:** two findings. (1) The same logic error recurring at two different budgets, reworded, suggests a real reasoning ceiling for this model on this task rather than a budget-starvation artifact — extending the budget didn't fix or dissolve it. (2) `soft_steps` isn't just "how far into the reasoning you get to observe" — on a fully deterministic, noise-free run, it changes the final generated text itself, since where the soft→hard switch lands shifts what the hard phase then greedily completes from. **Consequence:** the upcoming noise-condition comparison needs `soft_steps` frozen at one fixed value across every run (vanilla/Gumbel/Dirichlet, all seeds), not just "at least 100." **Decision: 100, frozen** — it already reaches the case-bifurcation region.

### Exp 7 — Gumbel noise sweep (τ=0.5), 100-step budget, seeds {0, 1, 2, 42, 123}
- **Prompt:** water-jug puzzle (same as Exp 2–6). **Noise:** Gumbel-softmax, τ=0.5. **Soft steps:** 100 (frozen, per Exp 6 decision). **Anchor:** Exp 5 (vanilla, 100-step, same prompt).
- **Bug found — frozen budget vs. EOS collide.** Seeds 2 and 123 both emit `<|im_end|>` well inside the 100-step budget (step 86, weight 0.850 and 0.996 respectively) but the loop has no early-stop under the frozen-budget design, so it keeps blending for 14 more steps. The hard phase then greedily continues from whatever post-EOS state that leaves, producing unrelated text (an unrelated word-problem for seed 2, an unrelated algebra problem for seed 123). **Steps 87–99 of seeds 2 and 123 are void for any downstream comparison** (weight-trace, entropy, lens, patching) until this is resolved. Decision on how to resolve it is open — see Next steps below.
- **Per-seed weight-trace summary** (lowest points; jug-choice digit tokens noted separately):
  - **Seed 0:** min weight 0.386 (step 5, `' transferring'`), second-lowest 0.498 (step 66, `':\n\n'`). Never cleanly states a first action — opens with garbled "the problem of transferring exactly 4 liters... from a 3-liter jug to a 5-liter jug" framing. Jug-identity digit tokens (steps 15, 21, 77, 83, 89, etc.) all pinned at 1.000. Output as captured trails off mid-sentence before a checkable final claim. Minor logging anomaly: the diagnostics printout for this seed skips step 79 (jumps 78→80) even though step 79 is present in the generation listing above it — worth a quick check on the diagnostic loop, may just be a display artifact.
  - **Seed 1:** min weight 0.415 (step 79, `' can'`). Explicitly frames two options in text, but the two options as written are incoherent — "We can either empty the 5-liter jug and pour water into the 5-liter jug until it is full, or... into the 3-liter jug until it is full" (same jug named as both the emptied and the filled target in option 1). **Notably, step 93 — the digit token distinguishing which jug gets poured into — has weight 0.758**, the closest any run in this sweep comes to a real sub-0.8 dip landing directly on a jug-identity token rather than a transition/framing word. This coincides with the one run that also produced jug-identity incoherence in its output. Single occurrence (n=1); not yet confirmed as causally meaningful.
  - **Seed 2 (EOS-contaminated after step 86):** min weight 0.342 (step 19, `' easiest'`), inside a garbled justification clause ("This is because [the] easiest... straightforward way..."). Commits early and cleanly to "fill the 5-liter jug first" (digit token at step 11 = 1.000, no hesitation). Ends with a clear capacity violation: "the 3-liter jug will still have 4 liters of water remaining" (step 80, `'4'`, weight 0.629 — the erroneous quantity token itself is one of the lower-confidence points in this run, though still >0.6).
  - **Seed 42:** min weight 0.284 (step 39, `'.\n\n'`) — **the lowest top-1 weight across the entire sweep**, landing on the paragraph break immediately before the model lists two explicit numbered scenarios. Second-lowest 0.481 (step 31, `' scenarios'`). Cleanly presents "1. Transfer water from the 5-liter jug to the 3-liter jug. 2. Transfer water from the 3-liter jug to the 5-liter jug," then works Scenario 1 into a capacity violation ("the 3-liter jug will have 5 liters of water"). **Step 55, a digit token (`'3'`, choosing the transfer amount) = 0.726** — second instance in this sweep of a sub-0.8 dip landing on a digit rather than a framing word.
  - **Seed 123 (EOS-contaminated after step 86):** min weight 0.510 (step 3, `' I'`). Opens with a fill/empty terminology confusion — "The first thing I would do is empty the 5-liter jug into the 3-liter jug" despite the prompt stating both jugs start empty. Given that (flawed) premise, the subsequent arithmetic is internally consistent through to "4 liters of water in the 3-liter jug," then hits EOS at step 86 with weight 0.996 (high-confidence stop, unlike seed 2's less-confident 0.850).
- **Cross-seed synthesis:**
  - Every run contains a distinguishable reasoning error, but they're not uniform in kind: seeds 2 and 42 show clear-cut capacity/volume-conservation violations (matching the Exp 5/6 finding exactly); seed 1 shows a jug-identity/redundant-option incoherence; seed 123 shows an initial-state/terminology confusion; seed 0's trace is inconclusive (cut off before a checkable claim). Noise does not fix or dissolve the reasoning ceiling identified in Exp 5/6 — it just reshuffles which specific error shows up.
  - Consistent with Exp 5/6 and the vanilla baseline: the large majority of sub-0.9 dips sit on transition, connective, or framing tokens (paragraph breaks, "either/or," justification clauses), not on the token that actually names which jug to act on — those stay pinned at ~1.0 in 8 of 10 digit-selection instances checked.
  - The two exceptions (seed 1 step 93 = 0.758, seed 42 step 55 = 0.726) are the first digit-level, plausibly decision-relevant dips seen anywhere in this project so far — more promising than anything in Exp 1–6. Both are single occurrences; nothing here yet distinguishes "genuine internal blending" from "noise happened to land there."
  - **Comparison to the Exp 5 vanilla-100 anchor:** Exp 5's lowest region was a *sustained* 4-consecutive-step sub-0.5 cluster (steps 93–96: 0.372, 0.159, 0.411, 0.381). No seed in this sweep reproduces a cluster of that shape — all five produce isolated single-step dips instead. **Caveat on this comparison:** because noise can promote a different candidate above the original top pick (per the Harness Update note below), the actual token sequence diverges from the vanilla path from very early in the run and differs seed-to-seed — so step-index alignment with Exp 5 isn't meaningful past the first few tokens. Any future comparison needs to be content-aligned (locate the analogous point in each run's own text), not step-aligned.
- **Interpretation:** no run in this sweep shows a balanced, decision-point-level weight split of the kind the superposition claim actually predicts. What noise appears to be perturbing is the same class of syntactic/framing uncertainty already flagged in vanilla runs and in Rizvi-Martel et al., with two possible early exceptions (seed 1 step 93, seed 42 step 55) that land on digit tokens specifically. This is not yet evidence for or against genuine superposition — only logit lens + causal patching on those two flagged steps can settle that.
- **Status:** Gumbel portion of Next Step 2 complete for 5 seeds, with 2 of 5 partially void past step 86. Dirichlet (γ=1.0) pass not yet run. EOS-handling decision (see Next steps) required before this experiment can be treated as final.

### Exp 7b — Follow-up: EOS fix validated, entropy added, full 5-seed confirmation (2026-08-27)

Re-ran all 5 seeds (`{0, 1, 2, 42, 123}`) on the fixed harness (see Harness To-Do — resolved, below, for the fix itself). Same noise mode/τ/budget as the Exp 7 entry above — this follow-up supersedes that entry's "void until resolved" caveat with clean, validated numbers.

- **EOS/void confirmation:** seeds 2 and 123 both cleanly fire `eos_fire_step = 86` (matching the original run exactly — the fix changes tracking, not the underlying computation), with steps 87–99 now correctly flagged `void` instead of silently contaminating the record. Seeds 0, 1, 42 confirmed `eos_fire_step = None`, 0 void steps.
- **Content unchanged from the original run, as expected:** seeds 2 and 123 still both land on the same capacity-violation claim (3-liter jug ending with 4 liters); seed 42 still claims the 3-liter jug holds 5 liters after a 3-liter transfer.
- **Entropy available for the first time across the full sweep.** Highest points: seed 2 step 19 `' easiest'` (1.917), seed 1 step 88 `' pour'` (1.516), seed 42 step 39 sentence-break token (1.418), seed 0 step 66 `':\n\n'` (1.407) — all phrasing/style choices, confirming the existing weight-based reading numerically rather than changing it.
- **Two additional digit-level candidates found**, alongside the original seed 1 step 93 / seed 42 step 55 pair (see Standing observations):
  - Seed 2, step 80: `'4'`, weight 0.629, entropy 0.689 — the erroneous quantity token itself, sitting right where the capacity-violation claim is made.
  - Seed 123, step 62: `'1'`, weight 0.676, entropy 0.923 — a quantity-value dip inside the (flawed-premise) arithmetic chain.
- **Status:** Gumbel half of Next Step 2 is now fully complete and clean — harness fix and entropy logging both validated. Dirichlet (`γ=1.0`) pass across the same 5 seeds is the remaining piece — in progress.

---

### Exp 8 — Dirichlet resampling sweep (γ=1.0), 100-step budget, seeds {0, 1, 2, 42, 123} (2026-08-28)
- **Prompt:** water-jug puzzle (same as Exp 2–7b). **Noise:** Dirichlet, γ=1.0 (repo default). **Soft steps:** 100 (frozen, per Exp 6). **Anchor:** Exp 5 (vanilla) and Exp 7b (Gumbel, τ=0.5) — same prompt/budget/seeds, so this is the third leg of the noise-condition comparison from Next Step 2.
- **Harness confirmation:** EOS-void tracking and entropy logging (landed in the 2026-08-26/27 harness update) both behaved correctly on this sweep — no new bugs found.

**Per-seed summary:**
- **Seed 0 (0 void steps):** Min weight 0.395 (step 13, `' contents'`). Digit dips: step 86 `'2'` = 0.680 — lands exactly on the erroneous "2 liters remaining" claim after pouring the 3-liter jug out. Content: opens with a garbled meta-strategy framing, then a first move ("pour from the 3-liter jug into the 5-liter jug") that's incoherent given both jugs start empty per the prompt, ending in an internal-arithmetic contradiction.
- **Seed 1 (0 void steps):** Min weight 0.383 (step 21, `' let'`). **Four digit dips <0.8** — step 55 `'5'`=0.583, step 93 `'4'`=0.613, step 69 `'4'`=0.733, step 88 `'4'`=0.762 — the most of any seed. Content: hallucinates a **nonexistent third jug** ("the 4-liter jug" — not in the problem) and states a physically impossible pour (4 liters into a 3-liter jug). Notably, the digit dips cluster tightly around the fabricated jug's references — confidence dips precisely where the model is inventing content that isn't in the prompt.
- **Seed 2 (EOS at step 62, weight 0.847; 37 void steps):** Qualitatively noisier throughout than the other four seeds — few extended 1.000 stretches. **Sustained low-confidence cluster at steps 20–25** (`' until' 0.531, ' until' 0.521, ' it' 0.998, ' reaches' 0.517, ' full' 0.489, '.\n\n' 0.510`) — the first multi-step sustained dip seen in *any* noise condition outside Exp 5's vanilla baseline, and it sits on a genuinely decision-relevant clause (when to stop pouring). One digit dip: step 47 `'7'` = 0.867, landing on a quantity that exceeds *both* jugs' total capacity — the most severe capacity violation seen in the project so far. EOS itself fired at lower confidence (0.847) than either Gumbel EOS event in Exp 7/7b (0.850, 0.996) — n=1, not conclusive, but consistent with this seed's generally noisier trace.
- **Seed 42 (0 void steps):** Min weight 0.405 (step 24, `'.'`) — highest entropy in the sweep (1.385). **Zero digit-level dips** — every digit token pinned ≥0.99, the only seed like this. Content: reaches the "Scenario 1/2" bifurcation structure (matching vanilla/Gumbel) and reproduces the same volume-conservation error type from Exp 5/6/7. This seed's overall shape (framing-only dips, no digit dips, familiar error type) looks the most like a vanilla/Gumbel run of any seed here.
- **Seed 123 (0 void steps):** Min weight 0.350 (step 17, `' possible'`) — lowest single weight in the sweep. Highest entropy value anywhere in the sweep: 1.449 (step 61, `' removing'`). **Six digit dips <0.8** — step 47 `'3'`=0.599, step 89 `'5'`=0.620, step 28 `'5'`=0.669, step 97 `'2'`=0.707, step 53 `'3'`=0.731, step 76 `'1'`=0.766. Content: states an impossible operation ("removing 4 liters from the 3-liter jug" — exceeds its capacity), then a non-sequitur quantity claim.

**Cross-seed synthesis:**
- **Digit-level dips are far more frequent under Dirichlet than Gumbel on the same prompt/budget/seeds.** Using the same <0.8 threshold Exp 7 used: this sweep has **11 digit dips across 3/5 seeds** (0, 1, 123) vs. **2 digit dips across 2/5 seeds** in the Gumbel sweep — roughly 5x more.
- **Caveat, stated plainly:** this is not yet evidence of more genuine superposition. Every seed's output is numerically broken somewhere (hallucinated jugs, impossible pours, contradictory quantities), so a dip landing on "an erroneous digit" is close to guaranteed by construction — nearly every digit in these outputs *is* part of a wrong claim. Can't distinguish "Dirichlet surfaces real quantity-level uncertainty" from "Dirichlet is just noisier everywhere and this is an arithmetic-heavy text" without logit lens + causal patching on these specific steps.
- **Seed-to-seed variance in overall noisiness is much larger than Gumbel showed** — seed 2 is choppy nearly throughout; seed 42 is close to indistinguishable from a vanilla-style run. This is the textbook signature of the γ tradeoff already documented in the Concepts doc (§3.3) and the 2026-07-22 review notes: at a mid γ, individual draws land anywhere between near-smooth and near-one-hot, and which one you get is unpredictable — not something γ=1.0 reliably splits the difference on.
- Every seed still fails to solve the puzzle correctly (5/5) — consistent with vanilla and Gumbel. If anything, error severity looks worse here: seed 1's hallucinated jug and seed 2's over-total-capacity quantity are both new failure modes not seen in vanilla or Gumbel runs.
- Only 1/5 seeds hit EOS early (seed 2, step 62) vs. Gumbel's 2/5 (both step 86) — small-n.

**Interpretation:** Same top-line conclusion as Exp 7/7b — no seed shows a clean, balanced decision-point split of the kind the superposition claim predicts; framing/transition-token dips still dominate. What's new is the digit-dip frequency gap between the two noise mechanisms, which is itself an interesting descriptive finding but **not yet interpretable as more or less "real" superposition** — that requires the logit lens + causal patching infrastructure this sweep was explicitly meant to motivate.

**Status:** Next Step 2 (noise-condition comparison — vanilla, Gumbel, Dirichlet, all at 5 seeds, all with entropy logging) is now **fully complete**. Priority candidates for the first logit-lens pass, in order of how decision-relevant they look: seed 1's four-dip cluster around the hallucinated jug (steps 55, 69, 88, 93), seed 123's six-dip cluster (steps 28, 47, 53, 76, 89, 97), seed 2's sustained steps 20–25 cluster, plus the two Gumbel candidates already flagged in Exp 7 (seed 1 step 93, seed 42 step 55).

---

## Standing observations (to revisit when drafting)

- So far, every genuine sub-0.9 dip has coincided with plausible, human-identifiable alternative continuations (synonym choice, punctuation choice, opening-phrase choice) — consistent with *some* real blending mechanism, but none of it yet at the level of "two different solution paths" that the superposition claim is actually about.
- Model verbosity/preamble length is a real confound for this experiment design — need enough soft-step budget to reach the actual fork, which varies a lot by prompt and needs to be checked per-prompt rather than assumed.
- Still pending: logit-lens decoding of hidden states (not just the already-known top-1 mixture weight) at each step, and causal patching to test whether a step's internal representation actually drives the outcome or is decorative.
- The Greedy Pitfall (Wu et al., arXiv:2508.03440) predicts exactly the shallow-dip pattern seen so far in noise-free soft thinking — treat as a competing/complementary hypothesis alongside the preamble-budget explanation, not a replacement for it. Needs a controlled comparison (vanilla vs. noised, same prompt, same budget) to separate the two.
- Water-jug hard-phase outputs (Exp 5, Exp 6) consistently produce the same volume-conservation logic error regardless of soft-phase budget (100 vs. 120 steps) — treat as a likely genuine reasoning ceiling for this model on this task, not a budget artifact, when interpreting later noise-condition results.
- `soft_steps` changes the final generated text even with no noise and a fixed seed, because it changes where the soft→hard switch lands (Exp 6). Any cross-condition comparison (vanilla vs. Gumbel vs. Dirichlet) must hold `soft_steps` fixed at one value, not just "at least N."
- Seed 1 (step 93) and seed 42 (step 55) in the Exp 7 Gumbel sweep are the first digit-level (jug/quantity-token) sub-0.8 weight dips seen in the project — every prior dip, vanilla or noised, has landed on a framing/transition token instead. Flag both as priority candidates for logit lens + causal patching once that infrastructure exists.
- Dirichlet (γ=1.0, Exp 8) produces ~5x more sub-0.8 digit-token dips than Gumbel (τ=0.5, Exp 7/7b) on the identical prompt/budget/seeds (11 vs. 2), but also much higher seed-to-seed variance in overall trace noisiness (seed 2 choppy throughout vs. seed 42 near-vanilla) — consistent with the known Dirichlet γ tradeoff rather than necessarily indicating more genuine blending. Not yet interpretable without logit lens, since nearly every digit in these outputs is part of a wrong claim, making dip-on-error correlation close to guaranteed rather than informative on its own.

## Review Notes — 2026-08-24 (third external check — is this still novel?)

Ran a fresh literature check (nothing since Aug 5 had closed the gap, but the picture got sharper). Decisions made this session:

**Finding 1 — Wu et al. already hints at a negative result.** Their logit-lens experiment (Section on branching points) manually built a *balanced* soft token (0.6/0.4 split) and ran logit lens across layers — a hand-made stand-in for what noise injection should produce. Result: both candidate paths show up in the first 2–3 layers, but the top-1 path still climbs to full dominance by later layers. Not causal patching, not real noise-injected generation, but a warning sign our own noise experiment may land the same way.

**Finding 2 — Rizvi-Martel already ran a lighter-weight intervention.** Their Section 4.2: every 50 soft-thinking steps, run a second forward pass swapping the soft mixture for the discrete argmax embedding, compare resulting hidden states via logit lens (KL divergence, cosine similarity). Found soft ≈ discrete — barely any difference. This is *not* the causal patching we're planning (no hidden-state overwrite, no check of whether the final decoded answer flips) and it's on vanilla (noise-free) Soft Thinking only. Our planned design is still not done anywhere.

**Finding 3 — real causal patching on latent CoT already exists, but on a different architecture.** Two papers run genuine do-interventions (hidden-state overwrite + downstream check) on **Coconut/CODI** (raw hidden-state recurrence — architecturally different from Soft Thinking's embedding-mixture method):
- Li et al., *Dynamics Within Latent Chain-of-Thought* (arXiv:2602.08783) — causal leverage is uneven across steps, a few steps dominate; there's a gap between early output bias and late representational commitment.
- Zhang et al., *Do Latent Tokens Think?* (arXiv:2512.21711) — Coconut tokens barely respond to steering, carry little reasoning-critical info, models lean on shortcuts instead.

Neither paper touches Soft Thinking or noise injection. But both are useful **comparison targets** once our own causal-patching results are in hand.

**Decision — no cross-architecture reimplementation.** Originally considered building a second harness for Coconut/CODI to compare directly. Decided against it — too much extra engineering for a CPU/solo setup. Instead: **compare our Soft Thinking + noise + causal-patching results against the *published* Coconut/CODI findings above** (Li et al., Zhang et al.), not against a re-run. Cheaper, still gives the paper a "how does this compare across the two latent-CoT families" angle.

**Decision — reframe the contribution.** Position the paper as closing a specific, named gap: Rizvi-Martel tested vanilla Soft Thinking with logit lens + one KL-swap check; Wu et al. tested noise for *performance* only; nobody has run noise-injected Soft Thinking through logit lens **and** real causal patching together. This is safer than a "we discovered X" framing — it's explicit about what's new and doesn't depend on being first.

**Decision — Track (3) formally moved to Future Work** (see section below). Not touched until Track (2) results exist.

**Correction:** an arXiv ID for Aswal et al. was written inconsistently in an earlier internal note (2606.12289 vs. the correct 2606.12689, which this file already uses correctly — no file change needed, just flagging so the wrong version doesn't get copied elsewhere).

References added: Li, Bai, Chen, Li, Yang, Lin & Zhang 2026, *Dynamics Within Latent Chain-of-Thought*, arXiv:2602.08783. Zhang, Tang, Ju, Duan & Liu 2025, *Do Latent Tokens Think?*, arXiv:2512.21711.

---

## Harness Update — 2026-08-26 (noise injection finalized + seeding added)

Closes out Next Step 1 below and settles the budget question Exp 5/6 raised.

**`soft_thinking_step` noise parameter (`None` / `"gumbel"` / `"dirichlet"`)** is implemented: inserted after top-k/top-p renormalization, before the embedding blend, as originally planned. `kept_idx` (which tokens are in the mixture) stays fixed; noise only reshuffles the weights.

**Bug found while validating it:** once noise is applied, `sorted_probs` is no longer guaranteed sorted high-to-low (noise can promote a lower-ranked candidate above the original top pick), so the diagnostic/logging code's earlier assumption that "index 0 is the top weight" silently broke. Fixed by always computing the true top weight/token dynamically (`weights[0].max(dim=-1)`) instead of assuming position 0.

**Seeding added** — a `SEED` constant plus `torch.manual_seed(SEED)` called immediately before the `generate_soft_then_hard` call. Important asymmetry to keep in mind going forward: the vanilla path (`NOISE_MODE=None`) never calls `F.gumbel_softmax` or `Dirichlet(...).sample()`, so it never touches the RNG — the blend is a pure deterministic weighted sum. **Seeding only has any effect once `NOISE_MODE` is `"gumbel"` or `"dirichlet"`.** Exp 5 and Exp 6's `seed=0` is recorded for provenance but was inert in both runs, which is exactly why they were reproducible byte-for-byte on their overlapping steps.

**Frozen settings for the noise-condition comparison** (per Exp 6's finding that `soft_steps` itself affects the output): `soft_steps=100`, `GUMBEL_TAU=0.5` (repo default; also what Exp 4 used), `DIRICHLET_GAMMA=1.0` (repo default). Only `SEED` varies, across `{0, 1, 2, 42, 123}` — a small-but-sufficient robustness check, not a tuned set. **Sweep complete — results logged as Exp 7b (Gumbel) and Exp 8 (Dirichlet).**

---

## Harness To-Do — RESOLVED (fixed and validated in Exp 7b, 2026-08-27; confirmed again in Exp 8, 2026-08-28)

**EOS/frozen-budget conflict:** 2 of 5 seeds in the original Exp 7 hit EOS inside the 100-step frozen budget (seed 2 at step 86, seed 123 at step 86) and the soft loop had no early-stop, so it kept blending past a completed answer, contaminating the hard-phase continuation. **Fix implemented:** kept the frozen `soft_steps=100` budget (still needed per Exp 6), and added tracking of the step index where EOS actually fires per seed, with all steps after that index flagged void for weight-trace, entropy, lens, and patching purposes — rather than discarding whole seeds or truncating the run early. This preserves budget comparability across seeds while keeping downstream analysis clean. **Validated:** Exp 7b re-confirmed clean void-tracking on seeds 2/123; Exp 8 confirmed it again on a fresh noise condition (seed 2's void tracking behaved correctly there too).

**Entropy logging (Next Step 3):** landed in the same harness edit as the EOS fix, as planned — `weights`/`kept_idx` were already in memory at each step, so no separate re-run was needed. **Validated:** working across both Exp 7b and Exp 8.

**Consequence — resolved:** the Dirichlet (γ=1.0) pass (Exp 8) and the Gumbel seed 2/123 re-run (Exp 7b) both completed on the same fixed harness version, as intended — no split between pre-fix and post-fix results.

---

## Next steps (Track 2 — active)

Steps 1–6 build the shared toolkit (noise injection → logit lens → causal patching) and answer Track 2. Step 7 is new as of 2026-08-24 (comparison against published Coconut/CODI results, not a reimplementation). **Steps 1–3 are now complete as of Exp 8 (2026-08-28) — step 4 (logit lens) is the current focus.**

1. **[DONE — 2026-08-26, see Harness Update above]** Add a `noise` parameter (`"gumbel"` / `"dirichlet"` / `None`) to `soft_thinking_step`, inserted after top-k/top-p renormalization and before the embedding blend. Default stays `None` so Exp 1–3 remain reproducible as a baseline.
2. **[DONE — Exp 7b (Gumbel, 2026-08-27) + Exp 8 (Dirichlet, 2026-08-28), both 5 seeds on the fixed harness.]** Frozen-budget/EOS collision (seeds 2 and 123 in the original Exp 7) resolved via EOS-index void-tracking (see Harness To-Do, resolved). Full three-condition comparison — vanilla (Exp 5), Gumbel (Exp 7b), Dirichlet (Exp 8) — now complete on the same prompt/budget/seeds.
3. **[DONE — landed with the EOS fix, validated across Exp 7b and Exp 8.]** Add per-step entropy logging (`-Σ p·log p` over the renormalized kept weights) alongside the existing weight log. Cheap, and turns fork-finding from manual scanning into something plottable across a whole run.
4. **(current focus)** Add `output_hidden_states=True` capture to `generate_soft_then_hard`; write a `logit_lens(model, hidden_vector, top_n)` utility that projects a hidden state through `model.model.norm` + `model.lm_head`. Note this is an approximation (the final norm wasn't trained for intermediate layers) — sanity-check by confirming the last layer's lens output exactly matches the model's actual argmax choice. Priority candidate steps to check first (from Exp 7/7b/8): seed 1 Dirichlet steps 55/69/88/93, seed 123 Dirichlet steps 28/47/53/76/89/97, seed 2 Dirichlet steps 20–25, seed 1 Gumbel step 93, seed 42 Gumbel step 55.
5. Once (4) is in place, decode the full top-k mixture *and* the internal hidden-state lens for the same step in the same run, side by side — first real look at whether they agree. Do this for vanilla **and** noised runs from step 2 — this comparison is the project's core contribution: does the noised variant's lens output show genuinely multi-candidate content that the vanilla variant doesn't?
6. Causal patching harness: a forward hook on `model.model.layers[L]` that overwrites the last-position hidden state at a chosen generation step; compare patched vs. unpatched continuation on the hard phase. Run on both vanilla and noised conditions — the key check is whether patching a noised-condition hidden state changes the outcome in the direction its lens content predicted (real, load-bearing multi-candidate content), versus changing nothing or changing unpredictably (noise adding beneficial randomness across rollouts without real per-step superposition).
7. **Comparison step (new):** once 6 is done, write up a short comparison section against Li et al. (arXiv:2602.08783) and Zhang et al. (arXiv:2512.21711) — same causal-patching *style* of question, different latent-CoT architecture (Soft Thinking's embedding-mixture vs. Coconut/CODI's hidden-state recurrence). Frame as: does noise change the picture for the embedding-mixture family the way it doesn't for the hidden-state-recurrence family (per their negative results)?

---

## Future Work — Track (3): Faithfulness, SFT vs. RL-only

**Status: deferred. Not started. Revisit only after Track (2) Next Steps 1–7 are complete.**

Plan (unchanged from earlier scoping, kept here for later): reuse the Track (2) harness on a second, RL-trained checkpoint. Small-scale reimplementation of Butt et al.'s noise-injection RL recipe (arXiv:2509.19170) — inject noise into the continuous input embedding during rollouts, train with a REINFORCE-style objective against a cheap, exactly-checkable reward (e.g. short arithmetic/logic tasks with a verifiable final answer) — on the same base model (Qwen2.5-0.5B or 1.5B) already used for the SFT/instruct-tuned baseline. Compare logit-lens decodability and causal-patching load-bearingness between the two checkpoints.

Before starting: read Jin, Yang & Wang (arXiv:2607.06648) closely — their training-trajectory faithfulness study is the closest neighbor — and note explicitly how the SFT-vs-RL-only axis here differs from their training-*paradigm* axis (CoT/NoCoT/CODI/Coconut). Also worth checking: SofT-GRPO (Zheng & Lee, arXiv:2511.06411) is a newer, better-performing alternative to Butt et al.'s RL recipe for soft-thinking-style models — worth a quick comparison before committing to Butt et al.'s method specifically.
