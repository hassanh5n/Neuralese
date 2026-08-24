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

---

## Standing observations (to revisit when drafting)

- So far, every genuine sub-0.9 dip has coincided with plausible, human-identifiable alternative continuations (synonym choice, punctuation choice, opening-phrase choice) — consistent with *some* real blending mechanism, but none of it yet at the level of "two different solution paths" that the superposition claim is actually about.
- Model verbosity/preamble length is a real confound for this experiment design — need enough soft-step budget to reach the actual fork, which varies a lot by prompt and needs to be checked per-prompt rather than assumed.
- Still pending: logit-lens decoding of hidden states (not just the already-known top-1 mixture weight) at each step, and causal patching to test whether a step's internal representation actually drives the outcome or is decorative.
- The Greedy Pitfall (Wu et al., arXiv:2508.03440) predicts exactly the shallow-dip pattern seen so far in noise-free soft thinking — treat as a competing/complementary hypothesis alongside the preamble-budget explanation, not a replacement for it. Needs a controlled comparison (vanilla vs. noised, same prompt, same budget) to separate the two.

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

## Next steps (Track 2 — active)

Steps 1–6 build the shared toolkit (noise injection → logit lens → causal patching) and answer Track 2. Step 7 is new as of 2026-08-24 (comparison against published Coconut/CODI results, not a reimplementation).

1. Add a `noise` parameter (`"gumbel"` / `"dirichlet"` / `None`) to `soft_thinking_step`, inserted after top-k/top-p renormalization and before the embedding blend. Default stays `None` so Exp 1–3 remain reproducible as a baseline.
2. Re-run the water-jug prompt at an extended step budget (~70–80) in three conditions — vanilla, Gumbel-Softmax (`τ=0.5`), Dirichlet (`γ=1.0`) — and compare weight/entropy traces at the fill-3-vs-fill-5 decision point specifically.
3. Add per-step entropy logging (`-Σ p·log p` over the renormalized kept weights) alongside the existing weight log. Cheap, and turns fork-finding from manual scanning into something plottable across a whole run.
4. Add `output_hidden_states=True` capture to `generate_soft_then_hard`; write a `logit_lens(model, hidden_vector, top_n)` utility that projects a hidden state through `model.model.norm` + `model.lm_head`. Note this is an approximation (the final norm wasn't trained for intermediate layers) — sanity-check by confirming the last layer's lens output exactly matches the model's actual argmax choice.
5. Once (4) is in place, decode the full top-k mixture *and* the internal hidden-state lens for the same step in the same run, side by side — first real look at whether they agree. Do this for vanilla **and** noised runs from step 2 — this comparison is the project's core contribution: does the noised variant's lens output show genuinely multi-candidate content that the vanilla variant doesn't?
6. Causal patching harness: a forward hook on `model.model.layers[L]` that overwrites the last-position hidden state at a chosen generation step; compare patched vs. unpatched continuation on the hard phase. Run on both vanilla and noised conditions — the key check is whether patching a noised-condition hidden state changes the outcome in the direction its lens content predicted (real, load-bearing multi-candidate content), versus changing nothing or changing unpredictably (noise adding beneficial randomness across rollouts without real per-step superposition).
7. **Comparison step (new):** once 6 is done, write up a short comparison section against Li et al. (arXiv:2602.08783) and Zhang et al. (arXiv:2512.21711) — same causal-patching *style* of question, different latent-CoT architecture (Soft Thinking's embedding-mixture vs. Coconut/CODI's hidden-state recurrence). Frame as: does noise change the picture for the embedding-mixture family the way it doesn't for the hidden-state-recurrence family (per their negative results)?

---

## Future Work — Track (3): Faithfulness, SFT vs. RL-only

**Status: deferred. Not started. Revisit only after Track (2) Next Steps 1–7 are complete.**

Plan (unchanged from earlier scoping, kept here for later): reuse the Track (2) harness on a second, RL-trained checkpoint. Small-scale reimplementation of Butt et al.'s noise-injection RL recipe (arXiv:2509.19170) — inject noise into the continuous input embedding during rollouts, train with a REINFORCE-style objective against a cheap, exactly-checkable reward (e.g. short arithmetic/logic tasks with a verifiable final answer) — on the same base model (Qwen2.5-0.5B or 1.5B) already used for the SFT/instruct-tuned baseline. Compare logit-lens decodability and causal-patching load-bearingness between the two checkpoints.

Before starting: read Jin, Yang & Wang (arXiv:2607.06648) closely — their training-trajectory faithfulness study is the closest neighbor — and note explicitly how the SFT-vs-RL-only axis here differs from their training-*paradigm* axis (CoT/NoCoT/CODI/Coconut). Also worth checking: SofT-GRPO (Zheng & Lee, arXiv:2511.06411) is a newer, better-performing alternative to Butt et al.'s RL recipe for soft-thinking-style models — worth a quick comparison before committing to Butt et al.'s method specifically.
