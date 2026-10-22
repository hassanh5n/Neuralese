# Neuralese: Paper Log (merged, validated)

Created 2026-10-17 by merging `neuralese-research-log.md` (Exp 1–8, literature notes) and
`neuralese-research-log-continuation.md` (Exp 9–23, audits, decisions). Both originals are unchanged in `docs/`
and in git; they remain the lab notebook. **This file is the paper's source of truth.**

**The rule.** A number may appear in the paper only if it has a row in the claim ledger (§2) and
`python verify_claims.py` recomputes it from `results/`. Status on 2026-10-17: **70/70 checks pass.**
Numbers in §6–§8 are history and must not be copied into the paper.

Contents: 1 Glossary · 2 Claim ledger · 3 Pre-registration record · 4 Setup · 5 Results (Exp 16, 21, 22, 23) ·
6 Pilot work and lessons · 7 Excluded results · 8 Errata · 9 Literature sources · 10 Open items · 11 What was cut

---

## 1. Glossary

| Term | Meaning (paper term in brackets) |
|---|---|
| Soft Thinking | During reasoning, feed the probability-weighted average of the top candidate tokens' embeddings instead of one sampled token (Zhang et al.). |
| Gumbel / Dirichlet noise | Perturb the blend weights before feeding: Gumbel-softmax τ = 0.5, or Dirichlet γ = 1.0 (Wu et al.). Vanilla = no noise. |
| Greedy Pitfall | Vanilla Soft Thinking collapsing into greedy, repetitive decoding (Wu et al.). |
| Cold Stop | Stop thinking when the pre-noise entropy stays below 0.01 for 256 steps (catches repetition loops). |
| Fork | A thinking step where the runner-up holds ≥ 25% of the pre-noise filtered weight. |
| A / B / C | Top candidate [chosen token] / runner-up [runner-up] / control token [control token]. |
| Fork kinds | num-num (digit vs digit), num-word, word-word, classified by whether A and B contain a digit. |
| Snap | How sharply the model resolves a blend α·A + (1−α)·B at a layer: mean\|c(α) − α\| over α = 0, 0.1, …, 1, divided by the same for a perfect switch at α = 0.5. 0 = carried linearly, 1 = clean switch. Rescaled 2026-10-15 (new = old × 1.1). |
| c (readback score) | Position of the model's next-token distribution on the line from pure-B (0) to pure-A (1). |
| Readback [runner-up readback] | Control − blend score when B's continuation is forced in. > 0 means B's own information is read back from the KV cache. |
| Remove / insert | KV patching at the fork position: blend cache with control K/V at layer L (is L needed?) / control cache with blend K/V at L (is L enough?). Reported as share of the blend–control gap closed. |
| Residual | Blend weight not on the top token. Exp 23 arms: hard = one-hot top token (= sampling), soft = full blend (= Gumbel Soft Thinking), rand = residual moved to a random vocabulary token. |
| Option A / Option B | [genuine superposition] / [sampling in disguise]. |

## 2. Claim ledger

Every row is checked by `verify_claims.py` (claim id in the first column). Items: GSM8K test set, by index.
CIs are 95% bootstrap over problems unless marked. Model: DeepSeek-R1-Distill-Qwen-1.5B, seed 0.

| Id | Claim | Value | Raw file (`results/`) |
|---|---|---|---|
| C16.1 | Discrete sampled CoT accuracy, items 0–29, seeds 0/1 | 52/60 (87%), no answer 2, Cold Stops 0 | exp16/gsm8k_eval.jsonl |
| C16.2 | Vanilla Soft Thinking accuracy | 22/30 (73%), no answer 5, **Cold Stops 14/30** | exp16 |
| C16.3 | Gumbel Soft Thinking accuracy, seeds 0/1 | 52/60 (87%), no answer 2, Cold Stops 0 | exp16 |
| C16.4 | Runs that produced a boxed answer | **141/150** | exp16 |
| C16.5 | Vanilla vs Gumbel seed 0, paired McNemar | 5 vs 1, p = 0.22 (**not significant**) | exp16 |
| C16.6 | Items where Gumbel and discrete differ | 12, 20, 21, 23 (cancel out) | exp16 |
| C21.1 | Forks found / unique, items 10–49 | 694 / 670 | exp21/step4_alpha_sweep_items10-49.jsonl |
| C21.2 | Layer-0 check: c(α) = α at the embedding | max error 1.5e-7 | exp21 step4 |
| C21.3 | Forks per kind | num-num 238 (40 problems), num-word 56 (24), word-word 376 (40) | exp21 step4 |
| C21.4 | Snap at layers 7 / 14 / 21 / final | num-num 0.37 / 0.59 / 0.76 / 0.84; num-word 0.70 / 0.80 / 0.90 / 0.98; word-word 0.44 / 0.54 / 0.65 / 0.77 | exp21 step4 |
| C21.5 | Final-layer snap CI | num-num [0.83, 0.86]; num-word [0.95, 1.00]; word-word [0.75, 0.79] | exp21 step4 |
| C21.6 | 50/50 input: next token is pure A's or pure B's (forks where they differ) | num-num 95%, num-word 98%, word-word 88% | exp21 step4 |
| C21.12 | num-num 50/50 next token: A / B / new / A=B | 96 / 80 / 10 / 52 | exp21 step4 |
| C21.13 | num-num off-line distance at 50/50, final layer (share of A–B gap) | 0.28 | exp21 step4 |
| C21.7–8 | Readback with B's text (control − blend), followed-A forks | num-num +0.44 [0.35, 0.53] (n = 96); num-word +0.11 [0.03, 0.19] (n = 34); word-word +0.33 [0.27, 0.40] (n = 139) | exp21/step5_kv_carryover_items10-49.jsonl |
| C21.9 | Same with A's own text (dormancy) | num-num −0.01, num-word +0.06, word-word +0.02 | exp21 carryover |
| C21.10 | Share of gap closed by layers 0–7, remove / insert | num-num 0.77 / 0.71; word-word 0.77 / 0.91 | exp21/step5_kv_patch_items10-49.jsonl |
| C21.11 | Share closed by layers 12–27, remove / insert | num-num 0.12 / 0.26; word-word 0.08 / 0.14 | exp21 patch |
| C22.1 | Vanilla, items 10–29: fork rate; fed top ≥ 0.9 all steps / forks; median fed top / runner-up at forks; winner flipped | 9.2%; 0.78 / 0.00; 0.56 / 0.33; 0.00 | exp22/exp22_mixtures_items10-29.jsonl |
| C22.2 | Gumbel, same measures | 13.0%; 0.85 / **0.47**; **0.88 / 0.10**; 0.45 | exp22 |
| C22.3 | Dirichlet, same measures | 13.3%; 0.83 / 0.37; 0.83 / 0.16; 0.42 | exp22 |
| C22.4 | Gumbel-max check: Gumbel flip rate vs sampling (1 − mean pre-noise top weight) | 0.45 vs 0.45 | exp22 |
| C22.5 | Vanilla fed weights = pre-noise weights at every step | true | exp22 |
| C23.1 | Accuracy, items 50–149: hard / soft / rand | 86 / 88 / 82 of 100 | exp23/exp23_residual_ablation_items50-149.jsonl |
| C23.2 | Soft vs hard, McNemar | soft-only 6, hard-only 4, p = 0.754; same final answer 84/100 | exp23 |
| C23.3 | Rand vs hard, McNemar | rand-only 3, hard-only 7, p = 0.344; same final answer 84/100 | exp23 |
| C23.4 | Median first token flip vs hard (step) | soft 35.5, rand 66 | exp23 |
| C23.5–6 | Which arm diverges from hard first (paired) | soft first 59, rand first 24, same 17; sign test p = 0.0002 | exp23 |
| C23.7–8 | Mean per-item KL(arm ‖ hard), matched steps (before either arm's first flip) | soft 0.0077, rand 0.0014; soft > rand on 94/100 items | exp23 |

Additional derived values, not yet scripted (add to `verify_claims.py` before citing): Wilson 95% CIs for accuracy
(52/60 [0.76, 0.93]; 22/30 [0.56, 0.86]; 88/100 [0.80, 0.93]; 86/100 [0.78, 0.91]; 82/100 [0.73, 0.88]).

## 3. Pre-registration record

Checked in git on 2026-10-17. Git dates are set by the local machine, so they show order, not an external timestamp.
For stronger evidence, cite the Kaggle notebook version times as well.

| Predictions | Committed | Results first logged | Evidence |
|---|---|---|---|
| P1–P6 (Exp 21) | `7d41cd4`, 2026-10-07 08:00 | 2026-10-07 | The committed log ends at "After Part A" (line 1256), before any Exp 21 result. |
| Exp 22 P1–P3, amendment A1–A3; Exp 23 E1–E2 | `d2e770f`, 2026-10-10 17:16 | 2026-10-11 | Plans and closed form committed; no results in that commit. |

The pilot (items 0–9) that set P1–P6's thresholds has no saved output, so the paper should say "thresholds set
on a 10-item pilot, then tested on 40 held-out items" and report no pilot numbers.

**Snap rescale vs P1.** P1 was written as "snap ≥ 0.6" on the old scale. The rescale came later (2026-10-15);
`neuralese_stats.py` converts the threshold exactly (0.66). P1 is met on both scales (old 0.77 ≥ 0.6, new 0.84 ≥ 0.66).

**Margins (Exp 21, items 10–49).**

| Prediction | Threshold | Observed | Met |
|---|---|---|---|
| P1 num-num final snap | ≥ 0.66 (new scale) | 0.84 | yes |
| P2 num-num 50/50 follows A or B | ≥ 85% | 95% | yes |
| P3 readback CI lower bound, num-num / word-word | > 0 | 0.35 / 0.27 | yes |
| P4 \|control − blend\| with A's text | < 0.1 | 0.01 / 0.02 | yes |
| P5 layers 0–7 share, min over remove/insert × kinds | ≥ 0.50 | 0.71 | yes |
| P6 (layers 0–7 − 12–27) CI lower bound, min | > 0 | 0.31 | yes |
| Exp 22 P1 / P2 (Gumbel / Dirichlet fed top ≥ 0.9 at ≥ 70% of forks) | ≥ 0.70 | 0.47 / 0.37 | no (expected by amendment) |
| Exp 22 P3 (Gumbel fed top ≥ 0.9 at ≥ 90% of steps) | ≥ 0.90 | 0.85 | **no** |
| Exp 22 A1 / A2 | 35–65% / 25–60% | 47% / 37% | yes |
| Exp 22 A3 (Gumbel flip within 0.05 of sampling) | ±0.05 | 0.45 vs 0.45 | yes |
| Exp 23 E1 (soft KL < rand KL on most items) | majority | 6/100 on matched steps | **no, opposite direction** |
| Exp 23 E2 (soft vs hard p > 0.05, diff ≤ 5 pts) | | p = 0.754, 2 pts | yes |

## 4. Setup (applies to Exp 16, 21, 22, 23)

- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, fp32, Kaggle T4 (Exp 14–23).
- **Method, following the Soft Thinking repo:** soft input only inside `<think>`; thinking ends when the mixture's
  top-1 is `</think>` or EOS, at Cold Stop, or at 8,192 steps; then a real `</think>` is fed and the answer is decoded
  greedily (the repo samples at T = 0.6; greedy chosen for reproducibility).
- **Settings:** temperature 0.6, top-k 10, top-p 0.95, min-p 0.001, Cold Stop entropy 0.01 for 256 steps on pre-noise
  weights, Gumbel τ = 0.5, Dirichlet γ = 1.0. Discrete arm = ordinary sampled CoT from the same filtered set, no Cold Stop.
  Differences from Wu et al. (top-k 30, 32,768-token cap, no Cold Stop) belong in the setup section of the paper.
- **Task:** GSM8K test items, prompt + "Please reason step by step, and put your final answer within \boxed{}."
  Scored by the last number inside the last `\boxed{}` (brace-matched). Item ranges: Exp 16 items 0–29; Exp 21
  items 10–49; Exp 22 items 10–29; Exp 23 items 50–149 (never used before). **Exp 22 is a subset of Exp 21's items and
  overlaps Exp 16**, so those are not independent samples; Exp 23 is disjoint from all.
- **Gates, all passed:** hand-written greedy loop = `model.generate()`; the final hidden state reconstructs the live
  logits under exactly one convention (`hidden_states[-1]` is already normalised); in vanilla, mixture top-1 = raw
  argmax at every step (C22.5); layer 0 gives c = α exactly (C21.2); forcing a text in one pass = generating it token
  by token; patching all 28 layers reproduces the donor; Exp 23's two hard arms agree for 64 steps (gate output not
  saved in the jsonl). Exp 21–23 do not use the logit lens, so the double-normalisation bug (§6) cannot affect them.

## 5. Results

### Exp 16: GSM8K accuracy (2026-10-03)

Conditions: discrete CoT seeds 0/1, vanilla, Gumbel seeds 0/1, items 0–29 (150 runs). Numbers: C16.1–C16.6.

- 141/150 runs produce a boxed answer, so the setup measures reasoning, not finishing.
- **Greedy Pitfall reproduced in behaviour:** vanilla ends in Cold Stop (a repetition loop) in 14/30 runs; Gumbel and
  discrete never do. Vanilla's 5 no-answer runs are all after Cold Stop (items 8, 19, 21, 22, 28); the other conditions
  solve 4 of those 5 items in every run. Vanilla also loops and recovers in 7 correct runs.
- **The accuracy gap is not significant at n = 30** (vanilla vs Gumbel seed 0: p = 0.22). Report the loop count as the
  Greedy Pitfall evidence, not the accuracy difference.
- **Gumbel = discrete sampling, 52/60 each;** they differ on 4 items and cancel. Items 3 and 7 are wrong in every condition.
- Caveat: 30 items, accuracy near ceiling. Mean time per run is not used in the paper.

### Exp 21: does the model carry a blend, store the runner-up, and where? (2026-10-07)

Replicates the pilot measurements (Exp 17, 18, 20, items 0–9) unchanged on fresh items 10–49; vanilla and Gumbel
traces, seed 0. 670 unique forks. Numbers: C21.1–C21.13. All six pre-registered predictions met (§3).

**Measurement 1 (α-sweep, `neuralese_sweep.py`).** At each fork, from the same cached prefix, feed α·emb(A) +
(1−α)·emb(B) for α = 0, 0.1, …, 1, and project every layer's state onto the line from the pure-B to the pure-A state.
- Blends collapse progressively: snap rises from 0 at the input to 0.37–0.70 at layer 7 and 0.77–0.98 at the output.
- The more A and B differ in kind, the harder the snap: num-word 0.98 > num-num 0.84 > word-word 0.77 (CIs do not overlap).
- At an exact 50/50 input the next token follows one candidate in 95% (num-num), 98% (num-word) and 88% (word-word) of forks.
- Blended states stay near the A–B line (num-num off-line distance 0.28 of the A–B gap): no "third thing".
- Noise does not change how a given blend is processed (vanilla and Gumbel final snap: num-num 0.85 / 0.84).

**Measurement 2 (KV carry-over, `neuralese_kv_carryover.py`).** At each fork, build caches for pure A, pure B, the
50/50 blend and a control blend of A with C (C = 3rd-ranked pre-noise candidate). Force A's greedy 64-token
continuation, then B's, into each branch and score the readback c at every later position.
- **The runner-up is stored and can be read back:** with B's text, the blend reads it more like B than the control does:
  num-num +0.44, word-word +0.33, num-word +0.11 (mostly erased, but CI above 0).
- **It is dormant on the model's own path:** with A's text, blend ≈ control (−0.01 to +0.06).

**Measurement 3 (KV patching, `neuralese_kv_patch.py`).** Swap only the fork position's keys/values between blend
and control caches, per layer and per 4-layer block.
- The runner-up is held mostly in layers 0–7 (71–91% of the gap) and barely in layers 12–27 (8–26%).
- Per-block (num-num remove / insert): L0–3 0.58 / 0.55, L4–7 0.19 / 0.16, L8–11 0.06 / 0.09, L12–15 0.12 / 0.17,
  L16–27 ≤ 0.04 each. Word-word: L0–3 0.66 / 0.75, L4–7 0.10 / 0.16, the rest ≤ 0.07. Notable single layers for
  num-num: L0 0.38 / 0.16, L7 0.13 / 0.12, L13 0.08 / 0.13 (fork-level CIs exclude 0). Blocks sum to ≈ 1 (0.95 / 1.06
  num-num), so layer effects are roughly additive. *These per-block values were recomputed on 2026-10-17 but are not
  yet in `verify_claims.py`; add them before citing.*
- Num-word: gap too small (+0.11) to localise.

**Reading.** The runner-up survives mostly as an echo of the input mix in early layers, plus a small processed trace
(layers 7 and 13) for numbers, and stays dormant unless later text turns toward it. No evidence that the model reasons
along both candidates. Caveats: one model, one seed per trace, GSM8K only, projection-based scores; exact 50/50 blends
are the most favourable case for the runner-up (Exp 22).

### Exp 22: what blends does noise actually feed? (2026-10-11)

`neuralese_mixtures.py` replays ordinary runs (same RNG use as Exp 16) for vanilla, Gumbel and Dirichlet, seed 0,
items 10–29, recording pre-noise and fed weights at every thinking step. Numbers: C22.1–C22.5.
- Forks are 9–13% of thinking steps.
- At forks, Gumbel feeds a near-one-token blend (fed top ≥ 0.9) only 47% of the time; the median fed blend is
  0.88 / 0.10. Dirichlet: 37%, median 0.83 / 0.16. Across all steps, 83–85% of fed blends are near one-token.
- **Gumbel-max holds in code:** the Gumbel winner flips at forks at exactly the sampling rate (0.45 vs 0.45). So Gumbel
  Soft Thinking = ordinary sampling of the top token + a residual on the others.
- Reading: real runs feed genuine but lopsided blends at about half of the forks. The exact 50/50 blends of Exp 21 are the
  extreme case; real carry-over is at most what Exp 21 measured.
- Pre-registered predictions: P1, P2 failed as the amendment predicted by arithmetic; P3 failed (0.85 < 0.90); A1–A3 met.
  The amendment's closed form (`neuralese_exp22_closed_form.py`) reruns and matches: P(fed top ≥ 0.9) = 0.50 at a 50/50
  fork, 0.60 at 75/25, 0.39–0.43 for three-candidate forks; Dirichlet 0.30–0.56.

### Exp 23: does the residual change answers? (2026-10-11, checks 2026-10-12)

`neuralese_residual_ablation.py`, items 50–149, seed 0. Three arms run in lockstep with shared Gumbel noise keyed by
(seed, step, token id): **hard** (one-hot top token = sampling), **soft** (full blend = Gumbel Soft Thinking),
**rand** (top token at its blend weight, residual moved onto a random vocabulary token). Cold Stop on in all arms.
Numbers: C23.1–C23.8.
- **The model reads the residual:** on matched steps, soft moves next-token predictions about 5× more than an
  equal-mass random residual (0.0077 vs 0.0014; soft > rand on 94/100 items), and soft diverges from sampling before
  rand does on 59 items vs 24 (p = 0.0002).
- **Answers do not change detectably:** 88 vs 86 correct, McNemar p = 0.754 (10 discordant pairs; cannot exclude effects
  of a few points). Rand vs hard: 82 vs 86, p = 0.344.
- Use matched-step KL, not the per-item KL logged at first (soft 0.545, rand 0.009), which averages each arm over its own
  steps up to its first flip and is dominated by the flip step.
- Pre-registered: E1 not met (opposite direction), E2 met.
- **Unbacked:** the embedding-norm check (random vocabulary rows median norm 1.195 vs 1.251 for generated tokens, so
  `rand` is a fair control) exists only in notebook output. Save it to `results/exp23/` or do not cite the numbers.

### Overall reading

Behaviourally, noisy Soft Thinking at 1.5B on GSM8K is **sampling in disguise** (Option B): Gumbel's top token is a
sampled token (Exp 22), and its accuracy equals sampling (Exp 16, 23). Mechanistically, the residual is not ignored:
it is read (Exp 23) and stored in early-layer KV (Exp 21), but collapses by the output, stays dormant on the model's own
path, and does not change answers detectably.

## 6. Pilot work and lessons (no citable numbers)

These shaped the design. None has saved output; do not cite any number from them.

- **Exp 1–8 (Qwen2.5-0.5B-Instruct, CPU, one water-jug prompt; vanilla, Gumbel, Dirichlet × 5 seeds).** Built and
  validated the harness. Lessons: `model.generate()` silently applies `generation_config` defaults (`repetition_penalty`
  is the one that acts under greedy decoding); the soft-step budget changes the final text even without noise, so it must
  be frozen across conditions; a frozen budget needs EOS void-tracking; noise reorders weights, so "index 0 is the top
  weight" breaks; a non-reasoning 0.5B model fails the task in every condition, and a single open-ended prompt cannot be
  scored. Eyeballed weight dips were mostly on framing tokens.
- **Exp 9, 11A, 12A, 13 (logit lens vs mixture): WITHDRAWN** (audit 2026-09-28). Two independent errors: (1) the final
  norm was applied twice (`hidden_states[-1]` is already normalised in this Transformers version); (2) the lens was read
  at the forward pass that *produced* the mixture, not the one that consumed it, so it measured how often noise flipped the
  model's own top-1. Lesson: validate a lens against live logits at many steps, and align the pass in time.
- **Exp 10, 12B (causal patching, 0.5B, 4 targets): pilot only.** Feeding the blend sometimes gave byte-identical text to
  feeding its top-1 token; a control-token arm showed any token change at these positions alters the output. Too small and
  coarse to cite; the question is answered properly by Exp 21 and 23.
- **Exp 11B** (abstract "flower and blade" prompt): the model refused or went meta. No result.
- **Exp 14** (R1-Distill smoke test, 4 runs): the setup works on a reasoning model; confirmed the double-norm bug at 1.5B.
- **Exp 17, 18, 20 (items 0–9 pilot of Exp 21):** set the thresholds for P1–P6. Raw outputs lost. Superseded by Exp 21.
  Two pilot observations did **not** replicate and must not be cited: a "prior bias for numbers" at 50/50 inputs
  (Exp 21: 96 vs 80, p = 0.26) and the pilot's layer-1 snap for digit forks.

## 7. Excluded results (disclose in the paper)

| Exp | What | Why excluded | Suggested disclosure |
|---|---|---|---|
| 15 | ProsQA accuracy, 53 of 150 runs, abandoned | Most runs never answer within 8,192 steps; the task is beyond the 1.5B model. | One sentence in setup: why GSM8K. |
| 19 | Answer-level fork test, items 0–9 | Saved file is a partial run (52 of 61 forks); recomputed p = 0.69 vs logged 0.34; both null. Decided 2026-10-17. | One sentence: an earlier underpowered fork-level test was null; Exp 23 replaces it. |
| 9, 11A, 12A, 13 | Logit-lens comparisons | Withdrawn for two measurement errors (§6). | Optional footnote or appendix on the double-norm pitfall. |
| 17, 18, 20 pilot numbers | Items 0–9 | Raw files lost; Exp 21 replicates on held-out items. | "Thresholds set on a 10-item pilot." |
| Part B | Exp 19 with several seeds per fork | Skipped 2026-10-07 (12–20 GPU h; effect expected too small). | Not needed. |

## 8. Errata (dated corrections; old log entries left as they were)

| Date | Correction | Source |
|---|---|---|
| 2026-07-22 | `repetition_penalty`, not temperature/top-k/top-p, caused the greedy mismatch. | Log 1 |
| 2026-08-24 | Aswal et al. arXiv ID is 2606.12689 (not 2606.12289). | Log 1 |
| 2026-09-19 / 09-28 | Exp 9, 11A, 12A, 13 withdrawn (double norm; temporal misalignment). | Log 2 |
| 2026-10-07 | Exp 21 corrects the pilot: num-word runner-up is "mostly erased" (+0.11), not erased; layer 13 is a real contributor for numbers. | Log 2 |
| 2026-10-12 | Exp 23 KL: report matched-step values (≈5×), not the logged ≈60×. | Log 2 |
| 2026-10-15 | Snap rescaled so a perfect step = 1 (new = old × 1.1); P1 threshold 0.6 → 0.66. | paper-writing-fixes.md |
| 2026-10-17 | Exp 19 dropped (partial file, numbers not reproducible). | Log 2 |
| **2026-10-17** | **Exp 16: 141/150 runs answered, not 145** (the log's own table sums to 9 no-answers). | this validation |
| **2026-10-17** | **Exp 16: vanilla's accuracy deficit is not significant at n = 30** (p = 0.22); the Cold Stop count carries the Greedy Pitfall claim. | this validation |
| **2026-10-17** | **Exp 23: median first flip is 35.5 (soft) and 66 (rand)**, not 36 and 67. | this validation |
| **2026-10-17** | **Pilot "prior bias for numbers" does not replicate** in Exp 21 (96 vs 80, p = 0.26). | this validation |
| 2026-10-17 | Exp 16: mean time per run is ~3 s lower than logged (not used in the paper). | Log 2 |

## 9. Literature sources (verify each before citing; see next-steps-figures-validation.md §3 Step 5)

- Zhang et al. 2025, *Soft Thinking*, arXiv:2505.15778. The method.
- Wu et al. 2025, arXiv:2508.03440 — **cite v4**, *LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism
  of Soft Thinking* (earlier versions titled *LLMs Have a Heart of Stone…*); ICLR 2026. Greedy Pitfall; Gumbel/Dirichlet
  fix; Gumbel-max motivation (§5.4); matched-sampling gains +0.4 to +1.1 points (details in paper-writing-fixes.md §4).
- Rizvi-Martel, Rabusseau & Mosbach 2026, *The Illusion of Superposition?*, arXiv:2604.06374. Vanilla Soft Thinking shows no
  superposition by logit lens; mixtures often blend syntactic alternatives. Open item: the tension with our gradual collapse.
- Backour, arXiv:2609.02049 (independent confirmation, per the open item; verify).
- Deng et al., arXiv:2510.15522 (listed in the citation plan; verify what it is cited for).
- Butt et al. 2025, *Soft Tokens, Hard Truths*, arXiv:2509.19170. RL for continuous CoT via embedding noise.
- Zhu et al. 2025, *Reasoning by Superposition*, arXiv:2505.12514. Theory origin of the superposition claim.
- Li et al. 2026, *Dynamics Within Latent Chain-of-Thought*, arXiv:2602.08783; Zhang et al. 2025, *Do Latent Tokens Think?*,
  arXiv:2512.21711. Causal patching on Coconut/CODI; comparison targets.
- Aswal et al. 2026, *Observable Patterns Are Not Explanations*, arXiv:2606.12689. Decodability ≠ mechanism.
- Jin, Yang & Wang 2026, arXiv:2607.06648; Zheng & Lee, SofT-GRPO, arXiv:2511.06411. Future work (faithfulness along training).
- Kool et al. 2019 (Gumbel-max / Gumbel-top-k); McNemar and bootstrap references.

## 10. Open items

1. Save the Exp 23 embedding-norm check to `results/exp23/`, or drop its numbers.
2. Add the Exp 21 per-block values and accuracy CIs to `verify_claims.py` before citing them.
3. Optional: rerun the items 0–9 pilot (all runs are seeded) if the paper should report pilot numbers; otherwise not needed.
4. Pin model revision, `transformers` version and Kaggle image in the paper's reproducibility section.
5. Writing items carried from paper-writing-fixes.md: shrink the Exp 17 framing and cite Backour; one paragraph on the
   Rizvi-Martel tension. Track 3 (faithfulness, SFT vs RL) stays future work.
6. Next: figure scripts (one per figure, reading `results/`).

## 11. What was cut and where to find it

Everything below remains in the original logs in `docs/` and in git; restore any of it by copying it back.

| Cut | Where it lives | Why |
|---|---|---|
| Exp 1–8 per-seed tables, weight dips, entropy values | `neuralese-research-log.md`, Experiment Log | No saved output; non-reasoning model; qualitative. Lessons kept in §6. |
| Exp 9 lens tables, "three patterns", Exp 11A layer readouts, Exp 12A table, Exp 13 tables | continuation log, Exp 9–13 | Withdrawn (measurement errors). |
| Exp 10 / 12B patching outputs | continuation log | Pilot, n = 4, no saved output. |
| Exp 11B outputs | continuation log | No result. |
| Exp 14 table, Exp 15 table | continuation log | Smoke test / abandoned; kept as one line each. |
| Exp 17, 18, 20 pilot tables (items 0–9) | continuation log | No raw files; replaced by Exp 21 values. |
| Exp 19 tables | continuation log | Dropped 2026-10-17. |
| Review notes 2026-07-22, 08-05, 08-24 (literature landscape, novelty checks, Track 2/3 framing) | research log | Superseded by the 2026-09-28 audit and the current framing; references kept in §9. |
| Harness updates, to-do lists, superseded "next steps" and synthesis sections | both logs | Process notes; the current setup is §4. |
| Exp 21 by-condition intermediate notes, Kaggle cell instructions | continuation log | Operational, not results. |
