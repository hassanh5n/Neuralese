# Neuralese: Paper Log (merged, validated)

Created 2026-10-17 by merging `neuralese-research-log.md` (Exp 1–8, literature notes) and
`neuralese-research-log-continuation.md` (Exp 9–23, audits, decisions). Both originals are unchanged in `docs/`
and in git; they remain the lab notebook. **This file is the paper's source of truth.**

**The rule.** A number may appear in the paper only if it has a row in the claim ledger (§2) and
`python verify_claims.py` recomputes it from `results/`. Status on 2026-10-26: **110/110 checks pass** (2026-11-03: 6 rows added, C16.14–15 and C23.16–19, from saved outputs; all pass) (16 added on 2026-10-22 for numbers quoted in §3 and §5, no value changed; 4 added on 2026-10-26 for the Exp 23 embedding-norm check; 2 added on 2026-10-26 for the snap order by layer in Fig 2; 3 added on 2026-10-26 for the readback CIs in Fig 3; 5 added on 2026-10-26 for the single-layer patching in Fig 4; 6 added on 2026-10-26 for Fig 5; 2 added on 2026-10-26 for the KL ratio in Fig 6; 2 added on 2026-10-26 for the McNemar tests vs sampling in Table 1). Robustness checks for Fig 4 and the appendix (C21.32–C21.40) run in `python verify_robustness.py`: **17/17 pass**; it also checks every cell of `figures/tabA1–A3.tex` and `figures/tab1.tex` (Tab1a–b, added 2026-10-26).
Numbers in §6–§8 are history and must not be copied into the paper.

> **Figure renumbering, 2026-10-26 (Hassan).** The four-questions figure is Fig 2. Entries written before this note
> use the old numbers: old Fig 2 → **3** (snap by layer), 3 → **4** (readback), 4 → **5** (KV patching), 5 → **6** (fed
> blends), 6 → **7** (residual ablation). Fig 1, Figs A1–A2 and Tables 1, A1–A3 keep their numbers. Files and scripts
> were renamed to match (`make_fig3.py` … `make_fig7.py`, `make_tabA_fig5.py`); all figures regenerated; 110/110
> and 17/17 checks pass.

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
| Top weight / second weight *(added 2026-10-26)* | Largest / second-largest weight in the blend actually fed (Exp 22 `fed1` / `fed2`). The residual (1 − top weight) is the paper term for Exp 22 as well; "second weight" only where the single next token matters. Not "runner-up", which is the pre-noise second candidate. |
| Balanced / lopsided / near one token *(added 2026-10-26)* | Fed top weight ≤ 0.75 / 0.75–0.9 / ≥ 0.9 (Fig 5a). |
| Flip, sampling rate *(added 2026-10-26)* | Fed winner is not the pre-noise top token; 1 − pre-noise top weight, the rate at which sampling would flip. |
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
| C16.7 | Vanilla runs that hit Cold Stop and still answered correctly (loop, then recover) | 7 | exp16 |
| C16.8 | Vanilla no-answer items; all ended in Cold Stop | 8, 19, 21, 22, 28; yes | exp16 |
| C16.9 | Of those, items solved in every discrete and Gumbel run | 8, 19, 22, 28 (4 of 5) | exp16 |
| C16.10 | Items wrong in every run of every condition | 3, 7 | exp16 |
| C16.11 | Wilson 95% CI: 52/60 (discrete, Gumbel); 22/30 (vanilla) | [0.76, 0.93]; [0.56, 0.86] | exp16 |
| C16.12 | Gumbel vs discrete, seed 0, paired McNemar *(added 2026-10-26, Table 1)* | Gumbel only 1, discrete only 2, p = 1.00 | exp16 |
| C16.13 | Vanilla vs discrete, seed 0, paired McNemar *(added 2026-10-26, Table 1)* | vanilla only 0, discrete only 5, p = 0.062 | exp16 |
| C16.14 | Per-seed accuracy: sampling seed 0 / 1; Gumbel seed 0 / 1; problems where Gumbel's two seeds disagree *(added 2026-11-03)* | 27 / 25; 26 / 26; 0 | exp16 |
| C16.15 | Problem-clustered 95% CI (bootstrap over the 30 problems, per-problem accuracy over both seeds): sampling; Gumbel; Gumbel − sampling *(added 2026-11-03, replaces the run-level Wilson CI for 60 runs)* | [0.75, 0.97]; [0.73, 0.97]; 0.00 [−0.10, 0.10] | exp16 |
| C16.16 | Vanilla − sampling, seed 0, paired, Newcombe hybrid-score 95% CI *(added 2026-11-03, Table 1)* | −0.17 [−0.33, −0.01] | exp16 |
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
| C21.14 | num-num final snap, vanilla / Gumbel traces | 0.85 / 0.84 | exp21 step4 |
| C21.15 | num-num per-block share, remove / insert: L0–3, L4–7, L8–11, L12–15 | 0.58 / 0.55, 0.19 / 0.16, 0.06 / 0.09, 0.12 / 0.17 | exp21 patch |
| C21.16 | num-num blocks L16–19, L20–23, L24–27 | ≤ 0.04 each | exp21 patch |
| C21.17 | word-word per-block share, remove / insert: L0–3, L4–7 | 0.66 / 0.75, 0.10 / 0.16 | exp21 patch |
| C21.18 | word-word blocks L8–27 | ≤ 0.07 each | exp21 patch |
| C21.19 | num-num single layers, remove / insert: L0, L7, L13 | 0.38 / 0.16, 0.13 / 0.12, 0.08 / 0.13 | exp21 patch |
| C21.20 | Fork-level 95% CIs of C21.19 exclude 0 (as `neuralese_kv_patch.py`) | yes | exp21 patch |
| C21.21 | num-num blocks sum to, remove / insert | 0.95 / 1.06 | exp21 patch |
| C21.22 | P6 margin: min CI lower bound of (layers 0–7 − 12–27) | 0.31 | exp21 patch |
| C21.23 | num-num vs word-word snap by layer: where the mean is below / where the CIs are apart | below at layers 1–12, above at 13–28 / CIs apart at 1–9 and 13–28 | exp21 step4 |
| C21.24 | num-word snap CI above both other kinds' CIs | every layer 1–28 | exp21 step4 |
| C21.25 | Readback CI with A's own text (dormancy), followed-A forks | num-num [−0.06, 0.04]; num-word [−0.00, 0.19]; word-word [−0.02, 0.07] | exp21 carryover |
| C21.26 | Readback with B's text, by forced-text position bin (1, 2, 3–4, 5–8, 9–16, 17–32, 33–64): CI above 0 in every bin | yes, num-num and word-word | exp21 carryover |
| C21.27 | Readback with A's own text, same bins: CI includes 0 in every bin | yes, num-num and word-word | exp21 carryover |
| C21.28 | Patch forks (followed-A): forks / problems; problem-level CIs of num-num L0, L7, L13 above 0 (remove and insert) | num-num 96 / 35, word-word 139 / 40; yes | exp21 patch |
| C21.29 | num-num single layers, remove / insert: L1, L2 | 0.21 / 0.07, 0.20 / 0.05 | exp21 patch |
| C21.30 | num-num single layers other than 0–3, 7, 13 | ≤ 0.05 each (remove and insert) | exp21 patch |
| C21.31 | num-num layers 0–3: sum of single layers vs the block, remove / insert | 0.84 vs 0.58 / 0.32 vs 0.55 | exp21 patch |
| C21.32 | Table A1 (Fig 4 robustness), every cell; forks with gap ≥ 0.1 | all match; num-num 78 of 96, word-word 108 of 139 | exp21 patch, `verify_robustness.py` |
| C21.33 | Largest change of a Table A1 pooled share when one problem is dropped | 0.04 | exp21 patch, `verify_robustness.py` |
| C21.34 | num-num layer 0: pooled share vs median fork (gap ≥ 0.1), remove / insert | 0.38 vs 0.12 / 0.16 vs 0.02 | exp21 patch, `verify_robustness.py` |
| C21.35 | Table A2 (num-num L0, L1, L2, L7, L13), every cell; Bonferroni lower bounds over 56 single-layer tests (20,000 problem resamples) all above 0; smallest | all match; yes; 0.01 (L1 and L2 insert) | exp21 patch, `verify_robustness.py` |
| C21.36 | Table A3 (num-word blocks), every cell; forks / problems; mean gap | all match; 34 / 21; 0.11 | exp21 patch, `verify_robustness.py` |
| C21.37 | Top 10 forks' share of the total gap; forks with gap ≤ 0 | num-num 0.27, 6; word-word 0.24, 19 | exp21 patch, `verify_robustness.py` |
| C21.38 | Fig A1: readback with B's text, mean above 0 at every single position 1–16 | yes, num-num and word-word | exp21 carryover, `verify_robustness.py` |
| C21.39 | num-num layer 0 remove share per fork (gap ≥ 0.1): quartiles | 0.01 / 0.12 / 0.68 | exp21 patch, `verify_robustness.py` |
| C21.40 | Largest change of a Table A1 pooled share when forks with gap < 0.1 are dropped | 0.03 | exp21 patch, `verify_robustness.py` |
| C22.1 | Vanilla, items 10–29: fork rate; fed top ≥ 0.9 all steps / forks; median fed top / runner-up at forks; winner flipped | 9.2%; 0.78 / 0.00; 0.56 / 0.33; 0.00 | exp22/exp22_mixtures_items10-29.jsonl |
| C22.2 | Gumbel, same measures | 13.0%; 0.85 / **0.47**; **0.88 / 0.10**; 0.45 | exp22 |
| C22.3 | Dirichlet, same measures | 13.3%; 0.83 / 0.37; 0.83 / 0.16; 0.42 | exp22 |
| C22.4 | Gumbel-max check: Gumbel flip rate vs sampling (1 − mean pre-noise top weight) | 0.45 vs 0.45 | exp22 |
| C22.5 | Vanilla fed weights = pre-noise weights at every step | true | exp22 |
| C22.6 | Amendment closed form, two-candidate Gumbel P(fed top ≥ 0.9) at 50/50 and 75/25 | 0.50, 0.60 | formula (no data) |
| C22.7 | Forks per arm: vanilla / Gumbel / Dirichlet; problems | 3,774 / 5,761 / 4,978; 20 | exp22 |
| C22.8 | Flip rate minus sampling rate (1 − pre-noise top weight) at forks, problem-level 95% CI: Gumbel; Dirichlet | 0.00 [−0.01, 0.02]; −0.03 [−0.04, −0.02] | exp22 |
| C22.9 | Gumbel, forks binned by sampling rate (width 0.05, 0.25–0.75): largest \|observed − sampling\| in bins with ≥ 100 forks | 0.04 | exp22 |
| C22.10 | Dirichlet observed flip rate below the sampling rate in every bin | yes (10 of 10) | exp22 |
| C22.11 | % of forks by fed top weight, ≤ 0.75 / 0.75–0.9 / ≥ 0.9: vanilla; Gumbel; Dirichlet | 100 / 0 / 0; 31 / 22 / 47; 38 / 25 / 37 | exp22 |
| C22.12 | Median residual (1 − fed top) at forks: vanilla / Gumbel / Dirichlet; median share of the residual on the second-largest token: Gumbel / Dirichlet | 0.44 / 0.12 / 0.17; 0.99 / 1.00 | exp22 |
| C23.1 | Accuracy, items 50–149: hard / soft / rand | 86 / 88 / 82 of 100 | exp23/exp23_residual_ablation_items50-149.jsonl |
| C23.2 | Soft vs hard, McNemar | soft-only 6, hard-only 4, p = 0.754; same final answer 84/100 | exp23 |
| C23.3 | Rand vs hard, McNemar | rand-only 3, hard-only 7, p = 0.344; same final answer 84/100 | exp23 |
| C23.4 | Median first token flip vs hard (step) | soft 35.5, rand 66 | exp23 |
| C23.5–6 | Which arm diverges from hard first (paired) | soft first 59, rand first 24, same 17; sign test p = 0.0002 | exp23 |
| C23.7–8 | Mean per-item KL(arm ‖ hard), matched steps (before either arm's first flip) | soft 0.0077, rand 0.0014; soft > rand on 94/100 items | exp23 |
| C23.9 | Wilson 95% CI: soft 88, hard 86, rand 82 of 100 | [0.80, 0.93], [0.78, 0.91], [0.73, 0.88] | exp23 |
| C23.14 | Matched-step KL: ratio of means soft / rand; top 5 problems' share of soft's total; ratio of means without them | 5.7; 0.54; 3.1 | exp23 |
| C23.15 | Matched-step KL: median per-problem ratio soft / rand, 95% CI over problems; median matched steps per problem | 3.9 [3.1, 5.0]; 31 | exp23 |
| C23.16 | E1 as registered: per-item KL averaged over each arm's own steps up to and including its first flip; soft < rand on; mean soft / rand *(added 2026-11-03)* | 14/100; 0.545 / 0.009 | exp23 |
| C23.17 | Paired accuracy difference vs hard, Newcombe hybrid-score 95% CI: soft − hard; rand − hard *(added 2026-11-03)* | +0.02 [−0.05, +0.09]; −0.04 [−0.11, +0.03] | exp23 |
| C23.18 | Soft vs rand, paired: final answer differs from hard's for both / soft only / rand only / neither; McNemar on soft only vs rand only; soft and rand same final answer *(added 2026-11-03)* | 10 / 6 / 6 / 78; p = 1.00; 85/100 | exp23 |
| C23.19 | Soft vs rand accuracy, paired: soft only / rand only; McNemar; difference, Newcombe 95% CI *(added 2026-11-03)* | 9 / 3; p = 0.146; +0.06 [−0.01, +0.13] | exp23 |
| C23.10 | Median embedding norm: all real-token rows / the 8,192 random rows Exp 23 drew / same, usage-weighted / reasoning tokens (sampled CoT, items 50–54) | 1.195 / 1.195 / 1.196 / 1.202 | exp23/exp23_embedding_norms.json |
| C23.11 | Random rows ÷ reasoning tokens, median / mean | 0.994 / 0.961 | exp23 norms |
| C23.12 | 10th percentile: random rows / reasoning tokens | 0.897 / 1.058 | exp23 norms |
| C23.13 | n: random draws / reasoning tokens / reasoning items | 8,192 / 9,168 / 5 | exp23 norms |

Not scripted: the Exp 22 closed form's three-candidate and Dirichlet values (simulation; rerun
`neuralese_exp22_closed_form.py`, needs numpy). C23.10–13 check the saved output of `neuralese_exp23_norms.py`
(a GPU run), not a recomputation from the model.

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
| Exp 23 E1 (soft KL < rand KL on most items) | majority | 14/100 as registered (own steps); 6/100 on matched steps | **no, opposite direction** |
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
- *Added 2026-10-26 (paper draft, from the code; no value changed):* bootstrap CIs resample problems 2,000 times and take the
  2.5th and 97.5th percentiles (`neuralese_stats.py` N_BOOT, `verify_claims.boot`, seed 0); the answer after `</think>` is
  decoded greedily for at most 512 tokens (`neuralese_r1.py` MAX_ANSWER); Exp 23 KL is computed on the full next-token
  distribution at the sampling temperature 0.6 (`neuralese_residual_ablation.py`); the random-residual token is one
  real-token id per step, shared by all arms, drawn by the same keyed generator as the Gumbel noise.
- *Added 2026-10-26 (paper draft, from the code; no value changed):* the readback score c places the model's next-token
  distribution p (temperature 0.6) on the line from pure B to pure A: c = (p − p_B)·(p_A − p_B) / |p_A − p_B|²; positions
  where p_A and p_B differ by less than 0.05 in total variation are skipped (`neuralese_kv_carryover.py` MIN_TV).

## 5. Results

### Exp 16: GSM8K accuracy (2026-10-03)

Conditions: discrete CoT seeds 0/1, vanilla, Gumbel seeds 0/1, items 0–29 (150 runs). Numbers: C16.1–C16.11.

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
traces, seed 0. 670 unique forks. Numbers: C21.1–C21.22. All six pre-registered predictions met (§3).

**Measurement 1 (α-sweep, `neuralese_sweep.py`).** At each fork, from the same cached prefix, feed α·emb(A) +
(1−α)·emb(B) for α = 0, 0.1, …, 1, and project every layer's state onto the line from the pure-B to the pure-A state.
- Blends collapse progressively: snap rises from 0 at the input to 0.37–0.70 at layer 7 and 0.77–0.98 at the output.
- The more A and B differ in kind, the harder the snap: num-word 0.98 > num-num 0.84 > word-word 0.77 (CIs do not overlap).
- *Added 2026-10-26 (Fig 2):* the order above holds **at the output**, not at every layer. Num-word snaps hardest at
  every layer (CI above both others at layers 1–28). Num-num starts slowest: it is below word-word at layers 1–12
  (CIs apart at 1–9) and above it from layer 13 to the output (CIs apart at 13–28). (C21.23–C21.24.)
- At an exact 50/50 input the next token follows one candidate in 95% (num-num), 98% (num-word) and 88% (word-word) of forks.
- Blended states stay near the A–B line (num-num off-line distance 0.28 of the A–B gap): no "third thing".
- Noise does not change how a given blend is processed (vanilla and Gumbel final snap: num-num 0.85 / 0.84).

**Measurement 2 (KV carry-over, `neuralese_kv_carryover.py`).** At each fork, build caches for pure A, pure B, the
50/50 blend and a control blend of A with C (C = 3rd-ranked pre-noise candidate). Force A's greedy 64-token
continuation, then B's, into each branch and score the readback c at every later position.
- **The runner-up is stored and can be read back:** with B's text, the blend reads it more like B than the control does:
  num-num +0.44, word-word +0.33, num-word +0.11 (mostly erased, but CI above 0).
- **It is dormant on the model's own path:** with A's text, blend ≈ control (−0.01 to +0.06).
- *Added 2026-10-26 (Fig 3):* the own-text CIs all include 0 (num-num [−0.06, 0.04], num-word [−0.00, 0.19],
  word-word [−0.02, 0.07]). Split by position in the forced text (log-spaced bins), the readback with B's text stays
  above 0 in every bin up to 64 tokens, and the own-text readback includes 0 in every bin (num-num and word-word;
  num-word has too few forks per bin). (C21.25–C21.27.)
- *Added 2026-10-26 (Fig A1):* without binning, the mean readback with B's text is above 0 at every single position
  1–16 (num-num, word-word; C21.38). Single positions hold few forks and their CIs are wide, which is why Fig 3 bins.

**Measurement 3 (KV patching, `neuralese_kv_patch.py`).** Swap only the fork position's keys/values between blend
and control caches, per layer and per 4-layer block.
- The runner-up is held mostly in layers 0–7 (71–91% of the gap) and barely in layers 12–27 (8–26%).
- Per-block (num-num remove / insert): L0–3 0.58 / 0.55, L4–7 0.19 / 0.16, L8–11 0.06 / 0.09, L12–15 0.12 / 0.17,
  L16–27 ≤ 0.04 each. Word-word: L0–3 0.66 / 0.75, L4–7 0.10 / 0.16, the rest ≤ 0.07. Notable single layers for
  num-num: L0 0.38 / 0.16, L7 0.13 / 0.12, L13 0.08 / 0.13 (fork-level CIs exclude 0). Blocks sum to ≈ 1 (0.95 / 1.06
  num-num), so layer effects are roughly additive. (C21.15–C21.21.)
- Num-word: gap too small (+0.11) to localise.
- *Added 2026-10-26 (Fig 4):* the single-layer CIs of L0, L7 and L13 also exclude 0 when resampling problems,
  not only forks (C21.28). The early effect sits in layers 0–2: L1 0.21 / 0.07 and L2 0.20 / 0.05 besides L0
  (C21.29); outside layers 0–3, 7 and 13 every single layer closes ≤ 0.05 (C21.30). **Inside layers 0–3, single
  layers are not additive:** removing one layer at a time sums to 0.84 against 0.58 for the block, inserting one at
  a time sums to 0.32 against 0.55 (C21.31). Reading: the runner-up is held redundantly across layers 0–2; taking
  out any one of them breaks the readback, but no single one is enough to carry it. The "roughly additive" line
  above holds across blocks (C21.21), not inside layers 0–3.
- *Added 2026-10-26 (robustness of Fig 4; Tables A1–A3, Fig A2):*
  - **Status:** the block split (layers 0–7 vs 12–27) is pre-registered (P5, P6). The single-layer results (L1, L2,
    L7, L13) and the non-additivity inside layers 0–3 are **exploratory**: they were found after looking.
  - **Multiple tests:** with a Bonferroni correction over all 56 single-layer tests, L0, L1, L2, L7 and L13 stay above
    0 for remove and insert; the weakest bound is 0.01 (L1 and L2 insert), so those two insert effects are only just
    clear (C21.35).
  - **Not one problem or a few forks:** dropping any one problem changes a pooled share by at most 0.04 (C21.33);
    the 10 largest-gap forks carry 0.27 (num-num) and 0.24 (word-word) of the total gap (C21.37); keeping only forks
    with gap ≥ 0.1 changes no share by more than 0.03 (C21.40).
  - **Layer 0 is uneven across forks:** pooled remove share 0.38, but the median fork is 0.12 (insert 0.16 vs 0.02;
    C21.34); quartiles 0.01 / 0.12 / 0.68 (C21.39). So "layer 0 closes 38%" is a pooled statement: layer 0 matters a
    lot at some forks and little at about half. The blocks do not have this gap (layers 0–3 median 0.53 vs pooled 0.58).
  - **Two readings of the non-additivity (C21.31):** (i) the runner-up is held redundantly across layers 0–2, or
    (ii) the readback score saturates (it is a projection that can pass 1), so large single-layer patches add up to
    more than their block. These data do not separate them; patching layers 0..k together would. No rerun planned
    unless a reviewer asks. In the paper: report the non-additivity as a fact and redundancy as one reading.
  - **Scope:** all of Fig 4 is 50/50 blends at forks where the model followed the chosen token (96 of 238 num-num
    forks), the most favourable case for the runner-up (Exp 22: median fed runner-up weight 0.10).
  - Num-word values are reported in Table A3 rather than dropped (mean gap 0.11, shares unstable; C21.36).

**Reading.** The runner-up survives mostly as an echo of the input mix in early layers, plus a small processed trace
(layers 7 and 13) for numbers, and stays dormant unless later text turns toward it. No evidence that the model reasons
along both candidates. Caveats: one model, one seed per trace, GSM8K only, projection-based scores; exact 50/50 blends
are the most favourable case for the runner-up (Exp 22).

### Exp 22: what blends does noise actually feed? (2026-10-11)

`neuralese_mixtures.py` replays ordinary runs (same RNG use as Exp 16) for vanilla, Gumbel and Dirichlet, seed 0,
items 10–29, recording pre-noise and fed weights at every thinking step. Numbers: C22.1–C22.6.
- Forks are 9–13% of thinking steps.
- At forks, Gumbel feeds a near-one-token blend (fed top ≥ 0.9) only 47% of the time; the median fed blend is
  0.88 / 0.10. Dirichlet: 37%, median 0.83 / 0.16. Across all steps, 83–85% of fed blends are near one-token.
- **Gumbel-max holds in code:** the Gumbel winner flips at forks at exactly the sampling rate (0.45 vs 0.45). So Gumbel
  Soft Thinking = ordinary sampling of the top token + a residual on the others.
- Reading: real runs feed genuine but lopsided blends at about half of the forks. The exact 50/50 blends of Exp 21 are the
  extreme case; real carry-over is at most what Exp 21 measured.
- *Added 2026-10-26 (Fig 5):* Gumbel-max also holds bin by bin. With forks binned by the sampling flip rate,
  Gumbel's flip rate is 0.00 [−0.01, 0.02] above sampling overall and within 0.04 of it in every bin with ≥ 100
  forks (C22.8–9; one of ten bin-level CIs misses the diagonal, about what chance gives). **New: Dirichlet flips less
  often than sampling**, −0.03 [−0.04, −0.02] overall and below the diagonal in all ten bins (C22.8, C22.10), so its
  winner is biased toward the pre-noise top token. Forks per arm: 3,774 / 5,761 / 4,978 (C22.7); each arm is its own
  run, so the forks differ.
- *Added 2026-10-26 (Fig 5a):* by fed top weight, Gumbel forks are 31% balanced (≤ 0.75), 22% lopsided, 47% near one
  token; Dirichlet 38 / 25 / 37%; vanilla 100% balanced by the fork rule (C22.11). Median residual 0.12 (Gumbel),
  0.17 (Dirichlet), 0.44 before noise; the second-largest token holds 99–100% of it at the median fork (C22.12).
- *Clarified 2026-10-26:* the "runner-up" weight in C22.1–C22.3 (`fed2`) is the **second-largest fed weight**, which
  can sit on a different token than the pre-noise runner-up. The paper should call it "second-largest fed weight".
- Pre-registered predictions: P1, P2 failed as the amendment predicted by arithmetic; P3 failed (0.85 < 0.90); A1–A3 met.
  The amendment's closed form (`neuralese_exp22_closed_form.py`) reruns and matches: P(fed top ≥ 0.9) = 0.50 at a 50/50
  fork, 0.60 at 75/25, 0.39–0.43 for three-candidate forks; Dirichlet 0.30–0.56.

### Exp 23: does the residual change answers? (2026-10-11, checks 2026-10-12)

`neuralese_residual_ablation.py`, items 50–149, seed 0. Three arms run in lockstep with shared Gumbel noise keyed by
(seed, step, token id): **hard** (one-hot top token = sampling), **soft** (full blend = Gumbel Soft Thinking),
**rand** (top token at its blend weight, residual moved onto a random vocabulary token). Cold Stop on in all arms.
Numbers: C23.1–C23.9.
- **The model reads the residual:** on matched steps, soft moves next-token predictions about 5× more than an
  equal-mass random residual (0.0077 vs 0.0014; soft > rand on 94/100 items), and soft diverges from sampling before
  rand does on 59 items vs 24 (p = 0.0002).
- **Answers do not change detectably:** 88 vs 86 correct, McNemar p = 0.754 (10 discordant pairs; cannot exclude effects
  of a few points). Rand vs hard: 82 vs 86, p = 0.344.
- Use matched-step KL, not the per-item KL logged at first (soft 0.545, rand 0.009), which averages each arm over its own
  steps up to its first flip and is dominated by the flip step.
- Pre-registered: E1 not met (opposite direction), E2 met.
- *Added 2026-10-26 (Fig 6):* the "about 5×" above is a ratio of means (5.7), and five problems hold 54% of soft's
  total; without them the ratio is 3.1 (C23.14). **Use the per-problem summary in the paper:** soft moves predictions
  a median 3.9× the random residual, 95% CI [3.1, 5.0], over a median of 31 matched steps (C23.15), and is higher on
  94 of 100 problems (C23.8). The direction is not in doubt; only the size quoted changes.
- **`rand` is a fair control** (`neuralese_exp23_norms.py`, rerun 2026-10-26; C23.10–13): the random rows Exp 23 actually
  drew (regenerated with the same keyed RNG on GPU) have median norm 1.195, the same as all real-token rows, and real
  reasoning tokens 1.202 (ratio 0.994; means 0.961). The random rows have a heavier low tail (10th percentile 0.897 vs
  1.058), so a minority of draws carry a weaker signal; this cannot explain a 5× KL gap.

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
| 2026-10-26 | Exp 21 snap order (num-word > num-num > word-word) is an output-layer result; num-num is below word-word at layers 1–12 (C21.23–24). Wording only, no value changed. | Fig 2 |
| 2026-10-26 | Exp 21 "layer effects are roughly additive" holds across 4-layer blocks only; inside layers 0–3 single layers are not additive (C21.31). | Fig 4 |
| 2026-10-26 | Exp 22 "median fed top / runner-up": the second number is the second-largest fed weight, not necessarily the pre-noise runner-up's weight (`neuralese_mixtures.py`: `fed2 = sorted(w)[-2]`). | Fig 5 |
| 2026-10-26 | Exp 23 "soft ≈ 5× rand" (matched-step KL) is a ratio of means driven by five problems; quote the median per-problem ratio 3.9 [3.1, 5.0] (C23.14–15). | Fig 6 |
| **2026-10-26** | **Exp 23 norm check: reasoning tokens median 1.202, not 1.251; random rows are ~0.6% smaller, not ~5%.** The old reference set was never saved; the new one is sampled CoT on items 50–54. Random-row median (1.195) unchanged. | `neuralese_exp23_norms.py` |

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
- *Verified 2026-10-26 (full text read):* Backour = *The Dynamics of Continuous Mixture Collapse in Language Models*,
  arXiv:2609.02049v1 (2 Sep 2026, single author, preprint). Its CALIB score is our snap measurement (snap = 1 − CALIB).
  Rizvi-Martel et al. = arXiv:2604.06374v2 (3 Aug 2026), **COLM 2026**. Details and drafts: paper-writing-fixes.md §7.

## 10. Open items

1. ~~Save the Exp 23 embedding-norm check to `results/exp23/`.~~ Done 2026-10-26 (C23.10–13).
2. ~~Add the Exp 21 per-block values and accuracy CIs to `verify_claims.py`.~~ Done 2026-10-22 (C16.7–C23.9, 86/86).
3. Optional: rerun the items 0–9 pilot (all runs are seeded) if the paper should report pilot numbers; otherwise not needed.
4. Pin model revision, `transformers` version and Kaggle image in the paper's reproducibility section. Partly done:
   the norm-check run recorded torch 2.11.0+cu128, transformers 5.20.2, CUDA 12.8, Python 3.13.15, Tesla T4. These are
   that run's versions only: every notebook ran `pip install -U transformers`, so Exp 16–23 may have used other versions
   (check each Kaggle version's log). Model revision not captured (`_commit_hash` is None in transformers 5.x); get it
   from `huggingface_hub.model_info(MODEL).sha` and pass `revision=` in `load()` from now on.
   - *2026-10-26:* Model revision pinned from the Hugging Face API: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at
     `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, last modified 2025-02-24, before every experiment, so Exp 14–23 all
     loaded this revision (the scripts load the Hub id, not a Kaggle copy). Package versions for Exp 16–23 still
     need the Kaggle version logs; none are in the repo.
   - *2026-10-26:* Kaggle logs of the committed runs saved in `kaggle_logs/` (Hassan; other runs were interactive and
     not saved). The logs print no versions (`pip -q`), but pip's download sizes identify the transformers wheel
     (matched against PyPI wheel sizes and upload dates):
     | Run (`kaggle_logs/`) | transformers wheel | Version | Reasoning |
     |---|---|---|---|
     | `exp21.log` | 12.6 MB | 5.19.0 | 5.18.0 (12.58 MB) also shows as 12.6, but the run followed the 10-07 pre-registration commit and 5.19.0 was out on 10-06 |
     | `exp23.log` | 12.6 MB | 5.19.0 | run 10-10/11; 5.20.0 was released 10-15 |
     | `exp16_exp22_exp19partial.log` | 12.8 MB | 5.20.0 or 5.20.1 | must be after 10-15: a later rerun of Exp 16 and 22, not the 10-03 / 10-11 originals; the Exp 19 part timed out at 12 h |
     | `exp23_norms.log` | 12.8 MB | 5.20.2 | recorded by the script (torch 2.11.0+cu128, CUDA 12.8, Python 3.13.15, T4) |
     Every log's summary matches the saved results and the ledger (C16.1–3, C22.1/C22.4, C21 readback +0.44/+0.11/+0.33,
     C23.1–3), so the files in `results/` come from these runs. Exp 16 and 22 reproduced their originally logged numbers
     exactly under a newer transformers. All logs run Python 3.13 (`dist-packages` path). torch, CUDA and the Kaggle image
     are recorded only for the norm check. Still open: the date of the Exp 16 + 22 Kaggle version (pins 5.20.0 vs 5.20.1).
   - *2026-10-26:* Exp 16 + 22 Kaggle version shows "9 days ago" (about 2026-10-17). PyPI uploads: 5.20.0 on 10-15
     18:01 UTC, 5.20.1 on 10-16 13:02 UTC, 5.20.2 on 10-19 17:46 UTC. So that run used **5.20.1**, unless it started on
     10-16 between about 11:50 and 13:02 UTC (exact time: hover over the date on Kaggle). Paper wording: transformers
     5.19.0 (Exp 21, 23) and 5.20.1 (Exp 16, 22), inferred from the installed wheel; model revision ad9f0ae.
5. Writing items carried from paper-writing-fixes.md: shrink the Exp 17 framing and cite Backour; one paragraph on the
   Rizvi-Martel tension. Track 3 (faithfulness, SFT vs RL) stays future work.
   - *2026-10-26:* Both drafted in paper-writing-fixes.md §7 after reading both papers. Backour's CALIB is our snap
     (snap = 1 − CALIB), so collapse is no longer a headline contribution; Rizvi-Martel's "first few layers" rests on
     logit-lens entropy and soft-vs-argmax KL over mostly near-one-token inputs, and their own §6 puts commitment in the
     last layers. Awaiting approval.
6. Next: figure scripts (one per figure, reading `results/`). Started 2026-10-26: `figures/style.py`, `figures/data.py`, Fig 2, Fig 3, Fig 4; appendix Tables A1–A3 (`make_tabA_fig4.py`), Fig A1, Fig A2, Fig 5, Fig 6.
   - *2026-10-26:* Fig 6 approved (Hassan) after a rework: the paired split 59 / 24 / 17 with the sign test is drawn in (a),
     the median ratio and its CI are a marker in (b), and the 31 matched steps are stated in the panel. The script now
     recomputes the sign test itself (C23.6).
   - *2026-10-26:* Table 1 drafted (`figures/make_tab1.py` → `figures/tab1.tex`, booktabs). Every McNemar compares an arm
     with sampling; in Exp 16 on seed 0, the only seed vanilla has (new rows C16.12–13). Every cell is checked by
     `verify_robustness.py` (Tab1a–b).
   - *2026-10-26:* Table 1 approved (Hassan). Fig 1 drafted in TikZ (`figures/fig1.tex`, standalone build
     `figures/fig1_standalone.tex`, compiled copy `figures/fig1.pdf`): (a) one thinking step, with the fed blend
     split into top token (= sampling, Gumbel-max) and residual, and the three Exp 23 arms; (b) the four questions with
     test, finding and figure. Weights in (a) are illustrative; the numbers in (b) are ledger values (C21.15–21, C22.8,
     C22.12, C23.15, C23.2–3). Awaiting approval.
   - *2026-10-26:* Fig 1 redrawn as a concept drawing after Hassan's review (the first draft was too text-heavy):
     (a) sampling and (b) Gumbel Soft Thinking unrolled over thinking steps, with the soft input drawn as a grey top
     token plus a blue residual slice, and the four questions as numbered marks with a one-line legend. No data
     numbers; weights are illustrative. Style modelled on Rizvi-Martel et al. Fig 1, not copied. Awaiting approval.
   - *2026-10-26:* Fig 1 redesigned from scratch (Hassan: the unrolled version was too close to Rizvi-Martel et al.).
     (a) a triangle of blends of three candidate tokens: sampling feeds a corner, Soft Thinking feeds that corner plus
     a short blue step (the residual), the control turns the step toward a random token; (b) one small picture per
     question: a blend point sliding to a corner over layers, a K,V grid with the runner-up stored in early layers and
     read later, a cloud of fed blends near the corners, and next-token bars shifted by each input. No data numbers.
     Awaiting approval.
   - *2026-10-26:* Split into two figures after Hassan's review (overlaps, untidy geometry). Fig 1 (`fig1.tex`) is now
     the concept alone: next-token probabilities branch into sampling (one grey token), Gumbel Soft Thinking (same
     token + blue residual) and the control (residual on a random token), with braces over the shared top token and
     the residual. The four questions are a separate figure (`fig_questions.tex`, standalone
     `fig_questions_standalone.tex`), one small drawing each. Fig 1 needs `\usetikzlibrary{decorations.pathreplacing}`.
     Placing the questions figure as Fig 2 would renumber the data figures (2-6 become 3-7). Awaiting approval.
   - *2026-10-26:* Fig 1 and the four-questions figure approved (Hassan). All figures and Table 1 are now done.
   - *2026-10-26:* The four-questions figure is **Fig 1.1** (Hassan), so Figs 2-6 keep their numbers. Files renamed to
     `figures/fig1_1.tex`, `fig1_1_standalone.tex`, `fig1_1.pdf`; the LaTeX for the 1.1 number (local \thefigure and
     \theHfigure, then \addtocounter{figure}{-1}) is in the file header, tested with hyperref: Figure 1, 1.1, 2.
7. *2026-10-26:* Outline drafted (`docs/paper-outline.md`). Hassan: the title must name superposition; venues are not
   TMLR only (one archival venue among TMLR, ACL 2027 via ARR, COLM 2027, ICML 2027, plus arXiv and non-archival
   workshops); a named preprint for professors' review comes first. Awaiting approval of title and contribution order.
   - *2026-10-26:* The Fig 1.1 bullet above is superseded by the renumbering note at the top of this log (Hassan, 17:18):
     questions = Fig 2, data figures 3–7, files renamed (`fig2.tex`, `make_fig3.py` … `make_fig7.py`, `make_tabA_fig5.py`).
     The outline and `next-chat-prompt-preprint.md` were written from the older decision and are now updated to match.
     The "Figure 5" references in Fig A2 and Tables A1–A3 are correct under the new numbers (KV patching).

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
