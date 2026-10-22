# Paper writing fixes (from the 2026-10-12 plan)

Status 2026-10-15. These are the five fixes from the handoff's "Writing fixes" list, each ready to paste into the paper. Old log entries stay as they are; this file and the code comments record the changes.

## 1. Rename the two "B"s

The logs use "B" both for the losing candidate token and for the hypothesis "Option B". The paper uses these names:

| Log term | Paper term | Notes |
|---|---|---|
| A (top candidate at a fork) | **chosen token** (or "top candidate") | the token whose path the model follows |
| B (losing candidate) | **runner-up** | the second candidate at a fork |
| C (control token) | **control token** | |
| "B-leftover test", "control − blend, B's text" | **runner-up readback** | Exp 18/21 |
| "residual" (Exp 23) | **residual**: the blend weight not on the chosen token | |
| Option A | **genuine superposition** | the model carries and uses both paths |
| Option B | **sampling in disguise** | noise picks one token per step, as sampling does |
| soft / hard / rand (Exp 23 arms) | **Soft Thinking / sampling / random-residual control** | keep the short names in tables |

Use "fork" exactly as defined: a step where the runner-up holds ≥ 25% of the pre-noise filtered weight.

## 2. Rescale snap so a perfect step scores 1 (done in code)

**Why.** Snap was `mean|c(α) − α| / 0.25`. On the 11-point α grid, a perfect step at α = 0.5 gives mean|c − α| = 2.5/11, so it scored 10/11 ≈ 0.91, not 1.

**New definition (paper).** Snap = mean|c(α) − α| divided by the same quantity for a perfect step at α = 0.5. Then 0 = the blend is carried linearly, 1 = a perfect switch to one candidate at the midpoint, and values above 1 mean overshoot. Conversion: **new = old × 1.1** exactly.

**Code changed (not run, no git):**
- `neuralese_sweep.py`: adds `STEP_SNAP = sum(abs((a > 0.5) - a) for a in ALPHAS) / len(ALPHAS)` (= 2.5/11) and divides by it instead of 0.25. The docstring explains the old scale.
- `neuralese_stats.py`: pre-registered P1 threshold 0.6 → 0.66 (the same threshold on the new scale), with a comment.

**Rescaled numbers.** The Exp 21 final-layer row is exact: it was recomputed on 2026-10-16 by `neuralese_stats.py` from the saved outputs in `results/exp21/`. All six predictions are met, and every other Exp 18/20 number matches the log. The by-layer rows (items 0–9) are the logged values × 1.1, so their second decimal can be off by ±0.01.

| Final-layer snap (Exp 21, items 10–49) | Old | New |
|---|---|---|
| Number vs number | 0.77 [0.75, 0.78] | **0.84 [0.83, 0.86]** |
| Number vs word | 0.89 [0.87, 0.90] | **0.98 [0.95, 1.00]** |
| Word vs word | 0.70 [0.68, 0.72] | **0.77 [0.75, 0.79]** |

| By layer (Exp 17, items 0–9) | L1 | L7 | L14 | L21 | Final |
|---|---|---|---|---|---|
| Number vs number | 0.28 | 0.39 | 0.62 | 0.77 | 0.85 |
| Number vs word | – | 0.68 | 0.79 | 0.89 | 0.98 |
| Word vs word | – | 0.43 | 0.54 | 0.64 | 0.76 |

Other values: near-synonyms 0.16–0.38 → 0.18–0.42, and the "≈ 0.25 at layer 1" in Exp 20 → ≈ 0.28. Drop the phrase "≈ 0.91 = perfect step" from the paper.

New reading in one line: "By the output layer, a 50/50 number-vs-word blend is 98% of the way to a clean switch; number-vs-number is 84%, word-vs-word 77%."

## 3. Explain +0.44 in plain words

Draft paragraph (Results, KV section):

> At each fork we feed a 50/50 blend of the chosen token and the runner-up. The model's next token follows the chosen token. We then force it to read text that continues the runner-up's line instead. We score how the model's next-token predictions on that text look on a 0–1 scale: 1 means "as if it had written the chosen token", 0 means "as if it had written the runner-up". We compare with a control blend that mixes the chosen token with an unrelated token instead of the runner-up. The real blend scores 0.44 lower than the control on number forks (95% CI [0.35, 0.53]; 0.33 for word forks, 0.11 for number-vs-word). In other words, **the stored runner-up moves the model 44% of the way toward reading the text as if it had written the runner-up itself**. On the model's own continuation the difference is −0.01 [−0.06, 0.04]: the runner-up is kept and can be read back, but it is not used unless later text turns toward it.

Add the caveat that this is an upper bound: it is measured at exact 50/50 blends, the most favourable case. At real forks, Gumbel feeds a median runner-up weight of only 0.10 (Exp 22).

## 4. One paragraph on Wu et al.

Checked against Wu et al. v4 (arXiv:2508.03440v4, ICLR 2026) on 2026-10-15.

**What their comparison is.**
- Three 32B reasoning models (DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, Skywork-OR1-32B) on eight benchmarks: AIME'24, AIME'25, MATH-500, AMC'23, GPQA-Diamond, HumanEval, MBPP and LiveCodeBench. GSM8K is not one of them.
- The baseline is matched. Sampling and vanilla Soft Thinking both use temperature 0.6, top-p 0.95 and top-k 30, with a 32,768-token cap (§3.1). The Gumbel runs follow the same protocol with τ = 0.5 (§5.2). They use no Cold Stop: soft thinking ends when the top token is the end-of-thinking token.
- The gain is small. Gumbel minus sampling, averaged over the eight benchmarks, is +1.05 for R1-Distill-32B, +0.69 for QwQ and +0.42 for Skywork (Table 2). Per-benchmark gains range from −0.61 to +5.05; the largest is on GPQA-Diamond (+2.5 to +5.0, a 198-question set).
- Uncertainty is not reported. There are no seeds, standard deviations or significance tests. AIME and AMC use the mean of 32 samples; everything else is a single Pass@1 run.
- Dirichlet does no better than sampling: averages −0.14, −0.96 and +0.13.
- They do invoke the Gumbel-max trick (§5.4), but as the reason Gumbel-Softmax is well-founded: each candidate is selected in proportion to its probability. They do not test whether the residual weight matters.

**Draft paragraph (Discussion):**

> Wu et al. (2025, v4) report that Gumbel Soft Thinking outperforms matched discrete sampling (same temperature, top-p and top-k) on three 32B reasoning models. The gain averages +0.4 to +1.1 points across eight benchmarks, with no variance or significance reported; GPQA-Diamond shows the largest gains (up to +5 points). At 1.5B on GSM8K we find no difference (Exp 16: 52/60 vs 52/60; Exp 23: 88 vs 86 of 100, McNemar p = 0.754). The two findings are compatible. Wu et al. motivate Gumbel-Softmax through the Gumbel-max trick (their §5.4). We make the consequence explicit and verify it in code (winner flip rate 0.45 vs 0.45 for sampling): the token that Gumbel Soft Thinking puts on top is distributed exactly as a sampled token. Any accuracy difference from sampling must therefore come from the residual weight on the other candidates. Our coupled ablation (Exp 23) holds the noise fixed and removes only that residual. The model reads the residual (on matched steps, about 5× the prediction shift of an equal-mass random residual), yet answers do not change detectably. Our test cannot exclude effects of one or two points (10 discordant pairs), which is the size of the gains Wu et al. report. Three things could separate the settings: model scale (a 32B model may use a residual that a 1.5B model only stores); task difficulty (GSM8K is near ceiling at 87–88%, whereas the largest reported gains are on harder benchmarks); and statistical power. Running the coupled ablation on a 32B model and a harder benchmark would show which one applies.

Note for yourself: our settings differ from theirs in top-k (10, from the Soft Thinking repo, vs their 30), budget (8,192 vs 32,768) and Cold Stop. These apply to both of our arms, so our comparison stays matched. Put them in the setup section, not in this paragraph.

## 5. Disclose AI assistance

TMLR allows LLMs as assistive tools; the author is fully responsible for all content, and LLMs can't be authors. Draft statement (Acknowledgements or an appendix section "Use of AI assistants"):

> The author used an AI assistant (Anthropic's Claude) for three tasks: summarising related literature, drafting Python scripts for the experiments and analyses, and summarising results and organising experimental data. The author designed and ran all experiments, reviewed and tested all code, checked every reported number against the saved outputs and every citation against its source, and takes full responsibility for the content of this paper.

---
**Still open from the plan (not writing fixes):** shrink Exp 17 and cite Backour (arXiv:2609.02049) as independent confirmation, and add one paragraph on the tension with Rizvi-Martel et al. (early-layer vs gradual collapse).
