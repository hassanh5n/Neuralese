# Paper writing fixes (from the 2026-10-12 plan)

> **Figure renumbering, 2026-10-26 (Hassan).** The four-questions figure is Fig 2. Entries written before this note
> use the old numbers: old Fig 2 → **3** (snap by layer), 3 → **4** (readback), 4 → **5** (KV patching), 5 → **6** (fed
> blends), 6 → **7** (residual ablation). Fig 1, Figs A1–A2 and Tables 1, A1–A3 keep their numbers. Files and scripts
> were renamed to match (`make_fig3.py` … `make_fig7.py`, `make_tabA_fig5.py`); all figures regenerated; 110/110
> and 17/17 checks pass.

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

> Wu et al. (2025, v4) report that Gumbel Soft Thinking outperforms matched discrete sampling (same temperature, top-p and top-k) on three 32B reasoning models. The gain averages +0.4 to +1.1 points across eight benchmarks, with no variance or significance reported; GPQA-Diamond shows the largest gains (up to +5 points). At 1.5B on GSM8K we find no difference (Exp 16: 52/60 vs 52/60; Exp 23: 88 vs 86 of 100, McNemar p = 0.754). The two findings are compatible. Wu et al. motivate Gumbel-Softmax through the Gumbel-max trick (their §5.4). We make the consequence explicit and verify it in code (winner flip rate 0.45 vs 0.45 for sampling): the token that Gumbel Soft Thinking puts on top is distributed exactly as a sampled token. Dirichlet noise, which Wu et al. find no better than sampling, has no such guarantee: its top token flips 0.03 less often than a sampled token would (95% CI [0.02, 0.04]), so it leans toward the model's top choice. Any accuracy difference from sampling must therefore come from the residual weight on the other candidates. Our coupled ablation (Exp 23) holds the noise fixed and removes only that residual. The model reads the residual: on matched steps it shifts next-token predictions a median 3.9× as much as an equal-mass random residual (95% CI [3.1, 5.0]), and more on 94 of 100 problems. Yet answers do not change detectably. Our test cannot exclude effects of one or two points (10 discordant pairs), which is the size of the gains Wu et al. report. Three things could separate the settings: model scale (a 32B model may use a residual that a 1.5B model only stores); task difficulty (GSM8K is near ceiling at 87–88%, whereas the largest reported gains are on harder benchmarks); and statistical power. Running the coupled ablation on a 32B model and a harder benchmark would show which one applies.

*Updated 2026-10-26:* "about 5×" replaced by the per-problem median 3.9× [3.1, 5.0], higher on 94/100 (C23.15, C23.8;
the 5.7 ratio of means is driven by five problems, C23.14); Dirichlet sentence added (C22.8). Ledger rows for every number
in the paragraph: C16.1, C16.3, C23.1–2, C22.4, C22.8, C23.8, C23.15.

Note for yourself: our settings differ from theirs in top-k (10, from the Soft Thinking repo, vs their 30), budget (8,192 vs 32,768) and Cold Stop. These apply to both of our arms, so our comparison stays matched. Put them in the setup section, not in this paragraph.

## 5. Disclose AI assistance

TMLR allows LLMs as assistive tools; the author is fully responsible for all content, and LLMs can't be authors. Draft statement (Acknowledgements or an appendix section "Use of AI assistants"):

> The author used an AI assistant (Anthropic's Claude) for three tasks: summarising related literature, drafting Python scripts for the experiments and analyses, and summarising results and organising experimental data. The author designed and ran all experiments, reviewed and tested all code, checked every reported number against the saved outputs and every citation against its source, and takes full responsibility for the content of this paper.

---
**Still open from the plan (not writing fixes):** shrink Exp 17 and cite Backour (arXiv:2609.02049) as independent confirmation, and add one paragraph on the tension with Rizvi-Martel et al. (early-layer vs gradual collapse).

## 6. Decisions from the figure pass (2026-10-26)

Read this section when planning the write-up. Every number below has a ledger row in `neuralese-paper-log.md`.

### 6.1 Terms for the fed blend (Exp 22) and the residual (Exp 22, 23)

| Use | Meaning | Do not say |
|---|---|---|
| **top weight** | Largest weight in the blend actually fed (`fed1`) | |
| **residual** | 1 − top weight: all blend weight not on the top token. Same word in Exp 22 and Exp 23 | "runner-up weight" for a fed blend |
| **second weight** | Second-largest fed weight (`fed2`); use only where the single next token matters | "runner-up" (it can be a different token) |
| **runner-up** | Pre-noise second candidate only (fork rule, Exp 21) | |
| balanced / lopsided / near one token | Fed top weight ≤ 0.75 / 0.75–0.9 / ≥ 0.9 (Fig 5a) | |
| **flip**, **sampling rate** | Fed winner is not the pre-noise top token; 1 − pre-noise top weight = how often sampling would flip | |

**Decided 2026-10-26 (Hassan): "residual" is the paper term.** Why "residual": at the median fork the second-largest token holds 99–100% of the residual (C22.12), so the two
numbers say the same thing, and one word then links the experiments: Gumbel feeds a median residual of 0.12
(Exp 22, C22.12), and the model reads the residual (Exp 23). Rewrite the ledger's "median fed top / runner-up
0.88 / 0.10" in the paper as "median top weight 0.88, residual 0.12".

### 6.2 Wording the claims (what the robustness checks allow)

- **Snap order** (Fig 2): number vs word > number vs number > word vs word holds **at the output**. Number vs number
  is below word vs word at layers 1–12 (C21.23). Say "at the output".
- **Readback** (Fig 3): lasts the whole 64-token continuation (C21.26); dormant on the model's own text, every CI
  includes 0 (C21.25, C21.27).
- **Patching shares** (Fig 4): always "pooled share of the gap". Layer 0 is uneven across forks (pooled 0.38, median
  fork 0.12; C21.34), so do not write "layer 0 holds 38% of the runner-up" as if it held at every fork.
- **Pre-registered vs exploratory:** the block split (layers 0–7 vs 12–27, P5, P6) was pre-registered. Single layers
  1, 2, 7, 13 and the non-additivity inside layers 0–3 are **exploratory**; cite the Bonferroni check (C21.35).
- **Non-additivity** (C21.31): report as a fact. Two readings, say both: the runner-up is held redundantly across
  layers 0–2, or the readback score saturates. These data do not separate them.
- **Scope of Exp 21:** exact 50/50 blends at forks where the model followed the chosen token. Real Gumbel forks feed
  a median residual of 0.12, so Exp 21 is an upper bound. Put this in each Exp 21 caption and in Limitations.
- **Gumbel-max:** holds pooled and bin by bin (C22.8–9). One bin-level CI of ten misses; do not claim "every bin".
- **New result, Dirichlet:** it flips less often than sampling (−0.03 [−0.04, −0.02], below sampling in all ten bins;
  C22.8, C22.10). So Dirichlet is *not* exactly sampling in disguise: its winner leans toward the model's top token.
  Worth one sentence in Results and a line in the Wu et al. paragraph (their Dirichlet does no better than sampling).
- **Size of the residual effect** (Fig 6): quote "a median 3.9× the random residual per problem, 95% CI [3.1, 5.0];
  higher on 94 of 100" (C23.15, C23.8), not "about 5×". The 5.7 ratio of means is driven by five problems (C23.14).
  **Update the §4 draft paragraph** ("about 5× the prediction shift") and the Exp 23 text accordingly. *(§4 done 2026-10-26.)*

### 6.3 Figure and table conventions (for the remaining figures and the paper)

- Exact final size: 6.5 in or 3.2 in wide, fonts Latin Modern 7–8 pt (pdflatex via pgf). Include with
  `\includegraphics{figN.pdf}` at natural size, never `width=\linewidth`, so text stays at 7–8 pt.
- Captions open with the takeaway sentence, state n (forks and problems), the CI method, and the scope.
- Same colour = same thing (see `figures/style.py`); filled = remove / runner-up's text, open = insert / own text.
- Appendix items that back the main text: Tables A1–A3 (Fig 4 robustness, single layers, number vs word), Fig A1
  (readback unbinned), Fig A2 (per-fork shares). Checked by `verify_robustness.py`.
- For the Rizvi-Martel paragraph: they read Qwen2-1.5B at layers 0/7/14/21/27 and report collapse "within the first
  few layers"; our Fig 2 shows snap rising gradually up to the output. Cite both figures side by side.


## 7. Related-work drafts: Backour and Rizvi-Martel (2026-10-26)

Both papers were read in full on 2026-10-26 (PDFs from arXiv, text extracted). Numbers from our runs carry ledger ids.

### 7.0 Citation check

*Checked 2026-10-26 (second pass):* every claim about both papers re-read against their text; snap = 1 − CALIB
confirmed against `neuralese_sweep.py` (c is the projection onto pure-B → pure-A, the same as Backour's α(w)); our Exp 17
ran 2026-10-03, after Backour's 2026-09-02 posting. Three wordings tightened (training "amplifies", not "causes most";
Rizvi-Martel inputs near one token "on average"; runner-up "mostly" in layers 0–7).

| Paper | Verified details | What it shows (for our purposes) |
|---|---|---|
| Backour, *The Dynamics of Continuous Mixture Collapse in Language Models*, arXiv:2609.02049v1, 2 Sep 2026, single author (MIT), preprint "under review" | Title, author, ID, version, date | Injects two-way embedding mixtures w·e1 + (1−w)·e2 into 1,000 synthetic prompts (7 categories: color, temperature, size, speed, hardness, weight, brightness). Per layer, projects h(w) onto the line h(0) to h(1): α(w); scores CALIB = 1 − E\|α(w) − w\|/Z, so exact preservation = 1 and a hard switch at 0.5 = 0. Trained Qwen3.5 (0.8B to 27B) and Gemma 4 models collapse far more than matched random-init controls, and more with scale. "Qwen accumulates collapse toward the output". Second source: softmax readout plus autoregressive feedback amplifies small imbalances until one component dominates; pretrained rollouts sit mostly on the amplifying side. No accuracy, no KV cache, no Gumbel noise. |
| Rizvi-Martel, Rabusseau & Mosbach, *The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models*, arXiv:2604.06374v2 (v1 7 Apr 2026, v2 3 Aug 2026), **COLM 2026** (cite as COLM 2026) | Title, authors, ID, version, venue (PDF header "Published as a conference paper at COLM 2026") | Training-free regime is **vanilla** Soft Thinking (top-k 30, soft top-k 15, no noise, 2,048 tokens, bf16). Main text QwQ-32B; Qwen2-1.5B and R1-Distill-Llama-70B in the appendix. Logit lens at 5 layers ({0, 7, 14, 21, 27} for Qwen2-1.5B). §4.1: logit-lens entropy of separate Soft Thinking and discrete runs, every 50 steps, is nearly identical. §4.2: replacing the current soft input by its argmax embedding gives KL ~1e-4 in middle layers, largest at the first and last layers; mean mixing-weight entropy 0.10 nats (Qwen2-1.5B), 0.18 (QwQ-32B). Their claim: collapse "within the first few layers". §6: next-token pretraining makes models "commit to a token in the last layers" (Pythia-1B, uniform mixtures). |

Correction to the note in §6.3 above: Rizvi-Martel's main-text model is QwQ-32B (Qwen2-1.5B is in their appendix), and
their own §6 locates commitment in the **last** layers; the "first few layers" claim comes from §4.1–4.2. The paragraph
in §7.2 uses the corrected reading.

### 7.1 Shrink the Exp 17 framing: snap is Backour's measurement

**Why shrink.** Our snap is the same measurement as Backour's: the projection c(α) is his α(w), and snap = 1 − CALIB
on our 11-point α grid (both normalise so that a hard switch at 0.5 is the reference). He posted it first (2 Sep 2026),
on more models, with random-init controls. So "blends collapse through the layers" is not a contribution of ours.
What is ours: the same measurement in a reasoning model, at natural forks from Soft Thinking runs, split by kind of
candidate, pre-registered, and with vanilla and Gumbel traces compared.

**What changes in the paper.**
- Contributions list: drop "blends collapse progressively" as a headline bullet. It becomes the set-up for the
  storage question.
- Fig 3 section: one short paragraph (below), opening with Backour's measurement, not ours.
- Exp 17 (the items 0–9 pilot) is not named in the paper at all; Exp 21 carries every number (see §3 of the log:
  "thresholds set on a 10-item pilot, then tested on 40 held-out items").

**Draft paragraph (Results, Fig 3):**

> **Blends snap toward one candidate across the layers.** We use the measurement of Backour (2026). At each fork we
> feed α·e(A) + (1 − α)·e(B) for α = 0, 0.1, …, 1 from the same cached prefix and project every layer's hidden state
> onto the line from the pure-B to the pure-A state. Our snap is 1 minus his CALIB score on this grid: 0 means the blend
> is carried linearly, 1 a clean switch at α = 0.5. Backour finds collapse for synthetic two-word mixtures in Qwen3.5
> and Gemma 4 models; we find it in a reasoning model at forks taken from Soft Thinking runs on GSM8K. Snap rises from 0
> at the input to 0.37–0.70 at layer 7 and 0.77–0.98 at the output, and the noise that produced the trace does not
> change it (number vs number at the output: 0.85 vanilla, 0.84 Gumbel). How far the two candidates differ matters at
> the output: number vs word snaps hardest (0.98), then number vs number (0.84), then word vs word (0.77). Collapse is
> therefore expected. The open question is what it leaves behind: is the runner-up gone, or only absent from the
> next-token prediction (§X)?

Ledger: snap = 0 at layer 0 (C21.2), layer 7 and output values (C21.4), output CIs separate (C21.5), vanilla vs Gumbel
(C21.14). If space allows, add "number vs number starts slowest and passes word vs word only from layer 13" (C21.23).

**Draft sentence (Related work):**

> Backour (2026) shows that pretrained models collapse injected two-way embedding mixtures through depth, that training
> strongly amplifies a distortion the architecture already causes, and that softmax feedback across steps amplifies small imbalances
> until one component dominates. Our snap score is his CALIB measure applied to natural forks in a reasoning model. We
> go further and ask whether the losing candidate is stored and read back, what blends the noise actually feeds, and
> whether the residual changes predictions and answers.

### 7.2 The Rizvi-Martel tension

**The tension.** They conclude that off-the-shelf models collapse Soft Thinking inputs "within the first few layers";
our Fig 3 shows a gradual collapse that completes only at the output. The two are measured differently, and their own
§6 agrees with us.

**Draft paragraph (Discussion or Related work):**

> Rizvi-Martel et al. (2026) conclude that off-the-shelf models collapse Soft Thinking inputs within the first few
> layers, while our Fig 3 shows a gradual collapse that completes only at the output. The two results measure different
> things. Their evidence for early collapse compares logit-lens entropy between separate Soft Thinking and discrete runs,
> and the KL divergence between forward passes with the soft input and with its top token, every 50 steps of vanilla
> Soft Thinking. Those inputs are close to one token on average (mean mixing entropy 0.10 to 0.18 nats), so there
> is little blend left to lose. We condition on forks, feed exact 50/50 blends of the two candidates, and track the
> position between them at every layer. On these inputs the blend is still far from a clean switch at layer 7 (snap
> 0.37 to 0.70). This agrees with their own test on uniform mixtures, where entropy stays high until the last layers
> (their §6), and with Backour (2026), who finds that Qwen models accumulate collapse toward the output. Collapse in the
> next-token readout also does not mean the runner-up is lost: it stays mostly in the keys and values of layers 0 to 7
> and can be read back later (§X).

Ledger: snap at layer 7 (C21.4); storage in layers 0–7 (C21.10, P5); readback (C21.7–8). Their numbers (0.10 / 0.18
nats, every 50 steps, vanilla) are from their Table 1, §4.2 and Appendix B.2. Keep the caveat from §6.2: our blends are
50/50, an upper bound on real forks (median residual 0.12, C22.12); say so where the paragraph lands.
