# Next step, figures, and validating the source of truth

Written 2026-10-16. The repo has raw outputs for **Exp 21** (`results/exp21/`) and **Exp 23** (`exp23_residual_ablation_items50-149.jsonl`) only.

## 1. What to do next, in order

1. **Collect every raw output into `results/`** (see §3). This is the only blocker, because the figures and the number checks both need these files.
2. **Validate the source of truth** (§3). Build a claims ledger and recompute every headline number from the raw files.
3. **Make the figures** (§2). Write one script per figure that reads from `results/`.
4. **Write the one-page outline**: title, 3–4 contribution bullets, the figure list below, and the claim each figure supports.
5. **Write the paper** in the TMLR template. Order: results → setup → introduction → abstract → related work (including the open items: shrink Exp 17 and cite Backour, plus the Rizvi-Martel tension paragraph) → limitations.

## 2. Designing the figures

There are two kinds, made with different tools.

**A. Concept diagrams** show how the method and the experiments work, not data. Draw them by hand in a vector tool such as draw.io (free), Inkscape, Figma, or TikZ if you like LaTeX. Export them as PDF or SVG, never PNG or JPG. Keep each one simple: boxes, arrows and few words. One is enough:
- **Fig 1, overview.** One think step of Gumbel Soft Thinking: logits → filter → Gumbel noise → blend weights. Split the blend into its *top token*, which is exactly what sampling picks, and the *residual*. Then mark the four questions with the experiment that tests each: does the blend collapse? (Exp 17/21); is the runner-up stored? (Exp 18/20); what does noise actually feed? (Exp 22); does the residual change answers? (Exp 23). A reader who sees only Fig 1 should understand the paper.

**B. Data plots**, made by Python scripts (matplotlib) from the saved jsonl files and never edited by hand afterwards. Proposed set, one claim each:

| Fig | Shows | Data | Claim it supports |
|---|---|---|---|
| 2 | Snap by layer (0 → final), one line per fork type, with CI bands | Exp 21 step4 | Blends collapse progressively; the harder the contrast, the harder the snap |
| 3 | Readback score over text positions: blend vs control, with the runner-up's text vs the model's own text | Exp 21 step5 carryover | The runner-up is stored and can be read back, but stays dormant on the model's own path |
| 4 | Share of the gap closed per layer block (bars), remove vs insert | Exp 21 step5 patch | The runner-up is stored mostly in layers 0–7 |
| 5 | Histogram of the fed top weight at forks for Gumbel, Dirichlet and vanilla, plus the flip rate vs sampling | Exp 22 | Real blends are lopsided, and Gumbel flips exactly like sampling |
| 6 | Paired flip order (soft vs random) and matched-step KL | Exp 23 | The model reads the residual |
| Table 1 | Accuracy: Exp 16 (3 arms) and Exp 23 (3 arms), with McNemar | Exp 16, 23 | No detectable accuracy difference |

**Style rules** (reviewers notice these):
- **Same colour = same thing in every figure.** For example: sampling = grey, Soft Thinking = blue, random-residual control = orange, and one colour per fork type. Use a colour-blind-safe palette (Okabe–Ito).
- **Plot at final size.** Make it 6.5 in wide for a full-width TMLR figure (3.2 in for half width), with text at 8–9 pt so it matches the paper font. Export as PDF.
- **Show uncertainty.** Add CI bands or error bars, and state the n in the caption.
- **The caption states the takeaway.** For example: "Blends collapse progressively: snap rises from 0 at the input to 0.84 (number vs number) at the output." Avoid captions like "Snap by layer."
- **Use plain labels**: "runner-up", "chosen token", "snap (1 = clean switch)", not variable names.
- **One script per figure**, `figures/make_fig2.py` etc., reading `results/` and writing `figures/fig2.pdf`. If a number changes, rerun the script; never edit the figure by hand.

## 3. Validating the source of truth

The source of truth is the **raw output files**, not the research log. The log is a record of what you saw at the time. Validation means proving that every number in the paper can be regenerated from a raw file by a script.

**Step 1: inventory.** For each experiment the paper uses, find its raw file and put it in `results/expNN/`:

| Exp | File | In repo? |
|---|---|---|
| 16 accuracy | `gsm8k_eval.jsonl` | Yes, `results/exp16/`. Matches the log. |
| 17/18/20 pilot (items 0–9) | `step4_alpha_sweep_items0-9.jsonl`, `step5_kv_carryover_items0-9.jsonl`, `step5_kv_patch_items0-9.jsonl` | **No**. Needed only if the paper reports the pilot; the items 10–49 replication can carry the claims alone. |
| 19 fork outcome | `step5_fork_outcome_items0-9.jsonl` | Partial run (52 of 61 forks). **Dropped from the paper** (decided 2026-10-17); Exp 23 covers the question. |
| 21 replication | three files | **Yes**, `results/exp21/` |
| 22 mixtures | `exp22_mixtures_items10-29.jsonl` | Yes, `results/exp22/`. Matches the log exactly. |
| 23 ablation | `exp23_residual_ablation_items50-149.jsonl` | Yes, `results/exp23/` |

Recover the missing ones from each notebook's Versions → Output tab, as you did for Exp 21. If one is truly lost, rerun it (all runs are seeded and deterministic) or drop that number from the paper.

**Step 2: claims ledger.** Make a `claims.md` table with one row per number in the paper: claim | value | script | raw file | log entry. Then run a small script, `verify_claims.py`, that recomputes each value from the raw file and asserts it matches. For Exp 21 this is already done: all values match the log and all six predictions are met.

**Step 3: rerun the built-in checks.** Run the assert gates in each script (determinism, layer-0 alignment, Gumbel-max flip rate) on the saved files where they can run without a GPU.

**Step 4: fix the log.** Where a recomputed number differs from the log, add a dated correction to the log and never overwrite the old entry. The snap rescale is the first such entry.

**Step 5: verify citations.** For every paper you cite, open it yourself and check four things: (a) the exact arXiv ID, title and authors in your .bib; (b) which version you cite (Wu et al. has changed title, so cite v4); (c) that the sentence where you cite it says what that paper actually says; (d) its publication status (e.g. Wu et al. is ICLR 2026). Current list: Zhang et al. 2505.15778, Wu et al. 2508.03440v4, Rizvi-Martel et al. 2604.06374, Backour 2609.02049, Deng et al. 2510.15522, Butt et al. 2509.19170, plus Gumbel-max / Kool et al. 2019 and any statistics references (McNemar, bootstrap).

**Step 6: fresh eyes.** Once the draft exists, have someone (or a simulated reviewer) check each figure and table against the claims ledger.
