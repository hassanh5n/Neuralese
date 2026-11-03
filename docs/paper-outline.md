# Paper outline (one page)

Drafted 2026-10-26. Figure numbers follow the renumbering note at the top of the paper log (questions = Fig 2,
data figures 3–7). Every number has a ledger row in `neuralese-paper-log.md` §2 (ids in brackets).
Approve or change: the title (A), the contribution order (C), the venue plan (G).

## A. Title (must name superposition)

1. **Superposition or Sampling in Disguise? What Soft Thinking Stores, Feeds and Uses** *(recommended: it is the
   question the research started from, in the paper's own two terms, and the subtitle lists the three tests)*
2. The Residual of Superposition: Soft Thinking Feeds a Sampled Token Plus a Residual the Model Reads but Does Not Use
3. Does Soft Thinking Reason in Superposition? Storage, Noise and a Coupled Ablation in a Reasoning Model

Avoid echoing Rizvi-Martel et al. (*The Illusion of Superposition?*) too closely.

## B. The answer in one sentence

In a 1.5B reasoning model, Gumbel Soft Thinking is a sampled token plus a residual: the model stores the runner-up and
reads the residual, but the residual does not change answers detectably, so we find no superposition that is used.

## C. Contributions (in this order)

1. **Gumbel Soft Thinking = sampling + residual.** By the Gumbel-max trick its top token is distributed exactly as a
   sampled token; verified in code (flip rate 0.45 vs 0.45 [C22.4], also bin by bin [C22.8–9]). Real runs feed lopsided
   blends: near one token at 47% of forks, median residual 0.12 [C22.2, C22.12]. Dirichlet is not even sampling: its top
   token leans toward the model's first choice [C22.8, C22.10]. *(Fig 1, Fig 6)*
2. **The runner-up is stored, not erased.** Blends snap toward one token through the layers (Backour's measure,
   confirmed here [C21.4]), yet the runner-up stays in the K,V of layers 0–7 (71–91% of the effect, pre-registered
   [C21.10]) and later text can read it back (+0.44 for number forks [C21.7]); on the model's own path it is dormant
   [C21.25]. *(Figs 3–5)*
3. **The model reads the residual but does not use it for answers.** A coupled ablation runs sampling, Soft Thinking
   and a random-residual control in lockstep with shared noise. Soft Thinking leaves sampling sooner (59 / 24 / 17,
   p = 0.0002 [C23.5–6]) and moves predictions a median 3.9× the random residual [3.1, 5.0] [C23.15], yet accuracy does
   not change (88 vs 86 of 100, p = 0.75 [C23.1–2]). *(Fig 7, Table 1)*
4. **A checkable study.** Predictions committed before the runs, failed ones reported, and every number recomputed from
   saved outputs by two scripts (110 + 17 checks). *(Appendix)*

## D. Figures and tables, in reading order

| # | Shows | The one claim |
|---|---|---|
| Fig 1 | Concept: sampling, Soft Thinking, control as bars | Soft Thinking = the sampled token + a residual |
| Fig 2 | The four questions | What we test and where |
| Fig 3 | Snap by layer, by fork kind | Blends collapse gradually, completing at the output (Q1) |
| Fig 4 | Readback over 64 tokens, runner-up's vs own text | The runner-up is stored and readable, dormant on the own path (Q2) |
| Fig 5 | Patching shares by layer block | Storage sits in layers 0–7 (Q2) |
| Fig 6 | Fed top weight; flip rate vs sampling | Real blends are lopsided; Gumbel flips exactly like sampling (Q3) |
| Fig 7 | Flip order; per-problem KL ratio | The model reads the residual (Q4) |
| Table 1 | Accuracy and McNemar, Exp 16 and 23 | No detectable accuracy change (Q4) |
| Appendix | Tables A1–A3, Figs A1–A2, pre-registration, versions, AI use | Robustness and reproducibility |

## E. Sections (about 12 pages of main text)

| § | Job | Length |
|---|---|---|
| Abstract | Question, method, the three findings with numbers, meaning | 200 words |
| 1 Introduction | The superposition hypothesis; Figs 1 and 2; answer; contributions | 1.25 p |
| 2 Background and related work | Soft Thinking, Wu et al., Rizvi-Martel et al., Backour, latent-CoT patching, superposition theory | 1 p |
| 3 Setup | Model, decoding, Cold Stop, GSM8K splits, forks, blends, gates, pre-registration, statistics | 1.25 p |
| 4 Results | One subsection per question (Q1–Q4), each opening with the question and closing with its answer | 5.5 p |
| 5 Discussion | Reconciling Wu et al. (gains at 32B) and Rizvi-Martel; what used superposition would need | 1 p |
| 6 Limitations | One model at 1.5B, GSM8K, 50/50 blends as an upper bound, power of the accuracy test, Exp 22 overlaps Exp 16/21 | 0.5 p |
| 7 Conclusion | Three sentences | 0.25 p |

Drafts already written: Wu et al. paragraph (paper-writing-fixes.md §4), Backour and Rizvi-Martel (§7), AI-use
statement (§5), +0.44 explanation (§3). Write in this order: Results → Setup → Introduction → Abstract → Related work →
Discussion and Limitations.

## F. The preprint for professors

- **One source, several wrappers.** Section files are shared; `preprint.tex` is the named version, and venue
  wrappers (anonymous, in each venue's style) are added later without touching the text.
- **Preprint layout:** article class, US letter, 1 in margins (6.5 in text width, so every figure stays at natural size),
  Latin Modern 10 pt, author name and affiliation, "Draft for discussion" and the date in the header, line numbers so
  reviewers can cite lines. Compiled here with pdflatex and also uploadable to Overleaf as a zip.
- **Same file later goes to arXiv** (without the draft header and line numbers), after the professors' feedback.

## G. Venues (not TMLR only)

Rule: one **archival** venue for these results; arXiv and non-archival workshops can be added to any of them.

| Venue | Type | Deadline (checked 2026-10-26) | Fit |
|---|---|---|---|
| arXiv preprint | Not peer-reviewed | Any time; aim after professor feedback | Yes, first. Timestamp, citable. Needs an endorser for a first cs submission (ask a professor). |
| TMLR | Journal, rolling, archival | Any time | Strong: judges correctness, not novelty or scale. No page limit pressure. |
| ACL 2027 via ARR | Conference, archival | 4 Jan 2027 (ARR); conference 20–22 Aug 2027 | Good audience (LM interpretability). 8-page limit forces cuts; check ARR's preprint rules. |
| COLM 2027 | Conference, archival | Not announced (2026 was late March) | Very good fit: Rizvi-Martel appeared there. |
| ICML 2027 | Conference, archival | About 22 Jan 2027 (estimate only) | Hard: one 1.5B model, one dataset. |
| ICLR 2027 workshops | Usually non-archival | About 1 Feb 2027 | Yes, alongside any of the above. In person only. |

Suggested path: professor review (November) → arXiv → choose by mid-December between ACL 2027 (if the professors
think the scope is enough for a conference) and TMLR (default; rolling, no deadline risk) → an ICLR 2027 workshop in
parallel.
