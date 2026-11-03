# Plausible-token control: pre-registration

Written 2026-11-03, before any run. Commit this file to git before launching the Kaggle notebook, and cite the
commit hash and the Kaggle version time in the paper's pre-registration appendix.

## Question
In the coupled ablation (items 50-149), the real residual of Gumbel Soft Thinking moves next-token predictions a
median 3.9 times as much as the same residual moved onto a uniformly random token (95% CI [3.1, 5.0]). Does the
model read *this* candidate, or does any context-plausible token move predictions this much?

## Design
- Script: `neuralese_plausible_ablation.py`, `PLAUS_MODE=kl_only` (primary). `full` is optional and descriptive.
- Same model, revision, decoding settings, seed 0, items 50-149 and keyed Gumbel noise as the coupled ablation.
- Arms in lockstep: hard (sampling), soft (Gumbel Soft Thinking), rand (residual on a uniformly random real token),
  plaus (residual on the most probable token that is neither the top token nor the token holding the largest share
  of the real residual). rand and plaus carry exactly the residual weight that soft carries.
- Measure: per-step KL(arm || hard) at the sampling temperature, averaged over matched steps (before either of the
  two compared arms first picks a token different from hard's).
- Statistics: median per-problem KL ratio, 95% percentile bootstrap over problems (2,000 resamples, seed 0);
  exact two-sided sign test on the number of problems where the first arm moves predictions more.
- Analysis: `python analyze_plausible.py <output>`; no other analysis is used for the verdicts below.

## Predictions
- **H1 (primary).** soft moves predictions more than plaus: median soft/plaus ratio > 1, with the 95% CI lower bound
  > 1, and soft > plaus on more than half of the problems.
- **H2.** plaus moves predictions more than rand: median plaus/rand ratio > 1, CI lower bound > 1.
- **H3.** H1 also holds on steps where the plausible token has pre-noise probability >= 0.05 (problems with at least
  one such matched step).
- **Replication.** soft/rand median ratio within the coupled ablation's CI [3.1, 5.0].

## How each outcome changes the paper
- H1 met: "the model reads the specific candidate that the residual carries, beyond any plausible token".
- H1 not met, H2 met: "the model is sensitive to plausible probability mass at that position, not specifically to the
  runner-up"; the abstract's "reads the residual" is weakened to that.
- Neither met: report as is; the 3.9x result is then attributed to the random token being off-context.

Accuracy (only in `full` mode) is descriptive: 100 problems cannot resolve differences of a few points.
