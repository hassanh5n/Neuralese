# Neuralese Research Log — Continuation

Picks up where `neuralese-research-log.md` currently ends: Exp 8 (Dirichlet sweep, 2026-08-28),
Review Notes 2026-08-24, and the Harness Update / Next Steps section, which marks Steps 1–3 done
and Step 4 (logit lens) as the current focus. Append to the canonical log once Exp 9 is actually run.

*(Correction note: an earlier version of this continuation file wrongly claimed Exp 7/7b/8 didn't
exist in the log — that was from reading a truncated view of the document and missed the middle
section entirely. They're all there, in full. This version is based on a complete read.)*

---

## Where the canonical log actually leaves off

- **Track 2 status:** vanilla (Exp 5), Gumbel (Exp 7b, EOS-fixed), and Dirichlet (Exp 8) are all
  complete on the same prompt, same frozen `soft_steps=100`, same five seeds `{0, 1, 2, 42, 123}`.
  This is the full three-condition comparison Next Step 2 called for.
- **Standing finding across all three conditions:** no seed, under any noise condition, produces a
  balanced decision-point weight split of the kind the superposition claim predicts. Dips are
  overwhelmingly framing/transition tokens. A handful of genuine digit-level dips exist (2 under
  Gumbel, 11 under Dirichlet across 3/5 seeds) but the log is explicit that these aren't yet
  interpretable as real superposition vs. noise-on-an-already-broken-answer — that's exactly what
  the lens + patching stage is for.
- **Reasoning ceiling confirmed, not resolved by noise:** every seed across all three conditions
  fails to solve the puzzle (0/15 correct). Noise changes *which* error shows up, not whether one
  occurs.
- **Harness is validated and stable:** EOS/void tracking and entropy logging both confirmed working
  cleanly across Exp 7b and Exp 8, no new bugs found in Exp 8.
- **Next Step 4 (logit lens) is marked "current focus"** with a priority step list written directly
  into the log:
  > seed 1 Dirichlet steps 55/69/88/93, seed 123 Dirichlet steps 28/47/53/76/89/97, seed 2
  > Dirichlet steps 20–25, seed 1 Gumbel step 93, seed 42 Gumbel step 55.

  Note this list, as written in the log's Next Steps section, **omits the two Gumbel candidates
  that Exp 7b itself found** (seed 2/step 80, seed 123/step 62) — see the discrepancy below.

---

## Real discrepancy found: code vs. log on Gumbel priority steps

This is the one actual gap, now correctly scoped:

| Source | Gumbel priority steps |
|---|---|
| Exp 7b entry in the log (body text) | seed 1/93, seed 42/55, **seed 2/80, seed 123/62** (four steps) |
| Next Steps section, Step 4 (log) | seed 1/93, seed 42/55 only (two steps) |
| `PRIORITY_STEPS` in `neuralese_harness.py` | `(\"gumbel\", 1): [93]`, `(\"gumbel\", 42): [55]`, **`(\"gumbel\", 2): [80]`, `(\"gumbel\", 123): [62]`** (four steps — **fixed before Exp 9**) |

**Fix applied before Exp 9:** the two missing Gumbel entries were added to `PRIORITY_STEPS` in the
harness code before the run. The code now matches Exp 7b's body text. The Next Steps prose in the
canonical log still only lists two Gumbel steps — this is a cosmetic lag, not a functional issue,
since `report_priority_steps` drives off the code dict, not the prose.

---

## Exp 9 — Logit-lens vs. mixture comparison (first lens pass) (2026-09-11)

**What this is:** Next Step 5 — the first real comparison of what the mixture weights show vs. what
the model's internal hidden state (decoded via logit lens) actually represents, at each of the 18
priority steps flagged across Exp 7b and Exp 8. This is the project's core contribution checkpoint:
does the noised variant's lens output show genuinely multi-candidate content that matches the
mixture, or does it diverge?

**Setup:** ran `main()` in the current `neuralese_harness.py`, which executes:
1. Sanity `.generate()` → manual baseline checkpoint (passed, byte-identical).
2. Default-condition soft-then-hard run (gumbel, seed=42, 100 soft steps) — soft-phase trace + hard
   output logged.
3. Step 7 diagnostics (full per-step weight/entropy table for the default condition).
4. `check_lens_checkpoint` — **passed: `hidden_states[-1]` is PRE-norm, `apply_norm=True`
   confirmed.** This was the hard gate; now cleared.
5. `run_priority_sweep` — loops every `(noise_mode, seed)` key in `PRIORITY_STEPS`, runs
   `generate_soft_then_hard` once each, then calls `report_priority_steps` to print mixture vs. lens
   side-by-side at every flagged step.

**Conditions swept** (7 total, each with its own priority steps):
- `(dirichlet, 1)`: steps 55, 69, 88, 93
- `(dirichlet, 2)`: steps 20–25
- `(dirichlet, 123)`: steps 28, 47, 53, 76, 89, 97
- `(gumbel, 1)`: step 93
- `(gumbel, 2)`: step 80
- `(gumbel, 42)`: step 55
- `(gumbel, 123)`: step 62

**EOS behavior:** Dirichlet seed=2 fired EOS at step 62 (matching Exp 8 exactly). Gumbel seed=2
fired EOS at step 86, Gumbel seed=123 at step 86 (matching Exp 7b). All other conditions: no EOS in
100 steps. No priority step fell inside a void region (all flagged steps < respective EOS fire
steps), so all 18 comparisons are clean.

---

### Results: lens vs. mixture at digit-level priority steps

#### Dirichlet, seed=1 (steps 55, 69, 88, 93)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 55 | '5' (0.583) | 5:0.583, 3:0.417 | '5' (0.769) | 5:0.769, 3:0.226, 4:0.004 | ✅ Agrees, **lens sharpens** |
| 69 | '4' (0.733) | 4:0.733, 3:0.201, 5:0.066 | '4' (0.397) | 4:0.397, 3:0.288, 5:0.288 | ✅ Top-1 agrees, **lens much flatter** (3 and 5 tied as runners-up) |
| 88 | '4' (0.762) | 4:0.762, 5:0.122, 3:0.107 | '4' (0.351) | 4:0.351, 1:0.301, 3:0.15 | ⚠️ Top-1 agrees, but **lens introduces '1' (0.301) not in mixture at all**; hidden state half-committed elsewhere |
| 93 | '4' (0.613) | 3:0.387, 4:0.613 | **'3' (0.767)** | 3:0.767, 4:0.121, 5:0.111 | ❌ **Top-1 DISAGREES.** Mixture says '4', hidden state says '3' at 0.767 |

#### Dirichlet, seed=2 (steps 20–25 — sustained low-confidence cluster)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 20 | ' until' (0.531) | until:0.531, ,:0.413 | ' until' (0.622) | until:0.622, and:0.062, to:0.042 | ✅ Agrees, lens more confident |
| 21 | ' until' (0.521) | until:0.521, it:0.448 | ' until' (0.301) | until:0.301, the:0.178, which:0.094 | ⚠️ Top-1 agrees but **lens extremely flat**, introduces new candidates |
| 22 | ' it' (0.998) | it:0.998 | ' it' (0.539) | it:0.539, the:0.294, they:0.025 | ⚠️ Top-1 agrees but **lens drops from 0.998 → 0.539** — hidden state far less certain than mixture |
| 23 | ' reaches' (0.517) | reaches:0.517, is:0.424 | ' reaches' (0.533) | reaches:0.533, is:0.174 | ✅ Close match on top-1 |
| 24 | ' full' (0.489) | full:0.489, exactly:0.439 | ' full' (0.317) | full:0.317, exactly:0.245, filled:0.065 | ✅ Roughly proportional, both uncertain |
| 25 | '.\n\n' (0.510) | .\n\n:0.510, ,:0.469 | '.' (0.141) | .:0.141, and:0.117, or:0.091 | ⚠️ Top-1 nominally agrees (both punctuation), but **lens is near-uniform across 5+ tokens** |

#### Dirichlet, seed=123 (steps 28, 47, 53, 76, 89, 97)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 28 | '5' (0.669) | 5:0.669, 3:0.331 | '5' (0.54) | 5:0.54, 3:0.434, 4:0.021 | ✅ Agrees, lens slightly flatter |
| 47 | '3' (0.599) | 5:0.401, 3:0.599 | **'5' (0.554)** | 5:0.554, 3:0.418 | ❌ **Top-1 DISAGREES.** Mixture says '3', hidden state says '5' |
| 53 | '3' (0.731) | 5:0.269, 3:0.731 | **'5' (0.538)** | 5:0.538, 3:0.46 | ❌ **Top-1 DISAGREES.** Same flip as step 47, stronger mixture confidence but lens still prefers '5' |
| 76 | '1' (0.766) | 1:0.766, 5:0.143, 3:0.066 | '1' (0.296) | 1:0.296, 3:0.259, 4:0.196, 0:0.137 | ⚠️ Top-1 agrees but **lens is near-flat (0.296)**, effectively 4-way uncertain |
| 89 | '5' (0.62) | 3:0.38, 5:0.62 | **'3' (0.796)** | 3:0.796, 5:0.193 | ❌ **Top-1 DISAGREES.** Mixture says '5', hidden state says '3' with high confidence |
| 97 | '2' (0.707) | 4:0.279, 2:0.707 | **'1' (0.415)** | 1:0.415, 4:0.281, 3:0.183, 2:0.112 | ❌ **Top-1 DISAGREES.** Mixture's '2' ranks 4th in lens at only 0.112; lens introduces '1' as top |

#### Gumbel, seed=1 (step 93)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 93 | '5' (0.758) | 5:0.758, 4:0.198, 3:0.039 | **'4' (0.512)** | 4:0.512, 5:0.195, 3:0.161 | ❌ **Top-1 DISAGREES.** Mixture's '5' falls to 0.195 in lens; '4' dominates internally |

#### Gumbel, seed=2 (step 80)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 80 | '4' (0.629) | 4:0.629, 2:0.365 | **'2' (0.623)** | 2:0.623, 4:0.212, 1:0.114 | ❌ **Top-1 DISAGREES.** Complete flip — lens strongly prefers '2' |

#### Gumbel, seed=42 (step 55)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 55 | '3' (0.726) | 3:0.726, 4:0.274 | '3' (0.875) | 3:0.875, 4:0.12 | ✅ Agrees, **lens sharpens strongly** |

#### Gumbel, seed=123 (step 62)

| Step | Mixture top-1 | Mixture dist | Lens top-1 | Lens dist (top-3) | Agreement? |
|------|--------------|-------------|-----------|-------------------|------------|
| 62 | '1' (0.676) | 1:0.676, 3:0.206, 2:0.088 | '1' (0.534) | 1:0.534, 3:0.254, 4:0.114, 2:0.035 | ✅ Agrees, lens slightly flatter |

---

### Cross-condition synthesis

**Scorecard:** 7 outright top-1 disagreements, 4 "agrees but lens is dramatically different in
shape" (⚠️), 7 clean agreements. This is across all 18 flagged steps.

**Three patterns, not two:**

1. **Lens sharpens** (model commits *harder* than the mixture). Seen at: Dir seed=1/step 55
   (0.583→0.769), Gumbel seed=42/step 55 (0.726→0.875). In these cases the mixture's "uncertainty"
   is decorative — the model's residual stream has already committed.

2. **Lens flattens** (model is *more* uncertain than the mixture). Seen at: Dir seed=1/step 69
   (top-1 drops 0.733→0.397), Dir seed=1/step 88 (0.762→0.351, plus introduces '1' at 0.301 — a
   candidate not in the mixture at all), Dir seed=123/step 76 (0.766→0.296, near-uniform over 4
   digits), Dir seed=2/step 22 (mixture nearly deterministic at 0.998, lens only 0.539). These
   suggest the hidden state is genuinely internally confused — but in a way that *doesn't match* the
   mixture's distribution. The model's uncertainty is real but structurally different from what the
   input blend encodes.

3. **Lens disagrees on top-1** (model has resolved to a *different* token entirely). The strongest
   cases:
   - Dir seed=123/step 89: mixture '5' (0.62) → lens '3' (0.796). High-confidence flip.
   - Dir seed=123/step 97: mixture '2' (0.707) → lens '1' (0.415), '2' only 0.112. Near-complete
     erasure of mixture's top pick.
   - Gumbel seed=2/step 80: mixture '4' (0.629) → lens '2' (0.623). Symmetric flip.
   - Dir seed=1/step 93: mixture '4' (0.613) → lens '3' (0.767). Lens more confident in the
     alternate.
   - Dir seed=123/steps 47+53: mixture says '3', lens says '5', consistently — the model's layers
     are systematically redirecting away from the Dirichlet-perturbed weight ordering.

**Key finding — Dirichlet shows more lens-mixture divergence than Gumbel.** Of the 7 top-1
disagreements, 5 are from Dirichlet conditions and 2 from Gumbel. This is consistent with the known
γ tradeoff: Dirichlet's higher per-sample variance pushes the mixture weights further from the
model's natural distribution, and the model's layers "correct" back toward their own preferences
rather than maintaining the blend. Gumbel, being a smoother perturbation of the existing log-probs,
stays closer to what the model would have done anyway, producing fewer outright disagreements (though
still not zero — seed=1/step 93 and seed=2/step 80 are genuine Gumbel-side flips).

---

### Interpretation: what this means for the superposition hypothesis

**This is direct evidence against genuine superposition in noise-injected soft thinking.** The
argument:

1. The superposition claim requires that when a blended embedding is fed in, the model's internal
   layers *maintain* the multi-candidate content through the residual stream — i.e., the hidden
   state at later layers should still encode both candidates in a way that influences downstream
   generation.

2. In 7/18 cases, the lens disagrees on top-1 entirely. The model's residual stream has resolved the
   blended input in a direction that doesn't reflect the intended blend. The mixture's
   "superposition" is surface-level — the model's layers collapse or redirect it before it reaches
   the output.

3. In the cases where lens agrees on top-1 but flattens dramatically (Pattern 2), the hidden state
   is genuinely uncertain — but the *shape* of its uncertainty doesn't match the mixture's
   distribution. The model is confused on its own terms, not faithfully carrying the blend's content.

4. In the cases where lens sharpens (Pattern 1), the model has committed *harder* than the mixture
   suggested. The mixture's lower weight on the top-1 candidate was noise-induced decoration — the
   model internally ignored it.

**Combined:** noise-injected soft thinking produces mixture weights that diverge from the model's
internal hidden-state distribution at decision-relevant (digit) tokens. The "superposition" visible
in the mixture is not faithfully maintained through the model's layers. This is directly novel
relative to both Wu et al. (who only tested performance, not lens) and Rizvi-Martel et al. (who only
tested vanilla, not noised).

**Caveat (important, must be in the paper):** this is lens evidence, not causal evidence. The lens
shows the hidden state doesn't *look like* the mixture — but it doesn't yet prove the hidden state
is what *drives* the final answer. Causal patching (Next Step 6) is the remaining piece: overwrite
the hidden state at a disagreement step and check whether the hard-phase output follows the lens
prediction, the mixture prediction, or neither.

---

### Additional observations from the soft-phase traces

- **Default condition (gumbel, seed=42) soft-phase trace** is byte-identical to the Exp 7/7b traces
  for the same condition, confirming reproducibility of the fixed harness across sessions.
- **EOS behavior reproduced exactly** across all re-run conditions (Dir seed=2 at step 62, Gumbel
  seeds 2 and 123 at step 86) — no new EOS anomalies.
- The Dirichlet seed=2 condition produced the most qualitatively interesting soft-phase *content*:
  "I will do the second jug. I start pour the water from the smaller jug into the larger jug until
  until it reaches full." — garbled but with a clear strategy commitment (pour small→large). EOS
  fires early at step 62 with high-ish confidence (0.847), producing a short but self-contained
  (wrong) answer. The sustained low-confidence cluster at steps 20–25 in the lens comparison above
  is from this run.
- The Gumbel seed=2 condition (EOS at step 86) drifts into "As a result, the 3-liter jug will still
  have 4 liters of water remaining" — the same capacity violation seen in Exp 7b. The lens
  comparison at step 80 (the erroneous '4' token) shows the hidden state actually preferred '2',
  which would have been the *correct* remaining amount after pouring 3 liters from a 5-liter jug.
  This is a tantalizing hint: the model's hidden state may have had the right answer internally, but
  the mixture weight (noise-perturbed to favor '4') overrode it. **Priority candidate for causal
  patching.**

---

## Exp 10 — Causal patching on Exp 9's lens-disagree steps (2026-09-11)

**What this is:** Next Step 6 — causal patching. For each of the 4 strongest lens-mixture
disagreement cases from Exp 9, re-run the model three ways:
1. **Unpatched** — normal noisy soft thinking (baseline)
2. **Patched with lens token** — at the target step, replace the blend with the discrete embedding of
   the token the model's hidden state preferred (per the logit lens)
3. **Patched with mixture token** — at the target step, replace the blend with the discrete embedding
   of the token noise pushed to top (control: is the blend equivalent to its discrete top-1?)

Then compare all three hard-phase outputs. If the lens patch changes the output, the step is causally
load-bearing. If the mixture patch matches unpatched, the blend is functionally equivalent to its
discrete top-1 (the secondary blend weights contribute nothing).

**Implementation:** `generate_soft_with_patch()` — identical to `generate_soft_then_hard()`, but at
one specified step, swaps `cur_embeds` from the blended vector to a single token's embedding. Same
seed → same noise draws at every step before the patch, so divergence is cleanly attributable to the
patched embedding alone.

---

### Target 1: Gumbel seed=2, step 80

**Context:** mixture='4' (weight 0.629), lens='2' (0.623). '2' would be the *correct* remaining
amount after pouring 3 liters from a 5-liter jug. This condition hits EOS at step 86 (per Exp 7b),
so the unpatched run has 14 void soft steps before the hard phase begins.

| Run | Output |
|-----|--------|
| **Unpatched** | `'. You are a helpful assistant. You should provide assistance to the best of able based on the information available.'` |
| **Lens '2'** | `' pour the remaining 2 liters of water from the 3-liter jug into the 5-liter jug, which will fill the 5-liter jug to 7 liters. Now, the 5-liter jug has exactly 4 liters of water, and the 3-liter jug is empty. This is the'` |
| **Mixture '4'** | `' the first step, you should fill the 5-liter jug.'` |

**Verdict:**
- ✅ Lens patch **CHANGED** the output -> step 80 is causally load-bearing
- ✅ Mixture patch also changed output -> discrete '4' != the blend (blend's secondary weights had some perturbation)
- ✅ Lens != mixture -> token choice matters

**Observations:**
- The unpatched output is **system-prompt leakage** caused by post-EOS drift after step 86.
- Patching with lens token '2' at step 80 steered the model out of system-prompt drift back into problem-solving arithmetic reasoning text.
- Patching with mixture token '4' produces distinct text ("the first step, you should fill the 5-liter jug.").
- All three outcomes differ; the position is causally active and token-sensitive.

---

### Target 2: Dirichlet seed=123, step 89

**Context:** mixture='5' (weight 0.620), lens='3' (0.796). High-confidence lens flip.

| Run | Output |
|-----|--------|
| **Unpatched** | `' the 5-liter jug to fill the 3-liter jug.\n   - Now, you have 1 liter in the 3-liter jug and 4 liters in the 5-liter jug.\n\n2. **If 4 liters is in the 3-liter jug:**\n   - After removing 4'` |
| **Lens '3'** | `' fill the 5-liter jug completely.\n   - Therefore, the first thing you do is to fill the 5-liter jug.\n\n2. **If 4 liters is in the 3-liter jug:**\n   - After removing 4 liters from the 3-liter jug, you will have 1'` |
| **Mixture '5'** | **(identical to unpatched)** |

**Verdict:**
- ✅ Lens patch **CHANGED** the output -> step 89 is causally load-bearing
- ❗ Mixture patch = unpatched -> **the blend behaves byte-identically to discrete '5' alone**
- ✅ Lens != mixture -> token choice matters

**Key finding:** Despite the blend holding 38% weight on '3' and 62% on '5', feeding the blend produces the exact same downstream output as feeding 100% discrete '5'. The 38% weight on '3' exerted zero causal influence on the generation.

---

### Target 3: Dirichlet seed=1, step 93

**Context:** mixture='4' (weight 0.613), lens='3' (0.767). In Exp 8, this run hallucinated a nonexistent "4-liter jug".

| Run | Output |
|-----|--------|
| **Unpatched** | `'Pour the remaining 1 liter from the 3-liter jug into the 4-liter jug.**\n   - This operation makes the 3-liter jug 2 liters shorter, but now you have a 4-liter jug and a 2-liter jug.\n   - At this point, you have 4'` |
| **Lens '3'** | `'Pour the remaining 1 liter from the 3-liter jug into the 5-liter jug.**\n   - This operation makes the 3-liter jug 2 liters shorter, but now you have a 4-liter jug and a 5-liter jug.\n   - At this point, you have 4'` |
| **Mixture '4'** | `'Pour the remaining 1 liter from the 3-liter jug into the 4-liter jug.**\n   - This operation makes the 4-liter jug 1 liter shorter, but now you have a 3-liter jug and a 4-liter jug.\n   - At this point, you have 4'` |

**Verdict:**
- ✅ Lens patch **CHANGED** the output -> step 93 is causally load-bearing
- ✅ Mixture patch also changed output (diverges slightly downstream after initial phrase)
- ✅ Lens != mixture -> token choice matters

**Observations:**
- Both unpatched and mixture patch '4' preserve the hallucinated "4-liter jug" entity.
- The lens patch ('3') alters the entity reference to "5-liter jug" (a valid problem entity).
- The blend acted primarily like discrete '4', failing to integrate the 38.7% '3' component to correct the hallucination.

---

### Target 4: Gumbel seed=1, step 93

**Context:** mixture='5' (weight 0.758), lens='4' (0.512). Gumbel-side flip.

| Run | Output |
|-----|--------|
| **Unpatched** | `", or we can empty the 5-liter jug and pour water into the 3-liter jug until it is full. Let's consider both scenarios.\n\n**Scenario 1: Empty the 5-liter jug and pour water into the 5-liter jug until it is full.**\n- After pouring, the"` |
| **Lens '4'** | `" fill the 5-liter jug and pour water into the 4-liter jug until it is full. Let's consider both scenarios.\n\n**Scenario 1: Empty 5-liter jug and pour water into 4-liter jug until it is full.**\n- After pouring, the 5-liter jug will have"` |
| **Mixture '5'** | **(identical to unpatched)** |

**Verdict:**
- ✅ Lens patch **CHANGED** the output -> step 93 is causally load-bearing
- ❗ Mixture patch = unpatched -> **the blend behaves byte-identically to discrete '5' alone**
- ✅ Lens != mixture -> token choice matters

**Same pattern as Target 2:** The blend (75.8% '5', 19.8% '4', 3.9% '3') produces output byte-identical to pure discrete '5'. The secondary candidate weights are causally inert.

---

### Cross-target synthesis & conclusions

1. **Blends act like discrete top-1:** In 2 of 4 cases (Target 2 and Target 4), feeding the blend produced byte-for-byte identical output to feeding the discrete top-1 token alone. In the other 2 cases, the discrete top-1 shared the core semantic trajectory (e.g. hallucinating the same 4-liter jug). Secondary weights in the mixture provide no parallel reasoning benefit.
2. **Steps are genuinely causally load-bearing:** In all 4 targets (4/4), intervening with the lens's top-1 candidate altered downstream text.
3. **Evidence against superposition:** Taken together with Exp 9 (where 7/18 steps showed lens-mixture top-1 disagreement), Exp 10 confirms that Soft Thinking with noise does not maintain active superposition over multiple reasoning trajectories. Instead, the model internally resolves to single discrete tokens, and secondary weights in the embedding mixture are largely ignored downstream.

---

## What's actually missing right now (updated post-Exp 10)

1. ~~Add missing Gumbel entries to `PRIORITY_STEPS`~~ — **done.**
2. ~~Run `check_lens_checkpoint`~~ — **done (PRE-norm, apply_norm=True).**
3. ~~Run `report_priority_steps` and log as Exp 9~~ — **done.**
4. ~~Build causal patching and run on priority targets (Exp 10)~~ — **done.**
5. **Comparison write-up vs. Li et al. / Zhang et al.** (Next Step 7) — ready to draft.
6. **Vanilla lens pass** (optional control check for three-way comparison).

---

## Recommended next steps, in order (updated post-Exp 10)

1. **Write the comparison section vs. Coconut / CODI literature (Li et al. arXiv:2602.08783, Zhang et al. arXiv:2512.21711):**
   - Coconut/CODI use latent recurrent hidden states and showed minimal causal steering.
   - Soft Thinking uses embedding mixtures; our Exp 9 & 10 show that even with noise injection, embedding mixtures collapse to discrete top-1 behavior and do not sustain superposition.
2. **Synthesize final experimental findings into paper draft structure:**
   - Section 1: Greedy Pitfall & noise injection motivation.
   - Section 2: Logit lens diagnostics (Exp 9) showing mixture vs. internal representation divergence.
   - Section 3: Causal patching (Exp 10) confirming blends act like discrete argmax.

---

## Exp 11 — Depth Probing (Part A) & Abstract Multi-Concept Blend (Part B) — Raw Data (2026-09-12)

### Part A: Layer-by-Layer Depth Probing (Layers 6, 12, 18, 24)

#### 1. (gumbel, seed=2, step 80)
- **Input Mixture (top > 2%):** `[('2', 0.365), ('4', 0.629)]`
- **Layer-by-layer logit lens readouts:**
  - **Layer 6:** `'出道'` (0.062), `'announce'` (0.028), `'xfff'` (0.016)
  - **Layer 12:** `':@"%@",'` (0.043), `'ifique'` (0.030), `'undance'` (0.016)
  - **Layer 18:** `'extrême'` (0.012), `'uary'` (0.008), `'举报'` (0.007)
  - **Layer 24:** `'2'` (0.623), `'4'` (0.212), `'1'` (0.114)

#### 2. (dirichlet, seed=123, step 89)
- **Input Mixture (top > 2%):** `[('3', 0.380), ('5', 0.620)]`
- **Layer-by-layer logit lens readouts:**
  - **Layer 6:** `' türlü'` (0.037), `'шедш'` (0.026), `'oretical'` (0.022)
  - **Layer 12:** `'玩家来说'` (0.069), `'унк'` (0.015), `'游戏里的'` (0.015)
  - **Layer 18:** `'看了一眼'` (0.045), `'>Date'` (0.027), `'addy'` (0.022)
  - **Layer 24:** `'3'` (0.796), `'5'` (0.193), `'4'` (0.005)

#### 3. (gumbel, seed=1, step 93)
- **Input Mixture (top > 2%):** `[('4', 0.198), ('3', 0.039), ('5', 0.758)]`
- **Layer-by-layer logit lens readouts:**
  - **Layer 6:** `'uvwxyz'` (0.078), `'вшие'` (0.033), `'碼'` (0.027)
  - **Layer 12:** `'玩家来说'` (0.106), `'性价'` (0.026), `'프로그'` (0.024)
  - **Layer 18:** `'叹了口气'` (0.043), `' StringTokenizer'` (0.022), `'addy'` (0.011)
  - **Layer 24:** `'4'` (0.512), `'5'` (0.195), `'3'` (0.161)

---

### Part B: Abstract Multi-Concept Blend (Prompt: "Describe an entity that is simultaneously a delicate blooming flower and a lethal razor-sharp blade.")

#### 1. (gumbel, seed=42)
- **Hard Completion:** `"abilities. If you have any questions or need information on a particular topic, I'd be happy to help."`
- **Step 9 (Entropy = 1.322):**
  - Mixture: `[(' generate', 0.081), (' create', 0.076), (' accurately', 0.272), (' produce', 0.522)]`
  - Layer 6: `'@hotmail'` (0.371), `',{"'` (0.115), `'.dds'` (0.025)
  - Layer 12: `'iếu'` (0.106), `'aket'` (0.055), `'antha'` (0.053)
  - Layer 18: `'untu'` (0.097), `'ập'` (0.022), `'PlainText'` (0.020)
  - Layer 24: `' generate'` (0.221), `' create'` (0.208), `' provide'` (0.164)
- **Step 24 (Entropy = 1.212):**
  - Mixture: `[(' model', 0.111), (' assistant', 0.070), (' and', 0.091), (' designed', 0.656), (' developed', 0.038)]`
  - Layer 6: `'\n \n'` (0.064), `" '(("` (0.037), `'esian'` (0.028)
  - Layer 12: `'强国'` (0.155), `'anggan'` (0.040), `'uang'` (0.036)
  - Layer 18: `'e'` (0.644), `' stemming'` (0.014), `'笑了笑'` (0.013)
  - Layer 24: `' language'` (0.263), `' and'` (0.226), `' model'` (0.147)
- **Step 30 (Entropy = 1.270):**
  - Mixture: `[(' to', 0.422), ('.', 0.136), (' in', 0.379), (' as', 0.031)]`
  - Layer 6: `' '` (0.163), `'人民服务'` (0.068), `'.Help'` (0.016)
  - Layer 12: `' '` (0.201), `'有所帮助'` (0.117), `' \n'` (0.057)
  - Layer 18: `'.'` (0.535), `','` (0.076), `'过'` (0.015)
  - Layer 24: `' to'` (0.325), `' based'` (0.158), `' in'` (0.110)

#### 2. (dirichlet, seed=1)
- **Hard Completion:** `'human psyche. \n\nIn literature, such entities are often used to explore themes of love, loss, and the fragility of life. They might be a flower that blooms with the'`
- **Step 14 (Entropy = 1.024):**
  - Mixture: `[(' that', 0.602), (',', 0.074), (' existing', 0.279)]`
  - Layer 6: `'ожет'` (0.166), `'trfs'` (0.062), `'化石'` (0.049)
  - Layer 12: `'trfs'` (0.349), `'化石'` (0.105), `'ожет'` (0.028)
  - Layer 18: `'具体的'` (0.018), `'trfs'` (0.014), `'.'` (0.014)
  - Layer 24: `' that'` (0.320), `' in'` (0.219), `' existing'` (0.054)
- **Step 27 (Entropy = 1.032):**
  - Mixture: `[(' concept', 0.180), (' themes', 0.635), (' whims', 0.154)]`
  - Layer 6: `'tas'` (0.149), `'ses'` (0.080), `'enkins'` (0.021)
  - Layer 12: `'ses'` (0.108), `'tas'` (0.080), `'捕捉'` (0.024)
  - Layer 18: `'ses'` (0.174), `' worlds'` (0.049), `' realms'` (0.009)
  - Layer 24: `' "'` (0.073), `' '` (0.055), `' two'` (0.042)
- **Step 29 (Entropy = 1.287):**
  - Mixture: `[(' beauty', 0.322), (' love', 0.375), (' poetry', 0.203), (' nature', 0.100)]`
  - Layer 6: `'魇'` (0.038), `'苓'` (0.038), `'碼'` (0.037)
  - Layer 12: `'思'` (0.156), `'各异'` (0.048), `' both'` (0.026)
  - Layer 18: `' both'` (0.672), `'both'` (0.058), `'前者'` (0.013)
  - Layer 24: `' the'` (0.345), `' beauty'` (0.092), `' love'` (0.068)

---

### Analysis & Interpretation of Exp 11

#### 1. Exp 11A (Depth Probing): The Logit Lens is Blind at Mid-Layers in 0.5B Models

- **Empirical Observation:** Across all three test cases, Layers 6, 12, and 18 decoded to low-probability (1–10%) noise and out-of-context multilingual fragments (Chinese characters, Cyrillic, Korean subwords, code fragments like `':@"%@"'` or `'StringTokenizer'`). The representations only snap into clean, high-confidence, semantically coherent vocabulary distributions at **Layer 24**.
- **Mechanistic Explanation:** The logit lens relies on the assumption that intermediate hidden states already reside in or near the unembedding manifold. In small models (0.5B, 24 layers), intermediate representations are non-linearly transformed and not aligned with `lm_head` until the very final layer.
- **Scientific Verdict on Mid-Layer Superposition:** **INCONCLUSIVE due to tool limitation.** We cannot confirm or rule out superposition at Layer 12 because the logit lens cannot resolve meaningful vocabulary tokens at intermediate depths for this parameter scale. Confirming mid-layer latent structure at 0.5B would require trained linear probes or contrastive probes rather than zero-shot unembedding projection.

#### 2. Exp 11B (Abstract Multi-Concept Blend): Alignment Refusal & Domain Clustering

- **Gumbel seed=42 (Refusal Triggered):** The prompt ("delicate blooming flower and a lethal razor-sharp blade") triggered the model's safety/alignment guardrails regarding weaponized/lethal descriptions. The model generated a boilerplate refusal (`"I apologize, but I'm not able to produce or describe any specific entity..."`). As a result, the high-entropy steps (9, 24, 30) reflect hedging and boilerplate syntax (`'produce'` vs `'create'` vs `'generate'`) rather than competition between floral and weapon concepts.
- **Dirichlet seed=1 (Meta-Commentary):** The model avoided direct refusal by shifting into abstract literary critique (`"fictional creation inspired by the themes of love, danger..."`).
  - At Step 29 (highest entropy, 1.287), the mixture competed across positive aesthetic terms (`'beauty'`, `'love'`, `'poetry'`, `'nature'`), clustering within a single semantic domain rather than balancing flower vs. blade.
  - **Notable Anomaly:** At Step 29, Layer 18 decoded `' both'` at **0.672** (with `'both'` at 0.058, totaling ~73% probability mass), reflecting an internal commitment to acknowledging duality at that intermediate layer, which subsequently collapsed to `' the'` at Layer 24. However, this represents syntactic framing rather than parallel token superposition across opposing domains.

---

## Exp 12 — Closing Two Review Gaps: Vanilla Baseline + Unbiased Sampling (12A), Control-Token Patching (12B) (2026-09-14)

**What this is:** a review pass on Exp 9 and Exp 10 flagged two gaps before either could be called
solid: (1) Exp 9 only ever measured lens-vs-mixture disagreement *under noise* — there was no
same-pipeline run of the noise-free (vanilla) condition to check whether the ~39% disagreement rate
was actually elevated by noise, or just a property of high-entropy steps in general; (2) Exp 10 only
compared "patch with lens token" vs. "patch with mixture token" vs. unpatched — with no control arm,
"lens patch changed the output" can't be told apart from "any different token here changes the
output," which is true of basically any generation step. Exp 12 adds both missing arms to the
existing harness (`run_exp12a_vanilla_baseline_and_unbiased_sampling`,
`run_causal_patches_with_control`), with no changes to Exp 9/10/11's own code or numbers.

---

### Exp 12A — Results: lens-vs-mixture disagreement rate, by condition and by selection method

| Condition | Selection | Disagree | Rate |
|---|---|---|---|
| Vanilla (noise=None) | own top-8 highest-entropy steps | 3/8 | 38% |
| Vanilla (noise=None) | random 8 steps | 2/8 | 25% |
| Gumbel, seed=42 | random 8 steps | 1/8 | 12% |
| Dirichlet, seed=1 | random 8 steps | 2/8 | 25% |
| *(reference, Exp 9)* Gumbel+Dirichlet, all seeds | hand-flagged high-entropy steps | 7/18 | 39% |

**Headline reading, at face value:** vanilla's own top-entropy steps (38%) look statistically
indistinguishable from Exp 9's flagged noise-condition rate (39%), and every random-sample condition
(12–25%) comes in well below the entropy-flagged rate. Read naively, this says lens/mixture
disagreement tracks *entropy*, not *noise* — the same disagreement rate shows up with no noise
injection at all, once you look at the noise-free run's own most uncertain moments; and cherry-picking
high-entropy steps roughly doubles the apparent disagreement rate relative to typical steps,
regardless of condition.

**This headline reading does not survive a sanity check, and should not yet be logged as a finding.**
See the flagged anomaly below — the vanilla number in particular is very likely a measurement
artifact, not a real result. Treat the table above as raw output pending Exp 12C, not as a validated
comparison.

---

### ⚠️ Flagged anomaly: the vanilla disagreement rate should be mathematically impossible

Worked through by hand before accepting the 38%/25% vanilla numbers:

1. Under `noise=None`, `soft_thinking_step` does top-k → top-p → renormalize, and **nothing else**.
   `torch.topk` returns candidates already sorted by probability; the top-p keep-mask only zeroes out
   a *trailing* suffix (it can never drop position 0, since the cumulative-sum-before-position-0 is
   always `0 ≤ top_p`); renormalizing divides every kept weight by the same constant. None of these
   three operations can ever change *which* candidate has the highest weight.
2. Therefore, for every vanilla step, **mixture top-1 must always equal `raw_argmax_id`** (the
   plain, pre-filter argmax of that step's own `logits`) — every single time, by construction, with
   zero exceptions. A disagreement between "mixture top-1" and anything claiming to reconstruct that
   same step's true distribution should be impossible in the vanilla condition.
3. Exp 9's `check_lens_checkpoint` already confirmed that `hidden_states[-1]` is pre-norm and that
   `lm_head(norm(hidden_states[-1]))` reproduces the model's real logits — meaning "lens top-1" is
   *supposed* to be reading out that exact same distribution, just recomputed by hand outside the
   forward call instead of read directly off `out.logits`.
4. Putting 1–3 together: a vanilla lens-vs-mixture disagreement should read **0/8, always**, not
   3/8 or 2/8. Getting a nonzero rate means the lens reconstruction and the live `logits` are
   diverging from each other on some steps — which is a property of the *measurement pipeline*, not
   a discovery about vanilla's internal representations.
5. The most likely explanation given the pattern in the data: `select_top_entropy_steps` deliberately
   selects steps where the distribution is *most spread out* — i.e., exactly the steps where the
   top-1 and top-2 candidates are closest together. A near-tie is precisely the regime where a tiny
   floating-point difference between two computation paths that are mathematically identical on
   paper (live forward-pass logits vs. hidden state recomputed through `norm` + `lm_head` after being
   detached and passed through a second, separate call) is most likely to flip which candidate comes
   out on top. That would explain why the *entropy-selected* steps show more disagreement (38%) than
   the *random* steps (25%) even within the same noise-free run — it's not that noise-free high-
   entropy steps have secretly-divergent internal representations, it's that near-ties are fragile to
   measure twice.
6. This matters beyond vanilla: Exp 9's own 18 priority steps were *also* chosen because they already
   looked like low-confidence dips — i.e., near-ties. If the same fragility applies there, some
   (not necessarily all) of Exp 9's "7/18 disagree" cases could be inflated by the same effect rather
   than reflecting real noise-driven divergence. This doesn't erase Exp 9 — several of its
   disagreements are large, clean flips (e.g. Dirichlet seed=123/step 89: mixture 0.62 vs. lens
   0.796, not a near-tie) that a floating-point wobble can't explain — but it means the 39% headline
   number needs the same diagnostic run on the noised conditions before being fully trusted either.

**Next action (Exp 12C, not yet run):** a small diagnostic — for the flagged vanilla steps, print
`raw_argmax_id` (already stored per-step) next to mixture-top-1 and lens-top-1. If `raw_argmax_id`
always equals mixture-top-1 (as it mathematically must) while lens-top-1 sometimes differs from both,
that confirms this is a lens-reconstruction precision artifact at near-ties, not a real vanilla
finding — and flags that the same check should be run against the *large, clean* Exp 9 disagreements
specifically (which are not near-ties and are far less likely to be explained this way) to separate
the genuine noise-driven flips from any near-tie noise in the measurement itself. Code for this is in
the harness update (`diagnose_vanilla_lens_anomaly`) — cheap to run, single vanilla pass only.

---

### Exp 12B — Results: control-token causal patch arm

Added a 4th patch arm (token `'1'`, chosen because it carries near-zero weight under both the mixture
and the lens at all 4 Exp 10 targets) to Exp 10's original 4 causal-patch targets.

| Target | Lens patch vs. unpatched | Mixture patch vs. unpatched | Control patch vs. unpatched | Lens vs. control |
|---|---|---|---|---|
| Gumbel s=2, step 80 | changed | changed | changed | different text |
| Dirichlet s=123, step 89 | changed | **unchanged (byte-identical)** | changed | different text |
| Dirichlet s=1, step 93 | changed | changed | changed | different text |
| Gumbel s=1, step 93 | changed | **unchanged (byte-identical)** | changed | different text |

**The control patch changed the output in 4/4 targets** — every bit as often as the lens patch did.

**Interpretation:**

1. **The "blend = discrete top-1" finding replicates exactly and is unaffected by this control.**
   Targets 2 and 4 reproduce Exp 10's byte-identical mixture-patch-vs-unpatched result precisely.
   This comparison was never confounded by the "any token change matters" issue (it's testing
   "does the blend behave like its own dominant component," not "does *some* token change the
   output"), so it stays the most solid finding to come out of the patching work.
2. **"Lens patch changed the output" is not, by itself, special evidence anymore.** Exp 10 read this
   as "the step is causally load-bearing," with the implication that the lens had located the
   internally-preferred content. Since an arbitrary control token — deliberately chosen to be
   favored by neither the mixture nor the lens — *also* changes the output in all 4/4 cases, the
   fair conclusion is narrower: these late-stage digit-decision positions are generically sensitive
   to whatever token is fed in, which is unsurprising for an autoregressive model and isn't unique to
   what the lens surfaced. Exp 10's claim "steps are genuinely causally load-bearing" should be read
   as "these positions are input-sensitive," not "the lens found the true internal answer."
3. **Weak, anecdotal, non-statistical note (n=1):** at Target 2 (Dirichlet s=123/step 89), the
   lens-patched continuation stays procedurally coherent ("fill the 5-liter jug completely... the
   first thing you do is fill the 5-liter jug"), while the control-patched continuation drifts into a
   nonexistent extra jug ("you can only use the 3-liter jug to fill the 1-liter jug completely").
   At the other 3 targets, lens- and control-patched continuations are comparably confused (both
   introduce their own hallucinated jug sizes; neither reaches a correct solution). This is far too
   thin a sample to claim the lens finds "better" content — flagged only as something worth checking
   again if more patch targets are ever added.
4. **Net effect on Exp 10's conclusions:** point 1 (blend ≈ discrete top-1) is strengthened by
   replication. Point 2 ("steps are causally load-bearing," 4/4) needs the caveat above — it's true
   in the trivial "the position matters" sense, not yet shown to be specific to what the lens
   predicted.

---

## Updated Comprehensive Project Synthesis (Experiments 1–12)

With Exp 12 folded in, two of the project's four core findings (Exp 9, Exp 10) need a caveat added,
one is strengthened, and one open methodological question (12C) is now blocking full confidence in
the headline disagreement numbers:

1. **Greedy Pitfall is real, and noise injects entropy (Exp 5 vs 7b/8):** unchanged. Standard soft
   thinking collapses immediately into greedy single-token traps; Gumbel and Dirichlet noise keep
   entropy non-zero and break greedy collapse.
2. **Noise provides rollout diversity, not per-step superposition (Exp 7b, 8, 9):** unchanged.
   Different seeds explore different trajectories; individual steps do not maintain stable, balanced
   multi-path states.
3. **Internal representations sometimes diverge from mixture weights (Exp 9) — magnitude now
   uncertain pending Exp 12C.** The large, clean flips (e.g. Dirichlet seed=123/step 89, mixture 0.62
   vs. lens 0.80) are very unlikely to be measurement noise and stand as real evidence. But Exp 12A's
   vanilla control run surfaced a mathematical impossibility — vanilla should show 0% lens/mixture
   disagreement by construction, and instead showed 25–38% — that points to some fraction of *all*
   near-tie disagreements (vanilla's and possibly some of Exp 9's) being a measurement artifact at
   close calls rather than a real finding. Exp 9's 7/18 headline number should be treated as an
   upper bound until Exp 12C separates the two.
4. **Embedding blends act causally like discrete top-1 tokens (Exp 10, replicated in Exp 12B):**
   strengthened. 2/4 targets produce byte-identical output whether fed the full blend or just its
   discrete top-1 component; this held again with a control arm added, and was never confounded by
   the generic-sensitivity issue below. Secondary blend weights are causally inert.
5. **"Steps are causally load-bearing" (Exp 10) needs a narrower reading (Exp 12B).** A control token
   that neither the mixture nor the lens favored changed the output in 4/4 targets — just as often as
   the lens token did. These positions are sensitive to input identity in general; Exp 10 correctly
   showed the position matters, but did not show the lens specifically located the model's "true"
   preferred content over an arbitrary alternative.
6. **Depth probing is constrained by model scale (Exp 11):** unchanged. Intermediate layers (6, 12,
   18) cannot be decoded via logit lens on a 0.5B model; only the final layer produces coherent
   readouts. This is a real methodological ceiling, not a finding about superposition either way.

**Core Paper Conclusion (updated):** the strongest, least-confounded evidence in the project is still
Exp 10/12B's "blend behaves like discrete top-1" result — that one survived a control-arm check
cleanly. The lens-divergence claims (Exp 9) and the "causally load-bearing" framing (Exp 10) are
real, but weaker than first written up: part of the disagreement rate looks like a measurement
artifact at near-ties (pending Exp 12C), and the causal-patching claim needs to drop the implication
that the lens found something a generic token substitution couldn't also find. **Net direction of the
evidence is unchanged — still no genuine superposition — but the case should be written up as
"blends collapse to their dominant discrete component" (solid) plus "some signs the internal state
sometimes disagrees with the surface mixture" (real but noisier than first measured), rather than
leaning on the causal-patching framing as independently strong support.

---

## Exp 13 — Qwen2.5-1.5B final-layer lens replication (2026-09-19)

**Purpose.** Re-run the final-layer mixture-vs-lens comparison on the larger
`Qwen/Qwen2.5-1.5B-Instruct` model, while fixing the measurement failure exposed by
Exp 12C. This is a replication/pilot, not a continuation of the 0.5B lens numbers:
the 0.5B final-layer lens results must be re-validated with live-logit reconstruction before
they are used as quantitative evidence.

**Harness / conditions.** New standalone `neuralese_harness_v2.py`; same water-jug prompt,
`top_k=15`, `top_p=0.95`, Gumbel temperature `τ=0.5`, `soft_steps=100`, and
`hard_steps=60`. Model ran CPU-only in float32. One deterministic vanilla trace and Gumbel
seeds `{0, 1, 2, 42, 123}` were collected across separate launches. At each condition, the
same fixed random 12 non-void soft-step indices were inspected:
`[6, 19, 20, 39, 41, 45, 55, 77, 79, 80, 84, 94]`.

### Measurement gate — passed

Before interpreting any lens result, V2 directly compared the `lm_head` logits reconstructed
from the captured final hidden state against the model's live `out.logits` at 8 fixed steps.

| Convention applied to `hidden_states[-1]` | Maximum absolute logit error | Allclose | Top-1 matches live logits |
|---|---:|---:|---:|
| Already normalized (no extra norm) | `1.38e-05` | 8/8 | 8/8 |
| Apply `model.model.norm` again | `10.5` | 0/8 | 7/8 |

The correct convention for this model / Transformers environment is therefore:
**`hidden_states[-1]` is already final-normalized; do not apply the final norm again.**
The `1.38e-05` maximum difference is ordinary separate-fp32-GEMM numerical variation; the
alternate path is orders of magnitude wrong. This removes the earlier vanilla anomaly:
on the fixed random sample, vanilla has **0/12** mixture-vs-final-lens top-1 disagreements,
as required when noise is absent.

### Results

| Condition | Final-lens / mixture disagreements | Notable mismatches |
|---|---:|---|
| Vanilla | 0/12 | None |
| Gumbel, seed 42 | 1/12 | step 84: mixture `'.\n'` (0.733) vs. lens `' completely'` (0.694) |
| Gumbel, seed 0 | 0/12 | None |
| Gumbel, seed 1 | 1/12 | step 55: mixture `'**'` (0.969) vs. lens `'**:'` (0.328) |
| Gumbel, seed 2 | 3/12 | step 20: `' we'` (0.286) vs. `' it'` (0.394); step 77: `' part'` (0.569) vs. `' step'` (0.827); step 80: `' transfer'` (0.582) vs. `' fill'` (0.354) |
| Gumbel, seed 123 | 1/12 | step 45: `' are'` (0.475) vs. `' have'` (0.826) |

**Aggregate:** 6/60 disagreements over the fixed random Gumbel samples, versus 0/12 in
vanilla. This is a descriptive pilot rate only: the sample is small, all conditions share the
same prompt, and most inspected locations are not decision points.

**Trace-level observations.** The great majority of Gumbel mixtures remain near one-hot. The
recorded disagreements are mostly grammatical, formatting, or procedural wording choices;
none is a clean fill-3-vs-fill-5 resolution. Seed 2 contains the closest action-word mismatch
(`'transfer'` vs. `'fill'`), but it is not itself a capacity choice and should not be treated as
evidence of parallel solution-path reasoning.

One uninspected candidate is worth preserving precisely: in **Gumbel seed 123, printed soft
step 10** (zero-indexed record 9), the input mixture was `'5'`: 0.669 and `'3'`: 0.331.
This is the first 1.5B trace point that directly names the two jug capacities. It was not part
of the fixed random lens sample, so no claim is made about its internal final-lens readout yet.

### Interpretation

Exp 13 fixes a critical measurement issue rather than providing a final superposition verdict.
With a validated final-layer readout, the no-noise control behaves exactly as expected and Gumbel
occasionally produces a real divergence between the input mixture and the model's final output
distribution. However, the current data show no sustained, balanced, decision-level
superposition. The safe statement is: **at 1.5B, Gumbel noise sometimes redirects the final
representation away from the mixture's top token, but the random-sample evidence so far is
predominantly syntactic/procedural rather than alternative solution-path content.**

### Runtime constraint

Each 100-soft-step + up-to-60-hard-step float32 CPU condition took roughly an hour in this
environment. The completed Gumbel sweep is therefore sufficient for screening; do not launch a
full Dirichlet sweep or activation-patching suite until a decision-relevant target is confirmed.

### Next test (pre-specified from this result)

Run a **short, target-only diagnostic** for vanilla and Gumbel seed 123: 12 soft steps,
no hard phase, then print the top-5 mixture and validated final-lens distributions at
zero-indexed step 9 (the printed step-10 `'5'`/`'3'` split). The first 12 soft steps are
unchanged by reducing the later budget, so this faithfully reproduces the candidate at a small
fraction of the full-run cost.

- If the final lens retains both `'5'` and `'3'` as meaningful candidates, repeat that exact
  target on at least one additional reasoning prompt before any causal claim.
- If it collapses sharply to one candidate or redirects to unrelated content, record this as a
  negative result for decision-level superposition and do not activate costly patching.
- Only if a replicated, decision-relevant lens divergence survives should genuine decoder-layer
  activation patching be run, with mixture, lens, and arbitrary-token control donors.

### Post-run correction — temporal alignment error in the Exp 13 comparison (2026-09-19)

The 12-step target probe reproduced the seed-123 raw numbers exactly:

| Source at zero-indexed step 9 (printed step 10) | `'5'` | `'3'` |
|---|---:|---:|
| Post-Gumbel input mixture | 0.669 | 0.331 |
| Reported final-layer lens | 0.790 | 0.210 |

The final-layer reconstruction itself remains validated: `hidden_states[-1]` is already normalized,
and its unembedding reproduces the **live logits of the same forward pass**. However, the comparison
was temporally misaligned for the mechanistic question. In the generation loop, forward pass `t`
produces logits; the code then derives mixture `t` from those logits and only feeds that blended
embedding into forward pass `t+1`. Consequently, the stored final hidden state at record `t` is the
state that **created** mixture `t`, not the state after the model consumed it. The correct pairing is:

`mixture_t` → blended embedding fed at forward pass `t+1` → `hidden_{t+1}` / output lens.

Therefore, the table above measures exactly what Gumbel noise does to the model's raw next-token
distribution (it broadened 5:0.790 / 3:0.210 into 5:0.669 / 3:0.331). It does **not** show whether
the model's representation preserves that blend after ingestion. The same issue reclassifies Exp
13's 6/60 Gumbel mismatch rate: it is a descriptive input-noise-vs-raw-logit statistic, not valid
evidence of final-state divergence or superposition. Vanilla's 0/12 is expected by construction,
because its mixture top-1 is the raw argmax from that same forward pass.

**What remains valid from Exp 13:** the final-norm validation gate; the 1.5B Gumbel traces; and the
observation that most post-noise mixtures are near one-hot. **What is withdrawn pending a corrected
run:** every inference about an internal representation agreeing or disagreeing with its same-index
mixture.

**Corrected next test:** modify the lightweight probe to carry mixture `t` forward and report the
validated final-layer lens at record `t+1`, alongside that incoming mixture. Re-run the 12-step
vanilla / Gumbel-seed-123 screen and inspect the pair `mixture_9` → `lens_10`. This still needs no
hard phase and should remain a minutes-scale run. Do not begin activation patching until this
time-aligned test is working and a decision-relevant effect replicates.
