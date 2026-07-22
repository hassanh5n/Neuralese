# Neuralese Research: Concepts Explained From First Principles

This is a standalone reference. Every concept your project actually uses is here, built up from
scratch with small worked examples — not just definitions. Read it top to bottom once, then use it
as a lookup later. Where useful, it leans on compiler/IR and algorithm framing, since that's
probably the fastest route to a real feel for what's happening rather than just a label for it.

---

## Part 1 — How a language model normally "thinks"

### 1.1 Tokens and embeddings

A model doesn't see words. It sees integers. "cat" might be token id `3212`. Before anything else
happens, that integer is used to look up a row in a big matrix — the **embedding matrix** — and
that row (a vector of, say, 1536 numbers for a small model) is what the network actually computes
with. So:

```
embedding_matrix.shape = [vocab_size, hidden_dim]   # e.g. [151936, 1536] for Qwen2.5-1.5B
embedding_matrix[3212]  = the vector representing "cat"
```

Think of it as a lookup table, exactly like an array indexed by an enum value. Every token in the
vocabulary has a fixed row; the network never invents new rows on the fly, it only ever combines
existing ones.

### 1.2 The normal generation loop

A model predicts one token at a time, then feeds its own output back in as the next input. This is
called **autoregressive generation** — "auto" because it conditions on its own past outputs.

```
prompt tokens → model → logits for "what comes next"
                            ↓
                      pick one token
                            ↓
              append it to the sequence, repeat
```

This is a simple loop with state (the growing sequence) — the same shape as a state machine that
emits one symbol per transition and feeds its own output back into its own input.

### 1.3 Logits → probabilities

At each step, the model's last layer produces one raw score per vocabulary entry — **logits**.
These aren't probabilities yet (they can be negative, and don't sum to 1). **Softmax** converts
them:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

Worked example — vocabulary of 3 words, logits `[2.0, 1.0, 0.1]`:

```
exp(2.0) = 7.39,  exp(1.0) = 2.72,  exp(0.1) = 1.11    sum = 11.22

softmax → cat: 0.66,  dog: 0.24,  bird: 0.10
```

That's the full next-token distribution: the model thinks "cat" is likeliest, but hasn't ruled out
"dog" or "bird."

### 1.4 Greedy decoding vs. sampling, and why top-k/top-p exist

**Greedy decoding**: always take `argmax` — the single highest-probability token. Deterministic,
reproducible, and what you want for a correctness checkpoint (this is exactly why your log's
validation step compares against `.generate(do_sample=False)`).

**Sampling**: draw a token according to the probability distribution instead of always taking the
max. Introduces variety, but raw sampling over the *entire* vocabulary is risky — the tail of a
50,000-word distribution has thousands of tokens with tiny but nonzero probability, and occasionally
sampling one of those produces garbage. Two standard fixes, both used in your `soft_thinking_step`:

- **Top-k**: only consider the `k` highest-probability tokens, discard the rest, renormalize.
- **Top-p / nucleus sampling**: sort by probability, keep adding tokens until their cumulative
  probability crosses `p` (e.g. 0.95), discard the rest, renormalize.

Your code applies both — top-k first as a hard cap, then top-p as a further cut inside that set.

### 1.5 Chain-of-Thought (CoT)

Instead of jumping straight to an answer, prompt or train the model to write out intermediate
reasoning steps in plain text first ("Let's think step by step..."). This works because it gives
the model more forward passes — more computation — to work with before it has to commit to an
answer, and because each written token becomes something the *next* token's prediction can condition
on. It's the current standard baseline your soft/continuous approach is being compared against.

---

## Part 2 — The continuous alternative: Soft Thinking

### 2.1 The core idea

Normal CoT collapses the full distribution to one token at every single step — even when the model
is genuinely unsure. **Soft Thinking** asks: what if, instead of collapsing to one token, you keep
a *blend* of the top candidates alive and feed that blend forward instead?

Concretely: instead of picking "cat" (embedding row 3212) and moving on, you compute a weighted
average of several candidate embeddings and feed *that* vector in as if it were a token embedding.

### 2.2 Concept tokens — the actual mechanism

Continuing the example from 1.3, but now with top-k=2 applied (keep only cat and dog, discard
bird), then renormalized:

```
before renorm:  cat 0.66, dog 0.24        (sum 0.90)
after renorm:   cat 0.73, dog 0.27
```

The **blended vector** ("concept token") is:

```
blended = 0.73 · embedding["cat"] + 0.27 · embedding["dog"]
```

This vector doesn't correspond to any single vocabulary word — it sits *between* "cat" and "dog" in
embedding space, weighted toward "cat." That blended vector is what gets fed back in as
`inputs_embeds` for the next step, instead of a normal embedding row.

Analogy: instead of committing a variable to one value and discarding the alternatives, you carry a
weighted superposition of a few candidate values forward and let downstream computation see all of
them at once, proportionally.

### 2.3 Why this might help — and the open question

The hoped-for benefit: if a reasoning path genuinely forks into two plausible continuations, keeping
both alive a little longer (rather than arbitrarily committing to one) might let the network
implicitly explore both before something downstream forces a real decision. This is the
**superposition** claim — and it's exactly what research question (2) in your project is testing:
does this actually happen, verifiably, against a domain with known ground truth? Or does it just
*look* like exploration because the weights are usually so lopsided (0.9+ on one candidate) that
there's nothing really being explored?

### 2.4 The soft → hard switch

You can't stay in "blended embedding" mode forever — eventually you need a real, readable token
sequence to hand back to a user. So the recipe is: run the blended-embedding process for a fixed
number of steps (the "soft phase"), then switch to ordinary greedy argmax decoding for the rest (the
"hard phase") so the final output is plain text again.

---

## Part 3 — The Greedy Pitfall and stochastic soft thinking

### 3.1 The problem: vanilla soft thinking is quietly greedy

Here's the catch discovered after the original method was published: language, most of the time,
just isn't that ambiguous at the token level. Even in the middle of a genuinely hard reasoning
problem, the *very next token* the model would write is usually still highly predictable (it's often
just the next word of a sentence, a comma, a connective). So the mixture weights end up looking like
`[0.95, 0.03, 0.01, ...]` almost everywhere — technically a "blend," but so lopsided it behaves just
like greedy decoding wearing a costume. This is the **Greedy Pitfall**: the model ends up relying
almost entirely on the single highest-probability token at every step, which is exactly what your
Exp 1–3 weight traces show.

### 3.2 Fix 1 — the Gumbel-Softmax trick

First, the building block: the **Gumbel-max trick**. If you want to *sample* a token according to a
probability distribution (rather than always taking argmax), one valid way is: add independent
random noise (drawn from a Gumbel distribution) to each log-probability, then take argmax of the
noisy scores. This provably samples exactly according to the original distribution — it converts
"randomly sample" into "add noise, then take the max," which is a useful reformulation because it's
differentiable-adjacent.

**Gumbel-softmax** replaces that final hard argmax with a softmax (at some temperature `τ`), turning
a discrete sample into a *soft*, continuous one — which is exactly what you want here, since you
need a smooth blend, not a single hard pick.

```
y_i = softmax_i( (log(π_i) + g_i) / τ ),      g_i ~ Gumbel(0,1)
```

Worked intuition with the cat/dog example (`π = [0.73, 0.27]`, `τ = 0.5`): each candidate gets a
random jolt added to its log-probability before the softmax. Sometimes the jolt on "dog" is large
enough to make it competitive despite starting lower — so across repeated steps, the mixture
genuinely varies instead of being pinned at `[0.73, 0.27]` every single time. Lower `τ` → jolts
matter less relative to the log-probs → closer to the original distribution (less random). Higher
`τ` → jolts dominate → closer to uniform noise (more random, less informative). `τ = 0.5` is a
middle ground.

### 3.3 Fix 2 — Dirichlet resampling

A **Dirichlet distribution** is a distribution *over* probability distributions — sampling from it
gives you back a full simplex vector (a set of numbers that sum to 1), which is exactly the shape
you need for a mixture. You feed it `γ · π` as its concentration parameters (`π` = your renormalized
top-k/top-p weights, `γ` = a scaling knob).

The catch, found empirically: at low `γ`, individual samples from `Dir(γπ)` tend to land near a
corner of the simplex (i.e., close to one-hot) even though *which* corner varies a lot run to run —
high randomness, but each individual sample is still basically a hard pick. At high `γ`, samples
converge toward `π` itself — smooth and blended, but barely random at all. You can't get "smooth and
random" simultaneously out of Dirichlet the way you can with Gumbel-softmax by tuning one knob. This
is why the paper (and your plan) treats Gumbel-softmax as the primary method to try, with Dirichlet
as a secondary comparison.

---

## Part 4 — Looking inside the model: interpretability tools

### 4.1 The residual stream (hidden states)

Picture the transformer's internal computation as a single running vector that starts at the input
embedding and gets updated, layer by layer, all the way to the output. Each layer *reads* the
current version of this vector, computes something (attention, then a small feed-forward network),
and *adds* its contribution back in — it doesn't replace the vector, it accumulates onto it. This
running vector is the **residual stream**, and its value after each layer is that layer's **hidden
state**.

If you know compiler IR: think of the residual stream as a single mutable IR value that survives
across a sequence of optimization passes, where each pass reads the current form of the value and
emits a delta on top of it, rather than each pass starting from scratch. That's structurally very
close to what's happening here — each transformer layer is a "pass" over the same running value.

`output_hidden_states=True` in a forward call gives you this vector's value *after every layer*, not
just the final one.

### 4.2 The logit lens

Here's the trick: the model's very last step is "take the final hidden state, run it through the
final normalization, then the unembedding matrix (`lm_head`), to get logits." What if you did that
*early* — took a hidden state from layer 10 of 28, and ran it through that same final norm +
unembedding, even though that's not what the model actually does?

You get a token distribution — a *guess* at what the model "would predict" if it stopped right
there. This is the **logit lens**. It works reasonably well in practice because transformer layers
tend to progressively refine an increasingly confident guess as information moves through them,
rather than computing something totally unrelated at each layer and only assembling the "real"
answer at the very end.

Caveat worth internalizing: the final norm layer was *trained* to work well on final-layer hidden
states specifically. Reusing it on an earlier layer's hidden state is an approximation, not a
guarantee — if a mid-layer readout looks like nonsense, that can be a property of the lens, not
necessarily a real fact about the model. (There's a fancier version, "tuned lens," that trains a
separate small probe per layer instead of reusing the final norm — worth knowing the name exists if
plain logit lens gives confusing results later.)

This is also why the blended *embedding* itself was never a lens target (per the Review Notes in
your log) — a lens is for decoding something whose content you don't already know. The blended
embedding's content is already fully known: it's a specific weighted sum of specific token rows, and
you already have those weights and indices directly from `soft_thinking_step`. The hidden states
*computed from* that input, on the other hand, genuinely need decoding.

### 4.3 Causal patching (activation patching)

Correlation isn't causation — just because a hidden state *decodes* to something meaningful via the
logit lens doesn't mean that value is actually what's driving the model's eventual output. It might
be a byproduct that the rest of the network ignores.

**Causal patching** tests this directly: run the model, save a hidden state at some (layer, token
position). Then run again, but surgically overwrite that one value with something else (a different
candidate's hidden state, a zeroed vector, noise) while leaving everything else — the input, the
other layers, the other positions — untouched. If the final output changes in the way you'd predict
from what the original value decoded to, that value was *causally* responsible. If nothing changes,
whatever the lens showed you was decorative — present, readable, but not actually used.

Compiler analogy: this is exactly bisection-style debugging of a dataflow graph — change one
intermediate value, re-run everything downstream of it, and check whether the final output diverges.
If it doesn't diverge no matter what you put there, that value was effectively dead code as far as
the final result is concerned, however meaningful it looked in isolation.

Mechanically, in PyTorch, this is done with a **forward hook**: a function you attach to a specific
layer (`module.register_forward_hook(fn)`) that gets called automatically every time that layer runs
its forward pass, and can inspect or overwrite its output before the rest of the network sees it.

### 4.4 Faithfulness

Putting 4.2 and 4.3 together defines **faithfulness** precisely: does what the logit lens shows you
match what causal patching says is actually driving the answer? Two distinct failure modes worth
telling apart:

- **Decorative reasoning**: the visible trace (or lens-decoded content) looks like sensible
  reasoning, but patching it away changes nothing — the real computation happened somewhere else,
  and what you're reading is a plausible-looking side effect.
- **Hidden/steganographic reasoning**: real computation is happening (patching it *does* change the
  output) but it never surfaces anywhere a lens can read it in human-recognizable terms — the model
  has its own internal shorthand.

This is research question (3), and the training-method angle (why RL-only training might make this
worse) is covered in Part 5.

---

## Part 5 — This project's two research questions, precisely

### 5.1 Superposition (question 2)

**Claim being tested:** continuous-thought vectors let a model implicitly hold multiple search
branches "in mind" at once, rather than committing to one path early — mirroring what an explicit
search algorithm does.

**Why math/graph benchmarks aren't rigorous enough:** on an open-ended math problem, "the model is
exploring multiple paths" is largely a matter of interpretation — there's no single, externally
verifiable list of "the branches a correct solver would have open at this exact point." A
constraint satisfaction problem gives you that for free.

**Constraint Satisfaction Problem (CSP), briefly:** a set of variables, each with a domain of
possible values, and constraints restricting which combinations of values are legal (e.g., exam
scheduling: variables = exams, domain = time slots, constraint = no two exams sharing a student can
share a slot).

**Backtracking search:** the classical way to solve a CSP — pick an unassigned variable, try a value
from its domain, recursively try to assign the rest; if you hit a dead end (no legal value for some
later variable), *backtrack* — undo the last assignment and try a different value. This is
depth-first search with pruning.

**MRV (Minimum Remaining Values) heuristic:** when picking which variable to assign next, choose the
one with the *fewest* legal values left in its domain. Intuition: that variable is most likely to
cause a dead end soonest, so resolving it first fails fast and prunes the search tree earlier rather
than wasting work on easier variables first.

The point of using your own CSP solver as ground truth: at any point in a backtracking run, you know
*exactly* which values are still live candidates for the current variable — no interpretation
needed. That's the yardstick the model's blended-vector behavior gets measured against: when the
model is at an equivalent decision point, do its mixture weights and hidden-state contents actually
track a similar live-candidate set, and does patching them shift the outcome the way patching a
branch choice would in the real solver? Or does it just look diffuse without carrying real,
usable information about alternatives?

### 5.2 Faithfulness and training method (question 3)

**Why training method might matter:** in supervised fine-tuning (SFT), every single output token is
directly trained to match a human-written demonstration (cross-entropy loss against real text). This
anchors the model's intermediate steps to human-legible language, because there's a constant,
token-by-token pressure to stay close to what a human actually wrote.

In reinforcement learning (RL) — e.g., training against a verifiable reward like "did the final
answer match" — only the *final outcome* is scored. The tokens along the way get credit only through
that end signal, with nothing forcing them to individually resemble anything a human would write.
Nothing stops the model from drifting toward whatever internal shorthand reliably gets a good score,
even if it stops verbalizing into recognizable human concepts. That's the mechanism behind the
worry that RL-trained models' visible reasoning traces might become less faithful — more decorative,
in the sense from 4.4 — than SFT models' traces, even if both produce fluent-looking text.

---

## Part 6 — Engineering pieces in the code

### 6.1 KV cache, and why the patching harness skips it

Normally, generating token N+1 requires attention over tokens 1..N, which involves computing "key"
and "value" tensors for every past token at every layer. Recomputing all of that from scratch at
every single step would be wasteful — so standard generation **caches** those key/value tensors
(`past_key_values`) and only computes the new ones for the newest token each step. This is exactly a
memoization optimization: don't recompute what you've already computed once.

For the patching harness, that cache is deliberately *not* used — every step recomputes the full
forward pass from scratch. This is slower, but far simpler to reason about and get right: no risk of
patching a cached tensor incorrectly and leaving stale, inconsistent state behind. At the sequence
lengths and model sizes here (tens of tokens, ~1B parameters, CPU), the wasted computation costs
seconds, not minutes — a reasonable trade until there's actual evidence it's too slow, rather than
optimizing preemptively.

### 6.2 Forward hooks

A **forward hook** is a callback you attach to a specific PyTorch module (e.g., one transformer
decoder layer) that fires automatically every time that module's `forward()` runs, and can read or
overwrite its output before the rest of the network sees it. It's the general-purpose mechanism used
for both the logit lens (read a layer's output) and causal patching (overwrite a layer's output).

### 6.3 Entropy as an uncertainty metric

**Entropy** of a probability distribution measures how spread out it is:

```
H(p) = -Σ p_i · log(p_i)
```

Worked examples over the cat/dog pair:

```
One-hot [1.0, 0.0]:        H = -(1·log 1 + 0·log 0) = 0            (fully certain)
Renormalized [0.73, 0.27]: H = -(0.73·log 0.73 + 0.27·log 0.27) ≈ 0.58
Uniform [0.5, 0.5]:        H = -(0.5·log 0.5 + 0.5·log 0.5) ≈ 0.69  (maximally uncertain, for 2 options)
```

Higher entropy = more genuinely spread across options; entropy near zero = effectively a hard
decision regardless of what the raw weights claim to be. Logging this per step turns "scan the
weight list by eye for a dip" into a number you can plot and threshold — a cheap, principled way to
flag candidate fork points automatically instead of manually.

---

## Quick-reference glossary

| Term | One-line meaning |
|---|---|
| Token / embedding | Integer id for a vocabulary entry / the vector row it looks up |
| Autoregressive generation | Predict one token, feed it back in, repeat |
| Logits | Raw, pre-softmax output scores |
| Softmax | Converts logits into a valid probability distribution |
| Greedy decoding | Always pick the single highest-probability token |
| Top-k / top-p (nucleus) | Truncate the distribution to the k most likely, or the smallest set covering probability mass p |
| Chain-of-Thought (CoT) | Writing reasoning out as discrete tokens before the final answer |
| Concept token / blended embedding | Weighted mixture of several candidate token embeddings, fed in as if it were one token |
| Soft Thinking | The method of feeding blended embeddings forward instead of discrete tokens |
| Superposition (this project's sense) | Whether a blended vector genuinely encodes multiple live candidate paths, not just one dominant one |
| Greedy Pitfall | Vanilla soft thinking collapsing into effectively-greedy, single-path behavior |
| Gumbel-max / Gumbel-softmax | Add Gumbel noise to log-probs then argmax (exact discrete sampling) / then softmax (smooth, differentiable version) |
| Dirichlet resampling | Sample a new mixture from Dir(γ·π); trades off randomness vs. smoothness via γ |
| Residual stream / hidden state | The running internal vector each transformer layer reads and updates |
| Logit lens | Decoding an intermediate hidden state by running it through the final norm + unembedding early |
| Causal / activation patching | Overwriting one internal value and observing whether the final output changes |
| Faithfulness | Whether lens-visible content matches what patching shows is actually driving the output |
| CSP | Constraint Satisfaction Problem — variables, domains, constraints |
| Backtracking search | DFS with pruning: assign, recurse, undo on dead end |
| MRV heuristic | Assign the most-constrained variable (fewest legal values) next, to fail fast |
| SFT vs. RL | Token-level supervision against human text vs. outcome-only reward — affects whether intermediate steps stay human-legible |
| KV cache | Cached per-layer key/value tensors so generation doesn't recompute the full sequence every step |
| Forward hook | A callback that intercepts a module's output during its forward pass |
| Entropy | `-Σ p·log(p)` — a single number for how spread out a distribution is |
