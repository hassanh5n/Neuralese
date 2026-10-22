# How a research paper gets published (for the Neuralese paper)

Written 2026-10-15 for Hassan, solo author. Venue facts were checked on the venues' own pages that day (links at the end); re-check dates before you rely on them.

## 1. The path, in one picture

1. **Freeze the claims.** Decide the 3–5 sentences the paper is allowed to say. Everything else supports them. (For you: the thesis and the "don't claim" list in the handoff notes.)
2. **Write the paper** in the venue's LaTeX template, with figures and tables made from saved data by scripts.
3. **Get outside eyes** before anyone official sees it.
4. **Post a preprint on arXiv** (public, timestamped, citable; protects you from being scooped).
5. **Submit to a peer-reviewed venue** (a journal like TMLR, or a conference/workshop) through OpenReview.
6. **Review and rebuttal.** Reviewers ask questions and request changes; you answer and revise.
7. **Decision.** Accept, accept with minor revisions, or reject. Rejection is normal; you revise and resubmit elsewhere.
8. **Camera-ready + code release**, then present (poster or talk) if it is a conference or workshop.

## 2. Writing standards ML reviewers actually check

**Structure (standard for an empirical ML paper).**
- *Abstract* (~150–250 words): question, method, main numbers, what it means.
- *Introduction* (~1 page): the problem, why it matters, your answer in one sentence, and a bullet list of contributions. Reviewers decide their opinion here.
- *Background / related work*: Soft Thinking (Zhang et al.), Wu et al., Rizvi-Martel et al., Backour, Deng et al. Say clearly what each did and what gap you fill.
- *Setup*: model, decoding settings, Cold Stop, data splits, fork definition, validation gates.
- *Results*: one subsection per question, each opening with the question and ending with the one-line answer.
- *Limitations*: one model, 1.5B, GSM8K, seeds, power of the accuracy test. Being upfront here earns trust; hiding it gets you rejected.
- *Conclusion*, *References*, *Appendix* (full tables, extra plots, pre-registration record, AI-use statement).

**Rules of thumb.**
- **Every claim has evidence next to it**: a number with a confidence interval, the n, and the test. TMLR's two acceptance questions are literally "are the claims supported by clear evidence?" and "would some readers care?".
- **Claim no more than the data shows.** "No detectable difference (p = 0.754, 10 discordant pairs)" is good. "Soft Thinking is useless" is not.
- **Define every term once, use it the same way everywhere.** (This is why the writing-fix list renames the two "B"s.)
- **Scales should be intuitive**: 0 = none, 1 = maximum (this is the snap rescale).
- **Figures carry the paper.** A reviewer skims figures first. Each figure has a caption that states the takeaway, not just what is plotted.
- **Report failed predictions as written.** Your pre-registration (timestamped by commits) and honest E1 failure are a strength; put them in an appendix and mention them in the main text.
- **Cite precisely**, with arXiv version numbers where titles changed (Wu et al. v4).
- **Plain, short sentences.** International reviewers read fast; clarity beats sophistication.

## 3. Where this paper fits

The council verdict (2026-10-08) stands, and the venue facts below back it.

| Venue | What it is | Fit for you |
|---|---|---|
| **arXiv** (cs.CL or cs.LG) | Preprint server, not peer-reviewed | **Yes, first.** Post as early as the paper is solid, because Backour (Sep 2026) is close to Exp 17. |
| **TMLR** (Transactions on Machine Learning Research) | Rolling-submission ML journal on OpenReview, double-blind. Judges correctness and interest, *not* novelty or "impact" | **Main target.** Ideal for a careful mechanistic study with a negative/null-leaning result. |
| **ICLR 2027 workshop** | Short paper at a workshop on reasoning / interpretability, usually non-archival | **Yes, second.** Gets you a poster and feedback. |
| ICML / NeurIPS / ACL main tracks | Top conferences, ~25% acceptance, want broad scale | Not now (one 1.5B model, one dataset). |

**TMLR facts that matter** ([author guide](https://jmlr.org/tmlr/author-guide.html), [editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html)):
- Must use the TMLR LaTeX style (on Overleaf). No hard page limit, but very long papers slow review; aim for roughly 12 pages of main text, with the rest in the appendix.
- Double-blind: the PDF and any code you upload must be anonymized. Posting to arXiv under your name is explicitly allowed, but the submission must not link to it.
- An action editor is assigned within about a week; reviews are public on OpenReview; then an open discussion and revision phase, then a decision.
- Supplementary material up to 100 MB (PDF or ZIP), and code is encouraged.
- LLM use is allowed as an assistive tool; you are fully responsible for every word and number, and LLMs can't be authors.
- A workshop or arXiv version does **not** block TMLR, because those are non-archival. An archival conference paper on the same results *would* block it.
- Accepted papers can get a **J2C certification**, which lets you present the TMLR paper at ICLR/ICML/NeurIPS through the Journal-to-Conference track. That is your route to a main-conference poster.

**ICLR 2027 facts** ([dates](https://iclr.cc/Conferences/2027/Dates), [workshop guidelines](https://iclr.cc/Conferences/2027/WorkshopGuidelines)):
- Main conference 26–28 April 2027, workshops 29–30 April; the location is not yet announced.
- The main-track deadline (25 Sep 2026) has passed, which is fine since it's not your target.
- Workshop papers: the suggested deadline is **1 Feb 2027**, and acceptance notices come by **26 Feb 2027**. Each workshop sets its own page limit (typically 4–8 pages) and archival status, so check each call.
- **Workshops are in-person only this year.** Some workshops may accept a poster displayed in your absence or a pre-recorded video, but don't count on it. If you need a visa, start as soon as you are accepted.
- The workshop list appears around December. Look for ones on reasoning, latent/continuous reasoning, or interpretability.

## 4. Step by step, from now to presentation

**A. Before writing (this week)**
- Finish the writing-fix list (started; see `docs/paper-writing-fixes.md` in the repo).
- Make a one-page outline: the title, the contribution bullets, and the figure list (one figure per claim). Write the paper around the figures.
- Create the paper's numbers from saved data with scripts, not by hand-copying from the log.

**B. Writing (about 2–3 weeks)**
- Use the TMLR template on Overleaf from day one. Converting later is painful.
- Order of writing: figures and tables → results → setup → introduction → abstract → related work → limitations.
- Keep a `claims.md`: every number in the paper → the script and file it came from. This is your own reproducibility check.

**C. Before submitting**
- **Outside readers.** As a solo author this is the single most valuable step. Ask 1–3 people: a professor at your university, someone from the ML Collective or EleutherAI Discord, or authors of the papers you build on (a short polite email with the PDF often works). Claude-based simulated review is a useful first filter, but not a substitute.
- **Reproducibility.** Clean the repo: a README, exact package versions, Kaggle instructions, one command per experiment, and the saved jsonl outputs. Make an anonymized copy for review (e.g. anonymous.4open.science) and give the public GitHub repo a Zenodo DOI at release.
- **AI-use statement** in the acknowledgements/appendix (drafted in the fix list).
- **Proofread** the PDF itself, checking that figures are readable in black-and-white and that every reference resolves.

**D. arXiv**
- Create an account; first-time submitters in cs categories may need an **endorsement** from an existing arXiv author. Ask early, since this is where a professor contact helps.
- Upload the LaTeX source (not just the PDF). It appears within about 1–2 days, and the timestamp is your priority claim.
- You can post v2, v3 later as the paper improves.

**E. TMLR submission**
- OpenReview account (use an institutional email if you have one; profiles with only gmail can take a while to be approved, so make it now).
- Submit the anonymized PDF + supplementary ZIP, and suggest action editors who work on LLM reasoning/interpretability.

**F. Review and rebuttal**
- Reply to every point, politely, with what you changed and where ("added in §4.2, Table 3"). Do the cheap requested analyses; for expensive ones, explain the limitation honestly.
- If a reviewer asks for a larger model, the plan is to port Exp 23 to 7B (not the KV suite), as already decided.

**G. Workshop version (Jan–Feb 2027)**
- A 4–8-page cut of the same paper, in that workshop's template, submitted by its deadline (around 1 Feb).
- If accepted: register for the conference (at least one author must), print an A0 poster (or the size the workshop asks for), and prepare a 3-minute verbal walk-through. Ask about **financial aid / volunteer programs**; ICLR usually offers registration help for students and authors from under-represented regions.

## 5. What a solo researcher should be doing alongside writing

- **Watch the literature weekly** (arXiv listings for "soft thinking", "latent reasoning", "continuous CoT"). If something close appears, cite it and adjust positioning; don't panic.
- **Keep the pre-registration trail intact.** Don't edit old log entries; add dated corrections.
- **Build a small network.** One or two senior readers make a large difference to the paper and to arXiv endorsement. Thank them in the acknowledgements.
- **Protect the scope.** No new experiments unless they remove a specific reviewer objection (the ponytail rule from the council).
- **Plan the money and logistics early**: conference registration, visa, travel, and the OpenReview profile.
- **Track deadlines in one place**: the arXiv target (asap, by ~30 Nov), the TMLR submission (~30 Nov 2026), the workshop list (~Dec), the workshop deadline (~1 Feb 2027), and the workshop decisions (26 Feb 2027).

## Sources
- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
- [TMLR editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html)
- [TMLR FAQ](https://jmlr.org/tmlr/faq.html)
- [ICLR 2027 dates](https://iclr.cc/Conferences/2027/Dates)
- [ICLR 2027 workshop guidelines](https://iclr.cc/Conferences/2027/WorkshopGuidelines)
