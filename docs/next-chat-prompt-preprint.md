# Prompt for the next chat: first full draft of the preprint (LaTeX)

Paste everything below the line into a new chat on this project, with the Neuralese folder linked.

---

Write the first full draft of my preprint as LaTeX. It must be the whole paper, every section and the appendix, ready to compile and to show my professors for review and discussion.

**Repo:** `C:\Users\hp\Desktop\Neuralese`. Work in the shell on my computer (Linux VM with python3, matplotlib, numpy, pdflatex, bibtex, pdftoppm; no scipy).

**Read first, in this order:**
1. `docs/paper-outline.md`: title, one-sentence answer, contribution order, figure list, section plan, preprint layout, venue plan. Use title 1 unless I change it.
2. `docs/neuralese-paper-log.md`: the source of truth. Claim ledger (§2), pre-registration (§3), setup (§4), results with dated additions (§5), errata (§8), literature (§9), open items with versions and model revision (§10).
3. `docs/paper-writing-fixes.md`: ready drafts (term table §1, snap §2, +0.44 §3, Wu et al. §4, AI-use statement §5, wording rules §6.2, Backour and Rizvi-Martel §7).
4. `docs/publishing-guide.md` (writing standards) and `figures/` (scripts, captions in docstrings, `tab1.tex`, `tabA1–A3.tex`, `fig1.tex`, `fig2.tex`). Figure numbers follow the renumbering note at the top of the paper log: Fig 1 concept, Fig 2 questions, Figs 3–7 data.

**Rules (non-negotiable):**
- A number goes in the paper only if it has a ledger row and a check that recomputes it. Put the ledger id in a LaTeX comment after each number (`% C23.15`). Numbers from other papers cite the paper instead.
- Use the paper terms from paper-writing-fixes.md §1 and §6.1 (chosen token, runner-up, residual, top weight, pooled share, sampling, Soft Thinking, random-residual control). Respect every scope note in §6.2: "at the output", "pooled share", 50/50 blends are an upper bound, exploratory vs pre-registered, Dirichlet result.
- No em dashes. Plain, short sentences. Professional academic tone. Claim no more than the data shows.
- Never edit old log entries; add dated bullets. Don't commit unless I ask.

**Learn how to write from past papers.** Download and read (curl arXiv to the VM, then `pdftotext`): Rizvi-Martel et al. arXiv:2604.06374v2 (COLM 2026), Backour arXiv:2609.02049v1, Wu et al. arXiv:2508.03440v4 (ICLR 2026), Zhang et al. arXiv:2505.15778. Take from them how the introduction states the question and the contributions, how each result subsection opens with its question and closes with its answer, how related work positions against prior work, and how limitations are stated. Do not copy wording or figure designs.

**Files to produce:**
- `paper/preprint.tex` (wrapper), `paper/sections/*.tex` (one file per section, shared by future venue versions), `paper/appendix/*.tex`, `paper/refs.bib`, plus copies or relative paths to `figures/`.
- Layout: article class, US letter, 1 in margins (6.5 in text width), Latin Modern 10 pt, `\usepackage{lineno}` line numbers, header "Draft for discussion, <date>", author Hassan Nafees with affiliation FAST NUCES Karachi (ask me for the exact affiliation line and email).
- Figures at natural size, never scaled: `\input{fig1.tex}` and `\input{fig2.tex}` (TikZ), `\includegraphics{figN.pdf}` for Figs 3–7 and A1–A2. Preamble: `tikz` with `arrows.meta,calc,decorations.pathreplacing`, `booktabs`, `hyperref`.
- Tables via `\input{tab1.tex}` and the appendix tables.
- Captions from each figure script's docstring.
- `paper/build.sh` (pdflatex, bibtex, pdflatex x2) and a zip for Overleaf.

**Content:** every section in outline §E, in this writing order: Results (one subsection per question Q1 to Q4, Figs 3 to 7 and Table 1) → Setup (model and revision, decoding settings, Cold Stop, item ranges and their overlaps, fork definition, blends A/B/C, validation gates, pre-registration, statistics: bootstrap over problems, Wilson, McNemar, sign test) → Introduction (superposition hypothesis, Figs 1 and 2, contributions) → Abstract → Related work (Soft Thinking, Wu et al., Rizvi-Martel et al., Backour, latent-CoT patching, superposition theory) → Discussion (Wu et al. reconciliation, Rizvi-Martel tension) → Limitations → Conclusion. Appendix: pre-registration record with failed predictions reported as written, Tables A1 to A3, Figs A1 and A2, embedding-norm check, reproducibility (model revision ad9f0ae, transformers 5.19.0 for Exp 21 and 23 and 5.20.1 for Exp 16 and 22 inferred from wheels, torch and CUDA from the norm check, `kaggle_logs/`), and the AI-use statement.

**Use the ARS skills where each helps most:**
- `academic-research-skills:academic-paper` to plan and draft the sections against the outline.
- Its abstract mode for the abstract.
- Its citation-check mode on `refs.bib`: every entry's arXiv ID, title, authors, version and venue checked at the source; Wu et al. v4; Rizvi-Martel COLM 2026.
- Its disclosure mode to finalise the AI-use statement from paper-writing-fixes.md §5.
- `academic-research-skills:academic-paper-reviewer` for a simulated peer review of the compiled draft. Fix what it finds that is correct, and list the rest for me.

**Then run every section through `anthropic-skills:humanizer`, keeping a professional academic tone:** remove AI patterns (em dashes, not-X-but-Y, staged openers, triads, filler, inflated claims) without changing any number, claim or citation. Re-check the numbers after the humanizer pass.

**Before delivering:**
- `python verify_claims.py` (110/110) and `python verify_robustness.py` (17/17) still pass.
- A script checks that every number in `paper/` with a `% C..` comment matches its ledger value.
- An independent agent that did not write the draft reads the PDF against the ledger and the scope rules.
- The PDF compiles with no errors, no undefined references and no overfull boxes over 1 pt.
- I look at every page image myself.

Work in stages with a task list. Send the PDF when the Results section is drafted, then the full draft. End with the list of open questions for my professors.
