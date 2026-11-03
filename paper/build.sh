#!/usr/bin/env bash
# Build the preprint: sync figures, then pdflatex, bibtex, pdflatex x2. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
# Figures and tables are made by ../figures/make_*.py; copy the files the paper uses (never edit the copies).
for f in fig1.tex fig2.tex fig3.pdf fig4.pdf fig5.pdf fig6.pdf fig7.pdf figA1.pdf figA2.pdf \
         tab1.tex tabA1.tex tabA2.tex tabA3.tex; do
  cp ../figures/$f figures/$f
done
run() { pdflatex -interaction=nonstopmode -halt-on-error preprint.tex > /dev/null || { grep -A8 "^!" preprint.log; exit 1; }; }
run
bibtex preprint > /dev/null || { cat preprint.blg; exit 1; }
run
run
for i in 1 2; do grep -q "Rerun to get" preprint.log && run; done
# Report problems: errors, undefined references or citations, overfull boxes over 1 pt.
grep -nE "^!|undefined|Overfull|Rerun to get" preprint.log | awk '
  /Overfull/ { if (match($0, /\(([0-9.]+)pt too/, m) && m[1] > 1.0) print; next } { print }' || true
echo "built preprint.pdf ($(pdfinfo preprint.pdf | awk '/Pages/ {print $2}') pages)"
