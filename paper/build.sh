#!/usr/bin/env bash
# Build the IEEE conference version: sync figures, then pdflatex, bibtex, pdflatex x2. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
# Figures and tables are made by ../figures/make_*.py; copy the files the paper uses (never edit the copies).
if [ -d ../figures ]; then
  for f in fig1.tex fig2.tex fig3.pdf fig4.pdf fig5.pdf fig6.pdf fig7.pdf figA1.pdf figA2.pdf \
           tab1.tex tabA1.tex tabA2.tex tabA3.tex; do
    cp ../figures/$f figures/$f
  done
fi
run() { pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { grep -A8 "^!" main.log; exit 1; }; }
run
bibtex main > /dev/null || { cat main.blg; exit 1; }
run
run
for i in 1 2; do grep -q "Rerun to get" main.log && run; done
# Report problems: errors, undefined references or citations, overfull boxes over 1 pt.
grep -nE "^!|undefined|Rerun to get" main.log || true
grep -oE "Overfull .hbox \(([0-9.]+)pt too wide\).*" main.log | awk '{v=$3; sub(/\(/,"",v); sub(/pt/,"",v); if (v+0 > 1.0) print}' || true
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/ {print $2}') pages)"
