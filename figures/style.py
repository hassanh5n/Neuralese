"""Shared style for every data figure. Import this first in each figures/make_figN.py.

Rules (docs/next-steps-figures-validation.md section 2):
- Same colour = same thing in every figure. Okabe-Ito palette (colour-blind safe); sampling is grey.
- Plot at final size: FULL = 6.5 in (TMLR text width), HALF = 3.2 in. Text 7-8 pt.
- Font matches the paper: TMLR loads lmodern (Latin Modern). With pdflatex on PATH, figures are typeset
  by pdflatex through matplotlib's pgf backend, so the PDF carries the same Type 1 fonts as the paper.
  Without pdflatex, falls back to matplotlib's bundled Computer Modern (cmr10), which looks nearly the same.
  Text in labels is LaTeX in the first case: escape %, &, _, #.
- save() writes the PDF (for the paper) and a 300 dpi PNG preview rasterised from that PDF.
"""
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
FULL, HALF = 6.5, 3.2  # inches
TEX = shutil.which("pdflatex") is not None

# One colour per thing, fixed across the paper.
COLORS = {
    # decoding arms
    "sampling": "#7F7F7F",  # discrete sampled CoT; Exp 23 "hard"
    "gumbel": "#0072B2",  # Gumbel Soft Thinking; Exp 23 "soft"
    "vanilla": "#56B4E9",  # vanilla Soft Thinking (no noise)
    "dirichlet": "#000000",  # Dirichlet Soft Thinking
    "rand": "#E69F00",  # random-residual control (Exp 23)
    # fork kinds (Exp 21)
    "nn": "#D55E00",  # number vs number
    "nw": "#009E73",  # number vs word
    "ww": "#CC79A7",  # word vs word
    # non-data elements
    "guide": "#9A9A9A",  # reference lines and their labels
    "grid": "#E6E6E6",
}
COLORS["hard"], COLORS["soft"] = COLORS["sampling"], COLORS["gumbel"]

LABELS = {
    "sampling": "sampling", "gumbel": "Gumbel Soft Thinking", "vanilla": "vanilla Soft Thinking",
    "dirichlet": "Dirichlet Soft Thinking", "rand": "random-residual control",
    "hard": "sampling", "soft": "Soft Thinking",
    "nn": "number vs number", "nw": "number vs word", "ww": "word vs word",
}

BAND_ALPHA = 0.2  # CI bands

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.labelpad": 2,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    # Exact final size: constrained layout fits labels inside the canvas; no cropping at save time.
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "savefig.bbox": None,
})
if TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    })
else:
    plt.rcParams.update({"font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})


def save(fig, name):
    """Write figures/<name>.pdf (paper) and figures/<name>.png (preview of that PDF)."""
    w = fig.get_size_inches()[0]
    assert min(abs(w - FULL), abs(w - HALF)) < 1e-6, f"{name}: width {w} in is not FULL or HALF"
    pdf, png = FIG_DIR / f"{name}.pdf", FIG_DIR / f"{name}.png"
    if TEX:
        os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # stable PDF dates across reruns
        fig.savefig(pdf, backend="pgf")
    else:
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(png.with_suffix(""))], check=True)
    elif not TEX:
        fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and figures/{name}.png ({'pdflatex, Latin Modern' if TEX else 'cmr10 fallback'})")


def plain_ticks(ax, axis="y"):
    """Tick labels as 0, 0.25, 1 (no trailing zeros), with a real minus sign."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"${v:g}$" if TEX else f"{v:g}".replace("-", "−"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)
