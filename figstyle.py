"""
figstyle.py -- one graphical style for every figure of the paper.

Reviewer 1 (comment 7) asked for larger fonts, clearer axis labels and legends
and a consistent style throughout; everything is set here so that no figure can
drift from the others.  Figures are sized for the IEEE two-column format:
single column = 3.5 in, double column = 7.16 in.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

COL1 = 3.5      # inches, one IEEE column
COL2 = 7.16     # inches, two IEEE columns

# colour-blind safe, ordered from the two "certificate" colours outwards
C_UP = "#1b6ca8"      # conforming / upper bound
C_LO = "#c1440e"      # equilibrated / lower bound
C_BAND = "#7fb3d5"    # certified band fill
C_MC = "#4d4d4d"      # Monte-Carlo cloud
PALETTE = ["#1b6ca8", "#c1440e", "#2e7d32", "#8e44ad", "#e08e0b", "#00838f"]


def use_paper_style(base=9.5):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 0.5,
        "figure.titlesize": base + 2,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7",
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": cycler(color=PALETTE),
    })


def row(n, height=3.0, width=COL2, sharey=False, ratios=None):
    """A row of n panels, sized and spaced so nothing collides.

    Reviewer 1 asked for readable figures.  The failure mode of the first
    submission was not the font size (9.5 pt at the printed width is right) but
    the layout: three panels crammed into 2.5 in with the legends dropped on top
    of the data.  Panels are therefore given room here, once, and the legend
    helper below keeps them off the curves.
    """
    fig, axes = plt.subplots(
        1, n, figsize=(width, height), sharey=sharey, layout="constrained",
        gridspec_kw=({"width_ratios": ratios} if ratios else None))
    if n == 1:
        axes = [axes]
    for ax, lab in zip(axes, "abcdefgh"):
        panel_label(ax, f"({lab})")
    return fig, list(axes)


def panel_label(ax, text, dx=-0.02, dy=1.10):
    """(a), (b), ... in a consistent position and weight."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"] + 1.5,
            fontweight="bold", va="top", ha="left")


def legend(ax, loc="best", headroom=0.0, **kw):
    """Compact legend, with optional extra room made for it inside the axes.

    `headroom` expands the y-axis by that fraction so the legend sits on empty
    space instead of over the data.
    """
    if headroom:
        lo, hi = ax.get_ylim()
        span = hi - lo
        if loc.startswith("upper"):
            ax.set_ylim(lo, hi + headroom * span)
        elif loc.startswith("lower"):
            ax.set_ylim(lo - headroom * span, hi)
    opts = dict(fontsize=plt.rcParams["font.size"] - 1.5, borderpad=0.35,
                labelspacing=0.3, handlelength=1.5, handletextpad=0.5,
                borderaxespad=0.35)
    opts.update(kw)
    return ax.legend(loc=loc, **opts)


def certified_band(ax, x, lo, hi, color=C_BAND, label=None, alpha=0.45):
    """Draw a guaranteed enclosure as a filled band with its two edges."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, label=label)
    ax.plot(x, hi, color=C_UP, lw=1.3)
    ax.plot(x, lo, color=C_LO, lw=1.3)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
