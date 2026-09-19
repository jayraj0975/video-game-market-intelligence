"""Shared matplotlib theme.

One place for colour and chrome so every figure in the repo reads as part of
the same system. Colours are a validated categorical palette: the slot order
is deliberate (adjacent slots stay distinguishable under the common forms of
colour-vision deficiency), so series are assigned slots in order and the order
is never cycled or reshuffled.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, used in this fixed order.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue ramp for continuous magnitude (heatmaps, intensity).
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Two-pole ramp for signed deviation, with a neutral (not coloured) midpoint.
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#b02b2a"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_theme() -> None:
    """Install the project theme into matplotlib's global rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        # Text stays as text in the SVG rather than being converted to path
        # outlines. That keeps the files about a quarter of the size, makes
        # the labels selectable and searchable, and lets the viewer's own
        # sans-serif render them.
        "svg.fonttype": "none",

        "font.family": "sans-serif",
        # DejaVu is the metric matplotlib lays the figure out with; the rest
        # are the fallbacks a browser reaches for when it renders the SVG.
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,

        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,

        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,

        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,

        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",

        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.labelcolor": INK_SECONDARY,

        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def titled(ax, title: str, subtitle: str | None = None) -> None:
    """Set a left-aligned title with an optional explanatory subtitle.

    The subtitle carries the units and the read-this-way hint, which keeps
    the title itself short enough to scan.
    """
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(
            0, 1.02, subtitle,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=INK_MUTED,
        )


def save(fig, path, close: bool = True) -> None:
    """Write a figure to disk as both SVG and PNG.

    SVG is the version the reports link to: it stays sharp at any zoom, and
    being text it diffs and stores sensibly in git. The PNG is kept alongside
    for anywhere that cannot render SVG -- slides, previews, pasting into a
    message.
    """
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path)
    fig.savefig(path.with_suffix(".png"))
    if close:
        plt.close(fig)
    print(f"  wrote {svg_path.name}")
