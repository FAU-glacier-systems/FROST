"""Shared plotting options for the experiment scripts.

Every plotting script accepts
    --dark   dark theme with transparent background (for slides)
    --pdf    save figures as PDF instead of PNG (the default)

Usage:
    from frost.visualization import plot_style

    parser = argparse.ArgumentParser()
    style = plot_style.setup(parser)       # parses args, applies the theme
    style.savefig(fig, HERE / "plots/name")  # -> plots/name.png or .pdf
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

# Inter for text, JetBrains Mono for tables where the columns must line up
# (SIL Open Font License, fonts/Inter-OFL.txt and fonts/OFL.txt)
FONT_DIR = Path(__file__).parent / "fonts"
FONT = "Inter"
MONO = "JetBrains Mono"

# Slide palette: ice flow blue, dh/dt red, mass balance yellow, calving
# green; observations in the foreground colour, old runs grey
COLORS = {
    "bg": "#0A0A0A", "fg": "#EDEDED", "muted": "#C8C8C8", "grid": "#4A4A4A",
    "blue": "#4FA3E0", "red": "#E5484D", "yellow": "#F2C230",
    "green": "#4CC38A", "grey": "#B4B4B4",
}

for _path in FONT_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(_path))

# Black background, Inter, no frame and no ticks, dashed grey grid;
# no text darker than the muted grey (readable on weak projectors)
DARK_THEME = {
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "savefig.facecolor": COLORS["bg"],
    "font.family": [FONT, "sans-serif"],
    "text.color": COLORS["fg"],
    "axes.labelcolor": COLORS["fg"],
    "axes.titlecolor": COLORS["fg"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["grid"],
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.facecolor": COLORS["bg"],
    "legend.edgecolor": COLORS["bg"],
    "legend.labelcolor": COLORS["fg"],
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=[COLORS[c] for c in ["blue", "yellow", "red", "green", "grey"]]),
}


def use_font():
    """Register the fonts and make Inter the default font (falls back to
    the generic sans-serif font if the files are missing). Tables: MONO."""
    for path in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = [FONT, "sans-serif"]


def fg() -> str:
    """Foreground (text/line) color of the active theme."""
    return plt.rcParams["text.color"]


def bg() -> str:
    """Background color of the active theme."""
    return plt.rcParams["axes.facecolor"]


@dataclass
class Style:
    dark: bool = False
    ext: str = "png"
    args: argparse.Namespace | None = None

    def savefig(self, fig, path, **kwargs):
        """Save with the chosen format; the suffix of `path` is replaced."""
        path = Path(path).with_suffix(f".{self.ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("dpi", 300)
        kwargs.setdefault("transparent", self.dark)
        (fig or plt).savefig(path, **kwargs)
        return path


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dark", action="store_true",
                        help="Dark theme with transparent background (for slides)")
    parser.add_argument("--pdf", action="store_true",
                        help="Save figures as PDF instead of PNG")
    return parser


def setup(parser: argparse.ArgumentParser | None = None) -> Style:
    """Parse --dark/--pdf (plus any arguments already on `parser`) and apply the theme."""
    parser = parser or argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    if args.dark:
        plt.rcParams.update(DARK_THEME)
    return Style(dark=args.dark, ext="pdf" if args.pdf else "png", args=args)
