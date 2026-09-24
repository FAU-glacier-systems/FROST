"""Shared plotting options for the central_europe scripts.

Every plotting script accepts
    --dark   dark theme with transparent background (for slides)
    --pdf    save figures as PDF instead of PNG (the default)

Usage:
    parser = argparse.ArgumentParser()
    style = plot_style.setup(parser)       # parses args, applies the theme
    style.savefig(fig, "plots/name")       # -> plots/name.png or plots/name.pdf
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

DARK_THEME = {
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "legend.facecolor": "black",
    "legend.edgecolor": "black",
    "legend.labelcolor": "white",
}


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
