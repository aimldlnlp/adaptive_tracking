from __future__ import annotations

from typing import Any

import matplotlib as mpl


COLORS = {
    "baseline": "#5B6472",
    "adaptive": "#00798C",
    "reference": "#D1495B",
    "accent": "#EDAe49",
    "grid": "#D6D6D6",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
