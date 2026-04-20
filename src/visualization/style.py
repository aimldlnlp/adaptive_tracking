from __future__ import annotations

from typing import Any

import matplotlib as mpl


COLORS = {
    "baseline": "#5B6472",
    "adaptive": "#00798C",
    "adaptive_mlp": "#1F6F8B",
    "adaptive_gru_nominal": "#2A9D8F",
    "adaptive_gru_uncertainty": "#0B6E4F",
    "reference": "#D1495B",
    "accent": "#EDAe49",
    "accent_alt": "#F4A261",
    "grid": "#D6D6D6",
    "paper": "#FBF8F1",
    "panel": "#FFFDF8",
    "panel_alt": "#F5F0E7",
    "baseline_soft": "#AAB3BC",
    "adaptive_soft": "#7AC2CE",
    "adaptive_mlp_soft": "#8EBFD0",
    "adaptive_gru_nominal_soft": "#89D2C7",
    "adaptive_gru_uncertainty_soft": "#8FBF9C",
    "reference_soft": "#F3B5BF",
    "post_shift": "#F9E6AE",
    "ink": "#263238",
}

FONT_FAMILY = "CMU Serif"
FONT_WEIGHT = "normal"


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": [FONT_FAMILY],
            "font.serif": [FONT_FAMILY],
            "font.weight": FONT_WEIGHT,
            "figure.facecolor": COLORS["paper"],
            "figure.titleweight": FONT_WEIGHT,
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["paper"],
            "axes.edgecolor": "#4C4B47",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.38,
            "grid.linewidth": 0.8,
            "axes.titleweight": FONT_WEIGHT,
            "axes.labelweight": FONT_WEIGHT,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
        }
    )


def text_style(**overrides: Any) -> dict[str, Any]:
    return {
        "fontfamily": FONT_FAMILY,
        "fontweight": FONT_WEIGHT,
        **overrides,
    }


def legend_style(**overrides: Any) -> dict[str, Any]:
    return {
        "prop": {
            "family": FONT_FAMILY,
            "weight": FONT_WEIGHT,
            **overrides,
        }
    }


def controller_color(name: str) -> str:
    return COLORS.get(name, COLORS["adaptive"])


def controller_soft_color(name: str) -> str:
    return COLORS.get(f"{name}_soft", COLORS["adaptive_soft"])
