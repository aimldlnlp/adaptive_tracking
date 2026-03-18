from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import aligned_error_curve
from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_json, load_npz
from src.visualization.style import COLORS, apply_publication_style


def create_all_figures(config: dict[str, Any]) -> list[Path]:
    apply_publication_style()
    figures_dir = get_output_dir(config, "figures")
    metrics_dir = get_output_dir(config, "metrics")
    ensure_dir(figures_dir)

    per_episode = pd.read_csv(metrics_dir / "per_episode_metrics.csv")
    aggregate = pd.read_csv(metrics_dir / "aggregate_metrics.csv")
    summary = pd.read_csv(metrics_dir / "controller_summary.csv")
    representatives = load_json(metrics_dir / "representative_cases.json")

    outputs = [
        _make_trajectory_comparison(config, figures_dir / "trajectory_comparison.png", representatives["trajectory_comparison"]),
        _make_tracking_error_vs_time(config, figures_dir / "tracking_error_vs_time.png", per_episode),
        _make_control_signal_vs_time(config, figures_dir / "control_signal_vs_time.png", representatives["control_signal"]),
        _make_robustness_plot(config, figures_dir / "robustness_under_dynamics_shift.png", aggregate),
        _make_rmse_boxplot(config, figures_dir / "rmse_boxplot_across_conditions.png", per_episode),
        _make_results_table(config, figures_dir / "results_table.png", summary),
    ]
    return outputs


def _make_trajectory_comparison(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    baseline = _load_rollout(config, episode_id, "baseline")
    adaptive = _load_rollout(config, episode_id, "adaptive")
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=config["figures"]["dpi"])
    ax.plot(baseline["ref_position"][:, 0], baseline["ref_position"][:, 1], color=COLORS["reference"], linewidth=2.4, label="Reference")
    ax.plot(baseline["true_state"][:, 0], baseline["true_state"][:, 1], color=COLORS["baseline"], linewidth=2.0, label="Baseline")
    ax.plot(adaptive["true_state"][:, 0], adaptive["true_state"][:, 1], color=COLORS["adaptive"], linewidth=2.0, label="Adaptive")
    shift_time = float(json.loads(baseline["metadata_json"].item())["shift_time"])
    shift_idx = int(np.searchsorted(baseline["time"], shift_time))
    ax.scatter(
        baseline["ref_position"][shift_idx, 0],
        baseline["ref_position"][shift_idx, 1],
        s=70,
        color=COLORS["accent"],
        label="Shift time",
        zorder=5,
    )
    ax.set_title("Trajectory Tracking Comparison")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_tracking_error_vs_time(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    grid = None
    curves: dict[str, list[np.ndarray]] = {"baseline": [], "adaptive": []}
    for row in per_episode.itertuples(index=False):
        rollout = _load_rollout(config, row.episode_id, row.controller)
        grid, curve = aligned_error_curve(rollout, horizon=(-2.0, 6.0), step=0.05)
        curves[row.controller].append(curve)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=config["figures"]["dpi"])
    for controller in ["baseline", "adaptive"]:
        curve_stack = np.stack(curves[controller], axis=0)
        mean = curve_stack.mean(axis=0)
        std = curve_stack.std(axis=0)
        ax.plot(grid, mean, color=COLORS[controller], linewidth=2.2, label=controller.capitalize())
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[controller], alpha=0.18)
    ax.axvline(0.0, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Shift")
    ax.set_title("Tracking Error Around Dynamics Shift")
    ax.set_xlabel("Time relative to shift [s]")
    ax.set_ylabel("Position error norm [m]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_control_signal_vs_time(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    baseline = _load_rollout(config, episode_id, "baseline")
    adaptive = _load_rollout(config, episode_id, "adaptive")
    shift_time = float(json.loads(baseline["metadata_json"].item())["shift_time"])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), dpi=config["figures"]["dpi"], sharex=True)
    labels = ["u_x", "u_y", "||u||"]
    series = [
        (baseline["command"][:, 0], adaptive["command"][:, 0]),
        (baseline["command"][:, 1], adaptive["command"][:, 1]),
        (np.linalg.norm(baseline["command"], axis=1), np.linalg.norm(adaptive["command"], axis=1)),
    ]
    for ax, label, (base_series, adapt_series) in zip(axes, labels, series):
        ax.plot(baseline["time"], base_series, color=COLORS["baseline"], linewidth=1.7, label="Baseline")
        ax.plot(adaptive["time"], adapt_series, color=COLORS["adaptive"], linewidth=1.7, label="Adaptive")
        ax.axvline(shift_time, color=COLORS["accent"], linestyle="--", linewidth=1.3)
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Control Signal Comparison")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_robustness_plot(config: dict[str, Any], path: Path, aggregate: pd.DataFrame) -> Path:
    shift_types = list(aggregate["shift_type"].unique())
    intensities = ["mild", "medium", "severe"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), dpi=config["figures"]["dpi"], sharey=True)
    for ax, shift_type in zip(axes.flatten(), shift_types):
        subset = aggregate[aggregate["shift_type"] == shift_type]
        x = np.arange(len(intensities))
        width = 0.34
        for offset, controller in [(-width / 2.0, "baseline"), (width / 2.0, "adaptive")]:
            values = []
            for intensity in intensities:
                row = subset[(subset["controller"] == controller) & (subset["shift_intensity"] == intensity)]
                values.append(float(row["rmse_mean"].iloc[0]))
            ax.bar(x + offset, values, width=width, color=COLORS[controller], label=controller.capitalize())
        ax.set_title(shift_type.replace("_", " ").title())
        ax.set_xticks(x, intensities)
        ax.set_xlabel("Shift intensity")
    axes[0, 0].set_ylabel("Mean RMSE [m]")
    axes[1, 0].set_ylabel("Mean RMSE [m]")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Robustness Under Dynamics Shift", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_rmse_boxplot(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    shift_types = list(per_episode["shift_type"].unique())
    fig, ax = plt.subplots(figsize=(11.0, 6.5), dpi=config["figures"]["dpi"])
    positions = []
    data = []
    labels = []
    for index, shift_type in enumerate(shift_types):
        for offset, controller in [(-0.18, "baseline"), (0.18, "adaptive")]:
            positions.append(index + 1 + offset)
            values = per_episode[(per_episode["shift_type"] == shift_type) & (per_episode["controller"] == controller)]["rmse"].to_numpy()
            data.append(values)
            labels.append(controller)
    box = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.55)
    ax.set_xticks(range(1, len(shift_types) + 1), [s.replace("_", "\n") for s in shift_types])
    ax.set_ylabel("RMSE [m]")
    ax.set_title("RMSE Distribution Across Shift Conditions")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_results_table(config: dict[str, Any], path: Path, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=config["figures"]["dpi"])
    ax.axis("off")
    display = summary.copy()
    for column in display.columns:
        if column != "controller":
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    table = ax.table(
        cellText=display.values,
        colLabels=[col.replace("_", " ").title() for col in display.columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    ax.set_title("Aggregate Controller Summary", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_rollout(config: dict[str, Any], episode_id: str, controller: str) -> dict[str, Any]:
    path = get_output_dir(config, "metrics", "rollouts") / f"{episode_id}__{controller}.npz"
    return load_npz(path)
