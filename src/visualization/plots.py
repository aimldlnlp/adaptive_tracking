from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import aligned_error_curve
from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_json, load_npz, save_dataframe
from src.utils.logging_utils import ProgressCallback
from src.visualization.style import (
    COLORS,
    apply_publication_style,
    controller_color,
    controller_soft_color,
    legend_style,
    text_style,
)


def create_all_figures(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    apply_publication_style()
    figures_dir = get_output_dir(config, "figures")
    metrics_dir = get_output_dir(config, "metrics")
    ensure_dir(figures_dir)

    per_episode = pd.read_csv(metrics_dir / "per_episode_metrics.csv")
    aggregate = pd.read_csv(metrics_dir / "aggregate_metrics.csv")
    bootstrap = _maybe_read_csv(metrics_dir / "bootstrap_intervals.csv")
    breakdown = _maybe_read_csv(metrics_dir / "condition_breakdown.csv")
    comparison = _maybe_read_csv(metrics_dir / "controller_comparison.csv")
    representatives = load_json(metrics_dir / "representative_cases.json")
    _save_focus_case_reports(metrics_dir, per_episode)
    figure_jobs = [
        lambda: _make_trajectory_comparison(config, figures_dir / "trajectory_comparison.png", representatives),
        lambda: _make_tracking_error_vs_time(config, figures_dir / "tracking_error_vs_time.png", per_episode),
        lambda: _make_control_signal_vs_time(config, figures_dir / "control_signal_vs_time.png", representatives),
        lambda: _make_robustness_plot(config, figures_dir / "robustness_under_dynamics_shift.png", aggregate),
        lambda: _make_rmse_boxplot(config, figures_dir / "rmse_boxplot_across_conditions.png", per_episode),
        lambda: _make_ablation_summary_dashboard(config, figures_dir / "ablation_summary_dashboard.png", breakdown),
        lambda: _make_id_unseen_compound(config, figures_dir / "id_vs_unseen_vs_compound.png", breakdown),
        lambda: _make_bootstrap_ci_forest(config, figures_dir / "bootstrap_ci_forest.png", bootstrap),
        lambda: _make_uncertainty_vs_gain(config, figures_dir / "uncertainty_vs_correction_gain.png", per_episode),
        lambda: _make_compound_recovery(config, figures_dir / "compound_shift_failure_recovery.png", representatives),
        lambda: _make_focus_case_dashboard(config, figures_dir / "focus_burst_dashboard.png", per_episode),
        lambda: _make_focus_gap_heatmap(config, figures_dir / "focus_burst_gap_heatmap.png", per_episode),
    ]
    outputs: list[Path] = []
    if progress_callback is not None:
        progress_callback(0.0, "starting figure export")
    for index, job in enumerate(figure_jobs, start=1):
        path = job()
        outputs.append(path)
        if progress_callback is not None:
            progress_callback(index / max(len(figure_jobs), 1), f"saved {path.name}")
    return outputs


def _make_trajectory_comparison(config: dict[str, Any], path: Path, representatives: dict[str, Any]) -> Path:
    episode_id = str(representatives["trajectory_comparison"])
    baseline = _load_rollout(config, episode_id, "baseline")
    primary = _load_rollout(config, episode_id, _primary_controller(config))
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=config["figures"]["dpi"])
    _style_axis(ax, COLORS["panel"])
    _plot_path(ax, baseline["ref_position"], COLORS["reference"], COLORS["reference_soft"], "Ref", 2.3, 2)
    _plot_path(ax, baseline["true_state"][:, :2], controller_color("baseline"), controller_soft_color("baseline"), "Base", 2.0, 3)
    _plot_path(
        ax,
        primary["true_state"][:, :2],
        controller_color(_primary_controller(config)),
        controller_soft_color(_primary_controller(config)),
        _controller_label(_primary_controller(config)),
        2.4,
        4,
    )
    shift_time = float(json.loads(primary["metadata_json"].item())["shift_time"])
    shift_idx = int(np.searchsorted(primary["time"], shift_time))
    _scatter_sequence(ax, primary["true_state"][:, :2], controller_color(_primary_controller(config)), step=7, alpha=0.09, size=12)
    ax.scatter(
        primary["true_state"][shift_idx, 0],
        primary["true_state"][shift_idx, 1],
        s=86,
        color=COLORS["accent"],
        edgecolors=COLORS["paper"],
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(*primary["true_state"][-1, :2], s=54, color=COLORS["accent_alt"], edgecolors="none", alpha=0.95, zorder=6)
    ax.set_xlabel("x", **text_style())
    ax.set_ylabel("y", **text_style())
    ax.axis("equal")
    ax.legend(loc="upper center", ncol=3, handlelength=1.8, columnspacing=1.2, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_tracking_error_vs_time(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=config["figures"]["dpi"])
    _style_axis(ax, COLORS["panel_alt"])
    _add_shift_band(ax, 0.0, 6.0)
    grid = None
    for controller in _controller_order(per_episode, config):
        curves: list[np.ndarray] = []
        subset = per_episode[per_episode["controller"] == controller]
        for row in subset.itertuples(index=False):
            rollout = _load_rollout(config, row.episode_id, row.controller)
            grid, curve = aligned_error_curve(rollout, horizon=(-2.0, 6.0), step=0.05)
            curves.append(curve)
        if not curves:
            continue
        stack = np.stack(curves, axis=0)
        mean = stack.mean(axis=0)
        lo = np.quantile(stack, 0.2, axis=0)
        hi = np.quantile(stack, 0.8, axis=0)
        ax.fill_between(grid, lo, hi, color=controller_soft_color(controller), alpha=0.20)
        ax.plot(grid, mean, color=controller_color(controller), linewidth=2.25, label=_controller_label(controller))
    ax.axvline(0.0, color=COLORS["accent"], linestyle=(0, (4, 2)), linewidth=1.2)
    ax.set_xlabel("time from shift", **text_style())
    ax.set_ylabel("error", **text_style())
    ax.legend(loc="upper right", ncol=2, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_control_signal_vs_time(config: dict[str, Any], path: Path, representatives: dict[str, Any]) -> Path:
    episode_id = str(representatives["control_signal"])
    baseline = _load_rollout(config, episode_id, "baseline")
    primary_name = _primary_controller(config)
    primary = _load_rollout(config, episode_id, primary_name)
    shift_time = float(json.loads(primary["metadata_json"].item())["shift_time"])
    fig, axes = plt.subplots(4, 1, figsize=(10.8, 8.6), dpi=config["figures"]["dpi"], sharex=True)
    series = [
        ("ux", baseline["command"][:, 0], primary["command"][:, 0]),
        ("uy", baseline["command"][:, 1], primary["command"][:, 1]),
        ("|u|", np.linalg.norm(baseline["command"], axis=1), np.linalg.norm(primary["command"], axis=1)),
        ("gain", np.ones(len(baseline["time"])), primary["correction_gain"].reshape(-1)),
    ]
    for ax, (label, base_series, primary_series) in zip(axes, series):
        _style_axis(ax, COLORS["panel"])
        _add_shift_band(ax, shift_time, float(primary["time"][-1]))
        ax.plot(baseline["time"], base_series, color=controller_color("baseline"), linewidth=1.55, label="Base")
        ax.plot(primary["time"], primary_series, color=controller_color(primary_name), linewidth=1.9, label=_controller_label(primary_name))
        ax.fill_between(primary["time"], base_series, primary_series, color=COLORS["accent"], alpha=0.10, linewidth=0.0)
        ax.set_ylabel(label, **text_style())
    axes[0].legend(loc="upper right", ncol=2, **legend_style())
    axes[-1].set_xlabel("time", **text_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_robustness_plot(config: dict[str, Any], path: Path, aggregate: pd.DataFrame) -> Path:
    shift_types = list(config["simulation"]["shift_types"])
    intensities = list(config["simulation"]["shift_intensities"].keys())
    controllers = [name for name in _controller_order(aggregate, config) if name in aggregate["controller"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=config["figures"]["dpi"], sharey=True)
    for ax, shift_type in zip(axes.flatten(), shift_types):
        _style_axis(ax, COLORS["panel"])
        subset = aggregate[(aggregate["shift_type"] == shift_type) & (aggregate["condition_group"] != "compound_shift_ood")]
        x = np.arange(len(intensities))
        width = 0.78 / max(len(controllers), 1)
        for idx, controller in enumerate(controllers):
            values = []
            for intensity in intensities:
                row = subset[(subset["controller"] == controller) & (subset["shift_intensity"] == intensity)]
                values.append(float(row["rmse_mean"].iloc[0]) if not row.empty else np.nan)
            offsets = x - 0.39 + width / 2.0 + idx * width
            ax.bar(offsets, values, width=width * 0.92, color=controller_color(controller), alpha=0.88, label=_controller_label(controller))
            ax.plot(offsets, values, color=COLORS["paper"], linewidth=0.8, alpha=0.7)
        ax.set_title(_short_shift_name(shift_type), pad=4, **text_style())
        ax.set_xticks(x, [_short_intensity_label(value) for value in intensities])
    axes[0, 0].set_ylabel("rmse", **text_style())
    axes[1, 0].set_ylabel("rmse", **text_style())
    axes[0, 0].legend(loc="upper left", ncol=2, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_rmse_boxplot(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    primary_name = _primary_controller(config)
    subset = per_episode[per_episode["controller"].isin(["baseline", primary_name])]
    shift_types = list(config["simulation"]["shift_types"])
    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=config["figures"]["dpi"])
    _style_axis(ax, COLORS["panel"])
    positions = []
    data = []
    labels = []
    scatters: list[tuple[np.ndarray, np.ndarray, str]] = []
    for index, shift_type in enumerate(shift_types):
        for offset, controller in [(-0.16, "baseline"), (0.16, primary_name)]:
            values = subset[(subset["shift_type"] == shift_type) & (subset["controller"] == controller)]["rmse"].to_numpy()
            positions.append(index + 1 + offset)
            data.append(values)
            labels.append(controller)
            jitter = np.linspace(-0.05, 0.05, len(values)) if len(values) > 1 else np.array([0.0])
            scatters.append((np.full_like(values, index + 1 + offset, dtype=float) + jitter, values, controller))
    violin = ax.violinplot(data, positions=positions, widths=0.32, showmeans=False, showmedians=False, showextrema=False)
    for body, controller in zip(violin["bodies"], labels):
        body.set_facecolor(controller_soft_color(controller))
        body.set_edgecolor("none")
        body.set_alpha(0.25)
    box = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True, showfliers=False)
    for patch, controller in zip(box["boxes"], labels):
        patch.set_facecolor(controller_color(controller))
        patch.set_alpha(0.48)
    for xs, ys, controller in scatters:
        ax.scatter(xs, ys, s=18, color=controller_color(controller), alpha=0.18, edgecolors="none")
    ax.set_xticks(range(1, len(shift_types) + 1), [_short_shift_name(value) for value in shift_types])
    ax.set_ylabel("rmse", **text_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_ablation_summary_dashboard(config: dict[str, Any], path: Path, breakdown: pd.DataFrame) -> Path:
    breakdown = breakdown[breakdown["condition_group"] == "overall"] if "condition_group" in breakdown.columns else breakdown
    controllers = _controller_order(breakdown, config)
    metrics = [
        ("rmse_mean", "RMSE"),
        ("success_rate", "Success"),
        ("robustness_score_mean", "Robust"),
        ("control_smoothness_mean", "Smooth"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), dpi=config["figures"]["dpi"])
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        _style_axis(ax, COLORS["panel_alt"])
        values = [float(breakdown[breakdown["controller"] == controller][metric].iloc[0]) for controller in controllers]
        x = np.arange(len(controllers))
        bars = ax.bar(x, values, color=[controller_color(name) for name in controllers], alpha=0.88)
        ax.scatter(x, values, s=42, color=COLORS["paper"], edgecolors=[controller_color(name) for name in controllers], linewidths=1.0)
        ax.set_title(title, pad=4, **text_style())
        ax.set_xticks(x, [_controller_label(name) for name in controllers])
        for bar in bars:
            ax.axhspan(0.0, bar.get_height(), color=COLORS["panel"], alpha=0.08, zorder=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_id_unseen_compound(config: dict[str, Any], path: Path, breakdown: pd.DataFrame) -> Path:
    buckets = ["in_distribution_single_shift", "hard_unseen_trajectory", "compound_shift_ood"]
    titles = {"in_distribution_single_shift": "ID", "hard_unseen_trajectory": "Unseen", "compound_shift_ood": "Compound"}
    controllers = _controller_order(breakdown, config)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), dpi=config["figures"]["dpi"], sharey=True)
    for ax, bucket in zip(axes, buckets):
        _style_axis(ax, COLORS["panel"])
        subset = breakdown[breakdown["condition_group"] == bucket]
        x = np.arange(len(controllers))
        rmse_values = [float(subset[subset["controller"] == controller]["rmse_mean"].iloc[0]) if not subset[subset["controller"] == controller].empty else np.nan for controller in controllers]
        bars = ax.bar(x, rmse_values, color=[controller_color(name) for name in controllers], alpha=0.86)
        success_values = [
            float(subset[subset["controller"] == controller]["success_rate"].iloc[0]) if not subset[subset["controller"] == controller].empty else np.nan for controller in controllers
        ]
        ax.plot(x, success_values, color=COLORS["accent"], linewidth=1.9, marker="o", markersize=4)
        ax.set_title(titles[bucket], pad=4, **text_style())
        ax.set_xticks(x, [_controller_label(name) for name in controllers], rotation=18)
        for bar in bars:
            ax.scatter(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), s=26, color=COLORS["paper"], edgecolors=COLORS["ink"], linewidths=0.4)
    axes[0].set_ylabel("rmse", **text_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_bootstrap_ci_forest(config: dict[str, Any], path: Path, bootstrap: pd.DataFrame) -> Path:
    subset = bootstrap[(bootstrap["metric"] == "rmse") & (bootstrap["condition_group"] == "overall")]
    controllers = _controller_order(subset, config)
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=config["figures"]["dpi"])
    _style_axis(ax, COLORS["panel"])
    y_positions = np.arange(len(controllers))
    for y_pos, controller in zip(y_positions, controllers):
        row = subset[subset["controller"] == controller]
        if row.empty:
            continue
        mean = float(row["mean"].iloc[0])
        low = float(row["ci_low"].iloc[0])
        high = float(row["ci_high"].iloc[0])
        ax.fill_betweenx([y_pos - 0.18, y_pos + 0.18], low, high, color=controller_soft_color(controller), alpha=0.38)
        ax.plot([low, high], [y_pos, y_pos], color=controller_color(controller), linewidth=3.0)
        ax.scatter(mean, y_pos, s=72, color=controller_color(controller), edgecolors=COLORS["paper"], linewidths=0.9, zorder=4)
    ax.set_yticks(y_positions, [_controller_label(name) for name in controllers])
    ax.set_xlabel("rmse", **text_style())
    ax.set_ylabel("", **text_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_uncertainty_vs_gain(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    primary_name = _primary_controller(config)
    subset = per_episode[per_episode["controller"] == primary_name]
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=config["figures"]["dpi"])
    _style_axis(ax, COLORS["panel_alt"])
    size = 80.0 + subset["rmse"].to_numpy() * 420.0
    colors = np.where(subset["condition_group"].to_numpy() == "compound_shift_ood", COLORS["accent"], controller_color(primary_name))
    ax.scatter(
        subset["mean_estimated_uncertainty"],
        subset["mean_correction_gain"],
        s=size,
        c=colors,
        alpha=0.44,
        edgecolors=COLORS["paper"],
        linewidths=0.8,
    )
    z = np.polyfit(subset["mean_estimated_uncertainty"], subset["mean_correction_gain"], deg=1)
    x_grid = np.linspace(float(subset["mean_estimated_uncertainty"].min()), float(subset["mean_estimated_uncertainty"].max()), 50)
    ax.plot(x_grid, np.polyval(z, x_grid), color=COLORS["ink"], linewidth=1.3, linestyle=(0, (4, 2)))
    ax.set_xlabel("uncertainty", **text_style())
    ax.set_ylabel("gain", **text_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_compound_recovery(config: dict[str, Any], path: Path, representatives: dict[str, Any]) -> Path:
    episode_id = str(representatives["compound_showcase"])
    baseline = _load_rollout(config, episode_id, "baseline")
    primary_name = _primary_controller(config)
    primary = _load_rollout(config, episode_id, primary_name)
    shift_time = float(json.loads(primary["metadata_json"].item())["shift_time"])
    limits = _trajectory_limits([baseline, primary], margin=0.35)
    fig = plt.figure(figsize=(11.4, 6.8), dpi=config["figures"]["dpi"])
    ax_traj = fig.add_axes([0.06, 0.12, 0.46, 0.76])
    ax_err = fig.add_axes([0.60, 0.56, 0.34, 0.28])
    ax_gain = fig.add_axes([0.60, 0.18, 0.34, 0.24])
    _style_axis(ax_traj, COLORS["panel"])
    _plot_path(ax_traj, baseline["ref_position"], COLORS["reference"], COLORS["reference_soft"], "Ref", 2.1, 2)
    _plot_path(ax_traj, baseline["true_state"][:, :2], controller_color("baseline"), controller_soft_color("baseline"), "Base", 1.8, 3)
    _plot_path(ax_traj, primary["true_state"][:, :2], controller_color(primary_name), controller_soft_color(primary_name), _controller_label(primary_name), 2.15, 4)
    ax_traj.set_xlim(limits[0], limits[1])
    ax_traj.set_ylim(limits[2], limits[3])
    ax_traj.set_aspect("equal")
    ax_traj.legend(loc="upper center", ncol=3, **legend_style())
    baseline_error = np.linalg.norm(baseline["ref_position"] - baseline["true_state"][:, :2], axis=1)
    primary_error = np.linalg.norm(primary["ref_position"] - primary["true_state"][:, :2], axis=1)
    _style_axis(ax_err, COLORS["panel_alt"])
    _add_shift_band(ax_err, shift_time, float(primary["time"][-1]))
    ax_err.fill_between(primary["time"], baseline_error, primary_error, color=COLORS["accent"], alpha=0.12)
    ax_err.plot(baseline["time"], baseline_error, color=controller_color("baseline"), linewidth=1.6)
    ax_err.plot(primary["time"], primary_error, color=controller_color(primary_name), linewidth=1.9)
    ax_err.set_ylabel("error", **text_style())
    _style_axis(ax_gain, COLORS["panel"])
    _add_shift_band(ax_gain, shift_time, float(primary["time"][-1]))
    ax_gain.fill_between(primary["time"], 0.0, primary["correction_gain"].reshape(-1), color=controller_soft_color(primary_name), alpha=0.28)
    ax_gain.plot(primary["time"], primary["correction_gain"].reshape(-1), color=controller_color(primary_name), linewidth=1.8)
    ax_gain.plot(primary["time"], primary["estimated_uncertainty"].reshape(-1), color=COLORS["accent"], linewidth=1.3, alpha=0.85)
    ax_gain.set_xlabel("time", **text_style())
    ax_gain.set_ylabel("gain", **text_style())
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_focus_case_dashboard(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    focus = _focus_case_breakdown(per_episode)
    focus_shifts = _focus_shift_types()
    intensities = list(config["simulation"]["shift_intensities"].keys())
    controllers = _controller_order(focus, config)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), dpi=config["figures"]["dpi"], sharex=True)
    metric_specs = [
        ("rmse_mean", "RMSE"),
        ("success_rate", "Success"),
    ]
    for col, shift_type in enumerate(focus_shifts):
        subset = focus[focus["shift_type"] == shift_type]
        for row, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row, col]
            _style_axis(ax, COLORS["panel"] if row == 0 else COLORS["panel_alt"])
            x = np.arange(len(intensities))
            for controller in controllers:
                values = []
                for intensity in intensities:
                    match = subset[(subset["controller"] == controller) & (subset["shift_intensity"] == intensity)]
                    values.append(float(match[metric].iloc[0]) if not match.empty else np.nan)
                ax.plot(
                    x,
                    values,
                    color=controller_color(controller),
                    linewidth=2.0,
                    marker="o",
                    markersize=4.5,
                    label=_controller_label(controller),
                )
                ax.fill_between(x, 0.0, values, color=controller_soft_color(controller), alpha=0.07)
            if row == 0:
                ax.set_title(_short_shift_name(shift_type), pad=4, **text_style())
            ax.set_ylabel(ylabel.lower(), **text_style())
            ax.set_xticks(x, [_short_intensity_label(value) for value in intensities])
    axes[0, 0].legend(loc="upper left", ncol=2, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_focus_gap_heatmap(config: dict[str, Any], path: Path, per_episode: pd.DataFrame) -> Path:
    focus = _focus_case_breakdown(per_episode)
    focus_shifts = _focus_shift_types()
    intensities = list(config["simulation"]["shift_intensities"].keys())
    primary = _primary_controller(config)
    comparison = _focus_pairwise_frame(per_episode, "adaptive_gru_nominal", primary)
    gain_subset = focus[focus["controller"] == primary]
    gap_grid = np.full((len(focus_shifts), len(intensities)), np.nan, dtype=float)
    gain_grid = np.full((len(focus_shifts), len(intensities)), np.nan, dtype=float)
    for row_idx, shift_type in enumerate(focus_shifts):
        for col_idx, intensity in enumerate(intensities):
            pair_row = comparison[
                (comparison["shift_type"] == shift_type)
                & (comparison["shift_intensity"] == intensity)
            ]
            gain_row = gain_subset[
                (gain_subset["shift_type"] == shift_type)
                & (gain_subset["shift_intensity"] == intensity)
            ]
            if not pair_row.empty:
                gap_grid[row_idx, col_idx] = float(pair_row["rmse_delta_primary_minus_reference"].iloc[0])
            if not gain_row.empty:
                gain_grid[row_idx, col_idx] = float(gain_row["mean_correction_gain"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=config["figures"]["dpi"])
    panels = [
        (axes[0], gap_grid, COLORS["accent_alt"], "dRMSE"),
        (axes[1], gain_grid, controller_color(primary), "Gain"),
    ]
    for ax, grid, color, title in panels:
        _style_axis(ax, COLORS["panel"])
        base = np.array([1.0, 1.0, 1.0, 1.0])
        if title == "dRMSE":
            vmax = max(float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1e-3, 1e-3)
            cmap = plt.get_cmap("coolwarm")
            image = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        else:
            image = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0.92, vmax=1.0)
        for row_idx in range(grid.shape[0]):
            for col_idx in range(grid.shape[1]):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=COLORS["ink"], **text_style(size=9))
        ax.set_title(title, pad=4, **text_style())
        ax.set_xticks(np.arange(len(intensities)), [_short_intensity_label(value) for value in intensities])
        ax.set_yticks(np.arange(len(focus_shifts)), [_short_shift_name(value) for value in focus_shifts])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_focus_case_reports(metrics_dir: Path, per_episode: pd.DataFrame) -> None:
    focus_breakdown = _focus_case_breakdown(per_episode)
    focus_pairwise = pd.concat(
        [
            _focus_pairwise_frame(per_episode, "adaptive_gru_nominal", "adaptive_gru_uncertainty"),
            _focus_pairwise_frame(per_episode, "adaptive_mlp", "adaptive_gru_uncertainty"),
        ],
        ignore_index=True,
    )
    focus_worst = _focus_worst_cases(per_episode, "adaptive_gru_nominal", "adaptive_gru_uncertainty")
    save_dataframe(focus_breakdown, metrics_dir / "focus_case_breakdown.csv")
    save_dataframe(focus_pairwise, metrics_dir / "focus_case_pairwise.csv")
    save_dataframe(focus_worst, metrics_dir / "focus_case_worst_episodes.csv")


def _focus_case_breakdown(per_episode: pd.DataFrame) -> pd.DataFrame:
    focus = per_episode[per_episode["shift_type"].isin(_focus_shift_types())].copy()
    if focus.empty:
        return pd.DataFrame()
    return (
        focus.groupby(["controller", "shift_type", "shift_intensity"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            success_rate=("success", "mean"),
            final_position_error_mean=("final_position_error", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            mean_estimated_uncertainty=("mean_estimated_uncertainty", "mean"),
            mean_correction_gain=("mean_correction_gain", "mean"),
        )
        .fillna(0.0)
    )


def _focus_pairwise_frame(per_episode: pd.DataFrame, reference: str, primary: str) -> pd.DataFrame:
    focus = per_episode[per_episode["shift_type"].isin(_focus_shift_types())].copy()
    if focus.empty:
        return pd.DataFrame()
    ref = focus[focus["controller"] == reference].set_index("episode_id")
    prim = focus[focus["controller"] == primary].set_index("episode_id")
    joined = ref[
        ["shift_type", "shift_intensity", "condition_group", "rmse", "success", "final_position_error", "control_smoothness"]
    ].join(
        prim[["rmse", "success", "final_position_error", "control_smoothness", "mean_correction_gain"]],
        lsuffix="_reference",
        rsuffix="_primary",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame()
    joined["rmse_delta_primary_minus_reference"] = joined["rmse_primary"] - joined["rmse_reference"]
    joined["success_delta_primary_minus_reference"] = joined["success_primary"] - joined["success_reference"]
    joined["final_error_delta_primary_minus_reference"] = (
        joined["final_position_error_primary"] - joined["final_position_error_reference"]
    )
    joined["smoothness_delta_primary_minus_reference"] = (
        joined["control_smoothness_primary"] - joined["control_smoothness_reference"]
    )
    return (
        joined.reset_index()
        .groupby(["shift_type", "shift_intensity", "condition_group"], as_index=False)
        .agg(
            reference_controller=("episode_id", lambda _: reference),
            primary_controller=("episode_id", lambda _: primary),
            paired_episodes=("episode_id", "count"),
            rmse_reference=("rmse_reference", "mean"),
            rmse_primary=("rmse_primary", "mean"),
            rmse_delta_primary_minus_reference=("rmse_delta_primary_minus_reference", "mean"),
            rmse_win_rate_primary=("rmse_delta_primary_minus_reference", lambda values: float((values < 0.0).mean())),
            success_rate_reference=("success_reference", "mean"),
            success_rate_primary=("success_primary", "mean"),
            success_delta_primary_minus_reference=("success_delta_primary_minus_reference", "mean"),
            final_error_delta_primary_minus_reference=("final_error_delta_primary_minus_reference", "mean"),
            smoothness_delta_primary_minus_reference=("smoothness_delta_primary_minus_reference", "mean"),
            mean_correction_gain_primary=("mean_correction_gain", "mean"),
        )
        .fillna(0.0)
    )


def _focus_worst_cases(per_episode: pd.DataFrame, reference: str, primary: str) -> pd.DataFrame:
    focus = per_episode[per_episode["shift_type"].isin(_focus_shift_types())].copy()
    ref = focus[focus["controller"] == reference].set_index("episode_id")
    prim = focus[focus["controller"] == primary].set_index("episode_id")
    joined = prim[
        ["shift_type", "shift_intensity", "condition_group", "rmse", "success", "mean_correction_gain", "mean_estimated_uncertainty"]
    ].join(
        ref[["rmse", "success"]],
        lsuffix="_primary",
        rsuffix="_reference",
        how="inner",
    )
    joined["rmse_gap_primary_minus_reference"] = joined["rmse_primary"] - joined["rmse_reference"]
    joined["success_gap_primary_minus_reference"] = joined["success_primary"] - joined["success_reference"]
    return (
        joined.reset_index()
        .sort_values("rmse_gap_primary_minus_reference", ascending=False)
        .head(24)
        .fillna(0.0)
    )


def _focus_shift_types() -> list[str]:
    return ["disturbance_burst", "actuator_delay+disturbance_burst"]


def _maybe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _style_axis(ax: plt.Axes, facecolor: str) -> None:
    ax.set_facecolor(facecolor)
    ax.grid(True, which="major", alpha=0.32)
    ax.minorticks_off()


def _plot_path(ax: plt.Axes, points: np.ndarray, color: str, glow_color: str, label: str, linewidth: float, zorder: int) -> None:
    ax.plot(points[:, 0], points[:, 1], color=glow_color, linewidth=linewidth * 2.6, alpha=0.18, zorder=zorder - 1)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, label=label, zorder=zorder)


def _scatter_sequence(ax: plt.Axes, points: np.ndarray, color: str, step: int, alpha: float, size: float) -> None:
    sampled = points[::step]
    ax.scatter(sampled[:, 0], sampled[:, 1], s=size, color=color, alpha=alpha, edgecolors="none", zorder=1)


def _add_shift_band(ax: plt.Axes, start: float, end: float) -> None:
    ax.axvspan(start, end, color=COLORS["post_shift"], alpha=0.18, zorder=0)
    ax.axvline(start, color=COLORS["accent"], linestyle=(0, (4, 2)), linewidth=1.2)


def _controller_order(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    configured = list(config.get("evaluation", {}).get("compare_controllers", []))
    available = list(frame["controller"].unique()) if "controller" in frame.columns else configured
    order = [name for name in configured if name in available]
    if order:
        return order
    return available


def _primary_controller(config: dict[str, Any]) -> str:
    return str(config.get("evaluation", {}).get("primary_controller", config.get("model", {}).get("name", "adaptive")))


def _controller_label(name: str) -> str:
    return {
        "baseline": "Base",
        "adaptive": "Adapt",
        "adaptive_mlp": "MLP",
        "adaptive_gru_nominal": "GRU-N",
        "adaptive_gru_uncertainty": "GRU-U",
    }.get(name, name.replace("_", " ").title())


def _short_shift_name(value: str) -> str:
    return {
        "friction_shift": "Friction",
        "mass_shift": "Mass",
        "actuator_delay": "Delay",
        "disturbance_burst": "Burst",
        "mass_shift+friction_shift": "Mass+Fric",
        "mass_shift+disturbance_burst": "Mass+Burst",
        "friction_shift+actuator_delay": "Fric+Delay",
        "actuator_delay+disturbance_burst": "Delay+Burst",
    }.get(value, value.replace("_", " ").replace("+", " + ").title())


def _short_intensity_label(value: str) -> str:
    return {"mild": "M", "medium": "Md", "severe": "S"}.get(value, value[:1].upper())


def _trajectory_limits(rollouts: list[dict[str, Any]], margin: float = 0.5) -> tuple[float, float, float, float]:
    xs = np.concatenate([np.concatenate([rollout["ref_position"][:, 0], rollout["true_state"][:, 0]]) for rollout in rollouts])
    ys = np.concatenate([np.concatenate([rollout["ref_position"][:, 1], rollout["true_state"][:, 1]]) for rollout in rollouts])
    return xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin


def _load_rollout(config: dict[str, Any], episode_id: str, controller: str) -> dict[str, Any]:
    return load_npz(get_output_dir(config, "metrics", "rollouts") / f"{episode_id}__{controller}.npz")
