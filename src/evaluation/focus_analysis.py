from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_npz, save_dataframe, save_json
from src.visualization.style import (
    COLORS,
    apply_publication_style,
    controller_color,
    controller_soft_color,
    legend_style,
    text_style,
)


def run_focus_case_analysis(config: dict[str, Any], top_k: int = 8) -> dict[str, str]:
    apply_publication_style()
    metrics_dir = get_output_dir(config, "metrics")
    analysis_dir = ensure_dir(get_output_dir(config, "analysis", "focus_cases"))
    tables_dir = ensure_dir(analysis_dir / "tables")
    episode_tables_dir = ensure_dir(tables_dir / "episodes")
    figures_dir = ensure_dir(analysis_dir / "figures")

    worst = pd.read_csv(metrics_dir / "focus_case_worst_episodes.csv")
    focus_pairwise = pd.read_csv(metrics_dir / "focus_case_pairwise.csv")
    top = worst.head(max(int(top_k), 1)).copy()
    reference_controller = "adaptive_gru_nominal"
    primary_controller = _primary_controller(config)

    step_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in top.itertuples(index=False):
        primary_rollout = _load_rollout(config, row.episode_id, primary_controller)
        reference_rollout = _load_rollout(config, row.episode_id, reference_controller)
        frame = _episode_step_frame(
            episode_id=str(row.episode_id),
            primary_rollout=primary_rollout,
            reference_rollout=reference_rollout,
            primary_controller=primary_controller,
            reference_controller=reference_controller,
        )
        step_frames.append(frame)
        episode_path = episode_tables_dir / f"{row.episode_id}.csv"
        save_dataframe(frame, episode_path)
        manifest_rows.append(
            {
                "episode_id": str(row.episode_id),
                "shift_type": str(row.shift_type),
                "shift_intensity": str(row.shift_intensity),
                "condition_group": str(row.condition_group),
                "step_trace_csv": str(episode_path),
            }
        )

    all_steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    aligned_summary = _aligned_focus_summary(config, focus_pairwise, primary_controller, reference_controller)

    save_dataframe(all_steps, tables_dir / "focus_case_step_traces.csv")
    save_dataframe(aligned_summary, tables_dir / "focus_case_alignment_summary.csv")
    save_dataframe(top, tables_dir / "focus_case_top_episodes.csv")
    save_json(manifest_rows, tables_dir / "focus_case_episode_manifest.json")

    alignment_path = _make_focus_alignment_dashboard(
        figures_dir / "focus_case_alignment_dashboard.png",
        aligned_summary,
        primary_controller,
        reference_controller,
    )
    worst_path = _make_focus_worst_episode_panels(
        figures_dir / "focus_case_worst_episode_panels.png",
        all_steps,
        primary_controller,
        reference_controller,
    )

    outputs = {
        "analysis_dir": str(analysis_dir),
        "step_traces": str(tables_dir / "focus_case_step_traces.csv"),
        "alignment_summary": str(tables_dir / "focus_case_alignment_summary.csv"),
        "top_episodes": str(tables_dir / "focus_case_top_episodes.csv"),
        "episode_manifest": str(tables_dir / "focus_case_episode_manifest.json"),
        "alignment_dashboard": str(alignment_path),
        "worst_episode_panels": str(worst_path),
    }
    save_json(outputs, analysis_dir / "focus_case_outputs.json")
    return outputs


def _episode_step_frame(
    episode_id: str,
    primary_rollout: dict[str, Any],
    reference_rollout: dict[str, Any],
    primary_controller: str,
    reference_controller: str,
) -> pd.DataFrame:
    metadata = _json_metadata(primary_rollout)
    shift_time = float(metadata["shift_time"])
    time = primary_rollout["time"].reshape(-1)
    primary_error = np.linalg.norm(primary_rollout["ref_position"] - primary_rollout["true_state"][:, :2], axis=1)
    reference_error = np.linalg.norm(reference_rollout["ref_position"] - reference_rollout["true_state"][:, :2], axis=1)
    primary_command_norm = np.linalg.norm(primary_rollout["command"], axis=1)
    reference_command_norm = np.linalg.norm(reference_rollout["command"], axis=1)
    baseline_command_norm = np.linalg.norm(primary_rollout["baseline_command"], axis=1)
    disturbance_true = primary_rollout["disturbance_force"]
    disturbance_true_norm = np.linalg.norm(disturbance_true, axis=1)
    disturbance_pred = primary_rollout["estimated_targets"][:, 3:5]
    disturbance_pred_norm = np.linalg.norm(disturbance_pred, axis=1)
    disturbance_pred_error = np.linalg.norm(disturbance_pred - disturbance_true, axis=1)
    delay_true = primary_rollout["delay_severity"].reshape(-1)
    delay_pred = primary_rollout["estimated_targets"][:, 2]
    delay_error = delay_pred - delay_true

    return pd.DataFrame(
        {
            "episode_id": episode_id,
            "primary_controller": primary_controller,
            "reference_controller": reference_controller,
            "trajectory_kind": str(metadata["trajectory_kind"]),
            "shift_type": str(metadata["shift_type"]),
            "shift_intensity": str(metadata["shift_intensity"]),
            "condition_group": str(metadata["condition_group"]),
            "time": time,
            "relative_time": time - shift_time,
            "shift_active": primary_rollout["shift_active"].reshape(-1),
            "error_primary": primary_error,
            "error_reference": reference_error,
            "error_gap_primary_minus_reference": primary_error - reference_error,
            "command_norm_primary": primary_command_norm,
            "command_norm_reference": reference_command_norm,
            "command_delta_from_baseline": primary_command_norm - baseline_command_norm,
            "applied_command_gap": np.linalg.norm(primary_rollout["applied_command"] - primary_rollout["command"], axis=1),
            "disturbance_true_norm": disturbance_true_norm,
            "disturbance_pred_norm": disturbance_pred_norm,
            "disturbance_pred_error_norm": disturbance_pred_error,
            "delay_true": delay_true,
            "delay_pred": delay_pred,
            "delay_error": delay_error,
            "mean_estimated_uncertainty": primary_rollout["estimated_uncertainty"].reshape(-1),
            "structure_estimated_uncertainty": primary_rollout["structure_estimated_uncertainty"].reshape(-1),
            "disturbance_estimated_uncertainty": primary_rollout["disturbance_estimated_uncertainty"].reshape(-1),
            "correction_gain": primary_rollout["correction_gain"].reshape(-1),
            "structure_gain": primary_rollout["structure_gain"].reshape(-1),
            "disturbance_gain": primary_rollout["disturbance_gain"].reshape(-1),
        }
    )


def _aligned_focus_summary(
    config: dict[str, Any],
    focus_pairwise: pd.DataFrame,
    primary_controller: str,
    reference_controller: str,
) -> pd.DataFrame:
    metrics_dir = get_output_dir(config, "metrics")
    rollouts_dir = metrics_dir / "rollouts"
    horizon = np.arange(-1.5, 6.05, 0.05)
    rows: list[dict[str, Any]] = []
    for shift_type in ["disturbance_burst", "actuator_delay+disturbance_burst"]:
        episode_ids = sorted(
            pd.read_csv(metrics_dir / "focus_case_worst_episodes.csv")
            .query("shift_type == @shift_type")["episode_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        if not episode_ids:
            continue
        primary_curves = {name: [] for name in ["error", "disturbance_pred_error", "gain", "uncertainty", "delay_error"]}
        reference_curves = {"error": []}
        for episode_id in episode_ids:
            primary = load_npz(rollouts_dir / f"{episode_id}__{primary_controller}.npz")
            reference = load_npz(rollouts_dir / f"{episode_id}__{reference_controller}.npz")
            metadata = _json_metadata(primary)
            shift_time = float(metadata["shift_time"])
            relative_time = primary["time"].reshape(-1) - shift_time
            primary_error = np.linalg.norm(primary["ref_position"] - primary["true_state"][:, :2], axis=1)
            reference_error = np.linalg.norm(reference["ref_position"] - reference["true_state"][:, :2], axis=1)
            disturbance_pred_error = np.linalg.norm(primary["estimated_targets"][:, 3:5] - primary["disturbance_force"], axis=1)
            delay_error = primary["estimated_targets"][:, 2] - primary["delay_severity"].reshape(-1)
            signals = {
                "error": primary_error,
                "disturbance_pred_error": disturbance_pred_error,
                "gain": primary["correction_gain"].reshape(-1),
                "uncertainty": primary["estimated_uncertainty"].reshape(-1),
                "delay_error": delay_error,
            }
            for name, values in signals.items():
                primary_curves[name].append(np.interp(horizon, relative_time, values, left=values[0], right=values[-1]))
            reference_curves["error"].append(
                np.interp(horizon, relative_time, reference_error, left=reference_error[0], right=reference_error[-1])
            )
        for idx, rel_t in enumerate(horizon):
            rows.append(
                {
                    "shift_type": shift_type,
                    "relative_time": float(rel_t),
                    "error_primary_mean": float(np.mean([curve[idx] for curve in primary_curves["error"]])),
                    "error_reference_mean": float(np.mean([curve[idx] for curve in reference_curves["error"]])),
                    "error_gap_mean": float(
                        np.mean([curve[idx] for curve in primary_curves["error"]])
                        - np.mean([curve[idx] for curve in reference_curves["error"]])
                    ),
                    "disturbance_pred_error_mean": float(np.mean([curve[idx] for curve in primary_curves["disturbance_pred_error"]])),
                    "gain_mean": float(np.mean([curve[idx] for curve in primary_curves["gain"]])),
                    "uncertainty_mean": float(np.mean([curve[idx] for curve in primary_curves["uncertainty"]])),
                    "delay_error_mean": float(np.mean([curve[idx] for curve in primary_curves["delay_error"]])),
                }
            )
    return pd.DataFrame(rows)


def _make_focus_alignment_dashboard(
    path: Path,
    aligned_summary: pd.DataFrame,
    primary_controller: str,
    reference_controller: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), dpi=180, sharex="col")
    shift_types = ["disturbance_burst", "actuator_delay+disturbance_burst"]
    for col, shift_type in enumerate(shift_types):
        subset = aligned_summary[aligned_summary["shift_type"] == shift_type]
        axes[0, col].set_facecolor(COLORS["panel"])
        axes[1, col].set_facecolor(COLORS["panel_alt"])
        axes[0, col].axvspan(0.0, subset["relative_time"].max() if not subset.empty else 1.0, color=COLORS["post_shift"], alpha=0.16)
        axes[1, col].axvspan(0.0, subset["relative_time"].max() if not subset.empty else 1.0, color=COLORS["post_shift"], alpha=0.16)
        axes[0, col].plot(subset["relative_time"], subset["error_reference_mean"], color=controller_color(reference_controller), linewidth=2.0, label="GRU-N")
        axes[0, col].plot(subset["relative_time"], subset["error_primary_mean"], color=controller_color(primary_controller), linewidth=2.0, label="GRU-U")
        axes[0, col].fill_between(
            subset["relative_time"],
            subset["error_reference_mean"],
            subset["error_primary_mean"],
            color=COLORS["accent"],
            alpha=0.12,
        )
        axes[0, col].set_title(_short_shift_name(shift_type), **text_style())
        axes[0, col].set_ylabel("error", **text_style())
        axes[1, col].plot(subset["relative_time"], subset["gain_mean"], color=controller_color(primary_controller), linewidth=1.9, label="gain")
        axes[1, col].plot(subset["relative_time"], subset["uncertainty_mean"], color=COLORS["accent"], linewidth=1.6, label="unc")
        axes[1, col].plot(subset["relative_time"], subset["disturbance_pred_error_mean"], color=COLORS["ink"], linewidth=1.4, linestyle=(0, (4, 2)), label="d err")
        axes[1, col].set_ylabel("diag", **text_style())
        axes[1, col].set_xlabel("time from shift", **text_style())
    axes[0, 0].legend(loc="upper left", ncol=2, **legend_style())
    axes[1, 0].legend(loc="upper left", ncol=3, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_focus_worst_episode_panels(
    path: Path,
    all_steps: pd.DataFrame,
    primary_controller: str,
    reference_controller: str,
) -> Path:
    if all_steps.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
        ax.text(0.5, 0.5, "no focus episodes", ha="center", va="center", **text_style())
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    top_episodes = list(all_steps["episode_id"].drop_duplicates().head(4))
    fig, axes = plt.subplots(len(top_episodes), 3, figsize=(12.4, 2.7 * len(top_episodes)), dpi=180, sharex=False)
    if len(top_episodes) == 1:
        axes = np.asarray([axes])
    for row_idx, episode_id in enumerate(top_episodes):
        subset = all_steps[all_steps["episode_id"] == episode_id]
        title = f"{_short_shift_name(str(subset['shift_type'].iloc[0]))} {_short_intensity_label(str(subset['shift_intensity'].iloc[0]))}"
        ax_err, ax_dist, ax_gain = axes[row_idx]
        for ax, face in [(ax_err, COLORS["panel"]), (ax_dist, COLORS["panel_alt"]), (ax_gain, COLORS["panel"])]:
            ax.set_facecolor(face)
            ax.grid(True, alpha=0.28)
            ax.axvspan(0.0, float(subset["relative_time"].max()), color=COLORS["post_shift"], alpha=0.12)
        ax_err.plot(subset["relative_time"], subset["error_reference"], color=controller_color(reference_controller), linewidth=1.7)
        ax_err.plot(subset["relative_time"], subset["error_primary"], color=controller_color(primary_controller), linewidth=1.9)
        ax_err.fill_between(subset["relative_time"], subset["error_reference"], subset["error_primary"], color=COLORS["accent"], alpha=0.10)
        ax_err.set_ylabel("error", **text_style())
        ax_err.set_title(title, **text_style())
        ax_dist.plot(subset["relative_time"], subset["disturbance_true_norm"], color=COLORS["accent_alt"], linewidth=1.6, label="true")
        ax_dist.plot(subset["relative_time"], subset["disturbance_pred_norm"], color=controller_color(primary_controller), linewidth=1.8, label="pred")
        ax_dist.plot(subset["relative_time"], subset["disturbance_pred_error_norm"], color=COLORS["ink"], linewidth=1.2, linestyle=(0, (4, 2)), label="err")
        ax_dist.set_ylabel("|d|", **text_style())
        ax_gain.plot(subset["relative_time"], subset["correction_gain"], color=controller_color(primary_controller), linewidth=1.8, label="gain")
        ax_gain.plot(subset["relative_time"], subset["mean_estimated_uncertainty"], color=COLORS["accent"], linewidth=1.4, label="unc")
        ax_gain.plot(subset["relative_time"], subset["delay_error"], color=COLORS["reference"], linewidth=1.2, linestyle=(0, (2, 2)), label="delay err")
        ax_gain.set_ylabel("diag", **text_style())
        ax_gain.set_xlabel("time from shift", **text_style())
        if row_idx == 0:
            ax_dist.legend(loc="upper left", ncol=3, **legend_style())
            ax_gain.legend(loc="upper left", ncol=3, **legend_style())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_rollout(config: dict[str, Any], episode_id: str, controller: str) -> dict[str, Any]:
    return load_npz(get_output_dir(config, "metrics", "rollouts") / f"{episode_id}__{controller}.npz")


def _json_metadata(rollout: dict[str, Any]) -> dict[str, Any]:
    raw = rollout["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _primary_controller(config: dict[str, Any]) -> str:
    return str(config.get("evaluation", {}).get("primary_controller", config.get("model", {}).get("name", "adaptive")))


def _short_shift_name(value: str) -> str:
    return {
        "disturbance_burst": "Burst",
        "actuator_delay+disturbance_burst": "Delay+Burst",
    }.get(value, value.replace("_", " ").replace("+", " + ").title())


def _short_intensity_label(value: str) -> str:
    return {"mild": "M", "medium": "Md", "severe": "S"}.get(value, value[:1].upper())
