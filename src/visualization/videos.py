from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_json, load_npz
from src.visualization.style import COLORS, apply_publication_style


def create_all_videos(config: dict[str, Any]) -> list[Path]:
    apply_publication_style()
    videos_dir = get_output_dir(config, "videos")
    metrics_dir = get_output_dir(config, "metrics")
    ensure_dir(videos_dir)
    reps = load_json(metrics_dir / "representative_cases.json")
    outputs = [
        _video_baseline_vs_adaptive(config, videos_dir / "01_baseline_vs_adaptive.mp4", reps["video_01"]),
        _video_adaptive_single(config, videos_dir / "02_adaptive_rollout_single_episode.mp4", reps["video_02"]),
        _video_shift_showcase(config, videos_dir / "03_dynamics_shift_showcase.mp4", reps["video_03"]),
        _video_unseen_generalization(config, videos_dir / "04_unseen_trajectory_generalization.mp4", reps["video_04"]),
        _video_failure_to_recovery(config, videos_dir / "05_failure_to_recovery.mp4", reps["video_05"]),
    ]
    if config["videos"]["make_gif_previews"]:
        _make_gif_preview(videos_dir / "01_baseline_vs_adaptive.gif", outputs[0], config)
        _make_gif_preview(videos_dir / "05_failure_to_recovery.gif", outputs[-1], config)
    return outputs


def _video_baseline_vs_adaptive(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    baseline = _load_rollout(config, episode_id, "baseline")
    adaptive = _load_rollout(config, episode_id, "adaptive")
    return _render_pair_video(config, path, baseline, adaptive, "Baseline vs Adaptive")


def _video_failure_to_recovery(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    baseline = _load_rollout(config, episode_id, "baseline")
    adaptive = _load_rollout(config, episode_id, "adaptive")
    return _render_pair_video(config, path, baseline, adaptive, "Failure to Recovery")


def _video_adaptive_single(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    rollout = _load_rollout(config, episode_id, "adaptive")
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_mp4_writer(path, fps)
    limits = _trajectory_limits([rollout])
    shift_time = float(json.loads(rollout["metadata_json"].item())["shift_time"])
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, rollout)
        fig.clf()
        ax_traj = fig.add_axes([0.05, 0.08, 0.56, 0.84])
        ax_err = fig.add_axes([0.68, 0.58, 0.27, 0.27])
        ax_u = fig.add_axes([0.68, 0.18, 0.27, 0.27])
        _draw_single_panel(ax_traj, rollout, sim_idx, limits, "Adaptive controller")
        error = np.linalg.norm(rollout["ref_position"] - rollout["true_state"][:, :2], axis=1)
        ax_err.plot(rollout["time"][: sim_idx + 1], error[: sim_idx + 1], color=COLORS["adaptive"], linewidth=2.0)
        ax_err.axvline(shift_time, color=COLORS["accent"], linestyle="--")
        ax_err.set_title("Tracking error")
        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error [m]")
        control_norm = np.linalg.norm(rollout["command"], axis=1)
        ax_u.plot(rollout["time"][: sim_idx + 1], control_norm[: sim_idx + 1], color=COLORS["adaptive"], linewidth=2.0)
        ax_u.axvline(shift_time, color=COLORS["accent"], linestyle="--")
        ax_u.set_title("Control norm")
        ax_u.set_xlabel("Time [s]")
        ax_u.set_ylabel("||u||")
        fig.suptitle("Adaptive rollout under changing dynamics", y=0.98)
        writer.append_data(_canvas_to_image(fig))
    writer.close()
    plt.close(fig)
    return path


def _video_shift_showcase(config: dict[str, Any], path: Path, mapping: dict[str, str]) -> Path:
    rollouts = {shift: _load_rollout(config, episode_id, "adaptive") for shift, episode_id in mapping.items()}
    fps = int(config["videos"]["fps"])
    reference_rollout = next(iter(rollouts.values()))
    frame_count = int(np.ceil(float(reference_rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_mp4_writer(path, fps)
    limits = _trajectory_limits(list(rollouts.values()))
    for frame_idx in range(frame_count):
        fig.clf()
        for axis_index, (shift_type, rollout) in enumerate(rollouts.items()):
            sim_idx = _frame_to_index(frame_idx, fps, rollout)
            row = axis_index // 2
            col = axis_index % 2
            ax = fig.add_axes([0.06 + col * 0.47, 0.55 - row * 0.45, 0.4, 0.34])
            _draw_single_panel(ax, rollout, sim_idx, limits, shift_type.replace("_", " ").title())
        fig.suptitle("Dynamics shift showcase", y=0.98)
        writer.append_data(_canvas_to_image(fig))
    writer.close()
    plt.close(fig)
    return path


def _video_unseen_generalization(config: dict[str, Any], path: Path, episode_id: str) -> Path:
    rollout = _load_rollout(config, episode_id, "adaptive")
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_mp4_writer(path, fps)
    limits = _trajectory_limits([rollout])
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, rollout)
        fig.clf()
        ax_traj = fig.add_axes([0.05, 0.1, 0.58, 0.8])
        ax_text = fig.add_axes([0.68, 0.15, 0.26, 0.72])
        ax_text.axis("off")
        _draw_single_panel(ax_traj, rollout, sim_idx, limits, "Unseen trajectory")
        metadata = json.loads(rollout["metadata_json"].item())
        current_error = np.linalg.norm(rollout["ref_position"][sim_idx] - rollout["true_state"][sim_idx, :2])
        ax_text.text(
            0.02,
            0.88,
            "\n".join(
                [
                    f"Trajectory: {metadata['trajectory_kind']}",
                    f"Shift: {metadata['shift_type']}",
                    f"Intensity: {metadata['shift_intensity']}",
                    f"Unseen: {bool(metadata['unseen'])}",
                    f"Time: {rollout['time'][sim_idx]:.2f} s",
                    f"Error: {current_error:.3f} m",
                ]
            ),
            fontsize=18,
            va="top",
        )
        fig.suptitle("Generalization on unseen trajectory variants", y=0.98)
        writer.append_data(_canvas_to_image(fig))
    writer.close()
    plt.close(fig)
    return path


def _render_pair_video(
    config: dict[str, Any],
    path: Path,
    baseline: dict[str, Any],
    adaptive: dict[str, Any],
    title: str,
) -> Path:
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(baseline["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_mp4_writer(path, fps)
    limits = _trajectory_limits([baseline, adaptive])
    shift_time = float(json.loads(baseline["metadata_json"].item())["shift_time"])
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, baseline)
        fig.clf()
        ax_l = fig.add_axes([0.05, 0.1, 0.4, 0.8])
        ax_r = fig.add_axes([0.55, 0.1, 0.4, 0.8])
        ax_e = fig.add_axes([0.2, 0.78, 0.6, 0.15])
        _draw_single_panel(ax_l, baseline, sim_idx, limits, "Baseline", COLORS["baseline"])
        _draw_single_panel(ax_r, adaptive, sim_idx, limits, "Adaptive", COLORS["adaptive"])
        error_base = np.linalg.norm(baseline["ref_position"] - baseline["true_state"][:, :2], axis=1)
        error_adapt = np.linalg.norm(adaptive["ref_position"] - adaptive["true_state"][:, :2], axis=1)
        ax_e.plot(baseline["time"][: sim_idx + 1], error_base[: sim_idx + 1], color=COLORS["baseline"], linewidth=1.7)
        ax_e.plot(adaptive["time"][: sim_idx + 1], error_adapt[: sim_idx + 1], color=COLORS["adaptive"], linewidth=1.7)
        ax_e.axvline(shift_time, color=COLORS["accent"], linestyle="--")
        ax_e.set_ylabel("Error [m]")
        ax_e.set_xlabel("Time [s]")
        ax_e.set_title(title)
        writer.append_data(_canvas_to_image(fig))
    writer.close()
    plt.close(fig)
    return path


def _draw_single_panel(
    ax: plt.Axes,
    rollout: dict[str, Any],
    sim_idx: int,
    limits: tuple[float, float, float, float],
    title: str,
    color: str = COLORS["adaptive"],
) -> None:
    ax.plot(rollout["ref_position"][:, 0], rollout["ref_position"][:, 1], color=COLORS["reference"], linewidth=1.6, alpha=0.55)
    ax.plot(rollout["true_state"][: sim_idx + 1, 0], rollout["true_state"][: sim_idx + 1, 1], color=color, linewidth=2.2)
    ax.scatter(rollout["true_state"][sim_idx, 0], rollout["true_state"][sim_idx, 1], color=COLORS["accent"], s=70, zorder=5)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _trajectory_limits(rollouts: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = np.concatenate([np.concatenate([r["ref_position"][:, 0], r["true_state"][:, 0]]) for r in rollouts])
    ys = np.concatenate([np.concatenate([r["ref_position"][:, 1], r["true_state"][:, 1]]) for r in rollouts])
    margin = 0.5
    return xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin


def _frame_to_index(frame_idx: int, fps: int, rollout: dict[str, Any]) -> int:
    time_value = frame_idx / float(fps)
    return int(np.clip(np.searchsorted(rollout["time"], time_value), 0, len(rollout["time"]) - 1))


def _canvas_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]


def _make_mp4_writer(path: Path, fps: int) -> Any:
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def _make_gif_preview(path: Path, source_video: Path, config: dict[str, Any]) -> None:
    reader = imageio.get_reader(source_video)
    frames = []
    stride = int(config["videos"]["gif_stride"])
    for index, frame in enumerate(reader):
        if index % stride == 0:
            frames.append(frame)
    reader.close()
    imageio.mimsave(path, frames, fps=max(1, int(config["videos"]["fps"]) // stride))


def _load_rollout(config: dict[str, Any], episode_id: str, controller: str) -> dict[str, Any]:
    return load_npz(get_output_dir(config, "metrics", "rollouts") / f"{episode_id}__{controller}.npz")
