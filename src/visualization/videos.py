from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_json, load_npz
from src.utils.logging_utils import ProgressCallback
from src.visualization.style import (
    COLORS,
    apply_publication_style,
    controller_color,
    controller_soft_color,
    text_style,
)


def create_all_videos(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    apply_publication_style()
    videos_dir = get_output_dir(config, "videos")
    metrics_dir = get_output_dir(config, "metrics")
    ensure_dir(videos_dir)
    _cleanup_stale_video_outputs(videos_dir)
    reps = load_json(metrics_dir / "representative_cases.json")
    video_jobs: list[tuple[str, Callable[[], Path]]] = [
        (
            "baseline_vs_adaptive.gif",
            lambda: _render_pair_video(
                config,
                videos_dir / "baseline_vs_adaptive.gif",
                str(reps["primary_pair"]),
                "baseline",
                _primary_controller(config),
                progress_callback=_nested_video_progress(progress_callback, 0, 5, "baseline_vs_adaptive.gif"),
            ),
        ),
        (
            "dynamics_shift_showcase.gif",
            lambda: _video_shift_showcase(
                config,
                videos_dir / "dynamics_shift_showcase.gif",
                reps["showcase_single_shift"],
                progress_callback=_nested_video_progress(progress_callback, 1, 5, "dynamics_shift_showcase.gif"),
            ),
        ),
        (
            "unseen_trajectory_generalization.gif",
            lambda: _video_unseen(
                config,
                videos_dir / "unseen_trajectory_generalization.gif",
                str(reps["unseen_generalization"]),
                progress_callback=_nested_video_progress(progress_callback, 2, 5, "unseen_trajectory_generalization.gif"),
            ),
        ),
        (
            "compound_shift_showcase.gif",
            lambda: _render_pair_video(
                config,
                videos_dir / "compound_shift_showcase.gif",
                str(reps["compound_showcase"]),
                "baseline",
                _primary_controller(config),
                progress_callback=_nested_video_progress(progress_callback, 3, 5, "compound_shift_showcase.gif"),
            ),
        ),
        (
            "uncertainty_aware_recovery.gif",
            lambda: _video_uncertainty_recovery(
                config,
                videos_dir / "uncertainty_aware_recovery.gif",
                str(reps["uncertainty_recovery"]),
                progress_callback=_nested_video_progress(progress_callback, 4, 5, "uncertainty_aware_recovery.gif"),
            ),
        ),
    ]

    outputs: list[Path] = []
    if progress_callback is not None:
        progress_callback(0.0, "starting gif export")
    for index, (name, job) in enumerate(video_jobs, start=1):
        outputs.append(job())
        if progress_callback is not None:
            progress_callback(index / len(video_jobs), f"saved {name}")
    return outputs


def _nested_video_progress(
    progress_callback: ProgressCallback | None,
    video_index: int,
    total_videos: int,
    label: str,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None

    def callback(progress: float, message: str | None = None) -> None:
        bounded = max(0.0, min(1.0, float(progress)))
        progress_callback((video_index + bounded) / max(total_videos, 1), message or f"rendering {label}")

    return callback


def _cleanup_stale_video_outputs(videos_dir: Path) -> None:
    stale_names = [
        "adaptive_rollout_single_episode.gif",
        "failure_to_recovery.gif",
        "01_baseline_vs_adaptive.gif",
        "05_failure_to_recovery.gif",
    ]
    stale_paths = [videos_dir / name for name in stale_names]
    stale_paths.extend(sorted(videos_dir.glob("*.mp4")))
    for path in stale_paths:
        if path.exists():
            path.unlink()


def _render_pair_video(
    config: dict[str, Any],
    path: Path,
    episode_id: str,
    left_controller: str,
    right_controller: str,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    left = _load_rollout(config, episode_id, left_controller)
    right = _load_rollout(config, episode_id, right_controller)
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(left["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_gif_writer(path, fps)
    limits = _trajectory_limits([left, right])
    shift_time = float(json.loads(left["metadata_json"].item())["shift_time"])
    if progress_callback is not None:
        progress_callback(0.0, f"rendering {path.name}")
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, left)
        fig.clf()
        ax_left = fig.add_axes([0.05, 0.10, 0.38, 0.75])
        ax_right = fig.add_axes([0.57, 0.10, 0.38, 0.75])
        ax_error = fig.add_axes([0.20, 0.80, 0.60, 0.13])
        _draw_single_panel(ax_left, left, sim_idx, limits, _label_for_video(left_controller), controller_color(left_controller))
        _draw_single_panel(ax_right, right, sim_idx, limits, _label_for_video(right_controller), controller_color(right_controller))
        left_error = np.linalg.norm(left["ref_position"] - left["true_state"][:, :2], axis=1)
        right_error = np.linalg.norm(right["ref_position"] - right["true_state"][:, :2], axis=1)
        _style_video_axis(ax_error, COLORS["panel_alt"])
        _add_video_shift_band(ax_error, shift_time, float(left["time"][-1]))
        ax_error.fill_between(left["time"], left_error, right_error, color=COLORS["accent"], alpha=0.10)
        ax_error.plot(left["time"][: sim_idx + 1], left_error[: sim_idx + 1], color=controller_color(left_controller), linewidth=1.65)
        ax_error.plot(right["time"][: sim_idx + 1], right_error[: sim_idx + 1], color=controller_color(right_controller), linewidth=1.75)
        ax_error.set_ylabel("e", **text_style())
        ax_error.set_xlabel("t", **text_style())
        writer.append_data(_canvas_to_image(fig))
        _emit_frame_progress(progress_callback, frame_idx, frame_count, path.name)
    writer.close()
    plt.close(fig)
    return path


def _video_shift_showcase(
    config: dict[str, Any],
    path: Path,
    mapping: dict[str, str],
    progress_callback: ProgressCallback | None = None,
) -> Path:
    controller = _primary_controller(config)
    rollouts = {shift: _load_rollout(config, episode_id, controller) for shift, episode_id in mapping.items()}
    fps = int(config["videos"]["fps"])
    reference_rollout = next(iter(rollouts.values()))
    frame_count = int(np.ceil(float(reference_rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_gif_writer(path, fps)
    limits_by_shift = {shift: _trajectory_limits([rollout], margin=0.32) for shift, rollout in rollouts.items()}
    if progress_callback is not None:
        progress_callback(0.0, f"rendering {path.name}")
    for frame_idx in range(frame_count):
        fig.clf()
        for axis_index, (shift_type, rollout) in enumerate(rollouts.items()):
            sim_idx = _frame_to_index(frame_idx, fps, rollout)
            row = axis_index // 2
            col = axis_index % 2
            left = 0.05 + col * 0.46
            bottom = 0.56 - row * 0.41
            ax = fig.add_axes([left, bottom, 0.40, 0.32])
            _draw_single_panel(
                ax,
                rollout,
                sim_idx,
                limits_by_shift[shift_type],
                _short_shift_label(shift_type),
                controller_color(controller),
                equal_aspect=False,
            )
        writer.append_data(_canvas_to_image(fig))
        _emit_frame_progress(progress_callback, frame_idx, frame_count, path.name)
    writer.close()
    plt.close(fig)
    return path


def _video_unseen(
    config: dict[str, Any],
    path: Path,
    episode_id: str,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    controller = _primary_controller(config)
    rollout = _load_rollout(config, episode_id, controller)
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_gif_writer(path, fps)
    limits = _trajectory_limits([rollout])
    metadata = json.loads(rollout["metadata_json"].item())
    shift_time = float(metadata["shift_time"])
    if progress_callback is not None:
        progress_callback(0.0, f"rendering {path.name}")
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, rollout)
        fig.clf()
        ax_traj = fig.add_axes([0.05, 0.10, 0.58, 0.78])
        ax_err = fig.add_axes([0.69, 0.56, 0.24, 0.22])
        ax_u = fig.add_axes([0.69, 0.24, 0.24, 0.22])
        _draw_single_panel(ax_traj, rollout, sim_idx, limits, "OOD", controller_color(controller))
        error = np.linalg.norm(rollout["ref_position"] - rollout["true_state"][:, :2], axis=1)
        control_norm = np.linalg.norm(rollout["command"], axis=1)
        _style_video_axis(ax_err, COLORS["panel_alt"])
        _add_video_shift_band(ax_err, shift_time, float(rollout["time"][-1]))
        ax_err.fill_between(rollout["time"][: sim_idx + 1], 0.0, error[: sim_idx + 1], color=controller_soft_color(controller), alpha=0.18)
        ax_err.plot(rollout["time"][: sim_idx + 1], error[: sim_idx + 1], color=controller_color(controller), linewidth=1.9)
        ax_err.set_title("e", **text_style())
        _style_video_axis(ax_u, COLORS["panel"])
        _add_video_shift_band(ax_u, shift_time, float(rollout["time"][-1]))
        ax_u.fill_between(rollout["time"][: sim_idx + 1], 0.0, control_norm[: sim_idx + 1], color=COLORS["accent"], alpha=0.12)
        ax_u.plot(rollout["time"][: sim_idx + 1], control_norm[: sim_idx + 1], color=controller_color(controller), linewidth=1.9)
        ax_u.set_title("|u|", **text_style())
        writer.append_data(_canvas_to_image(fig))
        _emit_frame_progress(progress_callback, frame_idx, frame_count, path.name)
    writer.close()
    plt.close(fig)
    return path


def _video_uncertainty_recovery(
    config: dict[str, Any],
    path: Path,
    episode_id: str,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    controller = _primary_controller(config)
    rollout = _load_rollout(config, episode_id, controller)
    fps = int(config["videos"]["fps"])
    frame_count = int(np.ceil(float(rollout["time"][-1]) * fps))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=config["videos"]["dpi"])
    writer = _make_gif_writer(path, fps)
    limits = _trajectory_limits([rollout])
    metadata = json.loads(rollout["metadata_json"].item())
    shift_time = float(metadata["shift_time"])
    if progress_callback is not None:
        progress_callback(0.0, f"rendering {path.name}")
    for frame_idx in range(frame_count):
        sim_idx = _frame_to_index(frame_idx, fps, rollout)
        fig.clf()
        ax_traj = fig.add_axes([0.05, 0.12, 0.54, 0.74])
        ax_err = fig.add_axes([0.66, 0.62, 0.26, 0.18])
        ax_uq = fig.add_axes([0.66, 0.38, 0.26, 0.18])
        ax_gain = fig.add_axes([0.66, 0.14, 0.26, 0.18])
        _draw_single_panel(ax_traj, rollout, sim_idx, limits, "Recovery", controller_color(controller))
        error = np.linalg.norm(rollout["ref_position"] - rollout["true_state"][:, :2], axis=1)
        uncertainty = rollout["estimated_uncertainty"].reshape(-1)
        gain = rollout["correction_gain"].reshape(-1)
        for axis, values, facecolor, color, title in [
            (ax_err, error, COLORS["panel_alt"], controller_color(controller), "e"),
            (ax_uq, uncertainty, COLORS["panel"], COLORS["accent"], "uq"),
            (ax_gain, gain, COLORS["panel_alt"], controller_color(controller), "gain"),
        ]:
            _style_video_axis(axis, facecolor)
            _add_video_shift_band(axis, shift_time, float(rollout["time"][-1]))
            axis.fill_between(rollout["time"][: sim_idx + 1], 0.0, values[: sim_idx + 1], color=color, alpha=0.18)
            axis.plot(rollout["time"][: sim_idx + 1], values[: sim_idx + 1], color=color, linewidth=1.8)
            axis.set_title(title, **text_style())
        writer.append_data(_canvas_to_image(fig))
        _emit_frame_progress(progress_callback, frame_idx, frame_count, path.name)
    writer.close()
    plt.close(fig)
    return path


def _draw_single_panel(
    ax: plt.Axes,
    rollout: dict[str, Any],
    sim_idx: int,
    limits: tuple[float, float, float, float],
    title: str,
    color: str,
    equal_aspect: bool = True,
) -> None:
    _style_video_axis(ax, COLORS["panel"])
    ref = rollout["ref_position"]
    state = rollout["true_state"][:, :2]
    shift_time = float(json.loads(rollout["metadata_json"].item())["shift_time"])
    shift_idx = int(np.searchsorted(rollout["time"], shift_time))
    trail = state[max(0, sim_idx - 42) : sim_idx + 1]
    ax.plot(ref[:, 0], ref[:, 1], color=COLORS["reference_soft"], linewidth=3.5, alpha=0.22, zorder=1)
    ax.plot(ref[:, 0], ref[:, 1], color=COLORS["reference"], linewidth=1.5, alpha=0.42, zorder=2)
    ax.plot(state[: sim_idx + 1, 0], state[: sim_idx + 1, 1], color=color, linewidth=4.4, alpha=0.16, zorder=2)
    ax.plot(state[: sim_idx + 1, 0], state[: sim_idx + 1, 1], color=color, linewidth=2.25, zorder=3)
    if len(trail) > 3:
        ax.scatter(trail[:, 0], trail[:, 1], c=np.linspace(0.15, 0.85, len(trail)), cmap="cividis", s=11, alpha=0.28, edgecolors="none", zorder=3)
    ax.scatter(ref[0, 0], ref[0, 1], color=COLORS["reference"], s=24, edgecolors="none", alpha=0.9, zorder=4)
    ax.scatter(ref[shift_idx, 0], ref[shift_idx, 1], color=COLORS["accent"], s=38, edgecolors=COLORS["paper"], linewidths=0.8, zorder=5)
    ax.scatter(state[sim_idx, 0], state[sim_idx, 1], color=COLORS["accent_alt"], s=74, edgecolors=COLORS["paper"], linewidths=0.8, zorder=6)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    if equal_aspect:
        ax.set_aspect("equal")
    ax.set_title(title, **text_style())
    ax.set_xlabel("x", **text_style())
    ax.set_ylabel("y", **text_style())


def _style_video_axis(ax: plt.Axes, facecolor: str) -> None:
    ax.set_facecolor(facecolor)
    ax.grid(True, alpha=0.28)


def _add_video_shift_band(ax: plt.Axes, shift_time: float, end_time: float) -> None:
    ax.axvspan(shift_time, end_time, color=COLORS["post_shift"], alpha=0.18, zorder=0)
    ax.axvline(shift_time, color=COLORS["accent"], linestyle=(0, (4, 2)), linewidth=1.2)


def _emit_frame_progress(
    progress_callback: ProgressCallback | None,
    frame_idx: int,
    frame_count: int,
    label: str,
) -> None:
    if progress_callback is None:
        return
    interval = max(1, frame_count // 10)
    if (frame_idx + 1) % interval == 0 or frame_idx + 1 == frame_count:
        progress_callback((frame_idx + 1) / max(frame_count, 1), f"{label} frame {frame_idx + 1}/{frame_count}")


def _trajectory_limits(rollouts: list[dict[str, Any]], margin: float = 0.5) -> tuple[float, float, float, float]:
    xs = np.concatenate([np.concatenate([rollout["ref_position"][:, 0], rollout["true_state"][:, 0]]) for rollout in rollouts])
    ys = np.concatenate([np.concatenate([rollout["ref_position"][:, 1], rollout["true_state"][:, 1]]) for rollout in rollouts])
    return xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin


def _frame_to_index(frame_idx: int, fps: int, rollout: dict[str, Any]) -> int:
    time_value = frame_idx / float(fps)
    return int(np.clip(np.searchsorted(rollout["time"], time_value), 0, len(rollout["time"]) - 1))


def _canvas_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]


def _make_gif_writer(path: Path, fps: int) -> Any:
    target = Path(path).resolve()
    if target.exists():
        target.unlink()
    return imageio.get_writer(target, format="GIF", mode="I", duration=1.0 / max(fps, 1), loop=0)


def _load_rollout(config: dict[str, Any], episode_id: str, controller: str) -> dict[str, Any]:
    return load_npz(get_output_dir(config, "metrics", "rollouts") / f"{episode_id}__{controller}.npz")


def _primary_controller(config: dict[str, Any]) -> str:
    return str(config.get("evaluation", {}).get("primary_controller", config.get("model", {}).get("name", "adaptive")))


def _label_for_video(controller: str) -> str:
    return {
        "baseline": "Base",
        "adaptive_mlp": "MLP",
        "adaptive_gru_nominal": "GRU-N",
        "adaptive_gru_uncertainty": "GRU-U",
    }.get(controller, controller.replace("_", " ").title())


def _short_shift_label(value: str) -> str:
    return {
        "friction_shift": "Friction",
        "mass_shift": "Mass",
        "actuator_delay": "Delay",
        "disturbance_burst": "Burst",
    }.get(value, value.replace("_", " ").title())
