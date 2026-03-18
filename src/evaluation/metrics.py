from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.math_utils import moving_average, safe_heading_from_velocity, wrap_angle


def compute_episode_metrics(config: dict[str, Any], rollout: dict[str, Any]) -> dict[str, float]:
    sim_cfg = config["simulation"]
    position_error = np.linalg.norm(rollout["ref_position"] - rollout["true_state"][:, :2], axis=1)
    rmse = float(np.sqrt(np.mean(position_error**2)))
    mae = float(np.mean(np.abs(position_error)))
    final_position_error = float(position_error[-1])
    headings = np.array(
        [
            safe_heading_from_velocity(rollout["true_state"][i, 2:4], float(rollout["ref_heading"][i]))
            for i in range(len(rollout["time"]))
        ],
        dtype=np.float32,
    )
    heading_error = float(np.mean(np.abs(wrap_angle(rollout["ref_heading"] - headings))))
    success = float(
        rmse <= float(sim_cfg["success_rmse_threshold"])
        and final_position_error <= float(sim_cfg["success_final_error_threshold"])
    )
    shift_time = float(json_metadata(rollout)["shift_time"])
    shift_index = int(np.searchsorted(rollout["time"], shift_time))
    recovery_time = _compute_recovery_time(
        errors=position_error,
        time=rollout["time"],
        shift_index=shift_index,
        threshold=float(sim_cfg["recovery_error_threshold"]),
        consecutive_steps=int(sim_cfg["recovery_consecutive_steps"]),
    )
    control_diff = np.diff(rollout["command"], axis=0)
    control_smoothness = float(np.mean(np.linalg.norm(control_diff, axis=1))) if len(control_diff) else 0.0
    control_energy_proxy = float(np.mean(np.sum(rollout["command"] ** 2, axis=1)))
    return {
        "rmse": rmse,
        "mae": mae,
        "final_position_error": final_position_error,
        "heading_error": heading_error,
        "success": success,
        "recovery_time": recovery_time,
        "control_smoothness": control_smoothness,
        "control_energy_proxy": control_energy_proxy,
        "peak_error_after_shift": float(np.max(position_error[shift_index:])),
        "mean_error_after_shift": float(np.mean(position_error[shift_index:])),
        "robustness_score": float(np.exp(-rmse) * (0.6 + 0.4 * success)),
    }


def aggregate_metrics(metrics_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        metrics_frame.groupby(["controller", "shift_type", "shift_intensity"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            final_position_error_mean=("final_position_error", "mean"),
            heading_error_mean=("heading_error", "mean"),
            success_rate=("success", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            control_energy_proxy_mean=("control_energy_proxy", "mean"),
            robustness_score_mean=("robustness_score", "mean"),
        )
        .fillna(0.0)
    )
    summary = (
        metrics_frame.groupby(["controller"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            final_position_error_mean=("final_position_error", "mean"),
            heading_error_mean=("heading_error", "mean"),
            success_rate=("success", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            control_energy_proxy_mean=("control_energy_proxy", "mean"),
            robustness_score_mean=("robustness_score", "mean"),
        )
        .fillna(0.0)
    )
    return aggregate, summary


def json_metadata(rollout: dict[str, Any]) -> dict[str, Any]:
    raw = rollout["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    import json

    return json.loads(raw)


def aligned_error_curve(rollout: dict[str, Any], horizon: tuple[float, float], step: float) -> tuple[np.ndarray, np.ndarray]:
    metadata = json_metadata(rollout)
    shift_time = float(metadata["shift_time"])
    relative_time = rollout["time"] - shift_time
    error = np.linalg.norm(rollout["ref_position"] - rollout["true_state"][:, :2], axis=1)
    smoothed = moving_average(error, window=3)
    grid = np.arange(horizon[0], horizon[1] + 1e-9, step)
    curve = np.interp(grid, relative_time, smoothed, left=smoothed[0], right=smoothed[-1])
    return grid, curve


def _compute_recovery_time(
    errors: np.ndarray,
    time: np.ndarray,
    shift_index: int,
    threshold: float,
    consecutive_steps: int,
) -> float:
    if shift_index >= len(errors):
        return 0.0
    count = 0
    for index in range(shift_index, len(errors)):
        if errors[index] <= threshold:
            count += 1
        else:
            count = 0
        if count >= consecutive_steps:
            return float(time[index] - time[shift_index])
    return float(time[-1] - time[shift_index])
