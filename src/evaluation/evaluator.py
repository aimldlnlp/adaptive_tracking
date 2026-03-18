from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.controllers.adaptive import AdaptiveController
from src.controllers.baseline import BaselineController
from src.dynamics.robot import EpisodeSimulator, sample_episode_spec
from src.evaluation.metrics import aggregate_metrics, compute_episode_metrics
from src.models.adaptive_estimator import load_checkpoint_bundle, resolve_device
from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, save_dataframe, save_json, save_npz


LOGGER = logging.getLogger(__name__)


def evaluate_from_config(config: dict[str, Any], checkpoint_path: str | Path | None = None) -> pd.DataFrame:
    metrics_dir = get_output_dir(config, "metrics")
    rollouts_dir = metrics_dir / "rollouts"
    ensure_dir(rollouts_dir)
    evaluation_specs = _build_evaluation_specs(config)

    simulator = EpisodeSimulator(config)
    baseline_controller = BaselineController(config)
    device = resolve_device(config)
    checkpoint = checkpoint_path or get_output_dir(config, "checkpoints") / "best_model.pt"
    adaptive_controller = AdaptiveController(config, load_checkpoint_bundle(str(checkpoint), device))

    metrics_rows: list[dict[str, Any]] = []
    for spec in tqdm(evaluation_specs, desc="evaluate", leave=False):
        for controller_name, controller in [
            ("baseline", baseline_controller),
            ("adaptive", adaptive_controller),
        ]:
            rollout = simulator.simulate_episode(spec=spec, controller=controller)
            metrics = compute_episode_metrics(config, rollout)
            row = {
                "episode_id": spec["episode_id"],
                "controller": controller_name,
                "trajectory_kind": spec["trajectory_kind"],
                "shift_type": spec["shift_type"],
                "shift_intensity": spec["shift_intensity"],
                "shift_time": spec["shift_time"],
                "unseen": int(spec["unseen"]),
                **metrics,
            }
            metrics_rows.append(row)
            payload = {key: value for key, value in rollout.items() if key != "metadata_json"}
            payload["metadata_json"] = np.array(json.dumps({**spec, "controller": controller_name}))
            save_npz(rollouts_dir / f"{spec['episode_id']}__{controller_name}.npz", **payload)

    metrics_frame = pd.DataFrame(metrics_rows)
    aggregate_frame, summary_frame = aggregate_metrics(metrics_frame)
    representatives = _select_representative_cases(metrics_frame)
    save_dataframe(metrics_frame, metrics_dir / "per_episode_metrics.csv")
    save_dataframe(aggregate_frame, metrics_dir / "aggregate_metrics.csv")
    save_dataframe(summary_frame, metrics_dir / "controller_summary.csv")
    save_json(evaluation_specs, metrics_dir / "evaluation_specs.json")
    save_json(representatives, metrics_dir / "representative_cases.json")
    LOGGER.info("saved_metrics=%s", metrics_dir)
    return metrics_frame


def _build_evaluation_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation_cfg = config["evaluation"]
    shift_types = list(config["simulation"]["shift_types"])
    intensities = list(config["simulation"]["shift_intensities"].keys())
    specs: list[dict[str, Any]] = []
    for seed in evaluation_cfg["seeds"]:
        for shift_index, shift_type in enumerate(shift_types):
            for intensity_index, intensity in enumerate(intensities):
                for episode_idx in range(int(evaluation_cfg["episodes_per_condition"])):
                    spec_seed = int(seed * 10_000 + shift_index * 1_000 + intensity_index * 100 + episode_idx)
                    rng = np.random.default_rng(spec_seed)
                    unseen = bool(episode_idx < max(1, round(evaluation_cfg["unseen_fraction"] * evaluation_cfg["episodes_per_condition"])))
                    spec = sample_episode_spec(
                        config,
                        split="test",
                        rng=rng,
                        episode_index=episode_idx,
                        unseen=unseen,
                        forced_shift_type=shift_type,
                        forced_intensity=intensity,
                    )
                    spec["episode_id"] = f"eval_{seed}_{shift_type}_{intensity}_{episode_idx:02d}"
                    specs.append(spec)
    return specs


def _select_representative_cases(metrics_frame: pd.DataFrame) -> dict[str, Any]:
    baseline = metrics_frame[metrics_frame["controller"] == "baseline"].set_index("episode_id")
    adaptive = metrics_frame[metrics_frame["controller"] == "adaptive"].set_index("episode_id")
    paired = baseline[
        [
            "trajectory_kind",
            "shift_type",
            "shift_intensity",
            "unseen",
            "rmse",
            "success",
        ]
    ].join(
        adaptive[["rmse", "success"]],
        lsuffix="_baseline",
        rsuffix="_adaptive",
    )
    paired["improvement"] = paired["rmse_baseline"] - paired["rmse_adaptive"]

    def choose(frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> str:
        if frame.empty:
            return str(paired.sort_values(["improvement"], ascending=[False]).index[0])
        return str(frame.sort_values(sort_cols, ascending=ascending).index[0])

    failure_recovery = paired[(paired["success_baseline"] < 0.5) & (paired["success_adaptive"] > 0.5)]
    general = paired[paired["unseen"] < 0.5]
    unseen = paired[paired["unseen"] > 0.5]

    showcase: dict[str, str] = {}
    for shift_type in paired["shift_type"].unique():
        subset = paired[(paired["shift_type"] == shift_type) & (paired["shift_intensity"] == "medium")]
        showcase[shift_type] = choose(subset, ["rmse_adaptive", "improvement"], [True, False])

    return {
        "trajectory_comparison": choose(general, ["improvement", "rmse_adaptive"], [False, True]),
        "control_signal": choose(general, ["improvement", "rmse_adaptive"], [False, True]),
        "video_01": choose(general, ["improvement", "rmse_adaptive"], [False, True]),
        "video_02": choose(general, ["rmse_adaptive", "improvement"], [True, False]),
        "video_03": showcase,
        "video_04": choose(unseen, ["rmse_adaptive"], [True]),
        "video_05": choose(failure_recovery, ["improvement"], [False]),
    }
