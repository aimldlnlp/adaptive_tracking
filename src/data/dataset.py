from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.controllers.baseline import BaselineController
from src.dynamics.robot import EpisodeSimulator, sample_episode_spec
from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, save_json, save_npz
from src.utils.logging_utils import ProgressCallback
from src.utils.math_utils import padded_stack, safe_heading_from_velocity, wrap_angle


TARGET_NAMES = [
    "mass_ratio",
    "friction_ratio",
    "delay_severity",
    "disturbance_x",
    "disturbance_y",
]
HISTORY_FEATURE_DIM = 17
CONTEXT_FEATURE_DIM = 12


@dataclass
class FeatureBuilder:
    history_steps: int
    dt: float

    def __post_init__(self) -> None:
        self.history_feature_dim = HISTORY_FEATURE_DIM
        self.context_feature_dim = CONTEXT_FEATURE_DIM
        self.reset()

    def reset(self) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.prev_obs_state: np.ndarray | None = None

    def build(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.build_inputs(*args, **kwargs)["flat_features"]

    def build_inputs(
        self,
        obs_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        ref_next: dict[str, np.ndarray | float],
        baseline_command: np.ndarray,
        prev_command: np.ndarray,
    ) -> dict[str, np.ndarray]:
        history_record = self._record(obs_state, ref_current, baseline_command, prev_command)
        history_sequence = padded_stack(
            (*self.history, history_record),
            self.history_steps,
            self.history_feature_dim,
        ).astype(np.float32)
        context_features = np.concatenate(
            [
                np.asarray(ref_current["position"], dtype=np.float32),
                np.asarray(ref_current["velocity"], dtype=np.float32),
                np.asarray(ref_current["acceleration"], dtype=np.float32),
                np.asarray(ref_next["velocity"], dtype=np.float32),
                np.asarray(ref_next["acceleration"], dtype=np.float32),
                np.array(
                    [
                        float(ref_current["heading"]),
                        float(ref_next["heading"]),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        flat_features = np.concatenate([history_sequence.reshape(-1), context_features], axis=0).astype(np.float32)
        return {
            "flat_features": flat_features,
            "sequence_features": history_sequence,
            "context_features": context_features,
        }

    def push(
        self,
        obs_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        baseline_command: np.ndarray,
        prev_command: np.ndarray,
    ) -> None:
        self.history.append(self._record(obs_state, ref_current, baseline_command, prev_command))
        self.prev_obs_state = obs_state.copy()

    def _record(
        self,
        obs_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        baseline_command: np.ndarray,
        prev_command: np.ndarray,
    ) -> np.ndarray:
        error_pos = np.asarray(ref_current["position"], dtype=np.float32) - obs_state[:2]
        error_vel = np.asarray(ref_current["velocity"], dtype=np.float32) - obs_state[2:4]
        accel_est = np.zeros(2, dtype=np.float32)
        if self.prev_obs_state is not None:
            accel_est = (obs_state[2:4] - self.prev_obs_state[2:4]) / self.dt
        obs_heading = safe_heading_from_velocity(obs_state[2:4], float(ref_current["heading"]))
        heading_error = wrap_angle(float(ref_current["heading"]) - obs_heading)
        return np.concatenate(
            [
                obs_state.astype(np.float32),
                error_pos.astype(np.float32),
                error_vel.astype(np.float32),
                baseline_command.astype(np.float32),
                prev_command.astype(np.float32),
                accel_est.astype(np.float32),
                (baseline_command - prev_command).astype(np.float32),
                np.array([heading_error], dtype=np.float32),
            ],
            axis=0,
        )


def get_feature_dim(history_steps: int) -> int:
    return history_steps * HISTORY_FEATURE_DIM + CONTEXT_FEATURE_DIM


def rollout_to_supervised_samples(
    rollout: dict[str, Any],
    feature_builder: FeatureBuilder,
) -> dict[str, np.ndarray]:
    feature_builder.reset()
    flat_features: list[np.ndarray] = []
    sequence_features: list[np.ndarray] = []
    context_features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prev_command = np.zeros(2, dtype=np.float32)
    for index in range(len(rollout["time"])):
        ref_current = {
            "position": rollout["ref_position"][index],
            "velocity": rollout["ref_velocity"][index],
            "acceleration": rollout["ref_acceleration"][index],
            "heading": float(rollout["ref_heading"][index]),
        }
        next_index = min(index + 1, len(rollout["time"]) - 1)
        ref_next = {
            "position": rollout["ref_position"][next_index],
            "velocity": rollout["ref_velocity"][next_index],
            "acceleration": rollout["ref_acceleration"][next_index],
            "heading": float(rollout["ref_heading"][next_index]),
        }
        baseline_command = rollout["baseline_command"][index]
        obs_state = rollout["observed_state"][index]
        inputs = feature_builder.build_inputs(obs_state, ref_current, ref_next, baseline_command, prev_command)
        target = np.array(
            [
                rollout["mass_ratio"][index],
                rollout["friction_ratio"][index],
                rollout["delay_severity"][index],
                rollout["disturbance_force"][index, 0],
                rollout["disturbance_force"][index, 1],
            ],
            dtype=np.float32,
        )
        flat_features.append(inputs["flat_features"])
        sequence_features.append(inputs["sequence_features"])
        context_features.append(inputs["context_features"])
        targets.append(target)
        feature_builder.push(obs_state, ref_current, baseline_command, prev_command)
        prev_command = baseline_command.astype(np.float32)
    return {
        "features": np.asarray(flat_features, dtype=np.float32),
        "sequence_features": np.asarray(sequence_features, dtype=np.float32),
        "context_features": np.asarray(context_features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
    }


def generate_datasets(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    simulation_cfg = config["simulation"]
    output_dir = get_output_dir(config, "datasets")
    ensure_dir(output_dir)
    simulator = EpisodeSimulator(config)
    baseline = BaselineController(config)
    feature_builder = FeatureBuilder(
        history_steps=int(simulation_cfg["history_steps"]),
        dt=float(simulation_cfg["dt"]),
    )

    dataset_paths: dict[str, Path] = {}
    specs_payload: dict[str, list[dict[str, Any]]] = {}
    total_episodes = (
        int(simulation_cfg["train_episodes"])
        + int(simulation_cfg["val_episodes"])
        + int(simulation_cfg["test_episodes"])
    )
    completed_episodes = 0
    if progress_callback is not None:
        progress_callback(0.0, "starting dataset generation")

    split_plan = [
        ("train", simulation_cfg["train_episodes"], simulation_cfg["train_seed"], float(simulation_cfg.get("train_unseen_fraction", 0.0))),
        ("val", simulation_cfg["val_episodes"], simulation_cfg["val_seed"], float(simulation_cfg.get("val_unseen_fraction", 0.15))),
        ("test", simulation_cfg["test_episodes"], simulation_cfg["test_seed"], float(simulation_cfg.get("test_unseen_fraction", 0.5))),
    ]
    for split, n_episodes, seed, unseen_fraction in split_plan:
        rng = np.random.default_rng(seed)
        split_flat_features: list[np.ndarray] = []
        split_sequence_features: list[np.ndarray] = []
        split_context_features: list[np.ndarray] = []
        split_targets: list[np.ndarray] = []
        split_specs: list[dict[str, Any]] = []
        iterator = tqdm(range(int(n_episodes)), desc=f"generate_{split}", leave=False)
        for episode_index in iterator:
            unseen = bool(rng.random() < unseen_fraction)
            spec = sample_episode_spec(config, split=split, rng=rng, episode_index=episode_index, unseen=unseen)
            rollout = simulator.simulate_episode(spec=spec, controller=baseline)
            samples = rollout_to_supervised_samples(rollout, feature_builder)
            split_flat_features.append(samples["features"])
            split_sequence_features.append(samples["sequence_features"])
            split_context_features.append(samples["context_features"])
            split_targets.append(samples["targets"])
            split_specs.append(spec)
            completed_episodes += 1
            if progress_callback is not None:
                progress_callback(
                    completed_episodes / max(total_episodes, 1),
                    f"{split} episode {episode_index + 1}/{n_episodes}",
                )

        dataset_path = output_dir / f"{split}_dataset.npz"
        save_npz(
            dataset_path,
            features=np.concatenate(split_flat_features, axis=0),
            sequence_features=np.concatenate(split_sequence_features, axis=0),
            context_features=np.concatenate(split_context_features, axis=0),
            targets=np.concatenate(split_targets, axis=0),
            feature_dim=np.array([get_feature_dim(int(simulation_cfg["history_steps"]))], dtype=np.int32),
            sequence_feature_dim=np.array([HISTORY_FEATURE_DIM], dtype=np.int32),
            context_feature_dim=np.array([CONTEXT_FEATURE_DIM], dtype=np.int32),
            target_names=np.asarray(TARGET_NAMES),
        )
        dataset_paths[split] = dataset_path
        specs_payload[split] = split_specs

    specs_path = output_dir / "episode_specs.json"
    save_json(specs_payload, specs_path)
    dataset_paths["specs"] = specs_path
    if progress_callback is not None:
        progress_callback(1.0, "dataset generation complete")
    return dataset_paths
