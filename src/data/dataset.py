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
from src.utils.math_utils import padded_stack, safe_heading_from_velocity, wrap_angle


TARGET_NAMES = [
    "mass_ratio",
    "friction_ratio",
    "delay_severity",
    "disturbance_x",
    "disturbance_y",
]


@dataclass
class FeatureBuilder:
    history_steps: int
    dt: float

    def __post_init__(self) -> None:
        self.feature_dim = 17
        self.reset()

    def reset(self) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.prev_obs_state: np.ndarray | None = None

    def build(
        self,
        obs_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        ref_next: dict[str, np.ndarray | float],
        baseline_command: np.ndarray,
        prev_command: np.ndarray,
    ) -> np.ndarray:
        record = self._record(obs_state, ref_current, baseline_command, prev_command)
        history_block = padded_stack((*self.history, record), self.history_steps, self.feature_dim).reshape(-1)
        ref_block = np.concatenate(
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
        )
        return np.concatenate([history_block, ref_block], axis=0).astype(np.float32)

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
    return history_steps * 17 + 12


def rollout_to_supervised_samples(
    rollout: dict[str, Any],
    feature_builder: FeatureBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    feature_builder.reset()
    features: list[np.ndarray] = []
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
        feature = feature_builder.build(obs_state, ref_current, ref_next, baseline_command, prev_command)
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
        features.append(feature)
        targets.append(target)
        feature_builder.push(obs_state, ref_current, baseline_command, prev_command)
        prev_command = baseline_command.astype(np.float32)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def generate_datasets(config: dict[str, Any]) -> dict[str, Path]:
    simulation_cfg = config["simulation"]
    output_dir = get_output_dir(config, "datasets")
    ensure_dir(output_dir)
    simulator = EpisodeSimulator(config)
    baseline = BaselineController(config)
    feature_builder = FeatureBuilder(
        history_steps=simulation_cfg["history_steps"],
        dt=simulation_cfg["dt"],
    )

    dataset_paths: dict[str, Path] = {}
    specs_payload: dict[str, list[dict[str, Any]]] = {}

    for split, n_episodes, seed, unseen_fraction in [
        ("train", simulation_cfg["train_episodes"], simulation_cfg["train_seed"], 0.0),
        ("val", simulation_cfg["val_episodes"], simulation_cfg["val_seed"], 0.15),
        ("test", simulation_cfg["test_episodes"], simulation_cfg["test_seed"], 0.5),
    ]:
        rng = np.random.default_rng(seed)
        split_features: list[np.ndarray] = []
        split_targets: list[np.ndarray] = []
        split_specs: list[dict[str, Any]] = []
        iterator = tqdm(range(n_episodes), desc=f"generate_{split}", leave=False)
        for episode_index in iterator:
            unseen = bool(rng.random() < unseen_fraction)
            spec = sample_episode_spec(config, split=split, rng=rng, episode_index=episode_index, unseen=unseen)
            rollout = simulator.simulate_episode(spec=spec, controller=baseline)
            features, targets = rollout_to_supervised_samples(rollout, feature_builder)
            split_features.append(features)
            split_targets.append(targets)
            split_specs.append(spec)

        dataset_path = output_dir / f"{split}_dataset.npz"
        save_npz(
            dataset_path,
            features=np.concatenate(split_features, axis=0),
            targets=np.concatenate(split_targets, axis=0),
            feature_dim=np.array([get_feature_dim(simulation_cfg["history_steps"])], dtype=np.int32),
            target_names=np.asarray(TARGET_NAMES),
        )
        dataset_paths[split] = dataset_path
        specs_payload[split] = split_specs

    specs_path = output_dir / "episode_specs.json"
    save_json(specs_payload, specs_path)
    dataset_paths["specs"] = specs_path
    return dataset_paths
