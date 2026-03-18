from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.controllers.baseline import BaselineController
from src.data.dataset import FeatureBuilder
from src.utils.math_utils import clip_vector_norm


class AdaptiveController:
    def __init__(self, config: dict[str, Any], checkpoint_bundle: dict[str, Any]) -> None:
        self.config = config
        self.baseline = BaselineController(config)
        sim_cfg = config["simulation"]
        controller_cfg = config["controller"]
        self.control_limit = float(sim_cfg["control_limit"])
        self.nominal_mass = float(sim_cfg["nominal_mass"])
        self.nominal_friction = float(sim_cfg["nominal_friction"])
        self.delay_lead_gain = float(controller_cfg["delay_lead_gain"])
        self.correction_blend = float(controller_cfg["correction_blend"])
        self.disturbance_blend = float(controller_cfg["disturbance_blend"])
        self.mass_ratio_clip = np.asarray(controller_cfg["mass_ratio_clip"], dtype=np.float32)
        self.friction_ratio_clip = np.asarray(controller_cfg["friction_ratio_clip"], dtype=np.float32)
        self.feature_builder = FeatureBuilder(
            history_steps=int(sim_cfg["history_steps"]),
            dt=float(sim_cfg["dt"]),
        )
        self.bundle = checkpoint_bundle
        self.device = checkpoint_bundle["model"].parameters().__next__().device
        self.model = checkpoint_bundle["model"]
        self.prev_command = np.zeros(2, dtype=np.float32)
        self.filtered_targets = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self) -> None:
        self.baseline.reset()
        self.feature_builder.reset()
        self.prev_command = np.zeros(2, dtype=np.float32)
        self.filtered_targets = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def compute_control(
        self,
        observed_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        ref_next: dict[str, np.ndarray | float],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        baseline_command, baseline_aux = self.baseline.compute_control(observed_state, ref_current, ref_next)
        features = self.feature_builder.build(
            obs_state=observed_state,
            ref_current=ref_current,
            ref_next=ref_next,
            baseline_command=baseline_command,
            prev_command=self.prev_command,
        )
        normalized = (features - self.bundle["feature_mean"]) / self.bundle["feature_std"]
        input_tensor = torch.from_numpy(normalized[None, :]).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]
        prediction = prediction * self.bundle["target_std"] + self.bundle["target_mean"]
        self.filtered_targets[:3] = (1.0 - self.correction_blend) * self.filtered_targets[:3] + self.correction_blend * prediction[:3]
        self.filtered_targets[3:] = (1.0 - self.disturbance_blend) * self.filtered_targets[3:] + self.disturbance_blend * prediction[3:]

        mass_ratio = float(np.clip(self.filtered_targets[0], self.mass_ratio_clip[0], self.mass_ratio_clip[1]))
        friction_ratio = float(np.clip(self.filtered_targets[1], self.friction_ratio_clip[0], self.friction_ratio_clip[1]))
        delay_severity = float(np.clip(self.filtered_targets[2], 0.0, 1.0))
        disturbance_force = self.filtered_targets[3:].astype(np.float32)

        desired_acceleration = baseline_aux["desired_acceleration"]
        corrected = (
            self.nominal_mass * mass_ratio * desired_acceleration
            + self.nominal_friction * friction_ratio * observed_state[2:4]
            - disturbance_force
            + self.delay_lead_gain * delay_severity * (baseline_command - self.prev_command)
        )
        command = clip_vector_norm(corrected.astype(np.float32), self.control_limit)
        self.feature_builder.push(
            obs_state=observed_state,
            ref_current=ref_current,
            baseline_command=baseline_command,
            prev_command=self.prev_command,
        )
        self.prev_command = command.copy()
        return command, {
            "baseline_command": baseline_command.astype(np.float32),
            "estimated_targets": self.filtered_targets.astype(np.float32),
        }
