from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.math_utils import clip_vector_norm


class BaselineController:
    def __init__(self, config: dict[str, Any]) -> None:
        controller_cfg = config["controller"]
        sim_cfg = config["simulation"]
        self.kp = float(controller_cfg["kp"])
        self.kd = float(controller_cfg["kd"])
        self.ki = float(controller_cfg["ki"])
        self.integral_limit = float(controller_cfg["integral_limit"])
        self.nominal_mass = float(sim_cfg["nominal_mass"])
        self.nominal_friction = float(sim_cfg["nominal_friction"])
        self.control_limit = float(sim_cfg["control_limit"])
        self.integral_error = np.zeros(2, dtype=np.float32)
        self.dt = float(sim_cfg["dt"])

    def reset(self) -> None:
        self.integral_error = np.zeros(2, dtype=np.float32)

    def compute_control(
        self,
        observed_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        ref_next: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        position_error = np.asarray(ref_current["position"], dtype=np.float32) - observed_state[:2]
        velocity_error = np.asarray(ref_current["velocity"], dtype=np.float32) - observed_state[2:4]
        self.integral_error += position_error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        desired_acceleration = (
            np.asarray(ref_current["acceleration"], dtype=np.float32)
            + self.kp * position_error
            + self.kd * velocity_error
            + self.ki * self.integral_error
        )
        command = self.nominal_mass * desired_acceleration + self.nominal_friction * observed_state[2:4]
        command = clip_vector_norm(command.astype(np.float32), self.control_limit)
        return command, {
            "desired_acceleration": desired_acceleration.astype(np.float32),
            "baseline_command": command.astype(np.float32),
        }
