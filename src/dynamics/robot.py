from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

from src.data.trajectories import sample_trajectory_params, trajectory_from_spec
from src.utils.math_utils import clip_vector_norm


SHIFT_INTENSITY_TO_LEVEL = {
    "mild": 0.5,
    "medium": 0.8,
    "severe": 1.1,
}


def sample_episode_spec(
    config: dict[str, Any],
    split: str,
    rng: np.random.Generator,
    episode_index: int,
    unseen: bool = False,
    forced_shift_type: str | None = None,
    forced_intensity: str | None = None,
    forced_compound_shift_types: list[str] | None = None,
) -> dict[str, Any]:
    sim_cfg = config["simulation"]
    duration = float(sim_cfg["episode_duration"])
    shift_types = forced_compound_shift_types or [forced_shift_type or str(rng.choice(sim_cfg["shift_types"]))]
    shift_label = "+".join(shift_types)
    intensity = forced_intensity or str(rng.choice(list(sim_cfg["shift_intensities"].keys())))
    all_kinds = list(sim_cfg["trajectory_kinds"])
    holdout_kinds = list(sim_cfg.get("hard_unseen_holdout_trajectory_kinds", []))
    if split == "test" and unseen and holdout_kinds:
        candidate_kinds = holdout_kinds
    else:
        candidate_kinds = [kind for kind in all_kinds if kind not in holdout_kinds]
        if not candidate_kinds:
            candidate_kinds = all_kinds
    kind = str(rng.choice(candidate_kinds))
    shift_time = float(rng.uniform(0.28, 0.68) * duration)
    pre_mass_ratio = float(rng.uniform(0.9, 1.08))
    pre_friction_ratio = float(rng.uniform(0.8, 1.2))
    pre_delay_steps = 0
    pre_lag_alpha = float(rng.uniform(0.9, 1.0))
    post_mass_ratio = pre_mass_ratio
    post_friction_ratio = pre_friction_ratio
    post_delay_steps = pre_delay_steps
    post_lag_alpha = pre_lag_alpha
    disturbance_amplitude = 0.0
    disturbance_frequency = float(rng.uniform(1.2, 2.1))
    disturbance_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    burst_duration = float(rng.uniform(1.6, 2.6))

    level = SHIFT_INTENSITY_TO_LEVEL[intensity]
    for shift_type in shift_types:
        if shift_type == "friction_shift":
            post_friction_ratio = float(post_friction_ratio * (1.0 + 1.25 * level))
        elif shift_type == "mass_shift":
            post_mass_ratio = float(post_mass_ratio * (1.0 + 1.15 * level))
        elif shift_type == "actuator_delay":
            post_delay_steps = max(post_delay_steps, int(np.clip(round(1 + 2 * level), 1, 3)))
            post_lag_alpha = float(min(post_lag_alpha, max(0.25, 0.92 - 0.48 * level)))
        elif shift_type == "disturbance_burst":
            disturbance_amplitude = float(max(disturbance_amplitude, 1.0 + 1.8 * level))
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")

    trajectory_params = sample_trajectory_params(
        kind=kind,
        rng=rng,
        duration=duration,
        split=split,
        unseen=unseen,
    )
    initial_offset = rng.normal(0.0, 0.22 if split != "test" else 0.28, size=4).astype(np.float32)

    return {
        "episode_id": f"{split}_{episode_index:04d}_{kind}_{shift_label.replace('+', '-')}_{intensity}",
        "episode_seed": int(rng.integers(0, 2**31 - 1)),
        "split": split,
        "unseen": bool(unseen),
        "trajectory_kind": kind,
        "trajectory_params": trajectory_params,
        "shift_type": shift_label,
        "shift_types": shift_types,
        "compound_shift": bool(len(shift_types) > 1),
        "shift_intensity": intensity,
        "condition_group": (
            "compound_shift_ood"
            if len(shift_types) > 1
            else ("hard_unseen_trajectory" if unseen else "in_distribution_single_shift")
        ),
        "shift_time": shift_time,
        "burst_duration": burst_duration,
        "initial_offset": initial_offset.tolist(),
        "pre_mass_ratio": pre_mass_ratio,
        "post_mass_ratio": post_mass_ratio,
        "pre_friction_ratio": pre_friction_ratio,
        "post_friction_ratio": post_friction_ratio,
        "pre_delay_steps": pre_delay_steps,
        "post_delay_steps": post_delay_steps,
        "pre_lag_alpha": pre_lag_alpha,
        "post_lag_alpha": post_lag_alpha,
        "disturbance_amplitude": disturbance_amplitude,
        "disturbance_frequency": disturbance_frequency,
        "disturbance_angle": disturbance_angle,
    }


class EpisodeSimulator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.sim_cfg = config["simulation"]
        self.dt = float(self.sim_cfg["dt"])
        self.duration = float(self.sim_cfg["episode_duration"])
        self.time = np.arange(0.0, self.duration + 1e-9, self.dt, dtype=np.float32)
        self.nominal_mass = float(self.sim_cfg["nominal_mass"])
        self.nominal_friction = float(self.sim_cfg["nominal_friction"])
        self.control_limit = float(self.sim_cfg["control_limit"])
        self.base_noise = float(self.sim_cfg["observation_noise_std"])
        self.noise_burst_multiplier = float(self.sim_cfg["noise_burst_multiplier"])

    def simulate_episode(
        self,
        spec: dict[str, Any],
        controller: Any,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        local_rng = rng or np.random.default_rng(int(spec["episode_seed"]))
        reference = trajectory_from_spec(spec, dt=self.dt, duration=self.duration)
        controller.reset()

        state = np.concatenate(
            [
                reference.position[0],
                reference.velocity[0],
            ],
            axis=0,
        ).astype(np.float32)
        state += np.asarray(spec["initial_offset"], dtype=np.float32)
        prev_command = np.zeros(2, dtype=np.float32)
        applied_command = np.zeros(2, dtype=np.float32)
        command_buffer: deque[np.ndarray] = deque(
            [np.zeros(2, dtype=np.float32) for _ in range(5)],
            maxlen=5,
        )

        history: dict[str, list[np.ndarray | float]] = {
            "time": [],
            "true_state": [],
            "observed_state": [],
            "command": [],
            "applied_command": [],
            "baseline_command": [],
            "ref_position": [],
            "ref_velocity": [],
            "ref_acceleration": [],
            "ref_heading": [],
            "error": [],
            "mass_ratio": [],
            "friction_ratio": [],
            "delay_severity": [],
            "disturbance_force": [],
            "shift_active": [],
            "estimated_targets": [],
            "estimated_uncertainty": [],
            "structure_estimated_uncertainty": [],
            "disturbance_estimated_uncertainty": [],
            "correction_gain": [],
            "structure_gain": [],
            "disturbance_gain": [],
        }

        for index, current_time in enumerate(self.time):
            state_before = state.copy()
            ref_current = {
                "position": reference.position[index],
                "velocity": reference.velocity[index],
                "acceleration": reference.acceleration[index],
                "heading": float(reference.heading[index]),
            }
            next_index = min(index + 1, len(self.time) - 1)
            ref_next = {
                "position": reference.position[next_index],
                "velocity": reference.velocity[next_index],
                "acceleration": reference.acceleration[next_index],
                "heading": float(reference.heading[next_index]),
            }
            mass_ratio, friction_ratio, delay_steps, lag_alpha, disturbance_force, noise_scale, shift_active = self._active_dynamics(
                spec,
                current_time,
            )
            obs_noise = np.concatenate(
                [
                    local_rng.normal(0.0, noise_scale, size=2),
                    local_rng.normal(0.0, noise_scale * 0.5, size=2),
                ]
            ).astype(np.float32)
            observed_state = state_before + obs_noise
            command, aux = controller.compute_control(observed_state, ref_current, ref_next)
            command = clip_vector_norm(command.astype(np.float32), self.control_limit)
            baseline_command = aux["baseline_command"].astype(np.float32)

            command_buffer.append(command.copy())
            delayed_target = command_buffer[-(delay_steps + 1)]
            applied_command = applied_command + lag_alpha * (delayed_target - applied_command)
            force = applied_command + disturbance_force.astype(np.float32)
            drag = self.nominal_friction * friction_ratio * state_before[2:4]
            acceleration = (force - drag) / (self.nominal_mass * mass_ratio)
            next_velocity = state_before[2:4] + self.dt * acceleration
            next_position = state_before[:2] + self.dt * next_velocity
            state = np.concatenate([next_position, next_velocity], axis=0).astype(np.float32)

            history["time"].append(float(current_time))
            history["true_state"].append(state_before.copy())
            history["observed_state"].append(observed_state.copy())
            history["command"].append(command.copy())
            history["applied_command"].append(applied_command.copy())
            history["baseline_command"].append(baseline_command.copy())
            history["ref_position"].append(reference.position[index].copy())
            history["ref_velocity"].append(reference.velocity[index].copy())
            history["ref_acceleration"].append(reference.acceleration[index].copy())
            history["ref_heading"].append(float(reference.heading[index]))
            history["error"].append(reference.position[index] - state_before[:2])
            history["mass_ratio"].append(float(mass_ratio))
            history["friction_ratio"].append(float(friction_ratio))
            history["delay_severity"].append(float(self._delay_severity(delay_steps, lag_alpha)))
            history["disturbance_force"].append(disturbance_force.astype(np.float32))
            history["shift_active"].append(float(shift_active))
            history["estimated_targets"].append(np.asarray(aux.get("estimated_targets", np.zeros(5)), dtype=np.float32))
            history["estimated_uncertainty"].append(
                np.asarray(aux.get("estimated_uncertainty", np.zeros(1)), dtype=np.float32)
            )
            history["structure_estimated_uncertainty"].append(
                np.asarray(aux.get("structure_estimated_uncertainty", np.zeros(1)), dtype=np.float32)
            )
            history["disturbance_estimated_uncertainty"].append(
                np.asarray(aux.get("disturbance_estimated_uncertainty", np.zeros(1)), dtype=np.float32)
            )
            history["correction_gain"].append(np.asarray(aux.get("correction_gain", np.ones(1)), dtype=np.float32))
            history["structure_gain"].append(np.asarray(aux.get("structure_gain", np.ones(1)), dtype=np.float32))
            history["disturbance_gain"].append(np.asarray(aux.get("disturbance_gain", np.ones(1)), dtype=np.float32))
            prev_command = command

        rollout = {
            key: np.asarray(value) for key, value in history.items() if key != "ref_heading"
        }
        rollout["ref_heading"] = np.asarray(history["ref_heading"], dtype=np.float32)
        rollout["metadata_json"] = json.dumps(spec)
        return rollout

    def _active_dynamics(
        self,
        spec: dict[str, Any],
        current_time: float,
    ) -> tuple[float, float, int, float, np.ndarray, float, bool]:
        shift_active = current_time >= float(spec["shift_time"])
        mass_ratio = float(spec["post_mass_ratio"] if shift_active else spec["pre_mass_ratio"])
        friction_ratio = float(spec["post_friction_ratio"] if shift_active else spec["pre_friction_ratio"])
        delay_steps = int(spec["post_delay_steps"] if shift_active else spec["pre_delay_steps"])
        lag_alpha = float(spec["post_lag_alpha"] if shift_active else spec["pre_lag_alpha"])
        disturbance_force = np.zeros(2, dtype=np.float32)
        noise_scale = self.base_noise

        shift_types = list(spec.get("shift_types", [spec["shift_type"]]))
        if "disturbance_burst" in shift_types:
            burst_end = float(spec["shift_time"]) + float(spec["burst_duration"])
            if float(spec["shift_time"]) <= current_time <= burst_end:
                phase = float(spec["disturbance_frequency"]) * (current_time - float(spec["shift_time"]))
                direction = np.array(
                    [
                        np.cos(phase + float(spec["disturbance_angle"])),
                        np.sin(phase + float(spec["disturbance_angle"])),
                    ],
                    dtype=np.float32,
                )
                envelope = 0.8 + 0.2 * np.cos(np.pi * (current_time - float(spec["shift_time"])) / float(spec["burst_duration"]))
                disturbance_force = float(spec["disturbance_amplitude"]) * envelope * direction
                noise_scale = self.base_noise * self.noise_burst_multiplier
                shift_active = True
        return mass_ratio, friction_ratio, delay_steps, lag_alpha, disturbance_force, noise_scale, shift_active

    @staticmethod
    def _delay_severity(delay_steps: int, lag_alpha: float) -> float:
        severity = 0.22 * float(delay_steps) + 0.9 * max(0.0, 0.95 - lag_alpha)
        return float(np.clip(severity, 0.0, 1.0))
