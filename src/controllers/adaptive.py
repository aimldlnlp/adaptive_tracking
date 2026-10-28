from __future__ import annotations

from typing import Any

import numpy as np

from src.controllers.baseline import BaselineController
from src.data.dataset import FeatureBuilder
from src.models.adaptive_estimator import predict_targets
from src.utils.math_utils import clip_vector_norm


class AdaptiveController:
    def __init__(self, config: dict[str, Any], checkpoint_bundle: dict[str, Any]) -> None:
        self.config = config
        self.bundle = checkpoint_bundle
        self.baseline = BaselineController(config)
        sim_cfg = config["simulation"]
        controller_cfg = config["controller"]
        model_cfg = config.get("model", {})
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
        uncertainty_stats = checkpoint_bundle.get("uncertainty_stats", {})
        use_uncertainty_calibration = bool(model_cfg.get("use_uncertainty_calibration", True))
        self.uncertainty_aware = bool(model_cfg.get("uncertainty_aware", False))
        self.structure_uncertainty_floor = float(
            uncertainty_stats.get("structure_floor", model_cfg.get("structure_uncertainty_floor", model_cfg.get("uncertainty_floor", 0.15)))
            if use_uncertainty_calibration
            else model_cfg.get("structure_uncertainty_floor", model_cfg.get("uncertainty_floor", 0.15))
        )
        self.structure_uncertainty_ceiling = float(
            uncertainty_stats.get(
                "structure_ceiling",
                model_cfg.get("structure_uncertainty_ceiling", model_cfg.get("uncertainty_ceiling", 1.75)),
            )
            if use_uncertainty_calibration
            else model_cfg.get("structure_uncertainty_ceiling", model_cfg.get("uncertainty_ceiling", 1.75))
        )
        self.disturbance_uncertainty_floor = float(
            uncertainty_stats.get(
                "disturbance_floor",
                model_cfg.get("disturbance_uncertainty_floor", model_cfg.get("uncertainty_floor", 0.15)),
            )
            if use_uncertainty_calibration
            else model_cfg.get("disturbance_uncertainty_floor", model_cfg.get("uncertainty_floor", 0.15))
        )
        self.disturbance_uncertainty_ceiling = float(
            uncertainty_stats.get(
                "disturbance_ceiling",
                model_cfg.get("disturbance_uncertainty_ceiling", model_cfg.get("uncertainty_ceiling", 1.75)),
            )
            if use_uncertainty_calibration
            else model_cfg.get("disturbance_uncertainty_ceiling", model_cfg.get("uncertainty_ceiling", 1.75))
        )
        self.structure_uncertainty_power = float(model_cfg.get("structure_uncertainty_power", 1.8))
        self.structure_min_gain = float(model_cfg.get("structure_min_gain", 0.84))
        self.disturbance_uncertainty_power = float(model_cfg.get("disturbance_uncertainty_power", 1.2))
        self.disturbance_min_gain = float(model_cfg.get("disturbance_min_gain", 0.52))
        self.uncertainty_smoothing = float(model_cfg.get("uncertainty_smoothing", 0.82))
        self.delay_support_scale = float(model_cfg.get("delay_support_scale", 0.0))
        self.delay_support_power = float(model_cfg.get("delay_support_power", 1.0))
        self.disturbance_blend_boost = float(model_cfg.get("disturbance_blend_boost", 0.0))
        self.disturbance_response_scale = float(model_cfg.get("disturbance_response_scale", 1.0))
        self.focus_burst_support_scale = float(model_cfg.get("focus_burst_support_scale", 0.0))
        self.focus_delay_burst_support_scale = float(model_cfg.get("focus_delay_burst_support_scale", 0.0))
        self.focus_delay_burst_power = float(model_cfg.get("focus_delay_burst_power", 1.0))
        self.prev_command = np.zeros(2, dtype=np.float32)
        self.filtered_targets = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_structure_uncertainty = 0.0
        self.filtered_disturbance_uncertainty = 0.0

    def reset(self) -> None:
        self.baseline.reset()
        self.feature_builder.reset()
        self.prev_command = np.zeros(2, dtype=np.float32)
        self.filtered_targets = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.filtered_structure_uncertainty = 0.0
        self.filtered_disturbance_uncertainty = 0.0

    def compute_control(
        self,
        observed_state: np.ndarray,
        ref_current: dict[str, np.ndarray | float],
        ref_next: dict[str, np.ndarray | float],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        baseline_command, baseline_aux = self.baseline.compute_control(observed_state, ref_current, ref_next)
        inputs = self.feature_builder.build_inputs(
            obs_state=observed_state,
            ref_current=ref_current,
            ref_next=ref_next,
            baseline_command=baseline_command,
            prev_command=self.prev_command,
        )

        model_type = str(self.bundle.get("model_type", "mlp")).lower()
        if model_type == "mlp":
            normalized_flat = (inputs["flat_features"] - self.bundle["feature_mean"]) / self.bundle["feature_std"]
            prediction, logvar = predict_targets(self.bundle, flat_inputs=normalized_flat.astype(np.float32))
        else:
            normalized_sequence = (
                inputs["sequence_features"] - self.bundle["sequence_mean"][None, :]
            ) / self.bundle["sequence_std"][None, :]
            normalized_context = (inputs["context_features"] - self.bundle["context_mean"]) / self.bundle["context_std"]
            prediction, logvar = predict_targets(
                self.bundle,
                sequence_inputs=normalized_sequence.astype(np.float32),
                context_inputs=normalized_context.astype(np.float32),
            )

        prediction = prediction * self.bundle["target_std"] + self.bundle["target_mean"]
        structure_uncertainty, disturbance_uncertainty, combined_uncertainty = self._predictive_uncertainty(logvar)
        structure_gain, disturbance_gain, _, normalized_disturbance = self._correction_gains(
            structure_uncertainty,
            disturbance_uncertainty,
        )
        effective_disturbance_blend = self._effective_disturbance_blend(prediction[3:], normalized_disturbance)
        self.filtered_targets[:3] = (
            (1.0 - self.correction_blend) * self.filtered_targets[:3] + self.correction_blend * prediction[:3]
        )
        self.filtered_targets[3:] = (
            (1.0 - effective_disturbance_blend) * self.filtered_targets[3:] + effective_disturbance_blend * prediction[3:]
        )
        mass_ratio = float(np.clip(self.filtered_targets[0], self.mass_ratio_clip[0], self.mass_ratio_clip[1]))
        friction_ratio = float(np.clip(self.filtered_targets[1], self.friction_ratio_clip[0], self.friction_ratio_clip[1]))
        delay_severity = float(np.clip(self.filtered_targets[2], 0.0, 1.0))
        disturbance_force = self.filtered_targets[3:].astype(np.float32)
        disturbance_support_gain = self._disturbance_support_gain(
            disturbance_gain=disturbance_gain,
            normalized_disturbance=normalized_disturbance,
            disturbance_force=disturbance_force,
            delay_severity=delay_severity,
        )
        delay_support_gain = self._delay_support_gain(
            disturbance_gain=disturbance_gain,
            normalized_disturbance=normalized_disturbance,
            delay_severity=delay_severity,
            disturbance_support_gain=disturbance_support_gain,
            disturbance_force=disturbance_force,
        )

        desired_acceleration = baseline_aux["desired_acceleration"]
        structured_command = (
            self.nominal_mass * mass_ratio * desired_acceleration
            + self.nominal_friction * friction_ratio * observed_state[2:4]
        )
        param_corrected_command = baseline_command + structure_gain * (structured_command - baseline_command)
        delay_term = delay_support_gain * self.delay_lead_gain * delay_severity * (baseline_command - self.prev_command)
        disturbance_term = disturbance_support_gain * disturbance_force
        blended_command = param_corrected_command - disturbance_term + delay_term
        command = clip_vector_norm(blended_command.astype(np.float32), self.control_limit)
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
            "estimated_uncertainty": np.array([combined_uncertainty], dtype=np.float32),
            "structure_estimated_uncertainty": np.array([structure_uncertainty], dtype=np.float32),
            "disturbance_estimated_uncertainty": np.array([disturbance_uncertainty], dtype=np.float32),
            "correction_gain": np.array([disturbance_gain], dtype=np.float32),
            "structure_gain": np.array([structure_gain], dtype=np.float32),
            "disturbance_gain": np.array([disturbance_gain], dtype=np.float32),
            "delay_support_gain": np.array([delay_support_gain], dtype=np.float32),
            "disturbance_support_gain": np.array([disturbance_support_gain], dtype=np.float32),
        }

    def _predictive_uncertainty(self, logvar: np.ndarray | None) -> tuple[float, float, float]:
        if logvar is None:
            return 0.0, 0.0, 0.0
        sigma = np.exp(0.5 * np.asarray(logvar, dtype=np.float32))
        raw_structure_uncertainty = float(np.mean(sigma[:2]))
        raw_disturbance_uncertainty = float(np.mean(sigma[2:]))
        self.filtered_structure_uncertainty = self.uncertainty_smoothing * self.filtered_structure_uncertainty + (
            1.0 - self.uncertainty_smoothing
        ) * raw_structure_uncertainty
        self.filtered_disturbance_uncertainty = self.uncertainty_smoothing * self.filtered_disturbance_uncertainty + (
            1.0 - self.uncertainty_smoothing
        ) * raw_disturbance_uncertainty
        combined_uncertainty = 0.35 * self.filtered_structure_uncertainty + 0.65 * self.filtered_disturbance_uncertainty
        return (
            float(self.filtered_structure_uncertainty),
            float(self.filtered_disturbance_uncertainty),
            float(combined_uncertainty),
        )

    def _correction_gains(
        self,
        structure_uncertainty: float,
        disturbance_uncertainty: float,
    ) -> tuple[float, float, float, float]:
        if not self.uncertainty_aware:
            return 1.0, 1.0, 0.0, 0.0
        structure_span = max(self.structure_uncertainty_ceiling - self.structure_uncertainty_floor, 1e-6)
        disturbance_span = max(self.disturbance_uncertainty_ceiling - self.disturbance_uncertainty_floor, 1e-6)
        normalized_structure = float(
            np.clip((structure_uncertainty - self.structure_uncertainty_floor) / structure_span, 0.0, 1.0)
        )
        normalized_disturbance = float(
            np.clip((disturbance_uncertainty - self.disturbance_uncertainty_floor) / disturbance_span, 0.0, 1.0)
        )
        structure_gain = 1.0 - (1.0 - self.structure_min_gain) * (normalized_structure**self.structure_uncertainty_power)
        disturbance_gain = 1.0 - (1.0 - self.disturbance_min_gain) * (
            normalized_disturbance**self.disturbance_uncertainty_power
        )
        return (
            float(np.clip(structure_gain, self.structure_min_gain, 1.0)),
            float(np.clip(disturbance_gain, self.disturbance_min_gain, 1.0)),
            normalized_structure,
            normalized_disturbance,
        )

    def _effective_disturbance_blend(self, predicted_disturbance: np.ndarray, normalized_disturbance: float) -> float:
        response_strength = np.tanh(float(np.linalg.norm(predicted_disturbance)) / max(self.disturbance_response_scale, 1e-6))
        boost = self.disturbance_blend_boost * response_strength * (1.0 - normalized_disturbance)
        return float(np.clip(self.disturbance_blend + boost, self.disturbance_blend, 0.95))

    def _delay_support_gain(
        self,
        disturbance_gain: float,
        normalized_disturbance: float,
        delay_severity: float,
        disturbance_support_gain: float,
        disturbance_force: np.ndarray,
    ) -> float:
        if not self.uncertainty_aware:
            return disturbance_gain
        headroom = max(1.0 - disturbance_gain, 0.0)
        boost = self.delay_support_scale * delay_severity * ((1.0 - normalized_disturbance) ** self.delay_support_power)
        response_strength = np.tanh(float(np.linalg.norm(disturbance_force)) / max(self.disturbance_response_scale, 1e-6))
        coupled_boost = self.focus_delay_burst_support_scale * response_strength * (delay_severity**self.focus_delay_burst_power)
        return float(
            np.clip(
                max(disturbance_support_gain, disturbance_gain + headroom * (boost + coupled_boost)),
                disturbance_gain,
                1.0,
            )
        )

    def _disturbance_support_gain(
        self,
        disturbance_gain: float,
        normalized_disturbance: float,
        disturbance_force: np.ndarray,
        delay_severity: float,
    ) -> float:
        if not self.uncertainty_aware:
            return disturbance_gain
        headroom = max(1.0 - disturbance_gain, 0.0)
        response_strength = np.tanh(float(np.linalg.norm(disturbance_force)) / max(self.disturbance_response_scale, 1e-6))
        burst_boost = self.focus_burst_support_scale * response_strength
        coupled_boost = self.focus_delay_burst_support_scale * response_strength * (delay_severity**self.focus_delay_burst_power)
        support = burst_boost + coupled_boost * max(0.25, 1.0 - normalized_disturbance)
        return float(np.clip(disturbance_gain + headroom * support, disturbance_gain, 1.0))
