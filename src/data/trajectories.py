from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class ReferenceTrajectory:
    kind: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: np.ndarray
    params: Dict[str, Any]


def sample_trajectory_params(
    kind: str,
    rng: np.random.Generator,
    duration: float,
    split: str,
    unseen: bool,
) -> dict[str, Any]:
    wide = split == "test" and unseen
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    rotation = float(rng.uniform(-0.55, 0.55))
    offset = rng.uniform(-0.75, 0.75, size=2).tolist()

    if kind == "circle":
        radius = float(rng.uniform(1.7, 2.5) if not wide else rng.uniform(2.6, 3.3))
        omega = float(rng.uniform(0.45, 0.7) if not wide else rng.uniform(0.25, 0.4))
        return {
            "radius": radius,
            "omega": omega,
            "phase": phase,
            "rotation": rotation,
            "offset": offset,
        }

    if kind == "figure8":
        amplitude = float(rng.uniform(1.3, 2.3) if not wide else rng.uniform(2.4, 3.1))
        omega = float(rng.uniform(0.45, 0.8) if not wide else rng.uniform(0.25, 0.42))
        return {
            "amplitude": amplitude,
            "omega": omega,
            "phase": phase,
            "rotation": rotation,
            "offset": offset,
        }

    if kind == "sinusoid":
        speed = float(rng.uniform(0.55, 0.9) if not wide else rng.uniform(0.95, 1.15))
        amplitude = float(rng.uniform(0.7, 1.4) if not wide else rng.uniform(1.45, 2.0))
        omega = float(rng.uniform(0.55, 1.0) if not wide else rng.uniform(0.28, 0.48))
        return {
            "speed": speed,
            "amplitude": amplitude,
            "omega": omega,
            "phase": phase,
            "rotation": rotation,
            "offset": offset,
        }

    if kind == "lane_change":
        speed = float(rng.uniform(0.6, 1.0) if not wide else rng.uniform(1.05, 1.2))
        amplitude = float(rng.uniform(0.8, 1.5) if not wide else rng.uniform(1.6, 2.2))
        sharpness = float(rng.uniform(0.7, 1.15) if not wide else rng.uniform(1.2, 1.55))
        midpoint = float(rng.uniform(0.35, 0.65) * duration)
        return {
            "speed": speed,
            "amplitude": amplitude,
            "sharpness": sharpness,
            "midpoint": midpoint,
            "rotation": rotation,
            "offset": offset,
        }

    if kind == "spline":
        n_points = 6
        base_speed = float(rng.uniform(0.55, 0.9) if not wide else rng.uniform(0.95, 1.2))
        x_points = np.linspace(0.0, base_speed * duration, n_points)
        x_points += rng.normal(0.0, 0.1 if not wide else 0.16, size=n_points)
        x_points = np.maximum.accumulate(x_points)
        y_scale = 1.4 if not wide else 2.1
        y_points = rng.uniform(-y_scale, y_scale, size=n_points)
        knot_times = np.linspace(0.0, duration, n_points)
        return {
            "knot_times": knot_times.tolist(),
            "x_points": x_points.tolist(),
            "y_points": y_points.tolist(),
            "rotation": rotation,
            "offset": offset,
        }

    raise ValueError(f"Unsupported trajectory kind: {kind}")


def build_trajectory(kind: str, params: dict[str, Any], time: np.ndarray) -> ReferenceTrajectory:
    if kind == "circle":
        radius = params["radius"]
        omega = params["omega"]
        phase = params["phase"]
        theta = omega * time + phase
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        vx = -radius * omega * np.sin(theta)
        vy = radius * omega * np.cos(theta)
        ax = -radius * (omega**2) * np.cos(theta)
        ay = -radius * (omega**2) * np.sin(theta)
    elif kind == "figure8":
        amplitude = params["amplitude"]
        omega = params["omega"]
        phase = params["phase"]
        theta = omega * time + phase
        x = amplitude * np.sin(theta)
        y = 0.7 * amplitude * np.sin(2.0 * theta)
        vx = amplitude * omega * np.cos(theta)
        vy = 1.4 * amplitude * omega * np.cos(2.0 * theta)
        ax = -amplitude * (omega**2) * np.sin(theta)
        ay = -2.8 * amplitude * (omega**2) * np.sin(2.0 * theta)
    elif kind == "sinusoid":
        speed = params["speed"]
        amplitude = params["amplitude"]
        omega = params["omega"]
        phase = params["phase"]
        x = speed * time
        y = amplitude * np.sin(omega * time + phase)
        vx = np.full_like(time, speed)
        vy = amplitude * omega * np.cos(omega * time + phase)
        ax = np.zeros_like(time)
        ay = -amplitude * (omega**2) * np.sin(omega * time + phase)
    elif kind == "lane_change":
        speed = params["speed"]
        amplitude = params["amplitude"]
        sharpness = params["sharpness"]
        midpoint = params["midpoint"]
        z = sharpness * (time - midpoint)
        sech2 = 1.0 / np.cosh(z) ** 2
        x = speed * time
        y = amplitude * np.tanh(z)
        vx = np.full_like(time, speed)
        vy = amplitude * sharpness * sech2
        ax = np.zeros_like(time)
        ay = -2.0 * amplitude * (sharpness**2) * sech2 * np.tanh(z)
    elif kind == "spline":
        knot_times = np.asarray(params["knot_times"], dtype=float)
        x_points = np.asarray(params["x_points"], dtype=float)
        y_points = np.asarray(params["y_points"], dtype=float)
        spline_x = CubicSpline(knot_times, x_points, bc_type="natural")
        spline_y = CubicSpline(knot_times, y_points, bc_type="natural")
        x = spline_x(time)
        y = spline_y(time)
        vx = spline_x.derivative(1)(time)
        vy = spline_y.derivative(1)(time)
        ax = spline_x.derivative(2)(time)
        ay = spline_y.derivative(2)(time)
    else:
        raise ValueError(f"Unsupported trajectory kind: {kind}")

    position = np.stack([x, y], axis=1)
    velocity = np.stack([vx, vy], axis=1)
    acceleration = np.stack([ax, ay], axis=1)
    position, velocity, acceleration = _apply_rigid_transform(position, velocity, acceleration, params)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    return ReferenceTrajectory(
        kind=kind,
        time=time,
        position=position.astype(np.float32),
        velocity=velocity.astype(np.float32),
        acceleration=acceleration.astype(np.float32),
        heading=heading.astype(np.float32),
        params=params,
    )


def trajectory_from_spec(spec: dict[str, Any], dt: float, duration: float) -> ReferenceTrajectory:
    time = np.arange(0.0, duration + 1e-9, dt, dtype=np.float32)
    return build_trajectory(spec["trajectory_kind"], spec["trajectory_params"], time)


def _apply_rigid_transform(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = float(params.get("rotation", 0.0))
    offset = np.asarray(params.get("offset", [0.0, 0.0]), dtype=float)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    position = position @ rotation.T + offset[None, :]
    velocity = velocity @ rotation.T
    acceleration = acceleration @ rotation.T
    return position, velocity, acceleration
