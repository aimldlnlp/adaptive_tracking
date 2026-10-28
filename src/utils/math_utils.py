from __future__ import annotations

from typing import Iterable

import numpy as np


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def safe_heading_from_velocity(velocity: np.ndarray, fallback: float) -> float:
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return fallback
    return float(np.arctan2(velocity[1], velocity[0]))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def padded_stack(history: Iterable[np.ndarray], target_length: int, feature_dim: int) -> np.ndarray:
    rows = list(history)
    if len(rows) < target_length:
        rows = [np.zeros(feature_dim, dtype=np.float32) for _ in range(target_length - len(rows))] + rows
    return np.stack(rows[-target_length:], axis=0)
