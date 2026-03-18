from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch
from torch import nn


class AdaptiveEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_model(config: dict[str, Any], input_dim: int, output_dim: int = 5) -> AdaptiveEstimator:
    training_cfg = config["training"]
    return AdaptiveEstimator(
        input_dim=input_dim,
        hidden_dims=list(training_cfg["hidden_dims"]),
        output_dim=output_dim,
        dropout=float(training_cfg["dropout"]),
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"]["device"]).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_checkpoint_bundle(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    try:
        bundle = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        # PyTorch 2.6 defaults to weights_only=True. Older checkpoints from this
        # project include numpy arrays in the metadata bundle, so fall back to
        # trusted full deserialization for locally generated checkpoints.
        bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AdaptiveEstimator(
        input_dim=int(bundle["input_dim"]),
        hidden_dims=list(bundle["hidden_dims"]),
        output_dim=int(bundle["output_dim"]),
        dropout=float(bundle["dropout"]),
    )
    model.load_state_dict(bundle["model_state"])
    model.to(device)
    model.eval()
    bundle["model"] = model
    bundle["feature_mean"] = _to_numpy_float32(bundle["feature_mean"])
    bundle["feature_std"] = _to_numpy_float32(bundle["feature_std"])
    bundle["target_mean"] = _to_numpy_float32(bundle["target_mean"])
    bundle["target_std"] = _to_numpy_float32(bundle["target_std"])
    return bundle


def _to_numpy_float32(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)
