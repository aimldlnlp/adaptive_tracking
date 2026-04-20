from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch
from torch import nn


class MLPAdaptiveEstimator(nn.Module):
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


class GRUUncertaintyEstimator(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        context_dim: int,
        hidden_size: int,
        head_hidden_size: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.sequence_encoder = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.instant_encoder = nn.Sequential(
            nn.Linear(sequence_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.structure_trunk = nn.Sequential(
            nn.Linear(hidden_size * 2, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delay_trunk = nn.Sequential(
            nn.Linear(hidden_size * 3, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.disturbance_trunk = nn.Sequential(
            nn.Linear(hidden_size * 3, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.structure_mean_head = nn.Linear(head_hidden_size, 2)
        self.structure_logvar_head = nn.Linear(head_hidden_size, 2)
        self.delay_mean_head = nn.Linear(head_hidden_size, 1)
        self.delay_logvar_head = nn.Linear(head_hidden_size, 1)
        self.disturbance_mean_head = nn.Linear(head_hidden_size, 2)
        self.disturbance_logvar_head = nn.Linear(head_hidden_size, 2)

    def forward(self, sequence_inputs: torch.Tensor, context_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.sequence_encoder(sequence_inputs)
        seq_summary = hidden[-1]
        instant_summary = self.instant_encoder(sequence_inputs[:, -1, :])
        context_summary = self.context_encoder(context_inputs)
        fused = self.fusion(torch.cat([seq_summary, instant_summary, context_summary], dim=-1))
        structure_features = self.structure_trunk(torch.cat([fused, context_summary], dim=-1))
        delay_features = self.delay_trunk(torch.cat([fused, instant_summary, context_summary], dim=-1))
        disturbance_features = self.disturbance_trunk(torch.cat([fused, seq_summary, instant_summary], dim=-1))
        mean = torch.cat(
            [
                self.structure_mean_head(structure_features),
                self.delay_mean_head(delay_features),
                self.disturbance_mean_head(disturbance_features),
            ],
            dim=-1,
        )
        logvar = torch.cat(
            [
                self.structure_logvar_head(structure_features),
                self.delay_logvar_head(delay_features),
                self.disturbance_logvar_head(disturbance_features),
            ],
            dim=-1,
        )
        logvar = torch.clamp(logvar, min=-6.0, max=3.0)
        return {"mean": mean, "logvar": logvar}


def build_model(
    config: dict[str, Any],
    input_dim: int | None = None,
    output_dim: int = 5,
    sequence_dim: int | None = None,
    context_dim: int | None = None,
) -> nn.Module:
    model_cfg = config.get("model", {})
    training_cfg = config["training"]
    model_type = str(model_cfg.get("type", "mlp")).lower()
    if model_type == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for the MLP estimator.")
        hidden_dims = list(model_cfg.get("hidden_dims", training_cfg.get("hidden_dims", [128, 128, 64])))
        dropout = float(model_cfg.get("dropout", training_cfg.get("dropout", 0.05)))
        return MLPAdaptiveEstimator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
    if model_type == "gru_uncertainty":
        if sequence_dim is None or context_dim is None:
            raise ValueError("sequence_dim and context_dim are required for the GRU estimator.")
        return GRUUncertaintyEstimator(
            sequence_dim=sequence_dim,
            context_dim=context_dim,
            hidden_size=int(model_cfg.get("hidden_size", 96)),
            head_hidden_size=int(model_cfg.get("head_hidden_size", model_cfg.get("hidden_size", 96))),
            num_layers=int(model_cfg.get("num_layers", 2)),
            output_dim=output_dim,
            dropout=float(model_cfg.get("dropout", training_cfg.get("dropout", 0.05))),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"]["device"]).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_checkpoint_bundle(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    try:
        bundle = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = str(bundle.get("model_type", "mlp")).lower()
    if model_type == "mlp":
        model = MLPAdaptiveEstimator(
            input_dim=int(bundle["input_dim"]),
            hidden_dims=list(bundle["hidden_dims"]),
            output_dim=int(bundle["output_dim"]),
            dropout=float(bundle["dropout"]),
        )
    elif model_type == "gru_uncertainty":
        model = GRUUncertaintyEstimator(
            sequence_dim=int(bundle["sequence_dim"]),
            context_dim=int(bundle["context_dim"]),
            hidden_size=int(bundle["hidden_size"]),
            head_hidden_size=int(bundle.get("head_hidden_size", bundle["hidden_size"])),
            num_layers=int(bundle["num_layers"]),
            output_dim=int(bundle["output_dim"]),
            dropout=float(bundle["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")
    model.load_state_dict(bundle["model_state"])
    model.to(device)
    model.eval()
    bundle["model"] = model
    for key in [
        "feature_mean",
        "feature_std",
        "target_mean",
        "target_std",
        "sequence_mean",
        "sequence_std",
        "context_mean",
        "context_std",
    ]:
        if key in bundle:
            bundle[key] = _to_numpy_float32(bundle[key])
    bundle["model_type"] = model_type
    return bundle


def predict_targets(
    bundle: dict[str, Any],
    *,
    flat_inputs: np.ndarray | None = None,
    sequence_inputs: np.ndarray | None = None,
    context_inputs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    device = bundle["model"].parameters().__next__().device
    model_type = str(bundle.get("model_type", "mlp")).lower()
    with torch.no_grad():
        if model_type == "mlp":
            if flat_inputs is None:
                raise ValueError("flat_inputs are required for the MLP estimator.")
            input_tensor = torch.from_numpy(flat_inputs[None, :]).to(device)
            mean = bundle["model"](input_tensor).cpu().numpy()[0]
            return mean, None
        if model_type == "gru_uncertainty":
            if sequence_inputs is None or context_inputs is None:
                raise ValueError("sequence_inputs and context_inputs are required for the GRU estimator.")
            seq_tensor = torch.from_numpy(sequence_inputs[None, :, :]).to(device)
            context_tensor = torch.from_numpy(context_inputs[None, :]).to(device)
            outputs = bundle["model"](seq_tensor, context_tensor)
            return outputs["mean"].cpu().numpy()[0], outputs["logvar"].cpu().numpy()[0]
    raise ValueError(f"Unsupported model type: {model_type}")


def _to_numpy_float32(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)
