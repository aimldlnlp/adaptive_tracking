from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.adaptive_estimator import build_model, resolve_device
from src.utils.config import get_output_dir
from src.utils.io import ensure_dir, load_npz, save_dataframe, save_json
from src.utils.logging_utils import ProgressCallback


LOGGER = logging.getLogger(__name__)


class ArrayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        sequence_features: np.ndarray,
        context_features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.sequence_features = torch.from_numpy(sequence_features.astype(np.float32))
        self.context_features = torch.from_numpy(context_features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "sequence_features": self.sequence_features[index],
            "context_features": self.context_features[index],
            "targets": self.targets[index],
        }


def train_from_config(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> Path:
    data_dir = get_output_dir(config, "datasets")
    train_data = load_npz(data_dir / "train_dataset.npz")
    val_data = load_npz(data_dir / "val_dataset.npz")
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "adaptive"))
    model_type = str(model_cfg.get("type", "mlp")).lower()

    x_train = train_data["features"].astype(np.float32)
    x_val = val_data["features"].astype(np.float32)
    x_train_seq = train_data["sequence_features"].astype(np.float32)
    x_val_seq = val_data["sequence_features"].astype(np.float32)
    x_train_ctx = train_data["context_features"].astype(np.float32)
    x_val_ctx = val_data["context_features"].astype(np.float32)
    y_train = train_data["targets"].astype(np.float32)
    y_val = val_data["targets"].astype(np.float32)
    target_names = [str(name) for name in train_data["target_names"].tolist()]

    feature_mean = x_train.mean(axis=0)
    feature_std = np.clip(x_train.std(axis=0), 1e-6, None)
    sequence_mean = x_train_seq.mean(axis=(0, 1))
    sequence_std = np.clip(x_train_seq.std(axis=(0, 1)), 1e-6, None)
    context_mean = x_train_ctx.mean(axis=0)
    context_std = np.clip(x_train_ctx.std(axis=0), 1e-6, None)
    target_mean = y_train.mean(axis=0)
    target_std = np.clip(y_train.std(axis=0), 1e-6, None)

    x_train = (x_train - feature_mean) / feature_std
    x_val = (x_val - feature_mean) / feature_std
    x_train_seq = (x_train_seq - sequence_mean[None, None, :]) / sequence_std[None, None, :]
    x_val_seq = (x_val_seq - sequence_mean[None, None, :]) / sequence_std[None, None, :]
    x_train_ctx = (x_train_ctx - context_mean) / context_std
    x_val_ctx = (x_val_ctx - context_mean) / context_std
    y_train = (y_train - target_mean) / target_std
    y_val = (y_val - target_mean) / target_std

    training_cfg = config["training"]
    device = resolve_device(config)
    model = build_model(
        config,
        input_dim=x_train.shape[1],
        sequence_dim=x_train_seq.shape[2],
        context_dim=x_train_ctx.shape[1],
        output_dim=y_train.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    loss_name = str(training_cfg.get("loss", "mse")).lower()
    selection_metric = str(training_cfg.get("selection_metric", "auto")).lower()
    if selection_metric == "auto":
        selection_metric = "val_mse" if model_type == "gru_uncertainty" else "val_loss"

    train_loader = DataLoader(
        ArrayDataset(x_train, x_train_seq, x_train_ctx, y_train),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        ArrayDataset(x_val, x_val_seq, x_val_ctx, y_val),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )

    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    history_rows: list[dict[str, Any]] = []
    total_epochs = int(training_cfg["epochs"])
    if progress_callback is not None:
        progress_callback(0.0, f"starting {model_name} training")

    for epoch in range(total_epochs):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            model_type=model_type,
            loss_name=loss_name,
            training_cfg=training_cfg,
            optimizer=optimizer,
            device=device,
            grad_clip=float(training_cfg["grad_clip"]),
        )
        val_loss, val_metrics, val_mae, uncertainty_stats = _evaluate(
            model,
            val_loader,
            model_type,
            loss_name,
            training_cfg,
            device,
            target_names,
        )
        selection_score = _resolve_selection_score(val_loss, val_metrics, selection_metric)
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
                "selection_score": selection_score,
                **{f"val_mae_{name}": value for name, value in val_mae.items()},
            }
        )
        LOGGER.info("model=%s epoch=%s train_loss=%.6f val_loss=%.6f", model_name, epoch + 1, train_loss, val_loss)
        if progress_callback is not None:
            progress_callback((epoch + 1) / max(total_epochs, 1), f"{model_name} epoch {epoch + 1}/{total_epochs}")

        score = selection_score
        if score < best_score:
            best_score = score
            best_state = {
                "model_state": model.state_dict(),
                "model_type": model_type,
                "model_name": model_name,
                "output_dim": y_train.shape[1],
                "dropout": float(model_cfg.get("dropout", training_cfg.get("dropout", 0.05))),
                "feature_mean": torch.from_numpy(feature_mean.copy()),
                "feature_std": torch.from_numpy(feature_std.copy()),
                "sequence_mean": torch.from_numpy(sequence_mean.copy()),
                "sequence_std": torch.from_numpy(sequence_std.copy()),
                "context_mean": torch.from_numpy(context_mean.copy()),
                "context_std": torch.from_numpy(context_std.copy()),
                "target_mean": torch.from_numpy(target_mean.copy()),
                "target_std": torch.from_numpy(target_std.copy()),
                "target_names": target_names,
                "config": config,
                "selection_metric": selection_metric,
                "best_val_loss": float(val_loss),
                "best_val_mse": float(val_metrics["val_mse"]),
                "best_val_focus_mse": float(val_metrics["val_focus_mse"]),
                "best_val_severe_focus_mse": float(val_metrics["val_severe_focus_mse"]),
                "best_val_delay_mse": float(val_metrics["val_delay_mse"]),
                "best_val_disturbance_mse": float(val_metrics["val_disturbance_mse"]),
                "uncertainty_stats": uncertainty_stats,
            }
            if model_type == "mlp":
                best_state["input_dim"] = x_train.shape[1]
                best_state["hidden_dims"] = list(model_cfg.get("hidden_dims", training_cfg.get("hidden_dims", [128, 128, 64])))
            else:
                best_state["sequence_dim"] = x_train_seq.shape[2]
                best_state["context_dim"] = x_train_ctx.shape[1]
                best_state["hidden_size"] = int(model_cfg.get("hidden_size", 96))
                best_state["head_hidden_size"] = int(model_cfg.get("head_hidden_size", model_cfg.get("hidden_size", 96)))
                best_state["num_layers"] = int(model_cfg.get("num_layers", 2))
            patience = 0
        else:
            patience += 1
            if patience >= int(training_cfg["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    metrics_dir = get_output_dir(config, "metrics")
    checkpoints_dir = get_output_dir(config, "checkpoints")
    ensure_dir(metrics_dir)
    ensure_dir(checkpoints_dir)
    history_path, summary_path, checkpoint_path = _artifact_paths(metrics_dir, checkpoints_dir, model_name)
    save_dataframe(pd.DataFrame(history_rows), history_path)
    save_json(
        {
            "model_name": model_name,
            "model_type": model_type,
            "loss": loss_name,
            "selection_metric": selection_metric,
            "best_selection_score": best_score,
            "best_val_loss": best_state["best_val_loss"],
            "best_val_mse": best_state["best_val_mse"],
            "best_val_focus_mse": best_state.get("best_val_focus_mse"),
            "best_val_severe_focus_mse": best_state.get("best_val_severe_focus_mse"),
            "best_val_delay_mse": best_state.get("best_val_delay_mse"),
            "best_val_disturbance_mse": best_state.get("best_val_disturbance_mse"),
            "uncertainty_stats": best_state.get("uncertainty_stats", {}),
            "device": str(device),
            "epochs_ran": len(history_rows),
            "target_names": target_names,
        },
        summary_path,
    )
    torch.save(best_state, checkpoint_path)
    LOGGER.info("saved_checkpoint=%s", checkpoint_path)
    if progress_callback is not None:
        progress_callback(1.0, f"{model_name} training complete")
    return checkpoint_path


def _artifact_paths(metrics_dir: Path, checkpoints_dir: Path, model_name: str) -> tuple[Path, Path, Path]:
    if model_name == "adaptive":
        return (
            metrics_dir / "training_history.csv",
            metrics_dir / "training_summary.json",
            checkpoints_dir / "best_model.pt",
        )
    return (
        metrics_dir / f"training_history_{model_name}.csv",
        metrics_dir / f"training_summary_{model_name}.json",
        checkpoints_dir / f"{model_name}.pt",
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    model_type: str,
    loss_name: str,
    training_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        features = batch["features"].to(device)
        sequence_features = batch["sequence_features"].to(device)
        context_features = batch["context_features"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions, logvar = _forward_model(model, model_type, features, sequence_features, context_features)
        loss = _loss_value(predictions, targets, logvar, loss_name, training_cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    model_type: str,
    loss_name: str,
    training_cfg: dict[str, Any],
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
    model.eval()
    loss_total = 0.0
    mse_total = 0.0
    focus_mse_total = 0.0
    severe_focus_mse_total = 0.0
    delay_disturbance_mse_total = 0.0
    mae_total = np.zeros(len(target_names), dtype=np.float64)
    per_dim_mse_total = np.zeros(len(target_names), dtype=np.float64)
    total_count = 0
    uncertainty_values: list[np.ndarray] = []
    structure_uncertainty_values: list[np.ndarray] = []
    disturbance_uncertainty_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            sequence_features = batch["sequence_features"].to(device)
            context_features = batch["context_features"].to(device)
            targets = batch["targets"].to(device)
            predictions, logvar = _forward_model(model, model_type, features, sequence_features, context_features)
            loss = _loss_value(predictions, targets, logvar, loss_name, training_cfg)
            diff = predictions - targets
            batch_size = int(targets.shape[0])
            loss_total += float(loss.item()) * batch_size
            per_dim_mse = torch.mean(diff**2, dim=0)
            mse_total += float(torch.mean(diff**2).item()) * batch_size
            focus_mse_total += float(_weighted_target_mse(diff, targets, training_cfg, prefix="selection_").item()) * batch_size
            severe_focus_mse_total += float(_severe_focus_mse(diff, targets, training_cfg, prefix="selection_").item()) * batch_size
            delay_disturbance_mse_total += float(torch.mean(per_dim_mse[2:]).item()) * batch_size
            mae_total += torch.mean(torch.abs(diff), dim=0).cpu().numpy() * batch_size
            per_dim_mse_total += per_dim_mse.cpu().numpy() * batch_size
            if logvar is not None:
                sigma = torch.exp(0.5 * logvar).cpu().numpy()
                uncertainty_values.append(sigma)
                structure_uncertainty_values.append(sigma[:, :2])
                disturbance_uncertainty_values.append(sigma[:, 2:])
            total_count += batch_size
    stats: dict[str, float] = {}
    if uncertainty_values:
        stacked = np.concatenate(uncertainty_values, axis=0).reshape(-1)
        structure = np.concatenate(structure_uncertainty_values, axis=0).reshape(-1)
        disturbance = np.concatenate(disturbance_uncertainty_values, axis=0).reshape(-1)
        stats = {
            "mean": float(np.mean(stacked)),
            "median": float(np.quantile(stacked, 0.5)),
            "floor": float(np.quantile(stacked, 0.35)),
            "ceiling": float(np.quantile(stacked, 0.9)),
            "structure_mean": float(np.mean(structure)),
            "structure_floor": float(np.quantile(structure, 0.35)),
            "structure_ceiling": float(np.quantile(structure, 0.9)),
            "disturbance_mean": float(np.mean(disturbance)),
            "disturbance_floor": float(np.quantile(disturbance, 0.35)),
            "disturbance_ceiling": float(np.quantile(disturbance, 0.9)),
        }
    count = max(total_count, 1)
    per_dim_mse_mean = per_dim_mse_total / count
    metric_values = {
        "val_mse": mse_total / count,
        "val_focus_mse": focus_mse_total / count,
        "val_severe_focus_mse": severe_focus_mse_total / count,
        "val_structure_mse": float(np.mean(per_dim_mse_mean[:2])),
        "val_delay_mse": float(per_dim_mse_mean[2]),
        "val_disturbance_mse": float(np.mean(per_dim_mse_mean[3:])),
        "val_delay_disturbance_mse": delay_disturbance_mse_total / count,
    }
    return (
        loss_total / count,
        metric_values,
        {name: float(value) for name, value in zip(target_names, mae_total / count)},
        stats,
    )


def _forward_model(
    model: nn.Module,
    model_type: str,
    features: torch.Tensor,
    sequence_features: torch.Tensor,
    context_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if model_type == "mlp":
        return model(features), None
    outputs = model(sequence_features, context_features)
    return outputs["mean"], outputs["logvar"]


def _loss_value(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    logvar: torch.Tensor | None,
    loss_name: str,
    training_cfg: dict[str, Any] | None,
) -> torch.Tensor:
    diff = predictions - targets
    mse = torch.mean(diff**2)
    weighted_mse = _weighted_target_mse(diff, targets, training_cfg)
    severe_focus_mse = _severe_focus_mse(diff, targets, training_cfg)
    if loss_name == "gaussian_nll":
        if logvar is None:
            raise ValueError("gaussian_nll requires predictive log-variance.")
        precision = torch.exp(-logvar)
        focus_weights = _target_weight_tensor(targets, training_cfg)
        nll = 0.5 * torch.mean(focus_weights * (precision * diff**2 + logvar))
        cfg = training_cfg or {}
        mse_aux_weight = float(cfg.get("mse_aux_weight", 0.35))
        logvar_reg_weight = float(cfg.get("logvar_reg_weight", 0.02))
        calibration_target = torch.sqrt(diff.detach() ** 2 + 1e-4)
        predicted_sigma = torch.exp(0.5 * logvar)
        calibration_weights = cfg.get("calibration_target_weights", [0.7, 0.7, 1.15, 1.4, 1.4])
        weight_tensor = predicted_sigma.new_tensor(calibration_weights).view(1, -1) * focus_weights
        calibration_error = torch.nn.functional.smooth_l1_loss(
            predicted_sigma,
            calibration_target,
            reduction="none",
        )
        calibration_loss = torch.mean(calibration_error * weight_tensor)
        calibration_loss_weight = float(cfg.get("calibration_loss_weight", 0.22))
        severe_focus_loss_weight = float(cfg.get("severe_focus_loss_weight", 0.22))
        return (
            nll
            + mse_aux_weight * weighted_mse
            + severe_focus_loss_weight * severe_focus_mse
            + calibration_loss_weight * calibration_loss
            + logvar_reg_weight * torch.mean(logvar**2)
        )
    return mse


def _resolve_selection_score(val_loss: float, val_metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "val_loss":
        return float(val_loss)
    if selection_metric in val_metrics:
        return float(val_metrics[selection_metric])
    if selection_metric in {"val_delay_disturbance_focus_mse", "val_head_focus_mse"}:
        return float(val_metrics["val_focus_mse"])
    if selection_metric in {"val_post_shift_severe_focus_mse", "val_delay_disturbance_severe_mse"}:
        return float(val_metrics["val_severe_focus_mse"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def _weighted_target_mse(
    diff: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> torch.Tensor:
    weights = _target_weight_tensor(targets, cfg, prefix=prefix)
    return torch.mean(weights * diff**2)


def _severe_focus_mse(
    diff: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> torch.Tensor:
    severe_weights = _severe_target_weight_tensor(targets, cfg, prefix=prefix)
    return torch.mean(severe_weights[:, 2:] * diff[:, 2:] ** 2)


def _target_weight_tensor(
    targets: torch.Tensor,
    cfg: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> torch.Tensor:
    if cfg is None or targets.shape[1] < 5:
        return torch.ones_like(targets)
    clip_value = float(cfg.get(f"{prefix}focus_magnitude_clip", cfg.get("focus_magnitude_clip", 3.0)))
    delay_mag = torch.clamp(torch.abs(targets[:, 2:3]), max=clip_value)
    disturbance_mag = torch.clamp(torch.linalg.norm(targets[:, 3:5], dim=1, keepdim=True), max=clip_value)
    coupled_mag = delay_mag * disturbance_mag
    structure_weight = float(cfg.get(f"{prefix}structure_head_weight", cfg.get("structure_head_loss_weight", 1.0)))
    delay_weight = float(cfg.get(f"{prefix}delay_head_weight", cfg.get("delay_head_loss_weight", 1.0)))
    disturbance_weight = float(cfg.get(f"{prefix}disturbance_head_weight", cfg.get("disturbance_head_loss_weight", 1.0)))
    delay_focus = float(cfg.get(f"{prefix}delay_target_focus_weight", cfg.get("delay_target_focus_weight", 0.0)))
    disturbance_focus = float(cfg.get(f"{prefix}disturbance_target_focus_weight", cfg.get("disturbance_target_focus_weight", 0.0)))
    coupled_focus = float(cfg.get(f"{prefix}coupled_focus_weight", cfg.get("coupled_focus_weight", 0.0)))
    weights = torch.ones_like(targets)
    weights[:, :2] = structure_weight
    weights[:, 2:3] = delay_weight * (1.0 + delay_focus * delay_mag + coupled_focus * coupled_mag)
    disturbance_multiplier = 1.0 + disturbance_focus * disturbance_mag + coupled_focus * coupled_mag
    weights[:, 3:5] = disturbance_weight * disturbance_multiplier
    return weights * _severe_target_weight_tensor(targets, cfg, prefix=prefix)


def _severe_target_weight_tensor(
    targets: torch.Tensor,
    cfg: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> torch.Tensor:
    if cfg is None or targets.shape[1] < 5:
        return torch.ones_like(targets)
    clip_value = float(cfg.get(f"{prefix}focus_magnitude_clip", cfg.get("focus_magnitude_clip", 3.0)))
    delay_mag = torch.clamp(torch.abs(targets[:, 2:3]), max=clip_value)
    disturbance_mag = torch.clamp(torch.linalg.norm(targets[:, 3:5], dim=1, keepdim=True), max=clip_value)
    coupled_mag = delay_mag * disturbance_mag
    delay_threshold = float(cfg.get(f"{prefix}severe_delay_threshold", cfg.get("severe_delay_threshold", 0.8)))
    disturbance_threshold = float(cfg.get(f"{prefix}severe_disturbance_threshold", cfg.get("severe_disturbance_threshold", 1.0)))
    coupled_threshold = float(cfg.get(f"{prefix}severe_coupled_threshold", cfg.get("severe_coupled_threshold", 1.0)))
    delay_excess = torch.relu(delay_mag - delay_threshold)
    disturbance_excess = torch.relu(disturbance_mag - disturbance_threshold)
    coupled_excess = torch.relu(coupled_mag - coupled_threshold)
    delay_severe_weight = float(cfg.get(f"{prefix}severe_delay_focus_weight", cfg.get("severe_delay_focus_weight", 0.0)))
    disturbance_severe_weight = float(
        cfg.get(f"{prefix}severe_disturbance_focus_weight", cfg.get("severe_disturbance_focus_weight", 0.0))
    )
    coupled_severe_weight = float(cfg.get(f"{prefix}severe_coupled_focus_weight", cfg.get("severe_coupled_focus_weight", 0.0)))
    weights = torch.ones_like(targets)
    weights[:, 2:3] = 1.0 + delay_severe_weight * delay_excess + coupled_severe_weight * coupled_excess
    disturbance_multiplier = 1.0 + disturbance_severe_weight * disturbance_excess + coupled_severe_weight * coupled_excess
    weights[:, 3:5] = disturbance_multiplier
    return weights
