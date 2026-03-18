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


LOGGER = logging.getLogger(__name__)


class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def train_from_config(config: dict[str, Any]) -> Path:
    data_dir = get_output_dir(config, "datasets")
    train_data = load_npz(data_dir / "train_dataset.npz")
    val_data = load_npz(data_dir / "val_dataset.npz")

    x_train = train_data["features"].astype(np.float32)
    y_train = train_data["targets"].astype(np.float32)
    x_val = val_data["features"].astype(np.float32)
    y_val = val_data["targets"].astype(np.float32)
    target_names = [str(name) for name in train_data["target_names"].tolist()]

    feature_mean = x_train.mean(axis=0)
    feature_std = np.clip(x_train.std(axis=0), 1e-6, None)
    target_mean = y_train.mean(axis=0)
    target_std = np.clip(y_train.std(axis=0), 1e-6, None)

    x_train = (x_train - feature_mean) / feature_std
    x_val = (x_val - feature_mean) / feature_std
    y_train = (y_train - target_mean) / target_std
    y_val = (y_val - target_mean) / target_std

    training_cfg = config["training"]
    device = resolve_device(config)
    model = build_model(config, input_dim=x_train.shape[1], output_dim=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        ArrayDataset(x_train, y_train),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        ArrayDataset(x_val, y_val),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(int(training_cfg["epochs"])):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=float(training_cfg["grad_clip"]),
        )
        val_loss, val_mae = _evaluate(model, val_loader, device, target_names)
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_mae_{name}": value for name, value in val_mae.items()},
            }
        )
        LOGGER.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "input_dim": x_train.shape[1],
                "output_dim": y_train.shape[1],
                "hidden_dims": list(training_cfg["hidden_dims"]),
                "dropout": float(training_cfg["dropout"]),
                "feature_mean": torch.from_numpy(feature_mean.copy()),
                "feature_std": torch.from_numpy(feature_std.copy()),
                "target_mean": torch.from_numpy(target_mean.copy()),
                "target_std": torch.from_numpy(target_std.copy()),
                "target_names": target_names,
                "config": config,
            }
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
    history_path = metrics_dir / "training_history.csv"
    summary_path = metrics_dir / "training_summary.json"
    checkpoint_path = checkpoints_dir / "best_model.pt"
    save_dataframe(pd.DataFrame(history_rows), history_path)
    save_json(
        {
            "best_val_loss": best_val,
            "device": str(device),
            "epochs_ran": len(history_rows),
            "target_names": target_names,
        },
        summary_path,
    )
    torch.save(best_state, checkpoint_path)
    LOGGER.info("saved_checkpoint=%s", checkpoint_path)
    return checkpoint_path


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, float]]:
    model.eval()
    mse_total = 0.0
    mae_total = np.zeros(len(target_names), dtype=np.float64)
    total_count = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            diff = predictions - targets
            batch_size = int(features.shape[0])
            mse_total += float(torch.mean(diff**2).item()) * batch_size
            mae_total += torch.mean(torch.abs(diff), dim=0).cpu().numpy() * batch_size
            total_count += batch_size
    return (
        mse_total / max(total_count, 1),
        {name: float(value) for name, value in zip(target_names, mae_total / max(total_count, 1))},
    )
