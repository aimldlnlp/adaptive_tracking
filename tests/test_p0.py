import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import _compute_recovery_metrics, aggregate_metrics, compute_episode_metrics
from src.training.trainer import train_from_config
from src.utils.config import load_config
from scripts.report_multiseed import paired_interval

ROOT = Path(__file__).resolve().parents[1]


class P0RegressionTests(unittest.TestCase):
    def test_paired_interval_keeps_identical_results_zero(self):
        np.testing.assert_array_equal(paired_interval(np.zeros((3, 12)), samples=100), [0.0, 0.0])
        low, high = paired_interval(np.tile([1.0, 2.0, 3.0], (8, 1)).T, samples=100)
        self.assertLessEqual(low, 2.0)
        self.assertGreaterEqual(high, 2.0)
        with self.assertRaises(ValueError):
            paired_interval(np.array([[float("nan")]]))

    def test_recovery_delayed_spike_relapse_and_failure(self):
        cases = [
            ([0.1] * 10, "no_loss", None),
            ([0.1] * 4 + [0.8, 0.1, 0.1, 0.1], "recovered", 0.7),
            ([0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1], "recovered", 0.7),
            ([0.8] * 8, "failed", None),
            ([0.8, 0.1, 0.1], "failed", None),
            ([0.8, 0.28, 0.28, 0.28], "recovered", 0.3),
        ]
        for errors, status, elapsed in cases:
            with self.subTest(status=status, errors=errors):
                result = _compute_recovery_metrics(np.array(errors), np.arange(len(errors)) * 0.1, 0, 0.28, 3)
                self.assertEqual(result["recovery_status"], status)
                if elapsed is None:
                    self.assertTrue(np.isnan(result["recovery_time"]))
                else:
                    self.assertAlmostEqual(result["recovery_time"], elapsed)
        # Errors before the shift must not count as loss after the shift.
        result = _compute_recovery_metrics(np.array([0.8, 0.1, 0.1]), np.arange(3), 1, 0.28, 2)
        self.assertEqual(result["recovery_status"], "no_loss")

    def test_aggregation_preserves_undefined_recovery(self):
        config = load_config(ROOT / "configs/default.yaml")
        def row(errors):
            n = len(errors)
            rollout = {"time": np.arange(n) * 0.1, "ref_position": np.zeros((n, 2)),
                       "true_state": np.column_stack([errors, np.zeros((n, 3))]),
                       "ref_heading": np.zeros(n), "command": np.zeros((n, 2)),
                       "metadata_json": json.dumps({"shift_time": 0.0})}
            return {"controller": "baseline", "condition_group": "single", "shift_type": "mass",
                    "shift_intensity": "severe", **compute_episode_metrics(config, rollout)}
        failed, no_loss = row([0.8] * 10), row([0.1] * 10)
        aggregate, summary = aggregate_metrics(pd.DataFrame([failed, no_loss]))
        for frame in (aggregate, summary):
            self.assertTrue(np.isnan(frame.iloc[0]["recovery_time_mean"]))
            self.assertEqual(frame.iloc[0]["recovery_needed_rate"], 0.5)
            self.assertEqual(frame.iloc[0]["recovery_success_rate"], 0.0)
        _, summary = aggregate_metrics(pd.DataFrame([no_loss]))
        self.assertTrue(np.isnan(summary.iloc[0]["recovery_success_rate"]))

    def test_saved_checkpoint_matches_best_epoch_after_later_updates(self):
        config = load_config(ROOT / "configs/default.yaml")
        config["training"].update(epochs=3, early_stopping_patience=2, device="cpu")
        model = torch.nn.Linear(2, 5)
        epoch = 0
        def train_epoch(**kwargs):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(epoch)
            return float(epoch)
        def evaluate(*args):
            metrics = {name: float(epoch) for name in (
                "val_mse", "val_focus_mse", "val_severe_focus_mse", "val_delay_mse", "val_disturbance_mse")}
            return float(epoch), metrics, {}, {"epoch": epoch}
        with tempfile.TemporaryDirectory() as directory:
            config["project_root"] = directory
            data_dir = Path(directory) / "outputs/datasets"
            data_dir.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for split in ("train", "val"):
                np.savez(data_dir / f"{split}_dataset.npz", features=rng.normal(size=(4, 2)),
                         sequence_features=rng.normal(size=(4, 2, 2)), context_features=rng.normal(size=(4, 2)),
                         targets=rng.normal(size=(4, 5)), target_names=np.array(["mass", "friction", "delay", "dx", "dy"]))
            with patch("src.training.trainer.build_model", return_value=model), \
                 patch("src.training.trainer._run_epoch", side_effect=train_epoch), \
                 patch("src.training.trainer._evaluate", side_effect=evaluate):
                checkpoint = train_from_config(config)
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(epoch, 3)
            self.assertEqual(bundle["best_epoch"], 1)
            self.assertEqual(bundle["uncertainty_stats"], {"epoch": 1})
            self.assertEqual(bundle["best_selection_score"], 1.0)
            for tensor in bundle["model_state"].values():
                self.assertTrue(torch.all(tensor == 1))
            summary = json.loads((Path(directory) / "outputs/metrics/training_summary.json").read_text())
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["epochs_ran"], 3)


if __name__ == "__main__":
    unittest.main()
