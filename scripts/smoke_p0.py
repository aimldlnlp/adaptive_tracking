"""Small CPU pipeline check; saves separate outputs and skips media generation."""
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data.dataset import generate_datasets
from src.evaluation.evaluator import evaluate_from_config
from src.training.trainer import train_from_config
from src.utils.config import get_output_dir, load_config, merge_config
from src.utils.io import save_json
from src.utils.seeding import set_seed


def main():
    torch.set_num_threads(1)
    suite = yaml.safe_load((ROOT / "configs/experiments/paper_best_smoke.yaml").read_text())["suite"]
    config = merge_config(load_config(ROOT / suite["base_config"]), suite["config_overrides"])
    config = merge_config(config, {
        "project": {"output_subdir": f"checks/p0_{datetime.now():%Y%m%d_%H%M%S}"},
        "simulation": {"train_episodes": 12, "val_episodes": 4, "test_episodes": 4},
        "training": {"device": "cpu", "epochs": 3, "batch_size": 256},
        "evaluation": {"bootstrap_samples": 20},
    })
    set_seed(config["simulation"]["train_seed"])
    generate_datasets(config)
    specs = []
    for experiment in suite["experiments"]:
        experiment_config = merge_config(config, {k: v for k, v in experiment.items() if k != "name"})
        experiment_config = merge_config(experiment_config, {
            "model": {"hidden_dims": [16, 16], "hidden_size": 16, "head_hidden_size": 16, "num_layers": 1},
            "training": {"epochs": 3},
        })
        checkpoint = train_from_config(experiment_config)
        specs.append({"name": experiment["name"], "checkpoint_path": str(checkpoint),
                      "config_overrides": {"model": experiment_config["model"]}})
    config["evaluation"].update(compare_controllers=["baseline", *[s["name"] for s in specs]],
                                primary_controller=specs[-1]["name"])
    frame = evaluate_from_config(config, controller_specs=specs)
    assert np.isfinite(frame["rmse"]).all()
    assert set(frame["controller"]) == set(config["evaluation"]["compare_controllers"])
    assert frame.loc[frame.recovery_status != "recovered", "recovery_time"].isna().all()
    metrics_dir = get_output_dir(config, "metrics")
    save_json(config, metrics_dir / "resolved_config.json")
    print(f"PASS: {len(frame)} rollouts; recovery={frame.recovery_status.value_counts().to_dict()}")
    print(f"Outputs: {metrics_dir}")


if __name__ == "__main__":
    main()
