from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import generate_datasets
from src.evaluation.evaluator import evaluate_from_config
from src.training.trainer import train_from_config
from src.utils.config import get_output_dir, load_config, merge_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging_utils import ProgressCallback
from src.utils.seeding import set_seed
from src.visualization.plots import create_all_figures
from src.visualization.videos import create_all_videos


def run(suite_path: str, progress_callbacks: dict[str, ProgressCallback] | None = None) -> dict[str, Any]:
    suite_file = Path(suite_path).resolve()
    with suite_file.open("r", encoding="utf-8") as handle:
        suite_payload = yaml.safe_load(handle)
    suite_cfg = suite_payload["suite"]
    base_config_path = (PROJECT_ROOT / Path(suite_cfg["base_config"])).resolve()
    base_config = load_config(base_config_path)
    output_subdir = str(suite_cfg.get("output_subdir", f"research/{suite_cfg['name']}"))
    shared_config = merge_config(
        base_config,
        {
            "project": {"output_subdir": output_subdir},
            "evaluation": {
                "compare_controllers": ["baseline", *[experiment["name"] for experiment in suite_cfg["experiments"]]],
                "primary_controller": suite_cfg["experiments"][-1]["name"],
            },
        },
    )
    shared_config = merge_config(shared_config, suite_cfg.get("config_overrides", {}))
    callbacks = progress_callbacks or {}

    set_seed(int(shared_config["simulation"]["train_seed"]))
    generate_datasets(shared_config, progress_callback=callbacks.get("generate_data"))

    controller_specs: list[dict[str, Any]] = []
    experiments = list(suite_cfg["experiments"])
    for experiment_index, experiment in enumerate(experiments):
        overrides = {
            key: value
            for key, value in experiment.items()
            if key not in {"name"}
        }
        overrides.setdefault("model", {})
        overrides["model"]["name"] = experiment["name"]
        experiment_config = merge_config(shared_config, overrides)
        checkpoint_path = train_from_config(
            experiment_config,
            progress_callback=_nested_experiment_progress(
                callbacks.get("train"),
                experiment_index,
                len(experiments),
                experiment["name"],
            ),
        )
        controller_specs.append(
            {
                "name": experiment["name"],
                "checkpoint_path": str(checkpoint_path),
                "config_overrides": {"model": experiment_config["model"]},
            }
        )

    set_seed(int(shared_config["simulation"]["test_seed"]))
    evaluate_from_config(shared_config, controller_specs=controller_specs, progress_callback=callbacks.get("evaluate"))
    create_all_figures(shared_config, progress_callback=callbacks.get("figures"))
    create_all_videos(shared_config, progress_callback=callbacks.get("videos"))

    manifest = {
        "suite_name": suite_cfg["name"],
        "suite_path": str(suite_file),
        "output_subdir": output_subdir,
        "controllers": ["baseline", *[experiment["name"] for experiment in experiments]],
    }
    metrics_dir = get_output_dir(shared_config, "metrics")
    ensure_dir(metrics_dir)
    save_json(manifest, metrics_dir / "suite_manifest.json")
    if callbacks.get("suite_summary") is not None:
        callbacks["suite_summary"](1.0, "suite manifest saved")
    return manifest


def _nested_experiment_progress(
    callback: ProgressCallback | None,
    experiment_index: int,
    total_experiments: int,
    label: str,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def nested(progress: float, message: str | None = None) -> None:
        bounded = max(0.0, min(1.0, float(progress)))
        callback((experiment_index + bounded) / max(total_experiments, 1), message or f"training {label}")

    return nested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/experiments/paper_best.yaml")
    args = parser.parse_args()
    run(args.suite)


if __name__ == "__main__":
    main()
