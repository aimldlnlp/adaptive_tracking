"""Retrain canonical estimators with fixed data/evaluation and multiple training seeds."""
import argparse
import hashlib
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import generate_datasets
from src.evaluation.evaluator import evaluate_from_config
from src.training.trainer import train_from_config
from src.utils.config import get_output_dir, load_config, merge_config
from src.utils.io import ensure_dir, save_json
from src.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("training seeds must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    suite = yaml.safe_load((ROOT / "configs/experiments/paper_best.yaml").read_text())["suite"]
    shared = merge_config(load_config(ROOT / suite["base_config"]), {
        "project": {"output_subdir": args.output_subdir}, "training": {"device": args.device}})
    run_dir = get_output_dir(shared)
    # Never overwrite a previous benchmark, even if its job stopped early.
    if run_dir.exists():
        raise FileExistsError(f"Use a fresh output subdir: {run_dir}")
    ensure_dir(run_dir)
    sources = sorted(p for folder in ("src", "configs", "scripts", "tests")
                     for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml"))
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "training_seeds": args.seeds,
                "data_seeds": {key: shared["simulation"][key] for key in ("train_seed", "val_seed", "test_seed")},
                "evaluation_seeds": shared["evaluation"]["seeds"], "python": sys.version,
                "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "device": args.device,
                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "status": "running"}
    save_json(manifest, run_dir / "run_manifest.json")
    save_json(shared, run_dir / "shared_config.json")
    set_seed(shared["simulation"]["train_seed"])
    logging.info("Generating fixed datasets")
    generate_datasets(shared)
    all_rows = []
    reference_specs = None
    for seed in args.seeds:
        config = merge_config(shared, {"project": {"output_subdir": f"{args.output_subdir}/seed_{seed}"}})
        data_dir = get_output_dir(config, "datasets")
        ensure_dir(data_dir.parent)
        data_dir.symlink_to(get_output_dir(shared, "datasets"), target_is_directory=True)
        controllers = []
        for experiment in suite["experiments"]:
            experiment_config = merge_config(config, {k: v for k, v in experiment.items() if k != "name"})
            set_seed(seed)
            logging.info("Training seed=%s model=%s", seed, experiment["name"])
            checkpoint = train_from_config(experiment_config)
            controllers.append({"name": experiment["name"], "checkpoint_path": str(checkpoint),
                                "config_overrides": {"model": experiment_config["model"]}})
            ensure_dir(get_output_dir(config, "configs"))
            save_json(experiment_config, get_output_dir(config, "configs", f"{experiment['name']}.json"))
        logging.info("Evaluating seed=%s", seed)
        frame = evaluate_from_config(config, controller_specs=controllers)
        assert np.isfinite(frame["rmse"]).all()
        specs = get_output_dir(config, "metrics", "evaluation_specs.json").read_bytes()
        if reference_specs is None:
            reference_specs = specs
        assert specs == reference_specs, "Evaluation specs differ across training seeds"
        frame["training_seed"] = seed
        all_rows.append(frame)
        manifest["completed_seeds"] = [int(rows.training_seed.iloc[0]) for rows in all_rows]
        save_json(manifest, run_dir / "run_manifest.json")
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(run_dir / "all_seed_episode_metrics.csv", index=False)
    per_seed = combined.groupby(["training_seed", "controller"], as_index=False).agg(
        rmse=("rmse", "mean"), success_rate=("success", "mean"),
        recovery_needed_rate=("recovery_needed", "mean"), recovery_success_rate=("recovery_success", "mean"))
    per_seed.to_csv(run_dir / "per_seed_summary.csv", index=False)
    per_seed.groupby("controller").agg(rmse_mean=("rmse", "mean"), rmse_seed_std=("rmse", "std"),
                                      success_rate_mean=("success_rate", "mean")).to_csv(run_dir / "multiseed_summary.csv")
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint_sha256={str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in run_dir.rglob("*.pt")})
    save_json(manifest, run_dir / "run_manifest.json")
    logging.info("Complete: %s", run_dir)


if __name__ == "__main__":
    main()
