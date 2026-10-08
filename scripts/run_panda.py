"""Train, evaluate, and render a torque-driven Panda adaptive tracking demo."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import mujoco
import numpy as np
import torch
import yaml

from src.panda.tracking import PandaEpisode, model_xml, train, load_bundle
from src.evaluation.metrics import _compute_recovery_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(ROOT / "configs/panda.yaml"), type=Path)
    parser.add_argument("--output", default=str(ROOT / "outputs/panda/demo"), type=Path)
    parser.add_argument("--reuse-checkpoint", type=Path)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text())
    if args.smoke:
        config.update(train_episodes=2, val_episodes=1, epochs=2, training_duration=6.0,
                      duration=6.0, payload_time=2.0, push_time=4.0, push_duration=1.0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    xml = model_xml(args.model_dir.resolve(), config)
    (output / "panda_torque.xml").write_text(xml)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    bundle = load_bundle(args.reuse_checkpoint) if args.reuse_checkpoint else train(xml, config, output)
    scenario = {key: config[key] for key in ("duration", "payload_time", "payload_kg", "push_time", "push_duration", "push_force")}
    scenario.update(noise_seed=20261008, amplitude=1.0, speed=1.0)
    (output / "demo_scenario.json").write_text(json.dumps(scenario, indent=2))
    rollouts, metrics = {}, {}
    for name, observer in (("baseline", None), ("adaptive_gru", bundle)):
        rollout = PandaEpisode(xml, config).run(scenario, observer)
        rollouts[name] = rollout
        np.savez_compressed(output / f"{name}_rollout.npz", **rollout)
        errors = rollout["error"]
        shift_index = np.searchsorted(rollout["time"], scenario["payload_time"])
        recovery = _compute_recovery_metrics(errors, rollout["time"], shift_index,
                                             config["recovery_threshold_m"], round(0.4 / config["control_dt"]))
        metrics[name] = {"rmse_m": float(np.sqrt(np.mean(errors**2))),
                         "pre_shift_rmse_m": float(np.sqrt(np.mean(errors[:shift_index]**2))),
                         "post_shift_rmse_m": float(np.sqrt(np.mean(errors[shift_index:]**2))),
                         "peak_error_m": float(errors.max()),
                         "orientation_rmse_deg": float(np.sqrt(np.mean(rollout["orientation_error"]**2)) * 180 / np.pi),
                         "torque_saturation_fraction": float(rollout["saturation"].mean()), **recovery}
        metrics[name] = {key: None if isinstance(value, float) and not np.isfinite(value) else value
                         for key, value in metrics[name].items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps(metrics, indent=2), flush=True)
    assert np.array_equal(rollouts["baseline"]["reference"], rollouts["adaptive_gru"]["reference"])
    assert np.array_equal(rollouts["baseline"]["push"], rollouts["adaptive_gru"]["push"])
    if not args.skip_render:
        from src.panda.render import render_demo
        render_demo(xml, config, scenario, rollouts, metrics, output)
    sources = [ROOT / "src/panda/tracking.py", ROOT / "src/panda/render.py", Path(__file__).resolve(), args.config.resolve()]
    manifest = {"mujoco": mujoco.__version__, "torch": torch.__version__, "seed": config["seed"],
                "best_epoch": bundle["best_epoch"], "validation_force_rmse_n": float(bundle["validation_force_rmse_n"]),
                "model": "Franka Emika Panda", "scenario": scenario,
                "asset_xml_sha256": hashlib.sha256((args.model_dir / "panda.xml").read_text(encoding="utf-8").encode()).hexdigest(),
                "asset_git_revision": subprocess.check_output(["git", "-C", str(args.model_dir), "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint_sha256": hashlib.sha256((args.reuse_checkpoint or output / "panda_gru.pt").read_bytes()).hexdigest(),
                "source_hash_encoding": "UTF-8 with LF newlines",
                "source_sha256": {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name:
                                  hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in sources},
                "fairness": "same reference, event schedule, sensor noise seed, impedance gains, nominal model, and torque limits",
                "limitations": "new Panda residual GRU; not the XY checkpoint; one showcase is not a broad robotics benchmark"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
