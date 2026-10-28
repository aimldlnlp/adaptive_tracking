"""Replay frozen GRU inference on stored nominal and closed-loop Panda features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.panda.tracking import load_bundle
import numpy as np
import pandas as pd
import torch


def shadow_inference(features, bundle, history_steps, force_limit):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 12 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Expected nonempty, finite N x 12 features")
    if history_steps < 1 or not np.isfinite(force_limit) or force_limit <= 0:
        raise ValueError("History and force limit must be positive")
    if np.any(bundle["feature_std"] <= 0):
        raise ValueError("Checkpoint normalization scale must be positive")
    padded = np.concatenate([np.repeat(features[:1], history_steps - 1, axis=0), features])
    windows = np.stack([padded[i:i + history_steps] for i in range(len(features))])
    inputs = torch.from_numpy((windows - bundle["feature_mean"]) / bundle["feature_std"]).float()
    with torch.inference_mode():
        raw = np.concatenate([bundle["model"](batch).numpy() * 30 for batch in inputs.split(512)])
    if raw.shape != (len(features), 3) or not np.isfinite(raw).all():
        raise ValueError("Expected finite N x 3 predictions")
    clipped = np.clip(raw, -force_limit, force_limit)
    correction = np.zeros(3)
    smoothed = []
    for prediction in clipped:
        correction = 0.65 * correction + 0.35 * prediction
        smoothed.append(correction.copy())
    return dict(gru_raw=raw, gru_clipped=clipped, gru_smoothed=np.asarray(smoothed),
                decoded_residual=-30 * features[:, :3])


def force_metrics(prediction, label, mask):
    if prediction.shape != label.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Predictions and labels must have matching N x 3 shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(label).all():
        raise ValueError("Force metrics require finite predictions and labels")
    if not np.any(mask):
        raise ValueError("Metric window must contain samples")
    error = prediction[mask] - label[mask]
    return dict(vector_rmse_n=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                z_bias_n=float(error[:, 2].mean()), mean_prediction_z_n=float(prediction[mask, 2].mean()),
                mean_label_z_n=float(label[mask, 2].mean()), samples=int(np.sum(mask)))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/panda/benchmark_final")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/panda_gru.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/panda/shadow_v1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if manifest["status"] != "complete" or manifest["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Require a completed benchmark and its original checkpoint")
    cases = json.loads((args.run_dir / "evaluation_scenarios.json").read_text())
    if len(cases) != manifest["protocol"]["scenarios"] or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Require the declared number of unique scenario IDs")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    bundle = load_bundle(args.checkpoint)
    config = manifest["config"]
    records, parity, input_hashes = [], [], {}
    for case in cases:
        previous = None
        for source in ("baseline", "adaptive_gru"):
            path = args.run_dir / f"case_{case['id']:02d}_{source}.npz"
            input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as arrays:
                rollout = {key: arrays[key] for key in ("time", "features", "target", "correction", "reference", "push")}
            if previous is not None:
                for key in ("time", "target", "reference", "push"):
                    np.testing.assert_array_equal(previous[key], rollout[key])
            previous = rollout
            predicted = shadow_inference(rollout["features"], bundle, config["history_steps"], config["force_limit"])
            if source == "adaptive_gru":
                difference = float(np.abs(predicted["gru_smoothed"] - rollout["correction"]).max())
                if difference > 2e-4:
                    raise RuntimeError(f"Inference replay mismatch for case {case['id']}: {difference} N")
                parity.append(difference)
            time = rollout["time"]
            masks = dict(pre=time < case["event_time"], post=time >= case["event_time"],
                         late=time >= case["duration"] - 3)
            for phase, mask in masks.items():
                for estimator, prediction in predicted.items():
                    records.append(dict(scenario_id=case["id"], group=case["group"], trajectory=case["trajectory"],
                                        source=source, estimator=estimator, phase=phase,
                                        **force_metrics(prediction, rollout["target"], mask)))
            trace = {**rollout, **predicted}
            np.savez_compressed(output / f"case_{case['id']:02d}_{source}_shadow.npz", **trace)
        print(f"shadow pair {case['id'] + 1}/{len(cases)}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "force_metrics.csv", index=False)
    summary = frame.groupby(["group", "source", "estimator", "phase"], as_index=False).agg(
        vector_rmse_n=("vector_rmse_n", "mean"), z_bias_n=("z_bias_n", "mean"), cases=("scenario_id", "count"))
    summary.to_csv(output / "summary.csv", index=False)
    raw = frame[(frame.estimator == "gru_raw") & (frame.phase == "late")]
    paired = raw.pivot(index="scenario_id", columns="source", values=["vector_rmse_n", "z_bias_n"])
    paired.columns = [f"{metric}_{source}" for metric, source in paired.columns]
    paired["rmse_increase_closed_loop_n"] = paired.vector_rmse_n_adaptive_gru - paired.vector_rmse_n_baseline
    paired.to_csv(output / "paired_late_metrics.csv")
    result = dict(status="complete", scenarios=len(cases), checkpoint_sha256=checkpoint_hash,
                  source_sha256={p: hashlib.sha256((ROOT / p).read_text().encode()).hexdigest()
                                 for p in ("scripts/diagnose_panda_gru.py", "src/panda/tracking.py")},
                  input_sha256=input_hashes, max_replay_difference_n=max(parity),
                  history_steps=config["history_steps"], force_limit_n=config["force_limit"],
                  smoothing="c[k] = 0.65*c[k-1] + 0.35*clipped_prediction[k], c[-1]=0",
                  metric="Vector force RMSE; means of episode metrics; phases use first disturbance and final 3 seconds",
                  limitations="Training force label is gravity minus push, not complete force truth; observational input-source comparison is not a causal intervention",
                  torch=torch.__version__)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(summary[(summary.group == "payload_only") & (summary.phase == "late")].to_string(index=False))
    print(f"Max replay difference: {max(parity):.8g} N", flush=True)


if __name__ == "__main__":
    main()
