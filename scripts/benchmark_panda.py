"""Calibrate classical compensation and compare four Panda controllers on paired cases."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.report_multiseed import paired_interval
from src.evaluation.metrics import _compute_recovery_metrics
from src.panda.tracking import PandaEpisode, load_bundle, model_xml

CONTROLLERS = ("baseline", "integral", "adaptive_gru", "residual_observer")


def scenarios(protocol, *, calibration=False):
    # Independent streams: gain selection never sees benchmark cases or their scores.
    rng = np.random.default_rng(protocol["seed"] + int(calibration))
    count = protocol["calibration_episodes"] if calibration else protocol["scenarios"]
    if count < 3 or count % 3:
        raise ValueError("Scenario counts must be positive multiples of three")
    cases = []
    for i in range(count):
        group = ("payload_only", "push_only", "compound")[i // (count // 3)]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        case = dict(id=i, group=group, duration=protocol["duration"],
                    payload_kg=float(rng.uniform(*([0.0, 2.0] if calibration else protocol["payload_range_kg"]))),
                    payload_time=float(rng.uniform(*([3.0, 6.0] if calibration else protocol["payload_time_range_s"]))),
                    push_time=float(rng.uniform(*([7.0, 9.0] if calibration else protocol["push_time_range_s"]))),
                    push_duration=float(rng.uniform(*protocol["push_duration_range_s"])),
                    push_force=(direction * rng.uniform(*protocol["push_peak_range_n"])).tolist(),
                    amplitude=float(rng.uniform(*([0.65, 1.15] if calibration else protocol["amplitude_range"]))),
                    speed=float(rng.uniform(*protocol["speed_range"])),
                    trajectory="figure_eight" if calibration else ("figure_eight", "circle", "tilted_ellipse")[i % 3],
                    noise_seed=int(rng.integers(2**31)))
        if group == "payload_only":
            case["push_force"] = [0.0, 0.0, 0.0]
        elif group == "push_only":
            case["payload_kg"] = 0.0
        case["event_time"] = case["push_time"] if group == "push_only" else case["payload_time"]
        if not 0 < case["event_time"] < case["duration"] or case["push_time"] + case["push_duration"] >= case["duration"]:
            raise ValueError("Events must end before the episode ends")
        cases.append(case)
    return cases


def episode_metrics(rollout, scenario, config):
    time, error = rollout["time"], rollout["error"]
    start = int(np.searchsorted(time, scenario["event_time"]))
    if not np.isfinite(error).all() or not 0 < start < len(error):
        raise ValueError("Expected finite tracking errors and an interior event")
    result = dict(rmse_m=float(np.sqrt(np.mean(error**2))),
                  pre_event_rmse_m=float(np.sqrt(np.mean(error[:start]**2))),
                  post_event_rmse_m=float(np.sqrt(np.mean(error[start:]**2))),
                  peak_error_m=float(error[start:].max()),
                  torque_saturation_fraction=float(rollout["saturation"].mean()),
                  torque_squared_integral_nm2_s=float(np.sum(rollout["command"]**2) * config["control_dt"]),
                  orientation_rmse_deg=float(np.sqrt(np.mean(rollout["orientation_error"]**2)) * 180 / np.pi))
    result.update(_compute_recovery_metrics(error, time, start, config["recovery_threshold_m"],
                                           round(0.4 / config["control_dt"])))
    return {key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in result.items()}


def report(frame, output):
    summaries, comparisons = [], []
    controllers = [name for name in CONTROLLERS if name in frame.controller.unique()]
    for group in ("overall", *sorted(frame.group.unique())):
        subset = frame if group == "overall" else frame[frame.group == group]
        for name in controllers:
            rows = subset[subset.controller == name]
            summaries.append(dict(group=group, controller=name, cases=len(rows),
                                  post_event_rmse_mm=rows.post_event_rmse_m.mean() * 1000,
                                  peak_error_mm=rows.peak_error_m.mean() * 1000,
                                  torque_saturation_fraction=rows.torque_saturation_fraction.mean(),
                                  torque_squared_integral_nm2_s=rows.torque_squared_integral_nm2_s.mean(),
                                  no_loss=int((rows.recovery_status == "no_loss").sum()),
                                  recovered=int((rows.recovery_status == "recovered").sum()),
                                  failed=int((rows.recovery_status == "failed").sum()),
                                  recovery_time_mean_s=rows.recovery_time.mean()))
        for a, b in itertools.combinations(controllers, 2):
            paired = subset.pivot(index="scenario_id", columns="controller", values="post_event_rmse_m")
            delta = paired[a].to_numpy() - paired[b].to_numpy()
            low, high = paired_interval(delta[None, :])
            comparisons.append(dict(group=group, reference_controller=a, controller=b,
                                    improvement_mm=float(delta.mean() * 1000),
                                    ci_low_mm=float(low * 1000), ci_high_mm=float(high * 1000),
                                    win_fraction=float((delta > 0).mean()), cases=len(delta)))
    summary, paired = pd.DataFrame(summaries), pd.DataFrame(comparisons)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    return summary, paired




def case_report(frame, cases, output):
    paired = frame.pivot(index="scenario_id", columns="controller", values="post_event_rmse_m") * 1000
    paired.columns = [f"{name}_rmse_mm" for name in paired.columns]
    paired["gru_minus_integral_mm"] = paired.adaptive_gru_rmse_mm - paired.integral_rmse_mm
    details = pd.DataFrame(cases).set_index("id")
    paired = paired.join(details[["group", "trajectory", "payload_kg", "amplitude", "speed", "noise_seed"]])
    paired["push_peak_n"] = details.push_force.map(np.linalg.norm)
    paired["gru_recovery_status"] = frame[frame.controller == "adaptive_gru"].set_index("scenario_id").recovery_status
    if "residual_observer_rmse_mm" in paired:
        paired["observer_minus_integral_mm"] = paired.residual_observer_rmse_mm - paired.integral_rmse_mm
        paired["observer_recovery_status"] = frame[frame.controller == "residual_observer"].set_index("scenario_id").recovery_status
        paired[(paired.observer_minus_integral_mm > 0) | (paired.observer_recovery_status == "failed")].to_csv(output / "observer_failure_cases.csv")
    paired.to_csv(output / "case_comparison.csv")
    failures = paired[(paired.gru_minus_integral_mm > 0) | (paired.gru_recovery_status == "failed")]
    failures.sort_values("gru_minus_integral_mm", ascending=False).to_csv(output / "failure_cases.csv")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/panda.yaml")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/panda_benchmark.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/panda/benchmark_v1")
    parser.add_argument("--reference-run", type=Path, help="Freeze integral calibration and verify identical benchmark cases")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Existing runs are evidence: do not overwrite their results accidentally.
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory for each benchmark")
    calibration, evaluation = scenarios(protocol, calibration=True), scenarios(protocol)
    xml = model_xml(args.model_dir.resolve(), config)
    bundle = load_bundle(args.checkpoint)
    reference_manifest = None
    if args.reference_run:
        reference_manifest = json.loads((args.reference_run / "manifest.json").read_text())
        if (reference_manifest["status"] != "complete" or reference_manifest["config"] != config
                or reference_manifest["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                or json.loads((args.reference_run / "evaluation_scenarios.json").read_text()) != evaluation
                or json.loads((args.reference_run / "calibration_scenarios.json").read_text()) != calibration):
            raise ValueError("Reference run must match checkpoint, configuration and both scenario sets")
    manifest = dict(status="running", config=config, protocol=protocol,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    asset_xml_sha256=hashlib.sha256((args.model_dir / "panda.xml").read_text(encoding="utf-8").encode()).hexdigest(),
                    asset_git_revision=subprocess.check_output(["git", "-C", str(args.model_dir), "rev-parse", "HEAD"], text=True).strip(),
                    mujoco=mujoco.__version__, torch=torch.__version__,
                    training_seeds=1, confidence_interval="Exploratory paired scenario bootstrap; fixed checkpoint",
                    fairness="Shared initial state, references, event schedule, sensor noise, impedance gains and torque limits",
                    source_sha256={p: hashlib.sha256((ROOT / p).read_text().encode()).hexdigest()
                                   for p in ("src/panda/tracking.py", "scripts/benchmark_panda.py",
                                             "configs/panda_benchmark.yaml", "configs/panda.yaml")})
    (output / "benchmark_script.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, cases in (("calibration", calibration), ("evaluation", evaluation)):
        (output / f"{name}_scenarios.json").write_text(json.dumps(cases, indent=2))
    scores = []
    for gain in ([] if reference_manifest else protocol["integral_gain_candidates"]):
        values = [episode_metrics(PandaEpisode(xml, config).run(case, integral_gain=gain), case, config)
                  for case in calibration]
        score = np.mean([v["post_event_rmse_m"] + v["torque_saturation_fraction"] for v in values])
        scores.append(dict(gain=gain, score_m=float(score), episodes=values))
        print(f"calibration Ki={gain}: score={score:.6f}", flush=True)
    if reference_manifest:
        gain = reference_manifest["selected_integral_gain"]
        (output / "integral_calibration.json").write_bytes((args.reference_run / "integral_calibration.json").read_bytes())
        manifest["reference_manifest_sha256"] = hashlib.sha256((args.reference_run / "manifest.json").read_bytes()).hexdigest()
    else:
        gain = min(scores, key=lambda row: row["score_m"])["gain"]
        (output / "integral_calibration.json").write_text(json.dumps(dict(selected_gain=gain, scores=scores,
            criterion="Mean post-event RMSE (m) plus 1 m times saturation fraction"), indent=2, allow_nan=False))
    observer_scores = []
    for cutoff in protocol["observer_cutoff_candidates_hz"]:
        values = [episode_metrics(PandaEpisode(xml, config).run(case, residual_cutoff_hz=cutoff), case, config)
                  for case in calibration]
        score = np.mean([v["post_event_rmse_m"] + v["torque_saturation_fraction"] for v in values])
        observer_scores.append(dict(cutoff_hz=cutoff, score_m=float(score), episodes=values))
        print(f"observer calibration {cutoff} Hz: score={score:.6f}", flush=True)
    cutoff = min(observer_scores, key=lambda row: row["score_m"])["cutoff_hz"]
    (output / "observer_calibration.json").write_text(json.dumps(dict(selected_cutoff_hz=cutoff, scores=observer_scores,
        criterion="Mean post-event RMSE (m) plus 1 m times saturation fraction"), indent=2, allow_nan=False))
    records = []
    for case in evaluation:
        reference, push = None, None
        for name in CONTROLLERS:
            rollout = PandaEpisode(xml, config).run(case, bundle if name == "adaptive_gru" else None,
                                                   integral_gain=gain if name == "integral" else 0,
                                                   residual_cutoff_hz=cutoff if name == "residual_observer" else None)
            if args.reference_run and name != "residual_observer":
                with np.load(args.reference_run / f"case_{case['id']:02d}_{name}.npz") as original:
                    for key in original.files:
                        np.testing.assert_allclose(rollout[key], original[key], rtol=0, atol=1e-12)
            if reference is not None:
                np.testing.assert_array_equal(reference, rollout["reference"])
                np.testing.assert_array_equal(push, rollout["push"])
            reference, push = rollout["reference"], rollout["push"]
            np.savez_compressed(output / f"case_{case['id']:02d}_{name}.npz", **rollout)
            records.append(dict(scenario_id=case["id"], group=case["group"], trajectory=case["trajectory"],
                                controller=name, **episode_metrics(rollout, case, config)))
        pd.DataFrame(records).to_csv(output / "episode_metrics.csv", index=False)
        print(f"paired case {case['id'] + 1}/{len(evaluation)} {case['group']} {case['trajectory']}", flush=True)
    frame = pd.DataFrame(records)
    summary, paired = report(frame, output)
    failures = case_report(frame, evaluation, output)
    manifest.update(status="complete", selected_integral_gain=gain, selected_observer_cutoff_hz=cutoff, paired_rollouts=len(records))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(paired[paired.group == "overall"].to_string(index=False), flush=True)
    print("Cases where GRU loses to integral or fails settling:", flush=True)
    print(failures.sort_values("gru_minus_integral_mm", ascending=False).head(5).to_string(), flush=True)


if __name__ == "__main__":
    main()
