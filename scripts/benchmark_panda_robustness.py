"""Fixed-controller development pilot for measurement noise and actuator delay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.benchmark_panda import CONTROLLERS, episode_metrics, report, scenarios
from src.panda.tracking import PandaEpisode, UnstableRollout, load_bundle, model_xml


def actuator_metrics(rollout, case, config):
    time, error, dt = rollout["time"], rollout["error"], config["control_dt"]
    post = time >= case["event_time"]
    warm = (time >= 2) & (time < case["event_time"])
    late = time >= time[-1] - 3 + dt / 2
    result = dict(late_rmse_m=float(np.sqrt(np.mean(error[late]**2))),
                  pre_warm_rmse_m=float(np.sqrt(np.mean(error[warm]**2))) if warm.any() else None,
                  post_error_p95_m=float(np.quantile(error[post], .95)),
                  post_loss_fraction=float(np.mean(error[post] > config["recovery_threshold_m"])),
                  saturation_duration_s=float(rollout["saturation"].sum()*dt),
                  min_joint_limit_margin_rad=float(rollout["joint_limit_margin"].min()),
                  max_sampled_contacts=int(rollout["contact_count"].max()),
                  sampled_contact_fraction=float(np.mean(rollout["contact_count"] > 0)),
                  applied_torque_squared_integral_nm2_s=float(np.sum(rollout["applied_command"]**2)*dt))
    for key in ("command", "applied_command", "correction"):
        rate = np.linalg.norm(np.diff(rollout[key], axis=0)/dt, axis=1)
        for statistic, value in (("rms", np.sqrt(np.mean(rate**2))), ("p95", np.quantile(rate, .95)), ("max", rate.max())):
            result[f"{key}_rate_{statistic}"] = float(value)
    for event in ("payload", "push"):
        active = case["payload_kg"] > 0 if event == "payload" else np.linalg.norm(case["push_force"]) > 0
        mask = (time >= case[f"{event}_time"]) & (time < case[f"{event}_time"]+1)
        result[f"{event}_first_second_peak_m"] = float(error[mask].max()) if active else None
    return result


def save_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize(frame, output):
    rows = []
    for (condition, controller), group in frame.groupby(["condition", "controller"], sort=False):
        complete = group[group.status == "complete"]
        values = dict(condition=condition, controller=controller, attempted=len(group),
                      complete=len(complete), unstable=int((group.status == "unstable").sum()))
        for status in ("no_loss", "recovered", "failed"):
            values[status] = int((complete.recovery_status == status).sum()) if len(complete) else 0
        for metric in ("post_event_rmse_m", "peak_error_m", "late_rmse_m", "command_rate_rms", "applied_command_rate_rms",
                       "command_rate_max", "saturation_duration_s", "min_joint_limit_margin_rad", "max_sampled_contacts"):
            aggregate = "min" if metric == "min_joint_limit_margin_rad" else "max" if metric in ("max_sampled_contacts", "command_rate_max") else "mean"
            values[metric] = float(getattr(complete[metric], aggregate)()) if len(complete) else None
        rows.append(values)
    pd.DataFrame(rows).to_csv(output / "condition_summary.csv", index=False)
    for condition, group in frame.groupby("condition", sort=False):
        folder = output / condition
        folder.mkdir(exist_ok=True)
        if group.status.eq("complete").all():
            report(group, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/panda.yaml")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/panda_robustness.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text())
    pilot = yaml.safe_load(args.protocol.read_text())
    protocol = yaml.safe_load((ROOT / "configs/panda_benchmark.yaml").read_text())
    protocol.update(seed=pilot["seed"], scenarios=pilot["scenarios"])
    cases = scenarios(protocol)
    conditions = pilot["conditions"]
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Condition names must be unique")
    bundle = load_bundle(args.checkpoint)
    xml = model_xml(args.model_dir.resolve(), config)
    sources = ["src/panda/tracking.py", "scripts/benchmark_panda_robustness.py", "scripts/benchmark_panda.py",
               "scripts/report_multiseed.py", "src/evaluation/metrics.py", "configs/panda.yaml",
               "configs/panda_benchmark.yaml", "configs/panda_robustness.yaml"]
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(), config=config, pilot=pilot,
                    scenario_protocol=protocol, planned_rollouts=len(cases)*len(conditions)*len(CONTROLLERS),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                    torch=torch.__version__, mujoco=mujoco.__version__, inference_device="cpu",
                    asset_revision=subprocess.check_output(["git", "-C", str(args.model_dir), "rev-parse", "HEAD"],text=True).strip(),
                    source_sha256={name:hashlib.sha256((ROOT/name).read_text().encode()).hexdigest() for name in sources},
                    delay_semantics="Whole episode, control-step FIFO initialized to zero; residual uses issued command, never delayed torque truth",
                    limitations="Development pilot; fixed parameters, one checkpoint; unstable attempts retained, not assigned successful metrics")
    save_json(args.output/"manifest.json", manifest)
    save_json(args.output/"evaluation_scenarios.json", cases)
    subprocess.run([sys.executable, "-m", "pip", "freeze"],stdout=(args.output/"environment.txt").open("w"),check=True)
    records=[]
    try:
        for case in cases:
            reference=push=None
            for condition in conditions:
                cfg={**config,"sensor_velocity_noise":condition["sensor_velocity_noise"]}
                for name in CONTROLLERS:
                    episode=PandaEpisode(xml,cfg)
                    row=dict(scenario_id=case["id"], group=case["group"], trajectory=case["trajectory"],
                             condition=condition["name"], controller=name)
                    try:
                        rollout=episode.run(case,bundle if name=="adaptive_gru" else None,
                            integral_gain=pilot["integral_gain"] if name=="integral" else 0,
                            residual_cutoff_hz=pilot["observer_cutoff_hz"] if name=="residual_observer" else None,
                            actuator_delay_steps=condition["actuator_delay_steps"])
                    except UnstableRollout as failure:
                        rollout=failure.rollout
                        row.update(status="unstable",failure_time_s=failure.time,recovery_status="unstable")
                    else:
                        row.update(status="complete",**episode_metrics(rollout,case,cfg),**actuator_metrics(rollout,case,cfg))
                    if reference is not None:
                        common = min(len(reference), len(rollout["time"]))
                        np.testing.assert_array_equal(reference[:common], rollout["reference"][:common])
                        np.testing.assert_array_equal(push[:common], rollout["push"][:common])
                    if reference is None or len(reference)<len(rollout["time"]):
                        reference,push=rollout["reference"].copy(),rollout["push"].copy()
                    np.savez_compressed(args.output/f"case_{case['id']:02d}_{condition['name']}_{name}.npz",**rollout)
                    records.append(row)
                    pd.DataFrame(records).to_csv(args.output/"episode_metrics.csv",index=False)
                    print(f"{len(records)}/{manifest['planned_rollouts']} case={case['id']} {condition['name']} {name} {row['status']}",flush=True)
            manifest.update(completed_attempts=len(records))
            save_json(args.output/"manifest.json",manifest)
        summarize(pd.DataFrame(records),args.output)
    except Exception as failure:
        manifest.update(status="failed",error=str(failure),completed_attempts=len(records))
        save_json(args.output/"manifest.json",manifest)
        raise
    manifest.update(status="complete",completed_attempts=len(records),finished_utc=datetime.now(timezone.utc).isoformat(),
                    unstable_attempts=sum(row["status"]=="unstable" for row in records))
    save_json(args.output/"manifest.json",manifest)
    print(pd.read_csv(args.output/"condition_summary.csv").to_string(index=False),flush=True)


if __name__=="__main__":
    main()
