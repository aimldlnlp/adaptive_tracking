"""Verify saved robustness traces without rerunning physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(run):
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed pilot can be reported")
    cases = json.loads((run/"evaluation_scenarios.json").read_text())
    frame = pd.read_csv(run/"episode_metrics.csv")
    assert len(frame) == manifest["planned_rollouts"]
    assert not frame.duplicated(["scenario_id", "condition", "controller"]).any()
    controllers = ("baseline", "integral", "adaptive_gru", "residual_observer")
    expected={(case["id"], condition["name"], name) for case in cases
              for condition in manifest["pilot"]["conditions"] for name in controllers}
    assert set(zip(frame.scenario_id,frame.condition,frame.controller)) == expected
    conditions={condition["name"]:condition for condition in manifest["pilot"]["conditions"]}
    dt=manifest["config"]["control_dt"]
    hashes={}
    for case in cases:
        reference=push=None
        for row in frame[frame.scenario_id == case["id"]].itertuples():
            file=run/f"case_{case['id']:02d}_{row.condition}_{row.controller}.npz"
            hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file) as trace:
                error=np.linalg.norm(trace["position"]-trace["reference"],axis=1)
                np.testing.assert_allclose(error,trace["error"],rtol=0,atol=1e-12)
                assert np.isfinite(trace["command"]).all()
                assert (np.abs(trace["command"]) <= np.array([87,87,87,87,12,12,12])+1e-12).all()
                delay=conditions[row.condition]["actuator_delay_steps"]
                if delay:
                    np.testing.assert_array_equal(trace["applied_command"][:delay],0)
                    np.testing.assert_array_equal(trace["applied_command"][delay:],trace["command"][:-delay])
                else:
                    np.testing.assert_array_equal(trace["applied_command"],trace["command"])
                if reference is not None:
                    common=min(len(reference),len(error))
                    np.testing.assert_array_equal(reference[:common],trace["reference"][:common])
                    np.testing.assert_array_equal(push[:common],trace["push"][:common])
                if reference is None or len(reference)<len(error):
                    reference,push=trace["reference"].copy(),trace["push"].copy()
                if row.status=="complete":
                    assert len(error)==round(case["duration"]/dt)
                    post=trace["time"] >= case["event_time"]
                    assert abs(np.sqrt(np.mean(error[post]**2))-row.post_event_rmse_m)<1e-12
                    rate=np.linalg.norm(np.diff(trace["command"],axis=0)/dt,axis=1)
                    assert abs(np.sqrt(np.mean(rate**2))-row.command_rate_rms)<1e-9
                    expected_status="no_loss" if not np.any(error[post]>.015) else None
                    if expected_status:
                        assert row.recovery_status==expected_status
    (run/"trace_verification.json").write_text(json.dumps(dict(verified_traces=len(hashes),
        checked="Pairing, delay FIFO, torque bounds, position-error norm, full-episode length, post-event RMSE and RMS torque rate",
        trace_sha256=hashes),indent=2))
    return frame




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    frame=verify(args.run)
    print(pd.read_csv(args.run/"condition_summary.csv").to_string(index=False))
    print(f"Verified {len(frame)} attempts")


if __name__=="__main__":
    main()
