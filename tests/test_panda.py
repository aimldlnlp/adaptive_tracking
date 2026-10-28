"""Run with requirements-panda installed and the public Panda assets fetched."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys
import unittest
try:
    MUJOCO_VERSION = importlib.metadata.version("mujoco")
except importlib.metadata.PackageNotFoundError:
    MUJOCO_VERSION = None

ASSETS = Path(__file__).resolve().parents[1] / "third_party/menagerie/franka_emika_panda"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(MUJOCO_VERSION == "3.6.0" and ASSETS.exists(),
                     "Requires pinned MuJoCo 3.6.0 and Panda assets")
class PandaPhysicsCheck(unittest.TestCase):
    def test_payload_preserves_state_and_zero_residual_matches_baseline(self):
        # Isolate MuJoCo's native plugins from the legacy suite's loaded DLLs.
        if "--isolated" not in sys.argv:
            subprocess.run([sys.executable, __file__, "--isolated"], check=True)
            return
        import mujoco
        import numpy as np
        import torch
        import yaml
        from src.panda.tracking import PandaEpisode, UnstableRollout, model_xml, reference, LIMITS
        from scripts.benchmark_panda import scenarios, episode_metrics, report, case_report
        from unittest.mock import patch
        import tempfile
        import pandas as pd
        config = yaml.safe_load((ASSETS.parents[2] / "configs/panda.yaml").read_text())
        xml = model_xml(ASSETS, config)
        episode = PandaEpisode(xml, config)
        before = episode.data.qpos.copy()
        episode.attach_payload(1.5)
        np.testing.assert_array_equal(episode.data.qpos, before)
        self.assertAlmostEqual(episode.model.body_mass[episode.payload], 1.52)
        self.assertAlmostEqual(episode.nominal.body_mass[episode.payload], 0.02)
        class ZeroResidual(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 3))
        bundle = {"model": ZeroResidual(), "feature_mean": np.zeros(12), "feature_std": np.ones(12)}
        scenario = dict(duration=1.0, payload_time=0.3, payload_kg=1.5, push_time=0.7,
                        push_duration=0.2, push_force=[2, -2, 0], amplitude=1, speed=1, noise_seed=123)
        baseline = PandaEpisode(xml, config).run(scenario)
        from scripts.view_panda import load_trace, main as view_main
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            trace_path = Path(folder) / "trace.npz"
            np.savez(trace_path, **baseline)
            np.testing.assert_array_equal(load_trace(trace_path)["qpos"], baseline["qpos"])
            view = SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(episode.model, maxgeom=1000),
                                   lock=nullcontext, sync=Mock(), is_running=lambda: True)
            with patch("scripts.view_panda.mujoco.viewer.launch_passive", return_value=nullcontext(view)), \
                 patch("scripts.view_panda.time.sleep"), \
                 patch.object(sys, "argv", ["view_panda", "--trace", str(trace_path), "--model-dir", str(ASSETS)]):
                view_main()
            self.assertEqual(view.sync.call_count, len(baseline["time"]))
            self.assertGreater(view.user_scn.ngeom, 0)
            np.savez(trace_path, **{**baseline, "time": np.zeros(len(baseline["time"]))})
            with self.assertRaises(ValueError):
                load_trace(trace_path)
            np.savez(trace_path, **{**baseline, "qpos": baseline["qpos"][:, :7]})
            with self.assertRaises(ValueError):
                load_trace(trace_path)
        np.testing.assert_array_equal(baseline["command"], baseline["applied_command"])
        delayed = PandaEpisode(xml, config).run(scenario, actuator_delay_steps=2)
        np.testing.assert_array_equal(delayed["applied_command"][:2], 0)
        np.testing.assert_array_equal(delayed["applied_command"][2:], delayed["command"][:-2])
        self.assertEqual(delayed["joint_limit_margin"].shape, (50, 7))
        self.assertTrue((delayed["contact_count"] >= 0).all())
        with patch("src.panda.tracking.mujoco.mj_step", side_effect=lambda model, data: data.qvel.__setitem__(0, 101)):
            with self.assertRaises(UnstableRollout) as failure:
                PandaEpisode(xml, config).run(scenario)
        self.assertEqual(failure.exception.time, 0)
        self.assertEqual(len(failure.exception.rollout["time"]), 1)
        for invalid_delay in (-1, 0.5, float("nan"), True):
            with self.assertRaises(ValueError):
                PandaEpisode(xml, config).run(scenario, actuator_delay_steps=invalid_delay)
        for invalid_noise in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PandaEpisode(xml, {**config, "sensor_velocity_noise": invalid_noise}).run(scenario)
        adaptive_zero = PandaEpisode(xml, config).run(scenario, bundle)
        np.testing.assert_array_equal(baseline["position"], adaptive_zero["position"])
        self.assertTrue(np.isfinite(baseline["qpos"]).all())
        self.assertTrue((np.abs(baseline["command"]) <= LIMITS + 1e-9).all())
        zero_integral = PandaEpisode(xml, config).run(scenario, integral_gain=0)
        np.testing.assert_array_equal(baseline["position"], zero_integral["position"])
        filtered = PandaEpisode(xml, config).run(scenario, residual_cutoff_hz=2)
        decay = np.exp(-2 * np.pi * 2 * config["control_dt"])
        expected = np.zeros(3)
        for features, correction in zip(filtered["features"], filtered["correction"]):
            expected = decay * expected + (1 - decay) * np.clip(-30 * features[:3], -config["force_limit"], config["force_limit"])
            np.testing.assert_allclose(correction, expected, atol=2e-6)
        self.assertTrue(np.isfinite(filtered["qpos"]).all())
        for cutoff in (0, -1, float("nan"), float("inf"), 25):
            with self.assertRaises(ValueError):
                PandaEpisode(xml, config).run(scenario, residual_cutoff_hz=cutoff)
        with self.assertRaises(ValueError):
            PandaEpisode(xml, config).run(scenario, bundle, residual_cutoff_hz=2)
        with self.assertRaises(ValueError):
            PandaEpisode(xml, config).run(scenario, integral_gain=80, residual_cutoff_hz=2)
        for gain in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PandaEpisode(xml, config).run(scenario, integral_gain=gain)
        with self.assertRaises(ValueError):
            PandaEpisode(xml, config).run(scenario, bundle, integral_gain=80)
        long_case = {**scenario, "duration": 12, "payload_time": 2, "push_force": [0, 0, 0]}
        nominal = PandaEpisode(xml, config).run(long_case)
        integral = PandaEpisode(xml, config).run(long_case, integral_gain=160)
        from scripts.benchmark_panda_robustness import actuator_metrics
        metrics = actuator_metrics(integral, {**long_case, "event_time": 2}, config)
        self.assertEqual(metrics["command_rate_rms"], metrics["applied_command_rate_rms"])
        self.assertIsNone(metrics["push_first_second_peak_m"])
        self.assertGreaterEqual(metrics["command_rate_max"], metrics["command_rate_p95"])
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "preserve.txt"
            sentinel.write_text("existing evidence")
            refused = subprocess.run([sys.executable, str(ASSETS.parents[2]/"scripts/run_panda.py"),
                                      "--model-dir", str(ASSETS), "--output", folder, "--skip-render"],capture_output=True,text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Choose a new empty output directory", refused.stderr)
            self.assertEqual(sentinel.read_text(), "existing evidence")
        self.assertLess(np.mean(integral["error"][-100:]), np.mean(nominal["error"][-100:]))
        self.assertTrue((np.abs(integral["correction"]) <= config["force_limit"]).all())
        limited_episode = PandaEpisode(xml, config)
        with patch("src.panda.tracking.LIMITS", np.zeros(7)):
            saturated = limited_episode.run(scenario, integral_gain=320)
        np.testing.assert_array_equal(saturated["correction"], 0)
        self.assertTrue(saturated["saturation"].all())
        for trajectory in ("figure_eight", "circle", "tilted_ellipse"):
            position, velocity = reference(1.3, np.zeros(3), 0.9, 1.2, trajectory)
            before, _ = reference(1.3 - 1e-5, np.zeros(3), 0.9, 1.2, trajectory)
            after, _ = reference(1.3 + 1e-5, np.zeros(3), 0.9, 1.2, trajectory)
            np.testing.assert_allclose(velocity, (after - before) / 2e-5, atol=1e-9)
        protocol = yaml.safe_load((ASSETS.parents[2] / "configs/panda_benchmark.yaml").read_text())
        cases = scenarios(protocol)
        pilot = yaml.safe_load((ASSETS.parents[2] / "configs/panda_robustness.yaml").read_text())
        development = scenarios({**protocol, "seed": pilot["seed"], "scenarios": pilot["scenarios"]})
        self.assertEqual(len(development)*len(pilot["conditions"])*4, 288)
        self.assertFalse({case["noise_seed"] for case in development} & {case["noise_seed"] for case in cases})
        for group in ("payload_only", "push_only", "compound"):
            self.assertEqual([sum(case["group"] == group and case["trajectory"] == trajectory for case in development)
                              for trajectory in ("figure_eight", "circle", "tilted_ellipse")], [2, 2, 2])
        self.assertEqual(cases, scenarios(protocol))
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({c["noise_seed"] for c in cases + scenarios(protocol, calibration=True)}), 36)
        self.assertEqual({c["trajectory"] for c in cases}, {"figure_eight", "circle", "tilted_ellipse"})
        for case in cases:
            if case["group"] == "payload_only":
                self.assertEqual(case["push_force"], [0, 0, 0])
            if case["group"] == "push_only":
                self.assertEqual(case["payload_kg"], 0)
                self.assertEqual(case["event_time"], case["push_time"])
        metric = episode_metrics(integral, {**long_case, "event_time": 2}, config)
        rows = [dict(scenario_id=i, group="payload_only", controller=name,
                     **{**metric, "post_event_rmse_m": error})
                for i in range(3) for name, error in (("baseline", 0.03), ("integral", 0.02), ("adaptive_gru", 0.01))]
        with tempfile.TemporaryDirectory() as folder:
            _, paired = report(pd.DataFrame(rows), Path(folder))
            losses = case_report(pd.DataFrame(rows), cases[:3], Path(folder))
            self.assertTrue(losses.empty)
            rows[2]["post_event_rmse_m"] = 0.04
            rows[2]["recovery_status"] = "failed"
            losses = case_report(pd.DataFrame(rows), cases[:3], Path(folder))
            self.assertEqual(list(losses.index), [0])
            self.assertAlmostEqual(losses.gru_minus_integral_mm.iloc[0], 20)
            expanded = rows + [dict(scenario_id=i, group="payload_only", controller="residual_observer",
                                    **{**metric, "post_event_rmse_m": 0.005, "recovery_status": "no_loss"}) for i in range(3)]
            summary_four, paired_four = report(pd.DataFrame(expanded), Path(folder))
            case_report(pd.DataFrame(expanded), cases[:3], Path(folder))
            self.assertEqual(len(paired_four[paired_four.group == "overall"]), 6)
            self.assertEqual(len(summary_four[summary_four.group == "overall"]), 4)
            self.assertTrue(pd.read_csv(Path(folder) / "observer_failure_cases.csv").empty)
        comparison = paired[(paired.group == "overall") & (paired.reference_controller == "integral")]
        self.assertAlmostEqual(comparison.improvement_mm.iloc[0], 10)
        self.assertAlmostEqual(comparison.ci_low_mm.iloc[0], 10)
        from scripts.diagnose_panda_gru import shadow_inference, force_metrics
        class HistoryMean(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, :, :3].mean(dim=1)
        features = np.zeros((3, 12), dtype=np.float32)
        features[:, 0] = [1, 2, 3]
        observer = {"model": HistoryMean(), "feature_mean": np.zeros(12, dtype=np.float32),
                    "feature_std": np.ones(12, dtype=np.float32)}
        predictions = shadow_inference(features, observer, 3, 35)
        np.testing.assert_allclose(predictions["gru_raw"][:, 0], [30, 40, 60])
        np.testing.assert_allclose(predictions["gru_clipped"][:, 0], [30, 35, 35])
        np.testing.assert_allclose(predictions["gru_smoothed"][:, 0], [10.5, 19.075, 24.64875])
        np.testing.assert_allclose(predictions["decoded_residual"][:, 0], [-30, -60, -90])
        invalid = features.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            shadow_inference(invalid, observer, 3, 35)
        with self.assertRaises(ValueError):
            shadow_inference(features, {**observer, "feature_std": np.zeros(12)}, 3, 35)
        force = force_metrics(np.zeros((1, 3)), np.array([[1, 2, 2]]), np.array([True]))
        self.assertEqual(force["vector_rmse_n"], 3)
        self.assertEqual(force["z_bias_n"], -2)
        with self.assertRaises(ValueError):
            force_metrics(np.zeros((1, 3)), np.array([[np.nan, 2, 2]]), np.array([True]))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
