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
        from src.panda.tracking import PandaEpisode, model_xml, LIMITS
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
        adaptive_zero = PandaEpisode(xml, config).run(scenario, bundle)
        np.testing.assert_array_equal(baseline["position"], adaptive_zero["position"])
        self.assertTrue(np.isfinite(baseline["qpos"]).all())
        self.assertTrue((np.abs(baseline["command"]) <= LIMITS + 1e-9).all())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
