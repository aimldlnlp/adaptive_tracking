"""Open a saved Panda rollout in the interactive MuJoCo viewer."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import mujoco.viewer
import numpy as np
import yaml

from src.panda.tracking import PandaEpisode, model_xml


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key].copy() for key in ("time", "qpos", "position", "reference")}
    times = trace["time"]
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Expected at least two strictly increasing finite timestamps")
    for key, width in (("qpos", 9), ("position", 3), ("reference", 3)):
        if trace[key].shape != (len(times), width) or not np.isfinite(trace[key]).all():
            raise ValueError(f"Invalid {key} samples")
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/panda.yaml")
    args = parser.parse_args()
    trace = load_trace(args.trace)
    config = yaml.safe_load(args.config.read_text())
    episode = PandaEpisode(model_xml(args.model_dir.resolve(), config), config)
    with mujoco.viewer.launch_passive(episode.model, episode.data) as viewer:
        viewer.cam.lookat[:] = [.35, 0, .39]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.25, 120, -32
        start = time.monotonic()
        for index, timestamp in enumerate(trace["time"]):
            if not viewer.is_running():
                break
            time.sleep(max(0, start + float(timestamp - trace["time"][0]) - time.monotonic()))
            with viewer.lock():
                episode.data.qpos[:] = trace["qpos"][index]
                mujoco.mj_forward(episode.model, episode.data)
                scene = viewer.user_scn
                scene.ngeom = 0
                for key, color in (("reference", [0, .65, .85, 1]), ("position", [.67, .27, .6, 1])):
                    path = trace[key][:index + 1:4]
                    for a, b in zip(path[:-1], path[1:]):
                        if scene.ngeom == scene.maxgeom:
                            break
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, a, b)
                        scene.ngeom += 1
            viewer.sync()


if __name__ == "__main__":
    main()
