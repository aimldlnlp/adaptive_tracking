"""Torque-driven Panda Cartesian impedance with a causal GRU residual observer.

The nominal inverse dynamics model never receives the changed payload or push.
Only the plant receives events. The learned observer sees noisy joint velocity,
previous applied command, nominal dynamics, and Cartesian tracking measurements.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from torch import nn

HOME = np.array([0, -0.4, 0, -2.1, 0, 1.7, 0.7854, 0.035, 0.035])
LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])


class UnstableRollout(RuntimeError):
    def __init__(self, time, records):
        super().__init__(f"Unstable rollout at t={time}")
        self.time = time
        self.rollout = {key: np.asarray(values) for key, values in records.items()}


def model_xml(model_dir, config):
    tree = ET.parse(Path(model_dir) / "panda.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((Path(model_dir) / "assets").resolve()))
    root.find("option").set("timestep", str(config["physics_dt"]))
    actuators = root.find("actuator")
    actuators.clear()
    for i, limit in enumerate(LIMITS, 1):
        ET.SubElement(actuators, "motor", name=f"torque{i}", joint=f"joint{i}",
                      gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}",
                      forcelimited="true", forcerange=f"{-limit} {limit}")
    # Fingers stay open through their native joint spring, not arm position servos.
    for joint in root.findall(".//joint"):
        if joint.get("name", "").startswith("finger_joint"):
            joint.set("stiffness", "100")
            joint.set("springref", "0.035")
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tracking_tip", pos="0 0 0.1034", size="0.007", rgba="0 0.9 0.8 1")
    payload = ET.SubElement(hand, "body", name="payload", pos="0 0 0.1034")
    ET.SubElement(payload, "geom", name="payload_visual", type="sphere", size="0.035", mass="0.02",
                  contype="0", conaffinity="0", rgba="0.3 0.35 0.4 0.35")
    world = root.find("worldbody")
    ET.SubElement(root.find("asset"), "texture", type="skybox", builtin="gradient",
                  rgb1="0.94 0.95 0.97", rgb2="0.99 0.99 0.99", width="512", height="3072")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1", rgba="0.92 0.93 0.94 1")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    # Existing home keys contain position actuator controls, now incompatible.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    return ET.tostring(root, encoding="unicode")


def reference(t, center, amplitude=1.0, speed=1.0, trajectory="figure_eight"):
    w = 2 * np.pi / 7.0 * speed
    phase = w * t
    if trajectory == "figure_eight":
        offset = np.array([0.09 * np.sin(phase), 0.07 * np.sin(2 * phase), 0.025 * (1 - np.cos(phase))])
        velocity = w * np.array([0.09 * np.cos(phase), 0.14 * np.cos(2 * phase), 0.025 * np.sin(phase)])
    elif trajectory == "circle":
        offset = np.array([0.07 * np.sin(phase), 0.07 * (1 - np.cos(phase)), 0])
        velocity = w * np.array([0.07 * np.cos(phase), 0.07 * np.sin(phase), 0])
    elif trajectory == "tilted_ellipse":
        offset = np.array([0.09 * np.sin(phase), 0.045 * (1 - np.cos(phase)), 0.035 * np.sin(phase)])
        velocity = w * np.array([0.09 * np.cos(phase), 0.045 * np.sin(phase), 0.035 * np.cos(phase)])
    else:
        raise ValueError(f"Unknown trajectory: {trajectory}")
    offset, velocity = amplitude * offset, amplitude * velocity
    return center + offset, velocity


class ResidualGRU(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(12, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, inputs):
        _, state = self.gru(inputs)
        return self.head(state[-1])


class PandaEpisode:
    def __init__(self, xml, config):
        self.config = config
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.nominal = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nominal_data = mujoco.MjData(self.nominal)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tracking_tip")
        self.payload = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.payload_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "payload_visual")
        self.data.qpos[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        self.center = self.data.site_xpos[self.site].copy()
        self.orientation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        ratio = config["control_dt"] / config["physics_dt"]
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.substeps = round(ratio)

    def attach_payload(self, mass):
        qpos, qvel, time = self.data.qpos.copy(), self.data.qvel.copy(), self.data.time
        self.model.body_mass[self.payload] = 0.02 + mass
        self.model.body_inertia[self.payload] = 0.4 * (0.02 + mass) * 0.035**2
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, time
        self.model.geom_rgba[self.payload_geom] = [1, 0.65, 0.12, 1]
        mujoco.mj_forward(self.model, self.data)

    def run(self, scenario, bundle=None, *, integral_gain=0.0, residual_cutoff_hz=None, actuator_delay_steps=0):
        if isinstance(actuator_delay_steps, bool) or not isinstance(actuator_delay_steps, (int, np.integer)) or actuator_delay_steps < 0:
            raise ValueError("Actuator delay must be a nonnegative integer number of control steps")
        if not np.isfinite(integral_gain) or integral_gain < 0 or (bundle is not None and integral_gain):
            raise ValueError("Integral gain must be finite/nonnegative and cannot be combined with GRU")
        cfg, data = self.config, self.data
        if not np.isfinite(cfg["sensor_velocity_noise"]) or cfg["sensor_velocity_noise"] < 0:
            raise ValueError("Sensor velocity noise must be finite and nonnegative")
        if residual_cutoff_hz is not None:
            if (not np.isfinite(residual_cutoff_hz) or not 0 < residual_cutoff_hz < 0.5 / cfg["control_dt"]
                    or bundle is not None or integral_gain):
                raise ValueError("Residual cutoff must be below Nyquist and cannot be combined with GRU/integral")
            residual_decay = np.exp(-2 * np.pi * residual_cutoff_hz * cfg["control_dt"])
        rng = np.random.default_rng(scenario["noise_seed"])
        history = deque(maxlen=cfg["history_steps"])
        previous_velocity = None
        previous_command = np.zeros(7)
        previous_mass = None
        previous_bias = None
        previous_jacobian = None
        correction = np.zeros(3)
        command_buffer = deque([np.zeros(7) for _ in range(actuator_delay_steps + 1)], maxlen=actuator_delay_steps + 1)
        arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        joint_ranges = self.model.jnt_range[arm_ids]
        records = {key: [] for key in ("time", "qpos", "position", "reference", "error", "command", "correction", "features", "target", "push", "orientation_error", "saturation", "applied_command", "joint_limit_margin", "contact_count", "qvel")}
        attached = False
        steps = round(scenario["duration"] / cfg["control_dt"])
        for k in range(steps):
            t = k * cfg["control_dt"]
            if t >= scenario["payload_time"] and not attached:
                self.attach_payload(scenario["payload_kg"])
                attached = True
            push = np.zeros(3)
            relative = (t - scenario["push_time"]) / scenario["push_duration"]
            if 0 <= relative <= 1:
                push = np.array(scenario["push_force"]) * np.sin(np.pi * relative)
            # Nominal model gets measured state, but never event mass/force.
            measured_velocity = data.qvel[:7] + rng.normal(0, cfg["sensor_velocity_noise"], 7)
            nd = self.nominal_data
            nd.qpos[:] = data.qpos
            nd.qvel[:] = data.qvel
            nd.qvel[:7] = measured_velocity
            mujoco.mj_forward(self.nominal, nd)
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.nominal, nd, jp, jr, self.site)
            jp, jr = jp[:, :7], jr[:, :7]
            mass = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.nominal, mass, nd.qM)
            bias = (nd.qfrc_bias - nd.qfrc_passive)[:7].copy()
            residual = np.zeros(3)
            if previous_velocity is not None:
                acceleration = (measured_velocity - previous_velocity) / cfg["control_dt"]
                joint_residual = previous_mass @ acceleration + previous_bias - previous_command
                residual = np.linalg.solve(previous_jacobian @ previous_jacobian.T + 0.015 * np.eye(3), previous_jacobian @ joint_residual)
            position = data.site_xpos[self.site].copy()
            desired, desired_velocity = reference(t, self.center, scenario["amplitude"], scenario["speed"],
                                                  scenario.get("trajectory", "figure_eight"))
            velocity = jp @ measured_velocity
            force = cfg["position_kp"] * (desired - position) + cfg["position_kd"] * (desired_velocity - velocity)
            previous_correction = correction.copy()
            if integral_gain:
                correction = np.clip(correction + integral_gain * (desired - position) * cfg["control_dt"],
                                     -cfg["force_limit"], cfg["force_limit"])
            if residual_cutoff_hz is not None:
                # Inverse-dynamics residual estimates disturbance; compensation has the opposite sign.
                correction = residual_decay * correction + (1 - residual_decay) * np.clip(
                    -residual, -cfg["force_limit"], cfg["force_limit"])
            features = np.concatenate([np.clip(residual / 30, -3, 3), velocity / 0.3,
                                       (position - self.center) / 0.12, force / 30]).astype(np.float32)
            history.append(features)
            if len(history) == 1:
                history.extend([features.copy() for _ in range(cfg["history_steps"] - 1)])
            if bundle is not None:
                inputs = (np.stack(history) - bundle["feature_mean"]) / bundle["feature_std"]
                with torch.no_grad():
                    predicted = bundle["model"](torch.from_numpy(inputs[None]).float()).numpy()[0] * 30
                predicted = np.clip(predicted, -cfg["force_limit"], cfg["force_limit"])
                correction = 0.65 * correction + 0.35 * predicted
            rotation = data.site_xmat[self.site].reshape(3, 3)
            orientation_error = sum(np.cross(rotation[:, i], self.orientation[:, i]) for i in range(3)) * 0.5
            moment = cfg["orientation_kp"] * orientation_error - cfg["orientation_kd"] * (jr @ measured_velocity)
            jacobian = np.vstack([jp, jr])
            nullspace = np.eye(7) - jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.001 * np.eye(6), jacobian)
            posture = nullspace @ (12 * (HOME[:7] - data.qpos[:7]) - 4 * measured_velocity)
            torque = jp.T @ (force + correction) + jr.T @ moment + nd.qfrc_bias[:7] + posture
            saturated = bool(np.any(np.abs(torque) > LIMITS))
            if integral_gain and saturated:
                # Freeze integration at actuator saturation to prevent windup.
                correction = previous_correction
                torque = jp.T @ (force + correction) + jr.T @ moment + nd.qfrc_bias[:7] + posture
            torque = np.clip(torque, -LIMITS, LIMITS)
            command_buffer.append(torque.copy())
            applied_command = command_buffer[0].copy()
            for key, value in {"time": t, "qpos": data.qpos.copy(), "position": position,
                               "reference": desired, "error": np.linalg.norm(desired - position),
                               "command": torque, "correction": correction.copy(), "features": features,
                               "applied_command": applied_command, "qvel": data.qvel[:7].copy(),
                               "joint_limit_margin": np.minimum(data.qpos[:7] - joint_ranges[:, 0], joint_ranges[:, 1] - data.qpos[:7]),
                               "contact_count": data.ncon,
                               "target": np.array([0, 0, scenario["payload_kg"] * 9.81 if attached else 0]) - push,
                               "push": push, "saturation": saturated,
                               "orientation_error": np.arccos(np.clip((np.trace(self.orientation.T @ rotation) - 1) / 2, -1, 1))}.items():
                records[key].append(value)
            # The estimator knows issued torque, not the hidden actuator delay.
            previous_velocity, previous_command = measured_velocity.copy(), torque.copy()
            previous_mass, previous_bias, previous_jacobian = mass[:7, :7].copy(), bias, jp.copy()
            data.ctrl[:] = applied_command
            data.qfrc_applied[:] = 0
            data.xfrc_applied[:] = 0
            data.xfrc_applied[self.payload, :3] = push
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, data)
            if not np.isfinite(data.qpos).all() or np.linalg.norm(data.qvel) > 100:
                raise UnstableRollout(t, records)
        return {key: np.asarray(values) for key, values in records.items()}


def train(xml, config, output):
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    data = {}
    for split, episodes in (("train", config["train_episodes"]), ("val", config["val_episodes"])):
        windows, targets = [], []
        scenarios = []
        for i in range(episodes):
            scenario = dict(duration=config["training_duration"], payload_time=float(rng.uniform(3, 6)),
                            payload_kg=float(rng.uniform(0, 2.0)), push_time=float(rng.uniform(7, 9)),
                            push_duration=float(rng.uniform(1, 3)), push_force=rng.uniform(-12, 12, 3).tolist(),
                            amplitude=float(rng.uniform(0.65, 1.15)), speed=float(rng.uniform(0.7, 1.3)),
                            noise_seed=int(rng.integers(1_000_000)))
            scenarios.append(scenario)
            rollout = PandaEpisode(xml, config).run(scenario)
            features = rollout["features"]
            padded = np.concatenate([np.repeat(features[:1], config["history_steps"] - 1, axis=0), features])
            windows.append(np.stack([padded[j:j + config["history_steps"]] for j in range(len(features))]))
            targets.append(rollout["target"] / 30)
            print(f"data {split} {i + 1}/{episodes}", flush=True)
        data[split] = (np.concatenate(windows).astype(np.float32), np.concatenate(targets).astype(np.float32))
        import json
        (output / f"{split}_scenarios.json").write_text(json.dumps(scenarios, indent=2))
    mean = data["train"][0].mean(axis=(0, 1))
    std = np.maximum(data["train"][0].std(axis=(0, 1)), 0.05)
    tensors = {split: (torch.from_numpy((x - mean) / std), torch.from_numpy(y)) for split, (x, y) in data.items()}
    model = ResidualGRU(config["hidden_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, state, best_epoch = float("inf"), None, None
    losses = []
    for epoch in range(config["epochs"]):
        model.train()
        x, y = tensors["train"]
        for indices in torch.randperm(len(x)).split(config["batch_size"]):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vx, vy = tensors["val"]
            score = sum(nn.functional.mse_loss(model(a), b, reduction="sum").item()
                        for a, b in zip(vx.split(512), vy.split(512))) / vy.numel()
        losses.append({"epoch": epoch + 1, "val_force_mse_n2": score * 900})
        if score < best:
            best, state, best_epoch = score, deepcopy(model.state_dict()), epoch + 1
        print(f"epoch {epoch + 1}: validation force RMSE={np.sqrt(score) * 30:.3f} N", flush=True)
    torch.save({"model_state": state, "feature_mean": mean, "feature_std": std,
                "hidden_size": config["hidden_size"], "best_epoch": best_epoch,
                "validation_force_rmse_n": np.sqrt(best) * 30}, output / "panda_gru.pt")
    import json
    (output / "training_history.json").write_text(json.dumps(losses, indent=2))
    return load_bundle(output / "panda_gru.pt")


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualGRU(bundle["hidden_size"])
    model.load_state_dict(bundle["model_state"])
    model.eval()
    bundle["model"] = model
    return bundle
