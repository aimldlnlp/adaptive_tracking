"""Headless MuJoCo playback with reference/path overlays and comparison HUD."""
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.panda.tracking import PandaEpisode, reference

COLORS = {"baseline": (213, 94, 0), "adaptive_gru": (0, 137, 110)}
INK = (30, 34, 39)
MUTED = (90, 96, 103)
FONT_DIR = Path(__file__).resolve().parents[2] / "assets/fonts/latin-modern"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / f"lmroman10-{'bold' if bold else 'regular'}.otf"), size)


def line(scene, start, end, color, radius=0.002):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    scene.ngeom += 1


def marker(scene, point, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.009] * 3), point,
                       np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def render_demo(xml, config, scenario, rollouts, metrics, output):
    episodes = {name: PandaEpisode(xml, config) for name in rollouts}
    renderers = {name: mujoco.Renderer(ep.model, height=720, width=960) for name, ep in episodes.items()}
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.35, 0, 0.39]
    camera.distance, camera.azimuth, camera.elevation = 1.25, 120, -32
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    fps = config["fps"]
    writers = {name: imageio.get_writer(output / f"{name}.mp4", fps=fps, codec="libx264", quality=None,
                                       macro_block_size=1, ffmpeg_params=["-crf", "24", "-movflags", "+faststart"])
               for name in (*rollouts, "comparison")}
    title, text, small = font(34, True), font(26), font(22)
    path = np.array([reference(t, next(iter(episodes.values())).center)[0]
                     for t in np.linspace(0, scenario["duration"], 120)])
    maximum = np.ceil(max(float(r["error"].max()) for r in rollouts.values()) * 1000 / 10) * 10 + 10
    try:
        for frame in range(round(scenario["duration"] * fps)):
            t = frame / fps
            panels = []
            for name, rollout in rollouts.items():
                ep, renderer = episodes[name], renderers[name]
                index = min(round(t / config["control_dt"]), len(rollout["time"]) - 1)
                ep.data.qpos[:] = rollout["qpos"][index]
                mujoco.mj_forward(ep.model, ep.data)
                if t >= scenario["payload_time"]:
                    ep.model.geom_rgba[ep.payload_geom] = [1, 0.65, 0.12, 1]
                renderer.update_scene(ep.data, camera=camera)
                scene = renderer.scene
                for a, b in zip(path[:-1], path[1:]):
                    line(scene, a, b, [0, 0.45, 0.7, 0.9], 0.0018)
                actual = rollout["position"][max(0, index - 200):index + 1:3]
                color = np.array([*COLORS[name], 255]) / 255
                for a, b in zip(actual[:-1], actual[1:]):
                    line(scene, a, b, color)
                marker(scene, rollout["reference"][index], [0.5, 0.8, 1, 1])
                if np.linalg.norm(rollout["push"][index]) > 0.1:
                    p = rollout["position"][index]
                    line(scene, p - rollout["push"][index] * 0.013, p, [1, 0.3, 0.2, 1], 0.008)
                panel = Image.new("RGB", (960, 1080), "white")
                panel.paste(Image.fromarray(renderer.render()), (0, 110))
                draw = ImageDraw.Draw(panel)
                heading = "(a) Nominal Cartesian impedance" if name == "baseline" else "(b) Adaptive impedance + residual GRU"
                draw.text((28, 15), heading, fill=COLORS[name], font=title)
                event = "Nominal dynamics" if t < scenario["payload_time"] else f"Added payload: {scenario['payload_kg']:.1f} kg"
                if scenario["push_time"] <= t <= scenario["push_time"] + scenario["push_duration"]:
                    event += "  /  External push"
                draw.text((28, 63), f"t = {t:04.1f} s     {event}", fill=INK, font=text)
                draw.line((28, 102, 932, 102), fill=(190, 195, 200), width=1)
                draw.text((28, 842), f"Tracking error: {rollout['error'][index] * 1000:5.1f} mm", fill=COLORS[name], font=title)
                draw.text((28, 884), f"Compensation: {np.linalg.norm(rollout['correction'][index]):4.1f} N", fill=INK, font=text)
                draw.text((535, 884), "Blue: reference; orange/green: actual", fill=MUTED, font=small)
                left, right, top, bottom = 60, 915, 935, 1025
                push_x = left + scenario["push_time"] / scenario["duration"] * (right - left)
                push_end = left + (scenario["push_time"] + scenario["push_duration"]) / scenario["duration"] * (right - left)
                draw.rectangle((push_x, top, push_end, bottom), fill=(242, 242, 242))
                draw.line((left, top, left, bottom, right, bottom), fill=(105, 110, 115), width=1)
                draw.text((12, 925), f"{maximum:.0f}", fill=MUTED, font=small)
                draw.text((15, 1013), "0", fill=MUTED, font=small)
                for controller, curve in rollouts.items():
                    indices = np.arange(0, index + 1, 3)
                    points = [(left + float(curve["time"][i]) / scenario["duration"] * (right - left),
                               bottom - float(curve["error"][i]) * 1000 / maximum * (bottom - top)) for i in indices]
                    if len(points) > 1:
                        draw.line(points, fill=COLORS[controller], width=2)
                for when in (scenario["payload_time"], scenario["push_time"]):
                    x = left + when / scenario["duration"] * (right - left)
                    draw.line((x, top, x, bottom), fill=(155, 155, 155))
                for seconds in (0, 9, 16, 24):
                    x = left + seconds / scenario["duration"] * (right - left)
                    draw.text((x - 8, 1028), str(seconds), fill=MUTED, font=small)
                draw.text((260, 1049), "Position error (mm) vs. time (s); shared scale", fill=MUTED, font=font(20))
                panels.append(panel)
                writers[name].append_data(np.asarray(panel))
            comparison = Image.new("RGB", (1920, 1080))
            for i, panel in enumerate(panels):
                comparison.paste(panel, (i * 960, 0))
            writers["comparison"].append_data(np.asarray(comparison))
            if frame in (0, round((scenario["payload_time"] + 2) * fps), round((scenario["push_time"] + 1) * fps)):
                comparison.save(output / f"preview_{frame:04d}.png")
            if frame % (fps * 4) == 0:
                print(f"render {t:.1f}/{scenario['duration']} s", flush=True)
        # Hold a readable result card for two seconds after the physical episode.
        comparison = Image.new("RGB", (1920, 1080))
        for i, (name, panel) in enumerate(zip(rollouts, panels)):
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 830, 960, 932), fill="white")
            draw.text((28, 843), f"Post-shift RMSE: {metrics[name]['post_shift_rmse_m'] * 1000:.1f} mm", fill=COLORS[name], font=title)
            draw.text((28, 888), "Identical trajectory, disturbances and torque limits", fill=INK, font=text)
            comparison.paste(panel, (i * 960, 0))
            for _ in range(2 * fps):
                writers[name].append_data(np.asarray(panel))
        for _ in range(2 * fps):
            writers["comparison"].append_data(np.asarray(comparison))
        comparison.save(output / "poster.png")
    finally:
        for writer in writers.values():
            writer.close()
        for renderer in renderers.values():
            renderer.close()
    paper_figure(rollouts, scenario, config, output)


def paper_figure(rollouts, scenario, config, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for path in FONT_DIR.glob("*.otf"):
        font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(FONT_DIR / "lmroman10-regular.otf")).get_name()
    with plt.rc_context({"font.family": family, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"}):
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        ref = rollouts["baseline"]["reference"]
        axes[0].plot(ref[:, 0], ref[:, 1], "--", color="#0072B2", label="Reference", linewidth=1.3)
        for name, rollout in rollouts.items():
            color = np.asarray(COLORS[name]) / 255
            label = "Nominal impedance" if name == "baseline" else "Residual GRU"
            axes[0].plot(rollout["position"][:, 0], rollout["position"][:, 1], color=color, label=label, linewidth=1)
            axes[1].plot(rollout["time"], rollout["error"] * 1000, color=color, label=label, linewidth=1.5)
        axes[0].set(xlabel="x (m)", ylabel="y (m)", title="(a) End-effector path: xy projection")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].legend(fontsize=9, frameon=False)
        axes[1].axvline(scenario["payload_time"], color="0.5", linestyle=":", label="Payload attached")
        axes[1].axvspan(scenario["push_time"], scenario["push_time"] + scenario["push_duration"], color="0.92", label="External push")
        axes[1].axhline(config["recovery_threshold_m"] * 1000, color="0.3", linestyle="--", linewidth=0.8, label="Loss threshold")
        axes[1].set(xlabel="Time (s)", ylabel="Position error (mm)", title="(b) Tracking under changed dynamics")
        axes[1].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
        figure.savefig(output / "tracking_summary.png", dpi=240)
        figure.savefig(output / "tracking_summary.svg")
        svg = output / "tracking_summary.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(figure)
