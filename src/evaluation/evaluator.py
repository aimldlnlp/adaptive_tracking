from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.controllers.adaptive import AdaptiveController
from src.controllers.baseline import BaselineController
from src.dynamics.robot import EpisodeSimulator, sample_episode_spec
from src.evaluation.metrics import aggregate_metrics, compute_episode_metrics
from src.models.adaptive_estimator import load_checkpoint_bundle, resolve_device
from src.utils.config import get_output_dir, merge_config
from src.utils.io import ensure_dir, save_dataframe, save_json, save_npz
from src.utils.logging_utils import ProgressCallback


LOGGER = logging.getLogger(__name__)


def evaluate_from_config(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    controller_specs: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    metrics_dir = get_output_dir(config, "metrics")
    rollouts_dir = metrics_dir / "rollouts"
    ensure_dir(rollouts_dir)
    evaluation_specs = _build_evaluation_specs(config)
    controllers = _build_controller_registry(config, checkpoint_path, controller_specs)

    simulator = EpisodeSimulator(config)
    metrics_rows: list[dict[str, Any]] = []
    total_rollouts = len(evaluation_specs) * len(controllers)
    completed_rollouts = 0
    if progress_callback is not None:
        progress_callback(0.0, "starting evaluation")

    controller_order = list(controllers)
    for spec in tqdm(evaluation_specs, desc="evaluate", leave=False):
        for controller_name in controller_order:
            controller = controllers[controller_name]
            rollout = simulator.simulate_episode(spec=spec, controller=controller)
            metrics = compute_episode_metrics(config, rollout)
            row = {
                "episode_id": spec["episode_id"],
                "controller": controller_name,
                "trajectory_kind": spec["trajectory_kind"],
                "shift_type": spec["shift_type"],
                "shift_intensity": spec["shift_intensity"],
                "shift_time": spec["shift_time"],
                "unseen": int(spec["unseen"]),
                "compound_shift": int(spec.get("compound_shift", False)),
                "condition_group": spec.get("condition_group", "in_distribution_single_shift"),
                **metrics,
            }
            metrics_rows.append(row)
            payload = {key: value for key, value in rollout.items() if key != "metadata_json"}
            payload["metadata_json"] = np.array(json.dumps({**spec, "controller": controller_name}))
            save_npz(rollouts_dir / f"{spec['episode_id']}__{controller_name}.npz", **payload)
            completed_rollouts += 1
            if progress_callback is not None:
                progress_callback(
                    completed_rollouts / max(total_rollouts, 1),
                    f"{controller_name} {completed_rollouts}/{total_rollouts}",
                )

    metrics_frame = pd.DataFrame(metrics_rows)
    aggregate_frame, summary_frame = aggregate_metrics(metrics_frame)
    condition_breakdown = _condition_breakdown(metrics_frame)
    bootstrap_frame = _bootstrap_intervals(
        metrics_frame,
        bootstrap_samples=int(config["evaluation"].get("bootstrap_samples", 500)),
    )
    comparison_frame = _controller_comparison(metrics_frame, list(controllers))
    representatives = _select_representative_cases(metrics_frame, config)
    suite_summary = _suite_summary(summary_frame, comparison_frame, config)

    save_dataframe(metrics_frame, metrics_dir / "per_episode_metrics.csv")
    save_dataframe(aggregate_frame, metrics_dir / "aggregate_metrics.csv")
    save_dataframe(summary_frame, metrics_dir / "controller_summary.csv")
    save_dataframe(condition_breakdown, metrics_dir / "condition_breakdown.csv")
    save_dataframe(bootstrap_frame, metrics_dir / "bootstrap_intervals.csv")
    save_dataframe(comparison_frame, metrics_dir / "controller_comparison.csv")
    save_dataframe(suite_summary, metrics_dir / "suite_summary.csv")
    save_json(evaluation_specs, metrics_dir / "evaluation_specs.json")
    save_json(representatives, metrics_dir / "representative_cases.json")
    LOGGER.info("saved_metrics=%s", metrics_dir)
    if progress_callback is not None:
        progress_callback(1.0, "evaluation complete")
    return metrics_frame


def _build_controller_registry(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    controller_specs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    registry: dict[str, Any] = {"baseline": BaselineController(config)}
    if controller_specs:
        for spec in controller_specs:
            if spec["name"] == "baseline":
                continue
            model_config = merge_config(config, spec.get("config_overrides", {}))
            bundle = load_checkpoint_bundle(str(spec["checkpoint_path"]), resolve_device(model_config))
            registry[spec["name"]] = AdaptiveController(model_config, bundle)
        return registry

    device = resolve_device(config)
    checkpoint = checkpoint_path or get_output_dir(config, "checkpoints") / "best_model.pt"
    controller_name = str(config.get("model", {}).get("name", "adaptive"))
    registry[controller_name] = AdaptiveController(config, load_checkpoint_bundle(str(checkpoint), device))
    return registry


def _build_evaluation_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation_cfg = config["evaluation"]
    shift_types = list(config["simulation"]["shift_types"])
    intensities = list(config["simulation"]["shift_intensities"].keys())
    specs: list[dict[str, Any]] = []
    seeds = list(evaluation_cfg["seeds"])
    episodes_per_condition = int(evaluation_cfg["episodes_per_condition"])
    unseen_fraction = float(evaluation_cfg["unseen_fraction"])

    for seed in seeds:
        for shift_index, shift_type in enumerate(shift_types):
            for intensity_index, intensity in enumerate(intensities):
                unseen_cutoff = max(1, round(unseen_fraction * episodes_per_condition))
                for episode_idx in range(episodes_per_condition):
                    spec_seed = int(seed * 10_000 + shift_index * 1_000 + intensity_index * 100 + episode_idx)
                    rng = np.random.default_rng(spec_seed)
                    unseen = bool(episode_idx < unseen_cutoff)
                    spec = sample_episode_spec(
                        config,
                        split="test",
                        rng=rng,
                        episode_index=episode_idx,
                        unseen=unseen,
                        forced_shift_type=shift_type,
                        forced_intensity=intensity,
                    )
                    spec["episode_id"] = f"eval_{seed}_{shift_type}_{intensity}_{episode_idx:02d}"
                    if unseen:
                        spec["condition_group"] = "hard_unseen_trajectory"
                    else:
                        spec["condition_group"] = "in_distribution_single_shift"
                    specs.append(spec)

    if bool(evaluation_cfg.get("include_compound_shifts", False)):
        compound_sets = list(evaluation_cfg.get("compound_shift_sets", []))
        compound_episodes = int(evaluation_cfg.get("compound_episodes_per_condition", max(2, episodes_per_condition // 2)))
        for seed in seeds:
            for shift_index, compound_entry in enumerate(compound_sets):
                compound_shift_types = list(compound_entry)
                compound_label = "+".join(compound_shift_types)
                for intensity_index, intensity in enumerate(intensities):
                    for episode_idx in range(compound_episodes):
                        spec_seed = int(seed * 20_000 + shift_index * 2_000 + intensity_index * 200 + episode_idx)
                        rng = np.random.default_rng(spec_seed)
                        spec = sample_episode_spec(
                            config,
                            split="test",
                            rng=rng,
                            episode_index=episode_idx,
                            unseen=False,
                            forced_intensity=intensity,
                            forced_compound_shift_types=compound_shift_types,
                        )
                        spec["episode_id"] = f"compound_{seed}_{compound_label.replace('+', '-')}_{intensity}_{episode_idx:02d}"
                        spec["condition_group"] = "compound_shift_ood"
                        spec["compound_shift"] = True
                        specs.append(spec)
    return specs


def _condition_breakdown(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    breakdown = (
        metrics_frame.groupby(["controller", "condition_group"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            final_position_error_mean=("final_position_error", "mean"),
            success_rate=("success", "mean"),
            robustness_score_mean=("robustness_score", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            mean_estimated_uncertainty=("mean_estimated_uncertainty", "mean"),
            mean_structure_uncertainty=("mean_structure_uncertainty", "mean"),
            mean_disturbance_uncertainty=("mean_disturbance_uncertainty", "mean"),
            mean_correction_gain=("mean_correction_gain", "mean"),
            mean_structure_gain=("mean_structure_gain", "mean"),
            mean_disturbance_gain=("mean_disturbance_gain", "mean"),
        )
        .fillna(0.0)
    )
    overall = (
        metrics_frame.groupby(["controller"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            final_position_error_mean=("final_position_error", "mean"),
            success_rate=("success", "mean"),
            robustness_score_mean=("robustness_score", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            mean_estimated_uncertainty=("mean_estimated_uncertainty", "mean"),
            mean_structure_uncertainty=("mean_structure_uncertainty", "mean"),
            mean_disturbance_uncertainty=("mean_disturbance_uncertainty", "mean"),
            mean_correction_gain=("mean_correction_gain", "mean"),
            mean_structure_gain=("mean_structure_gain", "mean"),
            mean_disturbance_gain=("mean_disturbance_gain", "mean"),
        )
        .assign(condition_group="overall")
    )
    return pd.concat([overall, breakdown], ignore_index=True)


def _bootstrap_intervals(metrics_frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260419)
    rows: list[dict[str, Any]] = []
    metrics = {
        "rmse": "mean",
        "final_position_error": "mean",
        "success": "mean",
        "robustness_score": "mean",
    }
    condition_groups = ["overall", *sorted(metrics_frame["condition_group"].unique())]
    for controller in metrics_frame["controller"].unique():
        controller_frame = metrics_frame[metrics_frame["controller"] == controller]
        for condition_group in condition_groups:
            if condition_group == "overall":
                subset = controller_frame
            else:
                subset = controller_frame[controller_frame["condition_group"] == condition_group]
            if subset.empty:
                continue
            for metric in metrics:
                values = subset[metric].to_numpy(dtype=np.float64)
                samples = []
                for _ in range(max(bootstrap_samples, 1)):
                    draw = rng.choice(values, size=len(values), replace=True)
                    samples.append(float(np.mean(draw)))
                rows.append(
                    {
                        "controller": controller,
                        "condition_group": condition_group,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "ci_low": float(np.quantile(samples, 0.025)),
                        "ci_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_samples": int(bootstrap_samples),
                    }
                )
    return pd.DataFrame(rows)


def _controller_comparison(metrics_frame: pd.DataFrame, controller_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["rmse", "final_position_error", "success", "robustness_score", "control_smoothness"]
    indexed = metrics_frame.set_index(["episode_id", "controller"])
    condition_groups = ["overall", *sorted(metrics_frame["condition_group"].unique())]
    for left_index, controller_a in enumerate(controller_order):
        for controller_b in controller_order[left_index + 1 :]:
            for condition_group in condition_groups:
                group_frame = metrics_frame if condition_group == "overall" else metrics_frame[metrics_frame["condition_group"] == condition_group]
                episode_ids = sorted(group_frame["episode_id"].unique())
                if not episode_ids:
                    continue
                paired_rows: list[dict[str, float]] = []
                for episode_id in episode_ids:
                    key_a = (episode_id, controller_a)
                    key_b = (episode_id, controller_b)
                    if key_a not in indexed.index or key_b not in indexed.index:
                        continue
                    row_a = indexed.loc[key_a]
                    row_b = indexed.loc[key_b]
                    paired_rows.append({metric: float(row_a[metric] - row_b[metric]) for metric in metrics})
                if not paired_rows:
                    continue
                paired_frame = pd.DataFrame(paired_rows)
                rows.append(
                    {
                        "controller_a": controller_a,
                        "controller_b": controller_b,
                        "condition_group": condition_group,
                        "rmse_delta": float(paired_frame["rmse"].mean()),
                        "final_position_error_delta": float(paired_frame["final_position_error"].mean()),
                        "success_rate_delta": float(paired_frame["success"].mean()),
                        "robustness_score_delta": float(paired_frame["robustness_score"].mean()),
                        "control_smoothness_delta": float(paired_frame["control_smoothness"].mean()),
                        "rmse_win_rate_a": float((paired_frame["rmse"] < 0.0).mean()),
                        "rmse_win_rate_b": float((paired_frame["rmse"] > 0.0).mean()),
                        "paired_episodes": int(len(paired_frame)),
                    }
                )
    return pd.DataFrame(rows)


def _select_representative_cases(metrics_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    primary_controller = str(config["evaluation"].get("primary_controller", metrics_frame["controller"].iloc[-1]))
    baseline = metrics_frame[metrics_frame["controller"] == "baseline"].set_index("episode_id")
    primary = metrics_frame[metrics_frame["controller"] == primary_controller].set_index("episode_id")
    paired = baseline[
        ["trajectory_kind", "shift_type", "shift_intensity", "unseen", "condition_group", "rmse", "success"]
    ].join(
        primary[["rmse", "success", "mean_estimated_uncertainty"]],
        lsuffix="_baseline",
        rsuffix="_primary",
    )
    paired["improvement"] = paired["rmse_baseline"] - paired["rmse_primary"]

    def choose(frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> str:
        if frame.empty:
            return str(paired.sort_values(["improvement"], ascending=[False]).index[0])
        return str(frame.sort_values(sort_cols, ascending=ascending).index[0])

    general = paired[paired["condition_group"] == "in_distribution_single_shift"]
    unseen = paired[paired["condition_group"] == "hard_unseen_trajectory"]
    compound = paired[paired["condition_group"] == "compound_shift_ood"]
    recovery = paired[(paired["success_baseline"] < 0.5) & (paired["success_primary"] > 0.5)]

    showcase: dict[str, str] = {}
    single_shift_names = [name for name in config["simulation"]["shift_types"]]
    for shift_type in single_shift_names:
        subset = general[(general["shift_type"] == shift_type) & (general["shift_intensity"] == "medium")]
        showcase[shift_type] = choose(subset, ["rmse_primary", "improvement"], [True, False])

    return {
        "primary_pair": choose(general, ["improvement", "rmse_primary"], [False, True]),
        "control_signal": choose(general, ["improvement", "rmse_primary"], [False, True]),
        "showcase_single_shift": showcase,
        "unseen_generalization": choose(unseen, ["rmse_primary", "improvement"], [True, False]),
        "compound_showcase": choose(compound, ["improvement", "rmse_primary"], [False, True]),
        "uncertainty_recovery": choose(recovery, ["improvement"], [False]),
        "trajectory_comparison": choose(general, ["improvement", "rmse_primary"], [False, True]),
        "video_01": choose(general, ["improvement", "rmse_primary"], [False, True]),
        "video_03": showcase,
        "video_04": choose(unseen, ["rmse_primary", "improvement"], [True, False]),
        "video_05": choose(recovery, ["improvement"], [False]),
    }


def _suite_summary(summary_frame: pd.DataFrame, comparison_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary_controller = str(config["evaluation"].get("primary_controller", summary_frame["controller"].iloc[-1]))
    gain_target_low = float(config["evaluation"].get("correction_gain_target_low", 0.92))
    gain_target_high = float(config["evaluation"].get("correction_gain_target_high", 0.97))
    rows: list[dict[str, Any]] = []
    primary_summary = summary_frame[summary_frame["controller"] == primary_controller]
    baseline_summary = summary_frame[summary_frame["controller"] == "baseline"]
    if not primary_summary.empty and not baseline_summary.empty:
        rows.append(
            {
                "item": "primary_vs_baseline",
                "primary_controller": primary_controller,
                "rmse_delta": float(primary_summary["rmse_mean"].iloc[0] - baseline_summary["rmse_mean"].iloc[0]),
                "success_rate_delta": float(primary_summary["success_rate"].iloc[0] - baseline_summary["success_rate"].iloc[0]),
                "control_smoothness_delta": float(
                    primary_summary["control_smoothness_mean"].iloc[0] - baseline_summary["control_smoothness_mean"].iloc[0]
                ),
                "mean_correction_gain": float(primary_summary["mean_correction_gain"].iloc[0]),
                "mean_structure_gain": float(primary_summary["mean_structure_gain"].iloc[0]),
                "mean_disturbance_gain": float(primary_summary["mean_disturbance_gain"].iloc[0]),
                "gain_target_low": gain_target_low,
                "gain_target_high": gain_target_high,
                "gain_target_hit": float(
                    gain_target_low <= float(primary_summary["mean_correction_gain"].iloc[0]) <= gain_target_high
                ),
            }
        )
        rows.append(
            {
                "item": "primary_gain_acceptance",
                "primary_controller": primary_controller,
                "rmse_delta": 0.0,
                "success_rate_delta": 0.0,
                "control_smoothness_delta": 0.0,
                "mean_correction_gain": float(primary_summary["mean_correction_gain"].iloc[0]),
                "mean_structure_gain": float(primary_summary["mean_structure_gain"].iloc[0]),
                "mean_disturbance_gain": float(primary_summary["mean_disturbance_gain"].iloc[0]),
                "gain_target_low": gain_target_low,
                "gain_target_high": gain_target_high,
                "gain_target_hit": float(
                    gain_target_low <= float(primary_summary["mean_correction_gain"].iloc[0]) <= gain_target_high
                ),
            }
        )
    if not comparison_frame.empty:
        for row in comparison_frame[comparison_frame["condition_group"] == "overall"].itertuples(index=False):
            rows.append(
                {
                    "item": f"{row.controller_a}_vs_{row.controller_b}",
                    "primary_controller": primary_controller,
                    "rmse_delta": float(row.rmse_delta),
                    "success_rate_delta": float(row.success_rate_delta),
                    "control_smoothness_delta": float(row.control_smoothness_delta),
                    "mean_correction_gain": np.nan,
                    "mean_structure_gain": np.nan,
                    "mean_disturbance_gain": np.nan,
                    "gain_target_low": gain_target_low,
                    "gain_target_high": gain_target_high,
                    "gain_target_hit": np.nan,
                }
            )
    return pd.DataFrame(rows)
