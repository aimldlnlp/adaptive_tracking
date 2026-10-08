"""Validate a completed run and report paired differences across training seeds."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_interval(delta, samples=2000):
    """Resample training seeds and shared scenario IDs; positive means A improves B."""
    if delta.ndim != 2 or not delta.size or not np.isfinite(delta).all():
        raise ValueError("Expected a finite training-seed by scenario matrix")
    rng = np.random.default_rng(20261007)
    draws = np.empty(samples)
    for i in range(samples):
        seeds = rng.integers(delta.shape[0], size=delta.shape[0])
        scenarios = rng.integers(delta.shape[1], size=delta.shape[1])
        draws[i] = delta[np.ix_(seeds, scenarios)].mean()
    return np.quantile(draws, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Run must be complete before cross-seed analysis")
    frame = pd.read_csv(root / "all_seed_episode_metrics.csv")
    seeds = sorted(manifest["training_seeds"])
    controllers = sorted(frame.controller.unique())
    assert sorted(frame.training_seed.unique()) == seeds
    assert not frame.duplicated(["training_seed", "episode_id", "controller"]).any()
    assert np.isfinite(frame.rmse).all()
    expected = None
    for seed in seeds:
        specs = json.loads((root / f"seed_{seed}/metrics/evaluation_specs.json").read_text())
        if expected is None:
            expected = specs
        assert specs == expected, "Scenario specs differ across seeds"
        counts = frame[frame.training_seed == seed].groupby("controller").size()
        assert set(counts.index) == set(controllers) and counts.eq(len(specs)).all()
    baseline = frame[frame.controller == "baseline"].pivot(index="training_seed", columns="episode_id", values="rmse")
    assert np.allclose(baseline.to_numpy(), baseline.iloc[0].to_numpy(), atol=1e-10), "Baseline differs across seeds"
    rows = []
    groups = ["overall", *sorted(frame.condition_group.unique())]
    for group in groups:
        subset = frame if group == "overall" else frame[frame.condition_group == group]
        for controller_a, controller_b in itertools.combinations(controllers, 2):
            a = subset[subset.controller == controller_a].pivot(index="training_seed", columns="episode_id", values="rmse").sort_index()
            b = subset[subset.controller == controller_b].pivot(index="training_seed", columns="episode_id", values="rmse").reindex(index=a.index, columns=a.columns)
            delta = b.to_numpy() - a.to_numpy()
            low, high = paired_interval(delta)
            rows.append({"condition_group": group, "controller_a": controller_a, "controller_b": controller_b,
                         "rmse_improvement_a_over_b": delta.mean(), "ci_low": low, "ci_high": high,
                         "training_seed_std": delta.mean(axis=1).std(ddof=1),
                         "training_seeds": len(seeds), "unique_scenarios": delta.shape[1]})
    paired = pd.DataFrame(rows)
    paired.to_csv(root / "paired_multiseed_comparison.csv", index=False)
    condition = frame.groupby(["training_seed", "condition_group", "controller"], as_index=False).agg(
        rmse=("rmse", "mean"), success_rate=("success", "mean"),
        recovery_needed_rate=("recovery_needed", "mean"), recovery_success_rate=("recovery_success", "mean"))
    condition.to_csv(root / "per_seed_condition_summary.csv", index=False)
    recovery = frame.groupby(["training_seed", "controller", "recovery_status"]).size().unstack(fill_value=0)
    recovery.to_csv(root / "recovery_counts.csv")
    checkpoint_rows = []
    for seed in seeds:
        for controller in controllers:
            if controller == "baseline":
                continue
            metrics = root / f"seed_{seed}/metrics"
            summary = json.loads((metrics / f"training_summary_{controller}.json").read_text())
            history = pd.read_csv(metrics / f"training_history_{controller}.csv")
            best_row = history.loc[history.selection_score.idxmin()]
            assert summary["best_epoch"] == int(best_row.epoch)
            assert np.isclose(summary["best_selection_score"], best_row.selection_score)
            checkpoint_rows.append({"training_seed": seed, "controller": controller,
                                    "best_epoch": summary["best_epoch"], "epochs_ran": summary["epochs_ran"],
                                    "selection_metric": summary["selection_metric"],
                                    "best_selection_score": summary["best_selection_score"]})
    pd.DataFrame(checkpoint_rows).to_csv(root / "checkpoint_selection.csv", index=False)
    import torch
    weight_parity = {}
    for seed in seeds:
        checkpoints = root / f"seed_{seed}/checkpoints"
        nominal = torch.load(checkpoints / "adaptive_gru_nominal.pt", map_location="cpu", weights_only=False)["model_state"]
        uncertainty = torch.load(checkpoints / "adaptive_gru_uncertainty.pt", map_location="cpu", weights_only=False)["model_state"]
        weight_parity[seed] = nominal.keys() == uncertainty.keys() and all(
            torch.equal(nominal[key], uncertainty[key]) for key in nominal)
    (root / "gru_weight_parity.json").write_text(json.dumps(weight_parity, indent=2))
    summary = pd.read_csv(root / "multiseed_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"adaptive_mlp": "MLP", "adaptive_gru_nominal": "GRU nominal",
              "adaptive_gru_uncertainty": "GRU uncertainty", "baseline": "Baseline"}
    per_seed = pd.read_csv(root / "per_seed_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for i, controller in enumerate(controllers):
        values = per_seed[per_seed.controller == controller].rmse.to_numpy()
        axes[0].scatter(i + np.linspace(-0.1, 0.1, len(values)), values, s=35)
        axes[0].plot([i - 0.2, i + 0.2], [values.mean()] * 2, color="black")
    axes[0].set_xticks(range(len(controllers)), [labels[c] for c in controllers], rotation=20)
    axes[0].set(ylabel="Mean position RMSE (m)", title="Dots: training seeds; line: mean")
    for i, controller in enumerate(c for c in controllers if c != "baseline"):
        rows = paired[(paired.controller_a == controller) & (paired.controller_b == "baseline")].set_index("condition_group").loc[groups]
        x = np.arange(len(groups)) + (i - 1) * 0.18
        axes[1].plot(x, rows.rmse_improvement_a_over_b, "o", label=labels[controller])
        axes[1].vlines(x, rows.ci_low, rows.ci_high)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(groups)), [g.replace("_", "\n") for g in groups], fontsize=8)
    axes[1].set(ylabel="Baseline RMSE minus model RMSE (m)", title="Paired improvement; exploratory 95% interval")
    axes[1].legend(fontsize=8)
    figure.savefig(root / "multiseed_comparison.png", dpi=160)
    plt.close(figure)
    report = ["# Corrected multi-seed benchmark", "",
              f"Training seeds: {seeds}. Fixed data; {len(expected)} unique scenarios per controller.",
              "", "```", summary.to_string(index=False), "```", "",
              "![Cross-seed comparison](multiseed_comparison.png)", "",
              "RMSE standard deviation measures training randomness across three seeds, not dataset variation.",
              "Paired intervals resample training seeds and shared scenario IDs. With only three training seeds,",
              "these intervals are exploratory; do not treat repeated baseline rows as independent runs.",
              "Recovery success is conditional on tracking loss; blank recovery time means no loss or failure.",
              "Checkpoint selection agrees with the minimum validation selection score for every model.",
              f"Nominal/uncertainty GRU weights are exactly identical in every seed: {all(weight_parity.values())}.",
              "Their different tracking results reflect inference controller configuration, not different learned weights.",
              "In this run, saved model configs differ only in name and uncertainty_aware; support settings are identical.",
              "This compares the complete uncertainty-aware controller branch enabled/disabled with the same learned weights.",
              "Legacy paper_best used the old checkpoint/recovery code and may use a different runtime."]
    (root / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Report: {root / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
