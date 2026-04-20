from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.focus_analysis import run_focus_case_analysis
from src.utils.config import load_config, merge_config


def _load_suite_config(suite_path: str) -> dict:
    suite_file = Path(suite_path).resolve()
    with suite_file.open("r", encoding="utf-8") as handle:
        suite_payload = yaml.safe_load(handle)
    suite_cfg = suite_payload["suite"]
    base_config = load_config((PROJECT_ROOT / Path(suite_cfg["base_config"])).resolve())
    output_subdir = str(suite_cfg.get("output_subdir", f"research/{suite_cfg['name']}"))
    config = merge_config(
        base_config,
        {
            "project": {"output_subdir": output_subdir},
            "evaluation": {
                "compare_controllers": ["baseline", *[experiment["name"] for experiment in suite_cfg["experiments"]]],
                "primary_controller": suite_cfg["experiments"][-1]["name"],
            },
        },
    )
    return merge_config(config, suite_cfg.get("config_overrides", {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/experiments/paper_best.yaml")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config = _load_suite_config(args.suite)
    outputs = run_focus_case_analysis(config, top_k=args.top_k)
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
