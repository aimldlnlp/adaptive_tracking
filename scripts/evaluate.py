from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_from_config
from src.utils.config import load_config
from src.utils.logging_utils import configure_logging
from src.utils.seeding import set_seed


def run(config_path: str) -> None:
    config = load_config(config_path)
    configure_logging()
    set_seed(int(config["simulation"]["test_seed"]))
    evaluate_from_config(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
