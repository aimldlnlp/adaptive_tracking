from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import run as run_evaluate
from scripts.generate_data import run as run_generate_data
from scripts.make_figures import run as run_make_figures
from scripts.make_videos import run as run_make_videos
from scripts.train import run as run_train
from src.utils.logging_utils import ProgressCallback


def run(config_path: str, progress_callbacks: dict[str, ProgressCallback] | None = None) -> None:
    callbacks = progress_callbacks or {}
    run_generate_data(config_path, progress_callback=callbacks.get("generate_data"))
    run_train(config_path, progress_callback=callbacks.get("train"))
    run_evaluate(config_path, progress_callback=callbacks.get("evaluate"))
    run_make_figures(config_path, progress_callback=callbacks.get("figures"))
    run_make_videos(config_path, progress_callback=callbacks.get("videos"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
