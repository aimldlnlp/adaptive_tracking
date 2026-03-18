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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_generate_data(args.config)
    run_train(args.config)
    run_evaluate(args.config)
    run_make_figures(args.config)
    run_make_videos(args.config)


if __name__ == "__main__":
    main()
