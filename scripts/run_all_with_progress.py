from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_all import run as run_pipeline
from src.utils.config import get_output_dir, load_config
from src.utils.logging_utils import ProgressTracker, configure_logging


LOGGER = logging.getLogger(__name__)

STAGE_WEIGHTS = {
    "bootstrap": 5.0,
    "generate_data": 20.0,
    "train": 25.0,
    "evaluate": 25.0,
    "figures": 10.0,
    "videos": 15.0,
}


def _default_log_path(config_path: str) -> Path:
    config = load_config(config_path)
    logs_dir = get_output_dir(config, "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"run_all_{timestamp}.log"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-path")
    parser.add_argument("--status-path")
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve() if args.log_path else _default_log_path(args.config).resolve()
    status_path = Path(args.status_path).resolve() if args.status_path else log_path.parent / "latest_progress.json"

    configure_logging(log_path=log_path, force=True)
    tracker = ProgressTracker(STAGE_WEIGHTS, status_path)

    LOGGER.info("pipeline_log=%s", log_path)
    LOGGER.info("progress_status=%s", status_path)
    tracker.emit("bootstrap", 1.0, "environment ready")
    run_pipeline(
        args.config,
        progress_callbacks={
            "generate_data": tracker.stage_callback("generate_data"),
            "train": tracker.stage_callback("train"),
            "evaluate": tracker.stage_callback("evaluate"),
            "figures": tracker.stage_callback("figures"),
            "videos": tracker.stage_callback("videos"),
        },
    )
    tracker.emit("videos", 1.0, "pipeline complete")


if __name__ == "__main__":
    main()
