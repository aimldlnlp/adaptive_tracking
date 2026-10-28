from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[float, str | None], None]


def configure_logging(level: str = "INFO", log_path: str | Path | None = None, force: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        target = Path(log_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=force,
    )


class ProgressTracker:
    def __init__(
        self,
        stage_weights: dict[str, float],
        status_path: str | Path,
    ) -> None:
        self.stage_weights = stage_weights
        self.stage_order = list(stage_weights)
        self.status_path = Path(status_path).resolve()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("progress")

    def stage_callback(self, stage: str) -> ProgressCallback:
        def callback(progress: float, message: str | None = None) -> None:
            self.emit(stage, progress, message)

        return callback

    def emit(self, stage: str, stage_progress: float, message: str | None = None) -> None:
        bounded_stage_progress = max(0.0, min(1.0, float(stage_progress)))
        overall_progress = self._compute_overall_progress(stage, bounded_stage_progress)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "stage_progress": round(bounded_stage_progress * 100.0, 2),
            "overall_progress": round(overall_progress, 2),
            "message": message or "",
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info(
            "stage=%s stage_progress=%.1f overall_progress=%.1f message=%s",
            stage,
            payload["stage_progress"],
            payload["overall_progress"],
            payload["message"] or "-",
        )

    def _compute_overall_progress(self, stage: str, stage_progress: float) -> float:
        completed = 0.0
        for current_stage in self.stage_order:
            weight = float(self.stage_weights[current_stage])
            if current_stage == stage:
                return completed + weight * stage_progress
            completed += weight
        raise KeyError(f"Unknown progress stage: {stage}")
