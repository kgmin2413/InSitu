"""Structured logging helpers for the Stage 1 pipeline scaffold."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


STAGE_NAMES = (
    "frame_extraction",
    "background_removal",
    "marker_detection",
    "colmap",
    "training",
    "scale_estimation",
    "proxy_generation",
)


def default_log_schema(object_id: str, preset: str) -> dict:
    return {
        "object_id": object_id,
        "preset": preset,
        "timing_sec": {
            "frame_extraction": 0.0,
            "background_removal": 0.0,
            "marker_detection": 0.0,
            "colmap": 0.0,
            "training": 0.0,
            "scale_estimation": 0.0,
            "proxy_generation": 0.0,
            "total": 0.0,
        },
        "frame_stats": {
            "raw_frames": 0,
            "selected_frames": 0,
            "removed_blurry": 0,
            "removed_duplicates": 0,
        },
        "colmap_stats": {
            "registered_images": 0,
            "registered_ratio": 0.0,
            "reprojection_error": 0.0,
            "sparse_points": 0,
        },
        "training_stats": {
            "iterations": 0,
            "final_gaussian_count": 0,
        },
        "status": "success",
        "warnings": [],
        "errors": [],
    }


class PipelineLogger:
    """Collects stage metrics, warnings, errors, and timing."""

    def __init__(self, object_id: str, preset: str) -> None:
        self.log = default_log_schema(object_id, preset)
        self._total_start = time.perf_counter()

    @contextmanager
    def stage_timer(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.add_error(stage_name, str(exc))
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.log["timing_sec"][stage_name] = round(elapsed, 6)

    def add_warning(self, stage_name: str, message: str) -> None:
        self.log["warnings"].append(f"{stage_name}: {message}")

    def add_error(self, stage_name: str, message: str) -> None:
        self.log["status"] = "failed"
        self.log["errors"].append(f"{stage_name}: {message}")

    def update_metrics(self, key: str, metrics: dict) -> None:
        if key in self.log:
            self.log[key].update(metrics)

    def mark_failed(self) -> None:
        self.log["status"] = "failed"

    def finalize(self) -> dict:
        total = time.perf_counter() - self._total_start
        self.log["timing_sec"]["total"] = round(total, 6)
        return self.log

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.finalize(), f, indent=2)
