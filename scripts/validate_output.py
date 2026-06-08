#!/usr/bin/env python3
"""Validate an InSitu output package contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_KEYS = {
    "object_id",
    "visual_model",
    "proxy_model",
    "unit",
    "scale_factor",
    "scale_method",
    "bbox_size_m",
    "object_center_m",
    "proxy_center_m",
    "coordinate_system",
    "up_axis",
    "created_at",
}

REQUIRED_LOG_KEYS = {
    "object_id",
    "preset",
    "timing_sec",
    "frame_stats",
    "colmap_stats",
    "training_stats",
    "status",
    "warnings",
    "errors",
}

REQUIRED_TIMING_KEYS = {
    "frame_extraction",
    "background_removal",
    "marker_detection",
    "colmap",
    "training",
    "scale_estimation",
    "proxy_generation",
    "total",
}

DEBUG_REPORTS = {
    "frame_selection_report.json",
    "capture_quality_report.json",
    "mask_manifest.json",
    "marker_detection_report.json",
    "colmap_summary.json",
    "training_report.json",
    "scale_report.json",
    "proxy_report.json",
    "selected_frames.txt",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_common(object_dir: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = object_dir / "metadata.json"
    log_path = object_dir / "processing_log.json"
    manifest_path = object_dir / "package_manifest.json"
    _require(metadata_path.exists(), "metadata.json is missing", errors)
    _require(log_path.exists(), "processing_log.json is missing", errors)
    _require(manifest_path.exists(), "package_manifest.json is missing", errors)
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    log = _read_json(log_path) if log_path.exists() else {}
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}

    _require(set(metadata) == REQUIRED_METADATA_KEYS, "metadata.json schema does not match required keys", errors)
    _require(set(log) == REQUIRED_LOG_KEYS, "processing_log.json schema does not match required keys", errors)
    _require(set(log.get("timing_sec", {})) == REQUIRED_TIMING_KEYS, "timing_sec keys are incomplete", errors)
    _require(manifest.get("schema_version") == 1, "package_manifest schema_version is not 1", errors)
    _require(manifest.get("object_id") == metadata.get("object_id"), "package_manifest object_id does not match metadata", errors)
    _require(manifest.get("unity_contract", {}).get("visual_and_physics_are_separate") is True, "package_manifest does not preserve visual/physics separation", errors)
    _require((object_dir / "preview").is_dir(), "preview/ is missing", errors)
    _require((object_dir / "debug").is_dir(), "debug/ is missing", errors)
    for report in DEBUG_REPORTS:
        _require((object_dir / "debug" / report).exists(), f"debug/{report} is missing", errors)
    return metadata, log


def _validate_default_safe(object_dir: Path, errors: list[str]) -> None:
    metadata, log = _validate_common(object_dir, errors)
    _require(log.get("status") == "success", "processing log status is not success", errors)
    manifest = _read_json(object_dir / "package_manifest.json") if (object_dir / "package_manifest.json").exists() else {}
    _require(not (object_dir / "point_cloud_metric.ply").exists(), "default-safe output unexpectedly has point_cloud_metric.ply", errors)
    _require(not (object_dir / "proxy_collider.obj").exists(), "default-safe output unexpectedly has proxy_collider.obj", errors)
    _require(metadata.get("scale_method") == "none_or_pending", "default-safe metadata scale_method changed", errors)
    _require((object_dir / "preview" / "input_thumbnail.jpg").exists(), "input thumbnail is missing", errors)
    _require((object_dir / "preview" / "mask_preview.jpg").exists(), "mask preview is missing", errors)
    _require((object_dir / "preview" / "selected_frames_contact_sheet.jpg").exists(), "selected frame contact sheet is missing", errors)
    _require(manifest.get("visual_representation", {}).get("exists") is False, "default-safe manifest visual asset should not exist", errors)
    _require(manifest.get("physics_representation", {}).get("exists") is False, "default-safe manifest physics asset should not exist", errors)


def _validate_sparse_proxy(object_dir: Path, errors: list[str]) -> None:
    metadata, log = _validate_common(object_dir, errors)
    _require(log.get("status") == "success", "processing log status is not success", errors)
    point_cloud = object_dir / "point_cloud_metric.ply"
    proxy = object_dir / "proxy_collider.obj"
    reconstruction_preview = object_dir / "preview" / "reconstruction_preview.png"
    _require(point_cloud.exists() and point_cloud.stat().st_size > 0, "point_cloud_metric.ply is missing or empty", errors)
    _require(proxy.exists() and proxy.stat().st_size > 0, "proxy_collider.obj is missing or empty", errors)
    _require(reconstruction_preview.exists() and reconstruction_preview.stat().st_size > 0, "reconstruction_preview.png is missing or empty", errors)
    _require(metadata.get("visual_model") == "point_cloud_metric.ply", "metadata visual_model is incorrect", errors)
    _require(metadata.get("proxy_model") == "proxy_collider.obj", "metadata proxy_model is incorrect", errors)
    _require(metadata.get("bbox_size_m") is not None, "metadata bbox_size_m is missing", errors)
    _require(metadata.get("object_center_m") is not None, "metadata object_center_m is missing", errors)
    _require(metadata.get("proxy_center_m") is not None, "metadata proxy_center_m is missing", errors)
    manifest = _read_json(object_dir / "package_manifest.json") if (object_dir / "package_manifest.json").exists() else {}
    _require(manifest.get("visual_representation", {}).get("exists") is True, "sparse-proxy manifest visual asset should exist", errors)
    _require(manifest.get("physics_representation", {}).get("exists") is True, "sparse-proxy manifest physics asset should exist", errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an InSitu output folder.")
    parser.add_argument("object_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("default_safe", "sparse_proxy"),
        default="default_safe",
        help="Validation contract to apply.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.object_dir.exists():
        errors.append(f"output folder does not exist: {args.object_dir}")
    elif args.mode == "default_safe":
        _validate_default_safe(args.object_dir, errors)
    else:
        _validate_sparse_proxy(args.object_dir, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.mode} contract valid for {args.object_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
