"""Shared utilities for the Stage 1 backend scaffold."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_OBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def validate_object_id(object_id: str) -> str:
    if not _OBJECT_ID_PATTERN.fullmatch(object_id):
        raise ValueError(
            "Invalid object_id. Use 1-128 characters: letters, numbers, "
            "underscores, or hyphens; start with a letter or number."
        )
    return object_id


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return data


def ensure_output_dirs(
    base_output_dir: Path,
    object_id: str,
    config: dict[str, Any] | None = None,
    create_intermediate: bool = True,
) -> dict[str, Path]:
    object_id = validate_object_id(object_id)
    config = config or {}
    intermediate_base = Path(config.get("intermediate_dir", "data/intermediate"))

    object_dir = base_output_dir / object_id
    preview_dir = object_dir / "preview"
    debug_dir = object_dir / "debug"
    intermediate_dir = intermediate_base / object_id
    images_original_dir = intermediate_dir / "images_original"
    images_masked_dir = intermediate_dir / "images_masked"

    preview_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    if create_intermediate:
        images_original_dir.mkdir(parents=True, exist_ok=True)
        images_masked_dir.mkdir(parents=True, exist_ok=True)

    return {
        "object": object_dir,
        "preview": preview_dir,
        "debug": debug_dir,
        "intermediate": intermediate_dir,
        "images_original": images_original_dir,
        "images_masked": images_masked_dir,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_metadata(object_id: str) -> dict[str, Any]:
    object_id = validate_object_id(object_id)
    return {
        "object_id": object_id,
        "visual_model": "point_cloud_metric.ply",
        "proxy_model": "proxy_collider.obj",
        "unit": "meter",
        "scale_factor": 1.0,
        "scale_method": "none_or_pending",
        "bbox_size_m": None,
        "object_center_m": None,
        "proxy_center_m": None,
        "coordinate_system": "backend_defined",
        "up_axis": "Y",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


_STAGE_REPORT_FILES = {
    "frame_extraction": "frame_selection_report.json",
    "background_removal": "mask_manifest.json",
    "marker_detection": "marker_detection_report.json",
    "colmap": "colmap_summary.json",
    "training": "training_report.json",
    "scale_estimation": "scale_report.json",
    "proxy_generation": "proxy_report.json",
}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _relative_to_object_dir(path: Path, object_dir: Path) -> str:
    try:
        return path.resolve().relative_to(object_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _asset_record(object_dir: Path, relative_path: str) -> dict[str, Any]:
    path = object_dir / relative_path
    return {
        "path": relative_path,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def build_package_manifest(output_dirs: dict[str, Path]) -> dict[str, Any]:
    """Build a Unity-facing manifest from current output files and reports."""
    object_dir = output_dirs["object"]
    preview_dir = output_dirs["preview"]
    debug_dir = output_dirs["debug"]
    metadata = _read_json_if_exists(object_dir / "metadata.json")
    processing_log = _read_json_if_exists(object_dir / "processing_log.json")

    stages = {}
    for stage_name, report_name in _STAGE_REPORT_FILES.items():
        report = _read_json_if_exists(debug_dir / report_name)
        stages[stage_name] = {
            "report": f"debug/{report_name}",
            "status": report.get("status", "missing"),
        }
        if "reason" in report:
            stages[stage_name]["reason"] = report["reason"]

    preview_assets = []
    for path in sorted(preview_dir.glob("*")):
        if path.is_file():
            preview_assets.append(
                {
                    "path": _relative_to_object_dir(path, object_dir),
                    "size_bytes": path.stat().st_size,
                }
            )

    visual_model = str(metadata.get("visual_model", "point_cloud_metric.ply"))
    proxy_model = str(metadata.get("proxy_model", "proxy_collider.obj"))
    return {
        "schema_version": 1,
        "object_id": metadata.get("object_id", processing_log.get("object_id")),
        "status": processing_log.get("status", "unknown"),
        "unit": metadata.get("unit", "meter"),
        "coordinate_system": metadata.get("coordinate_system", "backend_defined"),
        "up_axis": metadata.get("up_axis", "Y"),
        "created_at": metadata.get("created_at"),
        "visual_representation": {
            "type": "metric_point_cloud",
            **_asset_record(object_dir, visual_model),
        },
        "physics_representation": {
            "type": "obb_proxy_mesh",
            **_asset_record(object_dir, proxy_model),
        },
        "metadata": _asset_record(object_dir, "metadata.json"),
        "processing_log": _asset_record(object_dir, "processing_log.json"),
        "previews": preview_assets,
        "stages": stages,
        "unity_contract": {
            "visual_and_physics_are_separate": True,
            "raw_3dgs_point_cloud_used_as_collider": False,
        },
    }


def write_package_manifest(output_dirs: dict[str, Path]) -> None:
    write_json(output_dirs["object"] / "package_manifest.json", build_package_manifest(output_dirs))
