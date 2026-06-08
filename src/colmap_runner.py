"""COLMAP pose-estimation wrapper for selected pipeline frames."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from utils import write_json


_DEFAULT_COLMAP_STATS = {
    "registered_images": 0,
    "registered_ratio": 0.0,
    "reprojection_error": 0.0,
    "sparse_points": 0,
}


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_selected_frames(debug_dir: Path) -> list[str]:
    selected_frames_path = debug_dir / "selected_frames.txt"
    if not selected_frames_path.exists():
        raise RuntimeError("selected_frames.txt is missing; run frame extraction first")
    return [
        line.strip()
        for line in selected_frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_colmap_config(config: dict[str, Any]) -> dict[str, Any]:
    colmap_config = config.get("colmap", {})
    matcher = str(colmap_config.get("matcher", "exhaustive"))
    if matcher not in {"exhaustive", "sequential"}:
        raise ValueError("colmap.matcher must be exhaustive or sequential")

    quality_config = colmap_config.get("quality", {})
    min_registered_images = int(quality_config.get("min_registered_images", 0))
    min_registered_ratio = float(quality_config.get("min_registered_ratio", 0.0))
    min_sparse_points = int(quality_config.get("min_sparse_points", 0))
    max_reprojection_error_raw = quality_config.get("max_reprojection_error")
    max_reprojection_error = (
        None if max_reprojection_error_raw is None else float(max_reprojection_error_raw)
    )
    fail_on_low_quality = bool(quality_config.get("fail_on_low_quality", False))
    if min_registered_images < 0:
        raise ValueError("colmap.quality.min_registered_images must be zero or greater")
    if not 0.0 <= min_registered_ratio <= 1.0:
        raise ValueError("colmap.quality.min_registered_ratio must be between 0 and 1")
    if min_sparse_points < 0:
        raise ValueError("colmap.quality.min_sparse_points must be zero or greater")
    if max_reprojection_error is not None and max_reprojection_error < 0:
        raise ValueError("colmap.quality.max_reprojection_error must be zero or greater")

    return {
        "enabled": bool(colmap_config.get("enabled", False)),
        "matcher": matcher,
        "camera_model": str(colmap_config.get("camera_model", "SIMPLE_RADIAL")),
        "single_camera": bool(colmap_config.get("single_camera", True)),
        "use_gpu": bool(colmap_config.get("use_gpu", False)),
        "timeout_sec": int(colmap_config.get("timeout_sec", 1800)),
        "binary": str(colmap_config.get("binary", "colmap")),
        "quality": {
            "min_registered_images": min_registered_images,
            "min_registered_ratio": min_registered_ratio,
            "min_sparse_points": min_sparse_points,
            "max_reprojection_error": max_reprojection_error,
            "fail_on_low_quality": fail_on_low_quality,
        },
    }


def _command_for_summary(command: list[str]) -> str:
    return " ".join(command)


def _run_command(command: list[str], timeout_sec: int) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    record = {
        "command": _command_for_summary(command),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip()[-4000:],
        "stderr_tail": result.stderr.strip()[-4000:],
    }
    if result.returncode != 0:
        detail = record["stderr_tail"] or record["stdout_tail"] or "unknown COLMAP error"
        raise RuntimeError(f"COLMAP command failed: {record['command']}\n{detail}")
    return record


def _find_model_dir(sparse_dir: Path) -> Path | None:
    if not sparse_dir.exists():
        return None
    candidates = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "images.bin").exists() or (candidate / "images.txt").exists():
            return candidate
    return None


def _registered_image_count(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0
    non_comment_lines = [
        line
        for line in images_txt.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return len(non_comment_lines) // 2


def _mean_reprojection_error(points3d_txt: Path) -> float:
    if not points3d_txt.exists():
        return 0.0
    errors: list[float] = []
    for line in points3d_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 8:
            errors.append(float(parts[7]))
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def _sparse_point_count(points3d_txt: Path) -> int:
    if not points3d_txt.exists():
        return 0
    count = 0
    for line in points3d_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _quality_warnings(stats: dict[str, Any], quality_config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if stats["registered_images"] < quality_config["min_registered_images"]:
        warnings.append(
            "registered image count is below threshold "
            f"({stats['registered_images']} < {quality_config['min_registered_images']})"
        )
    if stats["registered_ratio"] < quality_config["min_registered_ratio"]:
        warnings.append(
            "registered image ratio is below threshold "
            f"({stats['registered_ratio']} < {quality_config['min_registered_ratio']})"
        )
    if stats["sparse_points"] < quality_config["min_sparse_points"]:
        warnings.append(
            "sparse point count is below threshold "
            f"({stats['sparse_points']} < {quality_config['min_sparse_points']})"
        )
    max_error = quality_config["max_reprojection_error"]
    if max_error is not None and stats["reprojection_error"] > max_error:
        warnings.append(
            "mean reprojection error is above threshold "
            f"({stats['reprojection_error']} > {max_error})"
        )
    return warnings


def _write_summary(output_dirs: dict[str, Path], summary: dict[str, Any]) -> None:
    write_json(output_dirs["debug"] / "colmap_summary.json", summary)


def _base_summary(selected_frames: list[str], colmap_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_frame_count": len(selected_frames),
        "configured_enabled": colmap_config["enabled"],
        "configured_matcher": colmap_config["matcher"],
        "camera_model": colmap_config["camera_model"],
        "single_camera": colmap_config["single_camera"],
        "use_gpu": colmap_config["use_gpu"],
        "quality_thresholds": colmap_config["quality"],
        **_DEFAULT_COLMAP_STATS,
    }


def run_colmap(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    selected_frames = _read_selected_frames(output_dirs["debug"])
    colmap_config = _read_colmap_config(config)
    stats = dict(_DEFAULT_COLMAP_STATS)

    if not selected_frames:
        summary = {
            "status": "skipped",
            "reason": "no selected frames available",
            "colmap_ran": False,
            "sparse_model_created": False,
            **_base_summary(selected_frames, colmap_config),
        }
        _write_summary(output_dirs, summary)
        return {
            "colmap_stats": stats,
            "warnings": ["COLMAP skipped because no selected frames are available"],
        }

    if not colmap_config["enabled"]:
        summary = {
            "status": "skipped",
            "reason": "colmap.enabled is false",
            "colmap_ran": False,
            "sparse_model_created": False,
            **_base_summary(selected_frames, colmap_config),
        }
        _write_summary(output_dirs, summary)
        return {
            "colmap_stats": stats,
            "warnings": ["COLMAP execution is disabled in config"],
        }

    binary_path = shutil.which(colmap_config["binary"])
    if binary_path is None:
        summary = {
            "status": "failed",
            "reason": f"COLMAP binary not found: {colmap_config['binary']}",
            "colmap_ran": False,
            "sparse_model_created": False,
            **_base_summary(selected_frames, colmap_config),
        }
        _write_summary(output_dirs, summary)
        raise RuntimeError(summary["reason"])

    images_original_dir = output_dirs["images_original"]
    if not images_original_dir.exists():
        raise RuntimeError(f"COLMAP image directory does not exist: {images_original_dir}")

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    workspace_dir = output_dirs["intermediate"] / "colmap" / run_name
    sparse_dir = workspace_dir / "sparse"
    text_model_dir = workspace_dir / "sparse_txt"
    database_path = workspace_dir / "database.db"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_model_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    summary = {
        "status": "running",
        "colmap_ran": True,
        "sparse_model_created": False,
        "workspace_dir": _relative_repo_path(workspace_dir),
        "database_path": _relative_repo_path(database_path),
        "image_path": _relative_repo_path(images_original_dir),
        "text_model_dir": _relative_repo_path(text_model_dir),
        "commands": commands,
        **_base_summary(selected_frames, colmap_config),
    }
    _write_summary(output_dirs, summary)

    binary = binary_path
    timeout_sec = colmap_config["timeout_sec"]
    try:
        commands.append(
            _run_command(
                [
                    binary,
                    "feature_extractor",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_original_dir),
                    "--ImageReader.camera_model",
                    colmap_config["camera_model"],
                    "--ImageReader.single_camera",
                    "1" if colmap_config["single_camera"] else "0",
                    "--SiftExtraction.use_gpu",
                    "1" if colmap_config["use_gpu"] else "0",
                ],
                timeout_sec,
            )
        )

        matcher_command = "sequential_matcher" if colmap_config["matcher"] == "sequential" else "exhaustive_matcher"
        commands.append(
            _run_command(
                [
                    binary,
                    matcher_command,
                    "--database_path",
                    str(database_path),
                    "--SiftMatching.use_gpu",
                    "1" if colmap_config["use_gpu"] else "0",
                ],
                timeout_sec,
            )
        )

        commands.append(
            _run_command(
                [
                    binary,
                    "mapper",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_original_dir),
                    "--output_path",
                    str(sparse_dir),
                ],
                timeout_sec,
            )
        )

        model_dir = _find_model_dir(sparse_dir)
        if model_dir is None:
            raise RuntimeError("COLMAP mapper did not create a sparse model")

        commands.append(
            _run_command(
                [
                    binary,
                    "model_converter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(text_model_dir),
                    "--output_type",
                    "TXT",
                ],
                timeout_sec,
            )
        )

        registered_images = _registered_image_count(text_model_dir / "images.txt")
        reprojection_error = _mean_reprojection_error(text_model_dir / "points3D.txt")
        sparse_points = _sparse_point_count(text_model_dir / "points3D.txt")
        registered_ratio = registered_images / len(selected_frames) if selected_frames else 0.0
        stats = {
            "registered_images": registered_images,
            "registered_ratio": round(registered_ratio, 6),
            "reprojection_error": round(reprojection_error, 6),
            "sparse_points": sparse_points,
        }
        quality_warnings = _quality_warnings(stats, colmap_config["quality"])
        quality_status = "low_quality" if quality_warnings else "pass"
        summary.update(
            {
                "status": "success",
                "quality_status": quality_status,
                "sparse_model_created": True,
                "sparse_model_dir": _relative_repo_path(model_dir),
                "quality_warnings": quality_warnings,
                **stats,
            }
        )
        if quality_warnings and colmap_config["quality"]["fail_on_low_quality"]:
            summary.update(
                {
                    "status": "failed",
                    "quality_status": "failed",
                    "error": "COLMAP quality gate failed: " + "; ".join(quality_warnings),
                }
            )
            _write_summary(output_dirs, summary)
            raise RuntimeError(summary["error"])
        _write_summary(output_dirs, summary)
        return {"colmap_stats": stats, "warnings": quality_warnings}
    except subprocess.TimeoutExpired as exc:
        summary.update(
            {
                "status": "failed",
                "error": f"COLMAP command timed out after {timeout_sec} seconds: {_command_for_summary(exc.cmd)}",
                "sparse_model_created": False,
            }
        )
        _write_summary(output_dirs, summary)
        raise RuntimeError(summary["error"]) from exc
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error": str(exc),
                "sparse_model_created": False,
            }
        )
        _write_summary(output_dirs, summary)
        raise
