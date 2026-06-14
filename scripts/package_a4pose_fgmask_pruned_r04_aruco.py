#!/usr/bin/env python3
"""Package an A4-trained point cloud with InSitu ArUco metric scale."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_proxy import run_proxy_generation  # noqa: E402
from marker_detection import detect_markers_in_images  # noqa: E402
from scale_estimation import run_scale_estimation  # noqa: E402
from utils import build_metadata, ensure_output_dirs, write_json, write_package_manifest  # noqa: E402


OBJECT_ID = "chair_a4pose_fgmask_pruned_r04_30000_aruco_20260611"
A4_DATASET = ROOT / "third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611"
A4_MARKER_IMAGE_DATASET = ROOT / "third_party/InSitu-A4/data/chair_insitu_a4true_20260611"
A4_MODEL = ROOT / "third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611"
RAW_PLY = A4_MODEL / "point_cloud/iteration_30000/point_cloud_maskvote_pruned_r04.ply"
MASKVOTE_REPORT = RAW_PLY.with_suffix(".json")
TEXT_MODEL_DIR = A4_DATASET / "sparse_txt"
MASK_UNDISTORTED_DIR = A4_DATASET / "mask_undistorted/images"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ply_vertex_count(path: Path) -> int:
    return int(PlyData.read(path)["vertex"].count)


def main() -> int:
    if not RAW_PLY.exists():
        raise FileNotFoundError(RAW_PLY)
    if not (TEXT_MODEL_DIR / "images.txt").exists():
        raise FileNotFoundError(TEXT_MODEL_DIR / "images.txt")

    config = {
        "intermediate_dir": "data/intermediate",
        "scale_estimation": {
            "method": "aruco_marker",
            "source": "training_point_cloud",
            "marker_id": 23,
            "marker_size_m": 0.12,
            "min_marker_observations": 2,
            "foreground_filter": {"enabled": False},
            "denoise": {"enabled": True, "opacity_threshold": 0.03, "statistical": {"enabled": False}},
            "object_filter": {"enabled": False},
        },
        "proxy_generation": {
            "method": "obb",
            "preprocess": {
                "max_points": 120000,
                "percentile": {"enabled": True, "lower": 1.0, "upper": 99.0},
            },
            "flat_bottom": {"enabled": True},
        },
    }

    output_dirs = ensure_output_dirs(ROOT / "output", OBJECT_ID, config)
    metadata = build_metadata(OBJECT_ID)
    write_json(output_dirs["object"] / "metadata.json", metadata)

    image_paths = sorted((A4_DATASET / "input").glob("*.jpg"))
    marker_image_paths = sorted((A4_MARKER_IMAGE_DATASET / "input").glob("*.jpg"))
    for mask_path in sorted(MASK_UNDISTORTED_DIR.glob("*.jpg")):
        target = output_dirs["images_masked"] / mask_path.name
        if not target.exists():
            target.write_bytes(mask_path.read_bytes())
    registered_image_count = sum(1 for line in (TEXT_MODEL_DIR / "images.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")) // 2
    sparse_points = sum(1 for line in (TEXT_MODEL_DIR / "points3D.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))
    seed_report_path = A4_DATASET / "foreground_seed_report.json"
    seed_report = json.loads(seed_report_path.read_text(encoding="utf-8")) if seed_report_path.exists() else {}
    maskvote_report = json.loads(MASKVOTE_REPORT.read_text(encoding="utf-8")) if MASKVOTE_REPORT.exists() else {}
    marker_report = detect_markers_in_images(
        marker_image_paths,
        dictionary_name="DICT_4X4_50",
        marker_ids=[23],
        marker_size_m=0.12,
        preview_path=output_dirs["preview"] / "aruco_detection_preview.jpg",
        max_images=None,
    )
    write_json(output_dirs["debug"] / "marker_detection_report.json", marker_report)
    if marker_report["status"] != "success":
        raise RuntimeError("No ArUco marker detections were found")

    raw_count = _ply_vertex_count(RAW_PLY)
    colmap_summary = {
        "status": "success",
        "sparse_model_created": True,
        "sparse_model_dir": _relative(A4_DATASET / "sparse/0"),
        "text_model_dir": _relative(TEXT_MODEL_DIR),
        "registered_images": registered_image_count,
        "registered_ratio": registered_image_count / max(1, len(image_paths)),
        "sparse_points": sparse_points,
        "camera_model": "OPENCV",
    }
    write_json(output_dirs["debug"] / "colmap_summary.json", colmap_summary)
    write_json(
        output_dirs["debug"] / "training_report.json",
        {
            "status": "success",
            "training_ran": True,
            "requested_iterations": 30000,
            "model_dir": _relative(A4_MODEL),
            "model_created": True,
            "point_cloud": _relative(RAW_PLY),
            "iterations": 30000,
            "final_gaussian_count": raw_count,
        },
    )
    write_json(
        output_dirs["debug"] / "frame_selection_report.json",
        {
            "status": "success",
            "source_video": "data/input/chair.mp4",
            "method": "InSitu-A4 preprocess_video.py --fps 4; full COLMAP poses retained",
            "selected_frame_count": len(image_paths),
        },
    )
    write_json(
        output_dirs["debug"] / "capture_quality_report.json",
        {"status": "success", "note": "A4 true run reused preprocessed frames"},
    )
    write_json(
        output_dirs["debug"] / "mask_manifest.json",
        {
            "status": "success",
            "method": "rembg via InSitu-A4 preprocess_video.py; masks undistorted with full COLMAP model; foreground sparse seed + alpha/background loss",
            "mask_count": len(list((A4_DATASET / "masks").glob("*.png"))),
            "undistorted_mask_count": len(list(MASK_UNDISTORTED_DIR.glob("*.jpg"))),
            "foreground_seed": seed_report,
            "maskvote_gaussian_pruning": maskvote_report,
        },
    )
    with (output_dirs["debug"] / "selected_frames.txt").open("w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{_relative(path)}\n")

    scale_result = run_scale_estimation(output_dirs, config)
    proxy_result = run_proxy_generation(output_dirs, config)

    timing = {
        "frame_extraction": 0.0,
        "background_removal": 0.0,
        "marker_detection": 0.0,
        "colmap": 0.0,
        "training": 0.0,
        "scale_estimation": 0.0,
        "proxy_generation": 0.0,
        "total": 0.0,
    }
    write_json(
        output_dirs["object"] / "processing_log.json",
        {
            "object_id": OBJECT_ID,
            "preset": "a4_full_pose_fgseed_alpha_bgpenalty_maskvote_pruned_r04_aruco_metric",
            "timing_sec": timing,
            "frame_stats": {"selected_frames": len(image_paths), "marker_detection_frames": len(marker_image_paths)},
            "colmap_stats": {"registered_images": registered_image_count, "sparse_points": sparse_points},
            "training_stats": {"iterations": 30000, "final_gaussian_count": raw_count},
            "status": "success",
            "warnings": scale_result.get("warnings", []) + proxy_result.get("warnings", []),
            "errors": [],
        },
    )
    write_package_manifest(output_dirs)
    print(json.dumps({"object_id": OBJECT_ID, "output": _relative(output_dirs["object"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
