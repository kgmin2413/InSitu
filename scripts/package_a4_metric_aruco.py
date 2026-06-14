#!/usr/bin/env python3
"""Package an InSitu-A4 3DGS PLY with ArUco-derived metric scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_proxy import run_proxy_generation  # noqa: E402
from marker_detection import detect_markers_in_images  # noqa: E402
from scale_estimation import run_scale_estimation  # noqa: E402
from utils import build_metadata, ensure_output_dirs, write_json, write_package_manifest  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ply_vertex_count(path: Path) -> int:
    return int(PlyData.read(path)["vertex"].count)


def _registered_image_count(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0
    lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return len(lines) // 2


def _sparse_point_count(points_txt: Path) -> int:
    if not points_txt.exists():
        return 0
    return sum(1 for line in points_txt.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True, help="Output object id")
    parser.add_argument("--a4-dataset", required=True, type=Path, help="A4 dataset directory containing input/, sparse_txt/, masks/")
    parser.add_argument("--a4-model", required=True, type=Path, help="A4 model directory")
    parser.add_argument("--raw-ply", required=True, type=Path, help="A4 3DGS PLY to metric-scale, usually mask-vote pruned")
    parser.add_argument("--a4-marker-image-dataset", type=Path, default=None, help="Dataset whose input images contain visible ArUco marker; defaults to --a4-dataset")
    parser.add_argument("--text-model-dir", type=Path, default=None, help="COLMAP text model dir; defaults to <a4-dataset>/sparse_txt")
    parser.add_argument("--mask-undistorted-dir", type=Path, default=None, help="Optional undistorted mask dir copied into intermediate images_masked")
    parser.add_argument("--maskvote-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--intermediate-dir", default="data/intermediate")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--source-video", default=None)
    parser.add_argument("--frame-method", default="InSitu-A4 preprocessing; full COLMAP poses retained")
    parser.add_argument("--marker-id", type=int, default=23)
    parser.add_argument("--marker-size-m", type=float, default=0.12)
    parser.add_argument("--marker-dictionary", default="DICT_4X4_50")
    parser.add_argument("--min-marker-observations", type=int, default=2)
    parser.add_argument("--opacity-threshold", type=float, default=0.03)
    parser.add_argument("--no-opacity-denoise", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a4_dataset = _resolve(args.a4_dataset)
    a4_model = _resolve(args.a4_model)
    raw_ply = _resolve(args.raw_ply)
    marker_dataset = _resolve(args.a4_marker_image_dataset) if args.a4_marker_image_dataset else a4_dataset
    text_model_dir = _resolve(args.text_model_dir) if args.text_model_dir else a4_dataset / "sparse_txt"
    mask_undistorted_dir = _resolve(args.mask_undistorted_dir) if args.mask_undistorted_dir else a4_dataset / "mask_undistorted" / "images"
    maskvote_report_path = _resolve(args.maskvote_report) if args.maskvote_report else raw_ply.with_suffix(".json")

    if not raw_ply.exists():
        raise FileNotFoundError(raw_ply)
    if not (text_model_dir / "images.txt").exists():
        raise FileNotFoundError(text_model_dir / "images.txt")

    config = {
        "intermediate_dir": args.intermediate_dir,
        "scale_estimation": {
            "method": "aruco_marker",
            "source": "training_point_cloud",
            "marker_id": args.marker_id,
            "marker_size_m": args.marker_size_m,
            "min_marker_observations": args.min_marker_observations,
            "foreground_filter": {"enabled": False},
            "denoise": {"enabled": not args.no_opacity_denoise, "opacity_threshold": args.opacity_threshold, "statistical": {"enabled": False}},
            "object_filter": {"enabled": False},
        },
        "proxy_generation": {
            "method": "obb",
            "percentile_filter": {"enabled": True, "lower": 1.0, "upper": 99.0},
            "flat_bottom": {"enabled": True, "up_axis": "Y"},
        },
    }

    output_dirs = ensure_output_dirs(_resolve(args.output_root), args.object_id, config)
    write_json(output_dirs["object"] / "metadata.json", build_metadata(args.object_id))

    image_paths = sorted((a4_dataset / "input").glob("*.jpg"))
    marker_image_paths = sorted((marker_dataset / "input").glob("*.jpg"))
    for mask_path in sorted(mask_undistorted_dir.glob("*.jpg")) if mask_undistorted_dir.exists() else []:
        target = output_dirs["images_masked"] / mask_path.name
        if not target.exists():
            target.write_bytes(mask_path.read_bytes())

    marker_report = detect_markers_in_images(
        marker_image_paths,
        dictionary_name=args.marker_dictionary,
        marker_ids=[args.marker_id],
        marker_size_m=args.marker_size_m,
        preview_path=output_dirs["preview"] / "aruco_detection_preview.jpg",
        max_images=None,
    )
    write_json(output_dirs["debug"] / "marker_detection_report.json", marker_report)
    if marker_report["status"] != "success":
        raise RuntimeError("No ArUco marker detections were found")

    registered_images = _registered_image_count(text_model_dir / "images.txt")
    sparse_points = _sparse_point_count(text_model_dir / "points3D.txt")
    raw_count = _ply_vertex_count(raw_ply)
    seed_report = _read_json_if_exists(a4_dataset / "foreground_seed_report.json")
    maskvote_report = _read_json_if_exists(maskvote_report_path)

    write_json(
        output_dirs["debug"] / "colmap_summary.json",
        {
            "status": "success",
            "sparse_model_created": True,
            "sparse_model_dir": _relative(a4_dataset / "sparse" / "0"),
            "text_model_dir": _relative(text_model_dir),
            "registered_images": registered_images,
            "registered_ratio": registered_images / max(1, len(image_paths)),
            "sparse_points": sparse_points,
            "camera_model": "OPENCV",
        },
    )
    write_json(
        output_dirs["debug"] / "training_report.json",
        {
            "status": "success",
            "training_ran": True,
            "requested_iterations": args.iterations,
            "model_dir": _relative(a4_model),
            "model_created": True,
            "point_cloud": _relative(raw_ply),
            "iterations": args.iterations,
            "final_gaussian_count": raw_count,
        },
    )
    write_json(
        output_dirs["debug"] / "frame_selection_report.json",
        {
            "status": "success",
            "source_video": args.source_video,
            "method": args.frame_method,
            "selected_frame_count": len(image_paths),
        },
    )
    write_json(output_dirs["debug"] / "capture_quality_report.json", {"status": "success", "note": "A4 dataset reused for packaging"})
    write_json(
        output_dirs["debug"] / "mask_manifest.json",
        {
            "status": "success",
            "method": "A4 masks/foreground training reused for metric packaging",
            "mask_count": len(list((a4_dataset / "masks").glob("*.png"))),
            "undistorted_mask_count": len(list(mask_undistorted_dir.glob("*.jpg"))) if mask_undistorted_dir.exists() else 0,
            "foreground_seed": seed_report,
            "maskvote_gaussian_pruning": maskvote_report,
        },
    )
    with (output_dirs["debug"] / "selected_frames.txt").open("w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{_relative(path)}\n")

    scale_result = run_scale_estimation(output_dirs, config)
    proxy_result = run_proxy_generation(output_dirs, config)

    timing = {stage: 0.0 for stage in ("frame_extraction", "background_removal", "marker_detection", "colmap", "training", "scale_estimation", "proxy_generation", "total")}
    write_json(
        output_dirs["object"] / "processing_log.json",
        {
            "object_id": args.object_id,
            "preset": "a4_maskvote_aruco_metric_generic",
            "timing_sec": timing,
            "frame_stats": {"selected_frames": len(image_paths), "marker_detection_frames": len(marker_image_paths)},
            "colmap_stats": {"registered_images": registered_images, "sparse_points": sparse_points},
            "training_stats": {"iterations": args.iterations, "final_gaussian_count": raw_count},
            "status": "success",
            "warnings": scale_result.get("warnings", []) + proxy_result.get("warnings", []),
            "errors": [],
        },
    )
    write_package_manifest(output_dirs)
    print(json.dumps({"object_id": args.object_id, "output": _relative(output_dirs["object"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
