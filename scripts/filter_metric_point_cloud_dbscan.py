#!/usr/bin/env python3
"""Copy an InSitu output folder and DBSCAN-filter its metric Gaussian PLY."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_proxy import run_proxy_generation  # noqa: E402
from scale_estimation import _bbox_and_center, _write_reconstruction_preview, _write_rgb_inspection_point_cloud  # noqa: E402
from utils import write_json, write_package_manifest  # noqa: E402


def _write_ply_like(source_ply: PlyData, vertices: np.ndarray, output_ply: Path) -> None:
    elements = []
    for element in source_ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertices, "vertex"))
        else:
            elements.append(element)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=source_ply.text, byte_order=source_ply.byte_order).write(output_ply)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Existing InSitu output folder with point_cloud_metric.ply")
    parser.add_argument("--object-id", required=True, help="New output object id")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--eps-m", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--keep-mode", choices=("largest", "min_points"), default="largest")
    parser.add_argument("--min-cluster-points", type=int, default=500, help="Only used when --keep-mode min_points")
    parser.add_argument("--skip-proxy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eps_m <= 0:
        raise ValueError("--eps-m must be greater than zero")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be at least one")

    source_dir = args.source_dir.resolve()
    dest_dir = (args.output_root / args.object_id).resolve()
    source_ply_path = source_dir / "point_cloud_metric.ply"
    if not source_ply_path.exists():
        raise FileNotFoundError(source_ply_path)
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    ply = PlyData.read(source_ply_path)
    vertices = ply["vertex"].data
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    labels = DBSCAN(eps=args.eps_m, min_samples=args.min_samples).fit_predict(xyz)
    cluster_labels = sorted(int(label) for label in set(labels.tolist()) if int(label) != -1)
    counts = {label: int(np.count_nonzero(labels == label)) for label in cluster_labels}
    if not counts:
        raise RuntimeError("DBSCAN found no non-noise clusters")

    if args.keep_mode == "largest":
        kept_labels = [max(counts, key=counts.get)]
    else:
        kept_labels = [label for label, count in counts.items() if count >= args.min_cluster_points]
        if not kept_labels:
            kept_labels = [max(counts, key=counts.get)]

    keep = np.isin(labels, kept_labels)
    if int(np.count_nonzero(keep)) < 3:
        raise RuntimeError("DBSCAN filter would leave too few points")
    filtered_vertices = vertices[keep]
    output_ply = dest_dir / "point_cloud_metric.ply"
    _write_ply_like(ply, filtered_vertices, output_ply)
    _write_rgb_inspection_point_cloud(filtered_vertices, dest_dir / "point_cloud_metric_rgb.ply")
    _write_reconstruction_preview(output_ply, dest_dir / "preview" / "reconstruction_preview.png")

    bbox_size_m, object_center_m = _bbox_and_center(filtered_vertices)
    point_count = int(len(filtered_vertices))
    removed = int(len(vertices) - point_count)
    filtered_xyz = np.column_stack([filtered_vertices["x"], filtered_vertices["y"], filtered_vertices["z"]]).astype(np.float64)

    metadata_path = dest_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({
        "object_id": args.object_id,
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_json(metadata_path, metadata)

    top_clusters = []
    for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]:
        points = xyz[labels == label]
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        top_clusters.append({
            "label": int(label),
            "count": int(count),
            "bbox_size_m": [round(float(v), 6) for v in (maximum - minimum)],
            "object_center_m": [round(float(v), 6) for v in ((maximum + minimum) / 2.0)],
        })

    report = {
        "status": "success",
        "method": "DBSCAN on metric Gaussian centers",
        "source_output_dir": str(source_dir),
        "object_id": args.object_id,
        "eps_m": args.eps_m,
        "min_samples": args.min_samples,
        "keep_mode": "largest_cluster_only" if args.keep_mode == "largest" else "clusters_with_min_points",
        "min_cluster_points": args.min_cluster_points if args.keep_mode == "min_points" else None,
        "input_points": int(len(vertices)),
        "kept_points": point_count,
        "removed_points": removed,
        "noise_points": int(np.count_nonzero(labels == -1)),
        "cluster_count": len(cluster_labels),
        "kept_labels": [int(v) for v in kept_labels],
        "selected_label": int(kept_labels[0]) if len(kept_labels) == 1 else None,
        "top_clusters": top_clusters,
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "bounds_m": {
            "min": [round(float(v), 6) for v in filtered_xyz.min(axis=0)],
            "max": [round(float(v), 6) for v in filtered_xyz.max(axis=0)],
        },
    }
    write_json(dest_dir / "debug" / "dbscan_pruning_report.json", report)

    scale_report_path = dest_dir / "debug" / "scale_report.json"
    if scale_report_path.exists():
        scale_report = json.loads(scale_report_path.read_text(encoding="utf-8"))
        scale_report.update({
            "point_cloud_metric": str(output_ply),
            "point_cloud_metric_rgb": str(dest_dir / "point_cloud_metric_rgb.ply"),
            "reconstruction_preview": str(dest_dir / "preview" / "reconstruction_preview.png"),
            "bbox_size_m": bbox_size_m,
            "object_center_m": object_center_m,
            "point_count": point_count,
        })
        scale_report.setdefault("postprocess", {})["dbscan_metric_space"] = {
            "enabled": True,
            "eps_m": args.eps_m,
            "min_samples": args.min_samples,
            "keep_mode": report["keep_mode"],
            "input_points": int(len(vertices)),
            "kept_points": point_count,
            "removed_points": removed,
            "selected_label": report["selected_label"],
        }
        write_json(scale_report_path, scale_report)

    log_path = dest_dir / "processing_log.json"
    if log_path.exists():
        processing_log = json.loads(log_path.read_text(encoding="utf-8"))
        processing_log["object_id"] = args.object_id
        processing_log["preset"] = processing_log.get("preset", "") + "_dbscan_metric"
        processing_log.setdefault("training_stats", {})["dbscan_metric_input_points"] = int(len(vertices))
        processing_log.setdefault("training_stats", {})["dbscan_metric_kept_points"] = point_count
        processing_log.setdefault("training_stats", {})["dbscan_metric_removed_points"] = removed
        write_json(log_path, processing_log)

    if not args.skip_proxy:
        output_dirs = {
            "object": dest_dir,
            "preview": dest_dir / "preview",
            "debug": dest_dir / "debug",
            "intermediate": ROOT / "data/intermediate" / args.object_id,
            "images_original": ROOT / "data/intermediate" / args.object_id / "images_original",
            "images_masked": ROOT / "data/intermediate" / args.object_id / "images_masked",
        }
        run_proxy_generation(output_dirs, {"proxy_generation": {"method": "obb", "percentile_filter": {"enabled": True, "lower": 1.0, "upper": 99.0}, "flat_bottom": {"enabled": True, "up_axis": "Y"}}})
        write_package_manifest(output_dirs)

    print(json.dumps({
        "object_id": args.object_id,
        "output": str(dest_dir),
        "input_points": int(len(vertices)),
        "kept_points": point_count,
        "removed_points": removed,
        "bbox_size_m": bbox_size_m,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
