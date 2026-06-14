#!/usr/bin/env python3
"""Copy an InSitu output folder and align its metric PLY for Unity/AR placement."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_proxy import run_proxy_generation  # noqa: E402
from scale_estimation import _bbox_and_center, _write_reconstruction_preview, _write_rgb_inspection_point_cloud  # noqa: E402
from utils import write_json, write_package_manifest  # noqa: E402


def _axis_vector(axis: str) -> np.ndarray:
    axis = axis.upper()
    mapping = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    if axis not in mapping:
        raise ValueError("front axes must be one of +X, -X, +Z, -Z")
    return mapping[axis]


def _yaw_from_to(current_front: str, target_front: str) -> tuple[np.ndarray, np.ndarray, float]:
    current = _axis_vector(current_front)
    target = _axis_vector(target_front)
    current_angle = float(np.arctan2(current[0], current[2]))
    target_angle = float(np.arctan2(target[0], target[2]))
    angle = target_angle - current_angle
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    quat_wxyz = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=np.float64)
    return rotation, quat_wxyz, angle


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _transform_vertices(vertices: np.ndarray, rotation: np.ndarray, yaw_quat: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    transformed = vertices.copy()
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    rotated = xyz @ rotation.T
    minimum = rotated.min(axis=0)
    maximum = rotated.max(axis=0)
    pivot = np.array([(minimum[0] + maximum[0]) * 0.5, minimum[1], (minimum[2] + maximum[2]) * 0.5], dtype=np.float64)
    aligned = rotated - pivot
    aligned[:, 1] -= aligned[:, 1].min()
    transformed["x"] = aligned[:, 0].astype(transformed["x"].dtype)
    transformed["y"] = aligned[:, 1].astype(transformed["y"].dtype)
    transformed["z"] = aligned[:, 2].astype(transformed["z"].dtype)
    names = transformed.dtype.names or ()
    changed = ["x", "y", "z"]
    if all(axis in names for axis in ("nx", "ny", "nz")):
        normals = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float64)
        aligned_normals = normals @ rotation.T
        transformed["nx"] = aligned_normals[:, 0].astype(transformed["nx"].dtype)
        transformed["ny"] = aligned_normals[:, 1].astype(transformed["ny"].dtype)
        transformed["nz"] = aligned_normals[:, 2].astype(transformed["nz"].dtype)
        changed += ["nx", "ny", "nz"]
    if all(axis in names for axis in ("rot_0", "rot_1", "rot_2", "rot_3")):
        q = np.column_stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).astype(np.float64)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn[qn == 0.0] = 1.0
        q = q / qn
        yaw = np.tile(yaw_quat, (len(q), 1))
        q2 = _quat_mul(yaw, q)
        q2n = np.linalg.norm(q2, axis=1, keepdims=True)
        q2n[q2n == 0.0] = 1.0
        q2 = q2 / q2n
        transformed["rot_0"] = q2[:, 0].astype(transformed["rot_0"].dtype)
        transformed["rot_1"] = q2[:, 1].astype(transformed["rot_1"].dtype)
        transformed["rot_2"] = q2[:, 2].astype(transformed["rot_2"].dtype)
        transformed["rot_3"] = q2[:, 3].astype(transformed["rot_3"].dtype)
        changed += ["rot_0", "rot_1", "rot_2", "rot_3"]
    post_xyz = np.column_stack([transformed["x"], transformed["y"], transformed["z"]]).astype(np.float64)
    report = {
        "translation_after_rotation_m": [round(float(v), 9) for v in (-pivot)],
        "original_bounds_m": {
            "min": [round(float(v), 6) for v in xyz.min(axis=0)],
            "max": [round(float(v), 6) for v in xyz.max(axis=0)],
            "bbox_size": [round(float(v), 6) for v in (xyz.max(axis=0) - xyz.min(axis=0))],
        },
        "aligned_bounds_m": {
            "min": [round(float(v), 6) for v in post_xyz.min(axis=0)],
            "max": [round(float(v), 6) for v in post_xyz.max(axis=0)],
            "bbox_size": [round(float(v), 6) for v in (post_xyz.max(axis=0) - post_xyz.min(axis=0))],
        },
        "transformed_fields": changed,
    }
    return transformed, report


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
    parser.add_argument("--source-dir", required=True, type=Path, help="Existing InSitu output directory with point_cloud_metric.ply")
    parser.add_argument("--object-id", required=True, help="New output object id")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--current-front", default="-Z", choices=["+X", "-X", "+Z", "-Z"])
    parser.add_argument("--target-front", default="+Z", choices=["+X", "-X", "+Z", "-Z"])
    parser.add_argument("--skip-proxy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dest_dir = (args.output_root / args.object_id).resolve()
    if not (source_dir / "point_cloud_metric.ply").exists():
        raise FileNotFoundError(source_dir / "point_cloud_metric.ply")
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    rotation, yaw_quat, yaw_angle = _yaw_from_to(args.current_front, args.target_front)
    source_ply = PlyData.read(source_dir / "point_cloud_metric.ply")
    vertices, alignment = _transform_vertices(source_ply["vertex"].data, rotation, yaw_quat)
    output_ply = dest_dir / "point_cloud_metric.ply"
    _write_ply_like(source_ply, vertices, output_ply)
    _write_rgb_inspection_point_cloud(vertices, dest_dir / "point_cloud_metric_rgb.ply")
    _write_reconstruction_preview(output_ply, dest_dir / "preview" / "reconstruction_preview.png")
    bbox_size_m, object_center_m = _bbox_and_center(vertices)

    metadata_path = dest_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({
        "object_id": args.object_id,
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "coordinate_system": f"Y-up, front={args.target_front}, bottom-center pivot at origin",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_json(metadata_path, metadata)

    report = {
        "status": "success",
        "source_output_dir": str(source_dir),
        "object_id": args.object_id,
        "aligned_point_cloud": str(output_ply),
        "unit": "meter",
        "metric_scale_preserved": True,
        "up_axis": "Y",
        "current_front_axis": args.current_front,
        "target_front_axis": args.target_front,
        "applied_transform": {
            "yaw_degrees": round(float(np.degrees(yaw_angle)), 6),
            "rotation_matrix": [[round(float(v), 6) for v in row] for row in rotation.tolist()],
            **alignment,
        },
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "unchanged_fields_note": "scale_0..2 and SH color coefficients are preserved; f_rest_* directional SH is not reprojected.",
    }
    write_json(dest_dir / "debug" / "alignment_report.json", report)

    scale_report_path = dest_dir / "debug" / "scale_report.json"
    if scale_report_path.exists():
        scale_report = json.loads(scale_report_path.read_text(encoding="utf-8"))
        scale_report.update({
            "point_cloud_metric": str(output_ply),
            "point_cloud_metric_rgb": str(dest_dir / "point_cloud_metric_rgb.ply"),
            "reconstruction_preview": str(dest_dir / "preview" / "reconstruction_preview.png"),
            "bbox_size_m": bbox_size_m,
            "object_center_m": object_center_m,
            "point_count": int(len(vertices)),
        })
        scale_report.setdefault("postprocess", {})["unity_alignment"] = report
        write_json(scale_report_path, scale_report)

    log_path = dest_dir / "processing_log.json"
    if log_path.exists():
        processing_log = json.loads(log_path.read_text(encoding="utf-8"))
        processing_log["object_id"] = args.object_id
        processing_log["preset"] = processing_log.get("preset", "") + "_unity_aligned"
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

    print(json.dumps({"object_id": args.object_id, "output": str(dest_dir), "bbox_size_m": bbox_size_m, "object_center_m": object_center_m}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
