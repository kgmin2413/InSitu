#!/usr/bin/env python3
"""Export a metric 3DGS PLY as a lightweight-ish GLB splat mesh for Unity preview."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scale_estimation import _vertices_to_colors  # noqa: E402
from utils import write_json  # noqa: E402

BASE_VERTS = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
], dtype=np.float32)
BASE_FACES = np.array([
    [0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0],
    [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5],
], dtype=np.int64)


def _quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    quat = quat / norm
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rotm = np.empty((len(quat), 3, 3), dtype=np.float32)
    rotm[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotm[:, 0, 1] = 2 * (x * y - z * w)
    rotm[:, 0, 2] = 2 * (x * z + y * w)
    rotm[:, 1, 0] = 2 * (x * y + z * w)
    rotm[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotm[:, 1, 2] = 2 * (y * z - x * w)
    rotm[:, 2, 0] = 2 * (x * z - y * w)
    rotm[:, 2, 1] = 2 * (y * z + x * w)
    rotm[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Existing InSitu output directory with point_cloud_metric.ply")
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--glb-name", default="chair_metric_splatmesh_flattened_pivot.glb")
    parser.add_argument("--opacity-min", type=float, default=0.04)
    parser.add_argument("--radius-multiplier", type=float, default=2.1)
    parser.add_argument("--min-radius-m", type=float, default=0.0035)
    parser.add_argument("--max-radius-m", type=float, default=0.035)
    parser.add_argument("--bottom-percentile", type=float, default=0.25)
    parser.add_argument("--bottom-flatten-band-m", type=float, default=0.018)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dest_dir = (args.output_root / args.object_id).resolve()
    source_ply = source_dir / "point_cloud_metric.ply"
    if not source_ply.exists():
        raise FileNotFoundError(source_ply)
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    glb_path = dest_dir / args.glb_name

    ply = PlyData.read(source_ply)
    vertices = ply["vertex"].data
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    rgb = _vertices_to_colors(vertices).astype(np.uint8)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float32)))
    keep = opacity >= args.opacity_min
    xyz = xyz[keep]
    rgb = rgb[keep]
    opacity = opacity[keep]
    scales = np.exp(np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).astype(np.float32))[keep]
    scales = np.clip(scales * args.radius_multiplier, args.min_radius_m, args.max_radius_m).astype(np.float32)
    quats = np.column_stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).astype(np.float32)[keep]
    rotm = _quat_to_rotmat_wxyz(quats)

    local = BASE_VERTS[None, :, :] * scales[:, None, :]
    splat_vertices = np.einsum("nvc,nkc->nvk", local, rotm) + xyz[:, None, :]
    mesh_vertices = splat_vertices.reshape(-1, 3).astype(np.float32)
    faces = (BASE_FACES[None, :, :] + (np.arange(len(xyz), dtype=np.int64)[:, None, None] * len(BASE_VERTS))).reshape(-1, 3)
    colors = np.repeat(rgb, len(BASE_VERTS), axis=0)
    alpha = np.clip((opacity * 255.0).astype(np.uint8), 32, 255)
    vertex_colors = np.column_stack([colors, np.repeat(alpha, len(BASE_VERTS))])

    floor_y = float(np.percentile(mesh_vertices[:, 1], args.bottom_percentile))
    flat_mask = mesh_vertices[:, 1] <= floor_y + args.bottom_flatten_band_m
    mesh_vertices[flat_mask, 1] = floor_y
    mesh_vertices[mesh_vertices[:, 1] < floor_y, 1] = floor_y
    pivot = np.array([(mesh_vertices[:, 0].min() + mesh_vertices[:, 0].max()) * 0.5, floor_y, (mesh_vertices[:, 2].min() + mesh_vertices[:, 2].max()) * 0.5], dtype=np.float32)
    mesh_vertices = mesh_vertices - pivot
    mesh_vertices[:, 1] -= mesh_vertices[:, 1].min()

    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    mesh.remove_unreferenced_vertices()
    _ = mesh.vertex_normals
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(glb_path)

    report = {
        "status": "success",
        "source_output_dir": str(source_dir),
        "object_id": args.object_id,
        "glb": str(glb_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "one oriented low-poly octahedron splat mesh per Gaussian",
        "unit": "meter",
        "metric_scale_preserved": True,
        "color_mode": "embedded vertex colors from 3DGS f_dc RGB; no external texture file",
        "input_points": int(len(vertices)),
        "kept_gaussians": int(len(xyz)),
        "removed_by_opacity": int(len(vertices) - len(xyz)),
        "opacity_min": args.opacity_min,
        "radius_multiplier": args.radius_multiplier,
        "min_radius_m": args.min_radius_m,
        "max_radius_m": args.max_radius_m,
        "final_vertices": int(len(mesh.vertices)),
        "final_triangles": int(len(mesh.faces)),
        "bounds_m": [[round(float(v), 6) for v in row] for row in mesh.bounds],
        "bbox_size_m": [round(float(v), 6) for v in mesh.extents],
        "bottom_flattening": {
            "up_axis": "Y",
            "floor_y_before_pivot_m": round(floor_y, 6),
            "bottom_percentile": args.bottom_percentile,
            "flatten_band_m": args.bottom_flatten_band_m,
            "flattened_vertices": int(np.count_nonzero(flat_mask)),
            "pivot_definition": "bottom footprint center at (0,0,0); object extends upward along +Y",
        },
        "file_size_bytes": int(glb_path.stat().st_size),
    }
    write_json(dest_dir / "debug" / "splatmesh_glb_report.json", report)
    write_json(dest_dir / "mobile_asset_manifest.json", {
        "schema_version": 1,
        "object_id": args.object_id,
        "source_output_dir": str(source_dir),
        "asset": args.glb_name,
        "asset_type": "glb_splatmesh_vertex_color",
        "unit": "meter",
        "pivot": "bottom_center_origin",
        "bottom_flattened": True,
        "metric_scale_1_to_1": True,
        "debug_report": "debug/splatmesh_glb_report.json",
    })
    print(json.dumps({"object_id": args.object_id, "glb": str(glb_path), "vertices": len(mesh.vertices), "triangles": len(mesh.faces), "size_bytes": glb_path.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
