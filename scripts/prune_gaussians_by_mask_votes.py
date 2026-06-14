#!/usr/bin/env python3
"""Prune 3DGS Gaussians by projecting centers into undistorted foreground masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def qvec_to_rotmat(q: list[float]) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def read_cameras(cameras_txt: Path) -> dict[int, tuple[int, int, float, float, float, float]]:
    cameras = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise RuntimeError(f"unsupported camera model for mask votes: {model}")
        cameras[camera_id] = (width, height, fx, fy, cx, cy)
    return cameras


def read_images(images_txt: Path, cameras: dict[int, tuple[int, int, float, float, float, float]]):
    records = []
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        idx += 1
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        qvec = [float(v) for v in parts[1:5]]
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        image_name = Path(" ".join(parts[9:])).name
        records.append((image_name, qvec_to_rotmat(qvec), tvec, cameras[camera_id]))
        idx += 1
    return records


def compute_votes(vertices: np.ndarray, colmap_text_dir: Path, mask_dir: Path, mask_threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cameras = read_cameras(colmap_text_dir / "cameras.txt")
    images = read_images(colmap_text_dir / "images.txt", cameras)
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    total = np.zeros(len(xyz), dtype=np.uint16)
    foreground = np.zeros(len(xyz), dtype=np.uint16)
    for image_name, rotation, translation, camera in images:
        mask_path = mask_dir / image_name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        width, height, fx, fy, cx, cy = camera
        points = xyz @ rotation.T + translation[None, :]
        depth = points[:, 2]
        valid = depth > 1e-6
        u = np.zeros(len(points), dtype=np.int32)
        v = np.zeros(len(points), dtype=np.int32)
        if np.any(valid):
            u[valid] = np.rint(fx * (points[valid, 0] / depth[valid]) + cx).astype(np.int32)
            v[valid] = np.rint(fy * (points[valid, 1] / depth[valid]) + cy).astype(np.int32)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(valid)
        if len(indices):
            total[indices] += 1
            foreground[indices] += (mask[v[indices], u[indices]] >= mask_threshold).astype(np.uint16)
    ratio = np.divide(foreground, total, out=np.zeros(len(foreground), dtype=np.float32), where=total > 0)
    return total, foreground, ratio


def write_filtered_ply(source_ply: Path, output_ply: Path, keep: np.ndarray) -> int:
    ply = PlyData.read(source_ply)
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(element.data[keep], "vertex"))
        else:
            elements.append(element)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(output_ply)
    return int(np.count_nonzero(keep))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--colmap-text-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--min-fg-observations", type=int, default=5)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.3)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--votes-npz", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ply = PlyData.read(args.input_ply)
    vertices = ply["vertex"].data
    total, foreground, ratio = compute_votes(vertices, args.colmap_text_dir, args.mask_dir, args.mask_threshold)
    keep = (foreground >= args.min_fg_observations) & (ratio >= args.min_foreground_ratio)
    kept = write_filtered_ply(args.input_ply, args.output_ply, keep)
    if args.votes_npz:
        args.votes_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.votes_npz, total=total, foreground=foreground, ratio=ratio, keep=keep)
    report = {
        "status": "success",
        "input_ply": str(args.input_ply),
        "output_ply": str(args.output_ply),
        "input_points": int(len(vertices)),
        "kept_points": kept,
        "removed_points": int(len(vertices) - kept),
        "min_fg_observations": args.min_fg_observations,
        "min_foreground_ratio": args.min_foreground_ratio,
        "visible_observations": {
            "min": int(total.min()) if len(total) else 0,
            "median": float(np.median(total)) if len(total) else 0.0,
            "max": int(total.max()) if len(total) else 0,
        },
        "ratio_percentiles": {str(p): float(np.percentile(ratio, p)) for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]},
    }
    report_path = args.report or args.output_ply.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
