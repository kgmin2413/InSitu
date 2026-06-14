#!/usr/bin/env python3
"""Prepare an A4 dataset that keeps full COLMAP poses but applies object masks for 3DGS."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
A4_ROOT = ROOT / "third_party/InSitu-A4"
SRC = A4_ROOT / "data/chair_insitu_a4true_20260611"
DST = A4_ROOT / "data/chair_insitu_a4pose_fgmask_20260611"
MASK_UNDIST = DST / "mask_undistorted"


def copy_dataset() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ["input", "images", "masks", "sparse", "sparse_txt", "distorted"]:
        src = SRC / name
        dst = DST / name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def make_mask_input() -> None:
    mask_input = DST / "mask_input"
    mask_input.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted((SRC / "input").glob("*.jpg")):
        mask_path = SRC / "masks" / f"{image_path.stem}.png"
        mask = Image.open(mask_path).convert("L")
        # COLMAP looks up files by the image names stored in the model, so keep .jpg names.
        rgb = Image.merge("RGB", (mask, mask, mask))
        rgb.save(mask_input / image_path.name, quality=100, subsampling=0)
        count += 1
    print(json.dumps({"mask_input": str(mask_input), "count": count}))


def _read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _parse_images_observations(images_txt: Path, mask_dir: Path, threshold: int = 128) -> dict[int, tuple[int, int]]:
    counts: dict[int, list[int]] = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_name = Path(" ".join(parts[9:])).name
        if idx >= len(lines):
            break
        points_line = lines[idx].strip()
        idx += 1
        mask_path = mask_dir / image_name
        if not mask_path.exists():
            continue
        mask = _read_mask(mask_path)
        height, width = mask.shape[:2]
        values = points_line.split()
        for p in range(0, len(values) - 2, 3):
            point_id = int(float(values[p + 2]))
            if point_id < 0:
                continue
            x = int(round(float(values[p])))
            y = int(round(float(values[p + 1])))
            fg = 0 <= x < width and 0 <= y < height and int(mask[y, x]) >= threshold
            rec = counts.setdefault(point_id, [0, 0])
            rec[0] += 1
            rec[1] += int(fg)
    return {pid: (total, fg) for pid, (total, fg) in counts.items()}


def _parse_points(points_txt: Path) -> dict[int, tuple[float, float, float, int, int, int]]:
    points = {}
    for line in points_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        pid = int(parts[0])
        points[pid] = (
            float(parts[1]), float(parts[2]), float(parts[3]),
            int(parts[4]), int(parts[5]), int(parts[6]),
        )
    return points


def make_foreground_seed(min_fg_obs: int = 2, min_fg_ratio: float = 0.6) -> None:
    images_txt = DST / "sparse_txt/images.txt"
    points_txt = DST / "sparse_txt/points3D.txt"
    mask_dir = MASK_UNDIST / "images"
    obs = _parse_images_observations(images_txt, mask_dir)
    points = _parse_points(points_txt)
    kept = []
    for pid, record in points.items():
        total, fg = obs.get(pid, (0, 0))
        ratio = fg / total if total else 0.0
        if fg >= min_fg_obs and ratio >= min_fg_ratio:
            kept.append(record)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    verts = np.empty(len(kept), dtype=dtype)
    for i, (x, y, z, r, g, b) in enumerate(kept):
        verts[i] = (x, y, z, 0.0, 0.0, 0.0, r, g, b)
    out_ply = DST / "sparse/0/points3D.ply"
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_ply)
    report = {
        "status": "success",
        "source_points": len(points),
        "observed_points": len(obs),
        "kept_points": len(kept),
        "removed_points": len(points) - len(kept),
        "min_fg_obs": min_fg_obs,
        "min_fg_ratio": min_fg_ratio,
        "seed_ply": str(out_ply),
    }
    (DST / "foreground_seed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def make_rgba_training_images(alpha_threshold: int = 8) -> None:
    count = 0
    for rgb_path in sorted((DST / "images").glob("*.jpg")):
        alpha_path = MASK_UNDIST / "images" / rgb_path.name
        rgb = Image.open(rgb_path).convert("RGB")
        alpha = Image.open(alpha_path).convert("L").resize(rgb.size, Image.Resampling.BILINEAR)
        alpha_arr = np.array(alpha)
        alpha_arr = np.where(alpha_arr >= alpha_threshold, alpha_arr, 0).astype(np.uint8)
        rgba = Image.merge("RGBA", (*rgb.split(), Image.fromarray(alpha_arr, "L")))
        rgba.save(rgb_path, format="PNG")
        count += 1
    print(json.dumps({"rgba_training_images": count}))


def main() -> int:
    copy_dataset()
    make_mask_input()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
