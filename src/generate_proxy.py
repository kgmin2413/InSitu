"""Proxy collider generation from metric point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from utils import write_json


_BOX_FACES = (
    (1, 2, 4, 3),
    (5, 7, 8, 6),
    (1, 5, 6, 2),
    (3, 4, 8, 7),
    (1, 3, 7, 5),
    (2, 6, 8, 4),
)
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_proxy_config(config: dict[str, Any]) -> dict[str, Any]:
    proxy_config = config.get("proxy_generation", {})
    method = str(proxy_config.get("method", "none_or_pending"))
    if method not in {"none", "none_or_pending", "pending", "obb", "pca_obb"}:
        raise ValueError("proxy_generation.method must be none_or_pending, obb, or pca_obb")

    voxel_size_m = float(proxy_config.get("voxel_size_m", 0.0))
    if voxel_size_m < 0:
        raise ValueError("proxy_generation.voxel_size_m must be zero or greater")

    percentile_config = proxy_config.get("percentile_filter", {})
    lower = float(percentile_config.get("lower", 0.0))
    upper = float(percentile_config.get("upper", 100.0))
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("proxy_generation.percentile_filter lower/upper must satisfy 0 <= lower < upper <= 100")

    statistical_config = proxy_config.get("statistical_filter", {})
    statistical_neighbors = int(statistical_config.get("neighbors", 12))
    statistical_std_ratio = float(statistical_config.get("std_ratio", 2.0))
    if statistical_neighbors < 2:
        raise ValueError("proxy_generation.statistical_filter.neighbors must be at least 2")
    if statistical_std_ratio <= 0:
        raise ValueError("proxy_generation.statistical_filter.std_ratio must be greater than 0")

    cluster_config = proxy_config.get("cluster_filter", {})
    eps = float(cluster_config.get("eps", 0.08))
    min_samples = int(cluster_config.get("min_samples", 8))
    if eps <= 0:
        raise ValueError("proxy_generation.cluster_filter.eps must be greater than 0")
    if min_samples < 1:
        raise ValueError("proxy_generation.cluster_filter.min_samples must be at least 1")

    flat_bottom_config = proxy_config.get("flat_bottom", {})
    up_axis = str(flat_bottom_config.get("up_axis", "Y")).upper()
    if up_axis not in _AXIS_INDEX:
        raise ValueError("proxy_generation.flat_bottom.up_axis must be X, Y, or Z")

    return {
        "method": method,
        "voxel_size_m": voxel_size_m,
        "percentile_filter": {
            "enabled": bool(percentile_config.get("enabled", False)),
            "lower": lower,
            "upper": upper,
        },
        "statistical_filter": {
            "enabled": bool(statistical_config.get("enabled", False)),
            "neighbors": statistical_neighbors,
            "std_ratio": statistical_std_ratio,
        },
        "cluster_filter": {
            "enabled": bool(cluster_config.get("enabled", False)),
            "eps": eps,
            "min_samples": min_samples,
        },
        "flat_bottom": {
            "enabled": bool(flat_bottom_config.get("enabled", False)),
            "up_axis": up_axis,
        },
    }


def _load_xyz(point_cloud_path: Path) -> np.ndarray:
    if not point_cloud_path.exists():
        raise RuntimeError(f"metric point cloud does not exist: {point_cloud_path}")
    ply = PlyData.read(point_cloud_path)
    vertices = ply["vertex"].data
    for axis in ("x", "y", "z"):
        if axis not in vertices.dtype.names:
            raise RuntimeError(f"PLY vertex data is missing required '{axis}' property")
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(float)
    if len(xyz) < 3:
        raise RuntimeError("at least 3 points are required to generate an OBB proxy")
    return xyz


def _filter_report(name: str, before: int, after: int, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "input_points": int(before),
        "kept_points": int(after),
        "removed_points": int(before - after),
        **extra,
    }


def _safe_filtered(original: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, str]:
    if len(candidate) < 3:
        return original, "skipped_too_few_points"
    return candidate, "success"


def _apply_percentile_filter(xyz: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    before = len(xyz)
    if not config["enabled"]:
        return xyz, _filter_report("percentile_filter", before, before, "disabled")
    lower_bounds = np.percentile(xyz, config["lower"], axis=0)
    upper_bounds = np.percentile(xyz, config["upper"], axis=0)
    keep = np.all((xyz >= lower_bounds) & (xyz <= upper_bounds), axis=1)
    filtered, status = _safe_filtered(xyz, xyz[keep])
    return filtered, _filter_report(
        "percentile_filter",
        before,
        len(filtered),
        status,
        lower=config["lower"],
        upper=config["upper"],
        lower_bounds=[round(float(v), 6) for v in lower_bounds],
        upper_bounds=[round(float(v), 6) for v in upper_bounds],
    )


def _apply_voxel_downsample(xyz: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    before = len(xyz)
    if voxel_size_m <= 0:
        return xyz, _filter_report("voxel_downsample", before, before, "disabled")
    minimum = xyz.min(axis=0)
    voxel_indices = np.floor((xyz - minimum) / voxel_size_m).astype(np.int64)
    _unique, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered, status = _safe_filtered(xyz, xyz[np.sort(first_indices)])
    return filtered, _filter_report(
        "voxel_downsample",
        before,
        len(filtered),
        status,
        voxel_size_m=voxel_size_m,
    )


def _apply_statistical_filter(xyz: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    before = len(xyz)
    if not config["enabled"]:
        return xyz, _filter_report("statistical_filter", before, before, "disabled")
    neighbors = min(config["neighbors"], before - 1)
    if before <= 3 or neighbors < 2:
        return xyz, _filter_report("statistical_filter", before, before, "skipped_too_few_points")
    nearest = NearestNeighbors(n_neighbors=neighbors + 1)
    nearest.fit(xyz)
    distances, _indices = nearest.kneighbors(xyz)
    mean_distances = distances[:, 1:].mean(axis=1)
    cutoff = float(mean_distances.mean() + config["std_ratio"] * mean_distances.std())
    filtered, status = _safe_filtered(xyz, xyz[mean_distances <= cutoff])
    return filtered, _filter_report(
        "statistical_filter",
        before,
        len(filtered),
        status,
        neighbors=neighbors,
        std_ratio=config["std_ratio"],
        distance_cutoff=cutoff,
    )


def _apply_cluster_filter(xyz: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    before = len(xyz)
    if not config["enabled"]:
        return xyz, _filter_report("cluster_filter", before, before, "disabled")
    if before < config["min_samples"]:
        return xyz, _filter_report("cluster_filter", before, before, "skipped_too_few_points")
    labels = DBSCAN(eps=config["eps"], min_samples=config["min_samples"]).fit_predict(xyz)
    cluster_labels = [label for label in sorted(set(labels)) if label != -1]
    if not cluster_labels:
        return xyz, _filter_report("cluster_filter", before, before, "skipped_no_clusters")
    counts = {int(label): int(np.count_nonzero(labels == label)) for label in cluster_labels}
    main_label = max(counts, key=counts.get)
    filtered, status = _safe_filtered(xyz, xyz[labels == main_label])
    return filtered, _filter_report(
        "cluster_filter",
        before,
        len(filtered),
        status,
        eps=config["eps"],
        min_samples=config["min_samples"],
        selected_label=main_label,
        cluster_counts=counts,
        noise_points=int(np.count_nonzero(labels == -1)),
    )


def _prepare_proxy_points(xyz: np.ndarray, proxy_config: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reports = []
    current, report = _apply_percentile_filter(xyz, proxy_config["percentile_filter"])
    reports.append(report)
    current, report = _apply_voxel_downsample(current, proxy_config["voxel_size_m"])
    reports.append(report)
    current, report = _apply_statistical_filter(current, proxy_config["statistical_filter"])
    reports.append(report)
    current, report = _apply_cluster_filter(current, proxy_config["cluster_filter"])
    reports.append(report)
    return current, reports


def _obb_from_points(xyz: np.ndarray) -> dict[str, Any]:
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local = centered @ axes
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    half_extents = (local_max - local_min) / 2.0
    local_center = (local_min + local_max) / 2.0
    world_center = centroid + local_center @ axes.T

    local_corners = np.array(
        [
            [local_min[0], local_min[1], local_min[2]],
            [local_max[0], local_min[1], local_min[2]],
            [local_min[0], local_max[1], local_min[2]],
            [local_max[0], local_max[1], local_min[2]],
            [local_min[0], local_min[1], local_max[2]],
            [local_max[0], local_min[1], local_max[2]],
            [local_min[0], local_max[1], local_max[2]],
            [local_max[0], local_max[1], local_max[2]],
        ]
    )
    corners = world_center + (local_corners - local_center) @ axes.T
    return {
        "center": world_center,
        "half_extents": half_extents,
        "size": half_extents * 2.0,
        "axes": axes,
        "corners": corners,
    }


def _enforce_flat_bottom(corners: np.ndarray, flat_bottom_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not flat_bottom_config["enabled"]:
        return corners, None
    adjusted = corners.copy()
    axis = _AXIS_INDEX[flat_bottom_config["up_axis"]]
    order = np.argsort(adjusted[:, axis])
    bottom_indices = order[:4]
    bottom_value = float(adjusted[bottom_indices, axis].min())
    adjusted[bottom_indices, axis] = bottom_value
    return adjusted, {
        "enabled": True,
        "up_axis": flat_bottom_config["up_axis"],
        "bottom_value": round(bottom_value, 6),
        "adjusted_vertices": [int(index) + 1 for index in bottom_indices],
    }


def _write_obj(path: Path, corners: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# InSitu OBB proxy collider"]
    for corner in corners:
        lines.append(f"v {corner[0]:.8f} {corner[1]:.8f} {corner[2]:.8f}")
    for face in _BOX_FACES:
        lines.append("f " + " ".join(str(index) for index in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def _update_metadata(output_dirs: dict[str, Path], proxy_center_m: list[float]) -> None:
    import json

    metadata_path = output_dirs["object"] / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata.update(
        {
            "proxy_model": "proxy_collider.obj",
            "proxy_center_m": proxy_center_m,
        }
    )
    write_json(metadata_path, metadata)


def run_proxy_generation(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    proxy_config = _read_proxy_config(config)
    method = proxy_config["method"]

    if method in {"none", "none_or_pending", "pending"}:
        report = {
            "status": "skipped",
            "reason": "proxy_generation.method is none_or_pending",
            "proxy_generation_ran": False,
            "configured_method": method,
            "proxy_model_created": False,
            "proxy_model": None,
            "proxy_center_m": None,
        }
        write_json(output_dirs["debug"] / "proxy_report.json", report)
        return {
            "metrics": {},
            "warnings": ["proxy collider generation is disabled in config"],
        }

    point_cloud_path = output_dirs["object"] / "point_cloud_metric.ply"
    xyz = _load_xyz(point_cloud_path)
    proxy_xyz, preprocessing = _prepare_proxy_points(xyz, proxy_config)
    obb = _obb_from_points(proxy_xyz)
    corners, flat_bottom_report = _enforce_flat_bottom(obb["corners"], proxy_config["flat_bottom"])
    proxy_path = output_dirs["object"] / "proxy_collider.obj"
    _write_obj(proxy_path, corners)
    proxy_center = _round_list(obb["center"])
    _update_metadata(output_dirs, proxy_center)

    report = {
        "status": "success",
        "proxy_generation_ran": True,
        "configured_method": method,
        "source_point_cloud": _relative_repo_path(point_cloud_path),
        "proxy_model_created": True,
        "proxy_model": _relative_repo_path(proxy_path),
        "proxy_center_m": proxy_center,
        "proxy_size_m": _round_list(obb["size"]),
        "point_count": int(len(proxy_xyz)),
        "input_point_count": int(len(xyz)),
        "preprocessing": preprocessing,
        "flat_bottom": flat_bottom_report,
    }
    write_json(output_dirs["debug"] / "proxy_report.json", report)
    return {"metrics": {"proxy_model_created": True}, "warnings": []}
