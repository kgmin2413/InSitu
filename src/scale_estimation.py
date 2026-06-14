"""Metric scale handling and point-cloud export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from utils import write_json


_COLMAP_VERTEX_DTYPE = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("reprojection_error", "f4"),
]


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_scale_config(config: dict[str, Any]) -> dict[str, Any]:
    scale_config = config.get("scale_estimation", {})
    method = str(scale_config.get("method", "none_or_pending"))
    scale_factor = float(scale_config.get("scale_factor", 1.0))
    source = str(scale_config.get("source", "training_point_cloud"))
    marker_id_raw = scale_config.get("marker_id")
    marker_id = int(marker_id_raw) if marker_id_raw is not None else None
    marker_size_m_raw = scale_config.get("marker_size_m")
    marker_size_m = float(marker_size_m_raw) if marker_size_m_raw is not None else None
    min_marker_observations = int(scale_config.get("min_marker_observations", 2))
    foreground_filter_config = scale_config.get("foreground_filter", {})
    foreground_filter = {
        "enabled": bool(foreground_filter_config.get("enabled", False)),
        "min_observations": int(foreground_filter_config.get("min_observations", 1)),
        "alpha_threshold": int(foreground_filter_config.get("alpha_threshold", 16)),
    }
    denoise_config = scale_config.get("denoise", {})
    statistical_config = denoise_config.get("statistical", {})
    object_filter_config = scale_config.get("object_filter", {})
    percentile_config = object_filter_config.get("percentile", {})
    cluster_config = object_filter_config.get("cluster", {})
    opacity_threshold_raw = denoise_config.get("opacity_threshold")
    opacity_threshold = float(opacity_threshold_raw) if opacity_threshold_raw is not None else None
    denoise = {
        "enabled": bool(denoise_config.get("enabled", False)),
        "opacity_threshold": opacity_threshold,
        "statistical_enabled": bool(statistical_config.get("enabled", False)),
        "statistical_neighbors": int(statistical_config.get("neighbors", 12)),
        "statistical_std_ratio": float(statistical_config.get("std_ratio", 2.0)),
    }
    object_filter = {
        "enabled": bool(object_filter_config.get("enabled", False)),
        "percentile_enabled": bool(percentile_config.get("enabled", True)),
        "percentile_lower": float(percentile_config.get("lower", 1.0)),
        "percentile_upper": float(percentile_config.get("upper", 99.0)),
        "cluster_enabled": bool(cluster_config.get("enabled", True)),
        "cluster_voxel_size_m": float(cluster_config.get("voxel_size_m", 0.01)),
        "cluster_eps_m": float(cluster_config.get("eps_m", 0.02)),
        "cluster_min_samples": int(cluster_config.get("min_samples", 8)),
        "cluster_min_keep_ratio": float(cluster_config.get("min_keep_ratio", 0.1)),
    }
    max_reprojection_error_raw = scale_config.get("max_reprojection_error")
    max_reprojection_error = None
    if max_reprojection_error_raw is not None:
        max_reprojection_error = float(max_reprojection_error_raw)
        if max_reprojection_error < 0:
            raise ValueError("scale_estimation.max_reprojection_error must be zero or greater")
    if scale_factor <= 0:
        raise ValueError("scale_estimation.scale_factor must be greater than 0")
    if source not in {"training_point_cloud", "colmap_sparse"}:
        raise ValueError("scale_estimation.source must be training_point_cloud or colmap_sparse")
    if marker_size_m is not None and marker_size_m <= 0:
        raise ValueError("scale_estimation.marker_size_m must be greater than 0")
    if min_marker_observations < 2:
        raise ValueError("scale_estimation.min_marker_observations must be at least 2")
    if foreground_filter["min_observations"] < 1:
        raise ValueError("scale_estimation.foreground_filter.min_observations must be at least 1")
    if not 0 <= foreground_filter["alpha_threshold"] <= 255:
        raise ValueError("scale_estimation.foreground_filter.alpha_threshold must be between 0 and 255")
    if denoise["opacity_threshold"] is not None and not 0 <= denoise["opacity_threshold"] <= 1:
        raise ValueError("scale_estimation.denoise.opacity_threshold must be between 0 and 1")
    if denoise["statistical_neighbors"] < 2:
        raise ValueError("scale_estimation.denoise.statistical.neighbors must be at least 2")
    if denoise["statistical_std_ratio"] <= 0:
        raise ValueError("scale_estimation.denoise.statistical.std_ratio must be greater than 0")
    if not 0.0 <= object_filter["percentile_lower"] < object_filter["percentile_upper"] <= 100.0:
        raise ValueError("scale_estimation.object_filter.percentile lower/upper must satisfy 0 <= lower < upper <= 100")
    if object_filter["cluster_voxel_size_m"] <= 0:
        raise ValueError("scale_estimation.object_filter.cluster.voxel_size_m must be greater than 0")
    if object_filter["cluster_eps_m"] <= 0:
        raise ValueError("scale_estimation.object_filter.cluster.eps_m must be greater than 0")
    if object_filter["cluster_min_samples"] < 1:
        raise ValueError("scale_estimation.object_filter.cluster.min_samples must be at least 1")
    if not 0.0 <= object_filter["cluster_min_keep_ratio"] <= 1.0:
        raise ValueError("scale_estimation.object_filter.cluster.min_keep_ratio must be between 0 and 1")
    return {
        "method": method,
        "scale_factor": scale_factor,
        "source": source,
        "marker_id": marker_id,
        "marker_size_m": marker_size_m,
        "min_marker_observations": min_marker_observations,
        "foreground_filter": foreground_filter,
        "denoise": denoise,
        "object_filter": object_filter,
        "max_reprojection_error": max_reprojection_error,
    }


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_training_point_cloud(output_dirs: dict[str, Path]) -> Path:
    training_report_path = output_dirs["debug"] / "training_report.json"
    if not training_report_path.exists():
        raise RuntimeError("training_report.json is missing; run training before scale export")
    training_report = _read_json(training_report_path)
    if training_report.get("status") != "success" or not training_report.get("model_created"):
        raise RuntimeError("successful 3DGS training point cloud is required before scale export")
    point_cloud = training_report.get("point_cloud")
    if not point_cloud:
        raise RuntimeError("training report does not contain a point cloud path")
    point_cloud_path = _resolve_repo_path(point_cloud)
    if not point_cloud_path.exists():
        raise RuntimeError(f"training point cloud does not exist: {point_cloud_path}")
    return point_cloud_path


def _load_colmap_points_path(output_dirs: dict[str, Path]) -> Path:
    colmap_summary_path = output_dirs["debug"] / "colmap_summary.json"
    if not colmap_summary_path.exists():
        raise RuntimeError("colmap_summary.json is missing; run COLMAP before sparse point export")
    colmap_summary = _read_json(colmap_summary_path)
    if colmap_summary.get("status") != "success" or not colmap_summary.get("sparse_model_created"):
        raise RuntimeError("successful COLMAP sparse model is required before sparse point export")
    text_model_dir = colmap_summary.get("text_model_dir")
    if not text_model_dir:
        raise RuntimeError("COLMAP summary does not contain text_model_dir")
    points_path = _resolve_repo_path(text_model_dir) / "points3D.txt"
    if not points_path.exists():
        raise RuntimeError(f"COLMAP points3D.txt does not exist: {points_path}")
    return points_path


def _scale_vertices(vertices: np.ndarray, scale_factor: float) -> np.ndarray:
    scaled = vertices.copy()
    for axis in ("x", "y", "z"):
        if axis not in scaled.dtype.names:
            raise RuntimeError(f"PLY vertex data is missing required '{axis}' property")
        scaled[axis] = scaled[axis] * scale_factor
    # 3DGS stores anisotropic Gaussian radii in log-space as scale_0..2.
    # Metric scaling must shrink/grow both centers and splat radii.
    log_scale = float(np.log(scale_factor))
    for axis in ("scale_0", "scale_1", "scale_2"):
        if axis in (scaled.dtype.names or ()):
            scaled[axis] = scaled[axis] + log_scale
    return scaled


def _bbox_and_center(vertices: np.ndarray) -> tuple[list[float], list[float]]:
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(float)
    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    bbox = maximum - minimum
    center = (minimum + maximum) / 2.0
    return [round(float(v), 6) for v in bbox], [round(float(v), 6) for v in center]


def _write_scaled_point_cloud(source_path: Path, output_path: Path, scale_factor: float) -> dict[str, Any]:
    ply = PlyData.read(source_path)
    vertex_element = ply["vertex"]
    scaled_vertices = _scale_vertices(vertex_element.data, scale_factor)
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(scaled_vertices, "vertex"))
        else:
            elements.append(element)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(output_path)
    bbox_size_m, object_center_m = _bbox_and_center(scaled_vertices)
    return {
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "point_count": int(len(scaled_vertices)),
    }


def _parse_colmap_points(points_path: Path, scale_factor: float, max_reprojection_error: float | None) -> np.ndarray:
    records = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        x, y, z = (float(parts[1]) * scale_factor, float(parts[2]) * scale_factor, float(parts[3]) * scale_factor)
        r, g, b = (int(parts[4]), int(parts[5]), int(parts[6]))
        error = float(parts[7])
        if max_reprojection_error is not None and error > max_reprojection_error:
            continue
        records.append((x, y, z, r, g, b, error))
    if not records:
        raise RuntimeError("COLMAP points3D.txt did not contain any usable points after filtering")
    return np.array(records, dtype=_COLMAP_VERTEX_DTYPE)


def _write_colmap_sparse_point_cloud(
    points_path: Path,
    output_path: Path,
    scale_factor: float,
    max_reprojection_error: float | None,
) -> dict[str, Any]:
    vertices = _parse_colmap_points(points_path, scale_factor, max_reprojection_error)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(output_path)
    bbox_size_m, object_center_m = _bbox_and_center(vertices)
    return {
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "point_count": int(len(vertices)),
    }



def _vertices_to_xyz(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(float)


def _vertices_to_colors(vertices: np.ndarray) -> np.ndarray:
    names = vertices.dtype.names or ()
    if all(name in names for name in ("red", "green", "blue")):
        return np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.uint8)
    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_c0 = 0.28209479177387814
        rgb = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(float)
        rgb = np.clip((rgb * sh_c0 + 0.5) * 255.0, 0, 255)
        return rgb.astype(np.uint8)
    return np.tile(np.array([[40, 96, 180]], dtype=np.uint8), (len(vertices), 1))


def _project_points(points: np.ndarray, axes: tuple[int, int], size: tuple[int, int], margin: int) -> np.ndarray:
    width, height = size
    coords = points[:, axes]
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = maximum - minimum
    span[span == 0] = 1.0
    usable_w = max(1, width - margin * 2)
    usable_h = max(1, height - margin * 2)
    normalized = (coords - minimum) / span
    projected = np.empty_like(normalized)
    projected[:, 0] = margin + normalized[:, 0] * usable_w
    projected[:, 1] = margin + (1.0 - normalized[:, 1]) * usable_h
    return projected.astype(int)


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    label: str,
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
) -> None:
    x0, y0 = origin
    width, height = panel_size
    draw.rectangle([x0, y0, x0 + width - 1, y0 + height - 1], outline=(210, 210, 210), width=1)
    draw.text((x0 + 10, y0 + 8), label, fill=(30, 30, 30))
    projected = _project_points(points, axes, panel_size, margin=18)
    for (px, py), color in zip(projected, colors):
        draw.point((x0 + int(px), y0 + int(py)), fill=tuple(int(v) for v in color))


def _write_reconstruction_preview(point_cloud_path: Path, preview_path: Path) -> str:
    ply = PlyData.read(point_cloud_path)
    vertices = ply["vertex"].data
    if len(vertices) == 0:
        raise RuntimeError("Cannot render reconstruction preview for an empty point cloud")
    points = _vertices_to_xyz(vertices)
    colors = _vertices_to_colors(vertices)
    max_points = 20000
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
        colors = colors[indices]

    panel_w, panel_h = 360, 300
    padding = 24
    image = Image.new("RGB", (panel_w * 3 + padding * 4, panel_h + padding * 2), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    panels = [
        ((padding, padding), "XY", (0, 1)),
        ((panel_w + padding * 2, padding), "XZ", (0, 2)),
        ((panel_w * 2 + padding * 3, padding), "YZ", (1, 2)),
    ]
    for origin, label, axes in panels:
        _draw_projection(draw, origin, (panel_w, panel_h), label, points, colors, axes)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path)
    return _relative_repo_path(preview_path)


def _qvec_to_rotmat(qvec: list[float]) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )


def _parse_colmap_cameras(cameras_txt: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def _parse_colmap_images(images_txt: Path) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    lines = [line.strip() for line in images_txt.read_text(encoding="utf-8").splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qvec = [float(value) for value in parts[1:5]]
        tvec = np.array([float(value) for value in parts[5:8]], dtype=float)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        rotation = _qvec_to_rotmat(qvec)
        center = -rotation.T @ tvec
        record = {
            "camera_id": camera_id,
            "rotation_world_to_camera": rotation,
            "translation_world_to_camera": tvec,
            "camera_center_world": center,
            "name": name,
        }
        images[name] = record
        images[Path(name).name] = record
        if index < len(lines):
            index += 1
    return images


def _camera_matrix_and_distortion(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = str(camera["model"]).upper()
    params = camera["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        dist = np.zeros(5, dtype=float)
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float), dist
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        dist = np.zeros(5, dtype=float)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float), dist
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        dist = np.array([k1, 0, 0, 0, 0], dtype=float)
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float), dist
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        dist = np.array([k1, k2, 0, 0, 0], dtype=float)
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float), dist
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        dist = np.array([k1, k2, p1, p2, 0], dtype=float)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float), dist
    raise RuntimeError(f"Unsupported COLMAP camera model for marker scale: {model}")


def _pixel_to_world_ray(pixel_xy: list[float], camera: dict[str, Any], image: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix, distortion = _camera_matrix_and_distortion(camera)
    pixel = np.array(pixel_xy, dtype=float).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(pixel, matrix, distortion).reshape(2)
    ray_camera = np.array([normalized[0], normalized[1], 1.0], dtype=float)
    ray_camera /= np.linalg.norm(ray_camera)
    rotation = image["rotation_world_to_camera"]
    ray_world = rotation.T @ ray_camera
    ray_world /= np.linalg.norm(ray_world)
    center = image["camera_center_world"]
    return center, ray_world


def _triangulate_rays(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if len(rays) < 2:
        raise RuntimeError("at least two rays are required for triangulation")
    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    identity = np.eye(3, dtype=float)
    for center, direction in rays:
        direction = direction / np.linalg.norm(direction)
        projector = identity - np.outer(direction, direction)
        matrix += projector
        vector += projector @ center
    return np.linalg.solve(matrix, vector)


def _marker_observation_name(observation: dict[str, Any]) -> str:
    return Path(str(observation["image"])).name


def _select_marker_id(observations: list[dict[str, Any]], configured_marker_id: int | None) -> int:
    if configured_marker_id is not None:
        return configured_marker_id
    counts: dict[int, int] = {}
    for observation in observations:
        marker_id = int(observation["marker_id"])
        counts[marker_id] = counts.get(marker_id, 0) + 1
    if not counts:
        raise RuntimeError("marker report has no marker observations")
    return max(counts.items(), key=lambda item: item[1])[0]


def _estimate_aruco_marker_scale(
    marker_report_path: Path,
    text_model_dir: Path,
    marker_id: int | None,
    marker_size_m: float | None,
    min_observations: int,
) -> dict[str, Any]:
    if not marker_report_path.exists():
        raise RuntimeError("marker_detection_report.json is missing; enable marker_detection before aruco scale estimation")
    marker_report = _read_json(marker_report_path)
    if marker_report.get("status") != "success" or not marker_report.get("detections_found"):
        raise RuntimeError("successful ArUco marker detections are required for aruco scale estimation")

    observations = marker_report.get("observations", [])
    selected_marker_id = _select_marker_id(observations, marker_id)
    selected = [observation for observation in observations if int(observation.get("marker_id", -1)) == selected_marker_id]
    if len(selected) < min_observations:
        raise RuntimeError(
            f"at least {min_observations} observations of marker {selected_marker_id} are required; got {len(selected)}"
        )

    effective_marker_size_m = marker_size_m or marker_report.get("marker_size_m")
    if effective_marker_size_m is None or float(effective_marker_size_m) <= 0:
        raise RuntimeError("marker_size_m must be configured for aruco scale estimation")
    effective_marker_size_m = float(effective_marker_size_m)

    cameras = _parse_colmap_cameras(text_model_dir / "cameras.txt")
    images = _parse_colmap_images(text_model_dir / "images.txt")
    corner_rays: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {index: [] for index in range(4)}
    used_observations = 0
    skipped_images: list[str] = []
    for observation in selected:
        image_name = _marker_observation_name(observation)
        image = images.get(image_name)
        if image is None:
            skipped_images.append(image_name)
            continue
        camera = cameras.get(int(image["camera_id"]))
        if camera is None:
            skipped_images.append(image_name)
            continue
        corners = observation.get("corners_px", [])
        if len(corners) != 4:
            continue
        for corner_index, corner in enumerate(corners):
            corner_rays[corner_index].append(_pixel_to_world_ray(corner, camera, image))
        used_observations += 1

    triangulated = []
    rays_per_corner = []
    for corner_index in range(4):
        rays = corner_rays[corner_index]
        rays_per_corner.append(len(rays))
        if len(rays) < 2:
            raise RuntimeError(f"marker corner {corner_index} has fewer than two valid rays")
        triangulated.append(_triangulate_rays(rays))

    side_lengths = [
        float(np.linalg.norm(triangulated[(index + 1) % 4] - triangulated[index]))
        for index in range(4)
    ]
    reconstructed_marker_size = float(np.median(side_lengths))
    if reconstructed_marker_size <= 0:
        raise RuntimeError("reconstructed marker size must be greater than zero")
    scale_factor = effective_marker_size_m / reconstructed_marker_size
    return {
        "scale_factor": float(scale_factor),
        "scale_method": "aruco_marker",
        "marker_id": selected_marker_id,
        "marker_size_m": effective_marker_size_m,
        "observations_total": len(selected),
        "observations_used": used_observations,
        "skipped_images": sorted(set(skipped_images)),
        "rays_per_corner": rays_per_corner,
        "triangulated_corners_colmap": [[round(float(value), 6) for value in point] for point in triangulated],
        "side_lengths_colmap": [round(length, 6) for length in side_lengths],
        "reconstructed_marker_size_colmap": round(reconstructed_marker_size, 6),
    }


def _distort_normalized_point(x: float, y: float, camera: dict[str, Any]) -> tuple[float, float]:
    model = str(camera["model"]).upper()
    params = camera["params"]
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return x, y
    if model == "SIMPLE_RADIAL":
        k1 = params[3]
        r2 = x * x + y * y
        scale = 1.0 + k1 * r2
        return x * scale, y * scale
    if model == "RADIAL":
        k1, k2 = params[3], params[4]
        r2 = x * x + y * y
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        return x * scale, y * scale
    if model == "OPENCV":
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd
    raise RuntimeError(f"Unsupported COLMAP camera model for projection: {model}")


def _project_world_point(point: np.ndarray, camera: dict[str, Any], image: dict[str, Any]) -> tuple[float, float] | None:
    rotation = image["rotation_world_to_camera"]
    translation = image["translation_world_to_camera"]
    camera_point = rotation @ point + translation
    if camera_point[2] <= 1e-8:
        return None
    x = float(camera_point[0] / camera_point[2])
    y = float(camera_point[1] / camera_point[2])
    x, y = _distort_normalized_point(x, y, camera)
    matrix, _distortion = _camera_matrix_and_distortion(camera)
    u = float(matrix[0, 0] * x + matrix[0, 2])
    v = float(matrix[1, 1] * y + matrix[1, 2])
    return u, v


def _load_mask_alpha(mask_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _filter_vertices_by_foreground_masks(
    vertices: np.ndarray,
    output_dirs: dict[str, Path],
    text_model_dir: Path,
    filter_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(vertices) == 0:
        return vertices, {"enabled": True, "input_points": 0, "kept_points": 0, "removed_points": 0}
    cameras = _parse_colmap_cameras(text_model_dir / "cameras.txt")
    images = _parse_colmap_images(text_model_dir / "images.txt")
    mask_records = []
    for name, image in images.items():
        if name != Path(name).name:
            continue
        mask_path = output_dirs["images_masked"] / name
        alpha = _load_mask_alpha(mask_path)
        if alpha is None:
            continue
        camera = cameras.get(int(image["camera_id"]))
        if camera is None:
            continue
        mask_records.append((name, image, camera, alpha))
    if not mask_records:
        return vertices, {
            "enabled": True,
            "status": "skipped",
            "reason": "no registered foreground masks were available",
            "input_points": int(len(vertices)),
            "kept_points": int(len(vertices)),
            "removed_points": 0,
        }

    points = _vertices_to_xyz(vertices)
    keep = np.zeros(len(points), dtype=bool)
    min_observations = int(filter_config["min_observations"])
    alpha_threshold = int(filter_config["alpha_threshold"])
    visibility_counts = np.zeros(len(points), dtype=np.uint16)
    for _name, image, camera, alpha in mask_records:
        height, width = alpha.shape[:2]
        for index, point in enumerate(points):
            projection = _project_world_point(point, camera, image)
            if projection is None:
                continue
            u, v = projection
            px = int(round(u))
            py = int(round(v))
            if 0 <= px < width and 0 <= py < height and int(alpha[py, px]) >= alpha_threshold:
                visibility_counts[index] += 1
    keep = visibility_counts >= min_observations
    if not np.any(keep):
        return vertices, {
            "enabled": True,
            "status": "skipped",
            "reason": "foreground filter would remove all points",
            "input_points": int(len(vertices)),
            "kept_points": int(len(vertices)),
            "removed_points": 0,
            "registered_masks": len(mask_records),
        }
    filtered = vertices[keep].copy()
    return filtered, {
        "enabled": True,
        "status": "success",
        "input_points": int(len(vertices)),
        "kept_points": int(len(filtered)),
        "removed_points": int(len(vertices) - len(filtered)),
        "registered_masks": len(mask_records),
        "min_observations": min_observations,
        "alpha_threshold": alpha_threshold,
        "min_visibility_count": int(visibility_counts[keep].min()),
        "max_visibility_count": int(visibility_counts[keep].max()),
    }


def _opacity_to_probability(opacity: np.ndarray) -> np.ndarray:
    raw = opacity.astype(float)
    return 1.0 / (1.0 + np.exp(-raw))


def _denoise_vertices(vertices: np.ndarray, denoise_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not denoise_config.get("enabled"):
        return vertices, None
    report: dict[str, Any] = {
        "enabled": True,
        "input_points": int(len(vertices)),
        "opacity_threshold": denoise_config.get("opacity_threshold"),
        "statistical_enabled": bool(denoise_config.get("statistical_enabled", False)),
        "statistical_neighbors": int(denoise_config.get("statistical_neighbors", 12)),
        "statistical_std_ratio": float(denoise_config.get("statistical_std_ratio", 2.0)),
    }
    if len(vertices) == 0:
        report.update({"status": "skipped", "reason": "empty point cloud", "kept_points": 0, "removed_points": 0})
        return vertices, report

    keep = np.ones(len(vertices), dtype=bool)
    opacity_threshold = denoise_config.get("opacity_threshold")
    if opacity_threshold is not None:
        if "opacity" in (vertices.dtype.names or ()):
            opacity_probability = _opacity_to_probability(vertices["opacity"])
            keep &= opacity_probability >= float(opacity_threshold)
            report["after_opacity_points"] = int(keep.sum())
        else:
            report["opacity_status"] = "skipped_missing_opacity"
            report["after_opacity_points"] = int(keep.sum())
    else:
        report["after_opacity_points"] = int(keep.sum())

    if denoise_config.get("statistical_enabled"):
        candidate_count = int(keep.sum())
        neighbors = min(int(denoise_config["statistical_neighbors"]), candidate_count - 1)
        if candidate_count <= 2 or neighbors < 2:
            report["statistical_status"] = "skipped_too_few_points"
            report["after_statistical_points"] = candidate_count
        else:
            xyz = _vertices_to_xyz(vertices[keep])
            nearest = NearestNeighbors(n_neighbors=neighbors + 1)
            nearest.fit(xyz)
            distances, _indices = nearest.kneighbors(xyz)
            mean_distances = distances[:, 1:].mean(axis=1)
            cutoff = float(mean_distances.mean() + float(denoise_config["statistical_std_ratio"]) * mean_distances.std())
            local_keep = mean_distances <= cutoff
            kept_indices = np.flatnonzero(keep)
            keep[kept_indices[~local_keep]] = False
            report["statistical_status"] = "success"
            report["statistical_distance_cutoff"] = cutoff
            report["after_statistical_points"] = int(keep.sum())
    else:
        report["after_statistical_points"] = int(keep.sum())

    if not np.any(keep):
        report.update({
            "status": "skipped",
            "reason": "denoise would remove all points",
            "kept_points": int(len(vertices)),
            "removed_points": 0,
        })
        return vertices, report

    filtered = vertices[keep].copy()
    report.update({
        "status": "success",
        "kept_points": int(len(filtered)),
        "removed_points": int(len(vertices) - len(filtered)),
    })
    return filtered, report



def _filter_vertices_to_main_object(vertices: np.ndarray, filter_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not filter_config.get("enabled"):
        return vertices, None
    report: dict[str, Any] = {
        "enabled": True,
        "input_points": int(len(vertices)),
        "percentile_enabled": bool(filter_config["percentile_enabled"]),
        "cluster_enabled": bool(filter_config["cluster_enabled"]),
    }
    if len(vertices) < 3:
        report.update({"status": "skipped", "reason": "too few points", "kept_points": int(len(vertices)), "removed_points": 0})
        return vertices, report

    current = vertices
    xyz = _vertices_to_xyz(current)
    if filter_config["percentile_enabled"]:
        lower_bounds = np.percentile(xyz, filter_config["percentile_lower"], axis=0)
        upper_bounds = np.percentile(xyz, filter_config["percentile_upper"], axis=0)
        keep = np.all((xyz >= lower_bounds) & (xyz <= upper_bounds), axis=1)
        if np.count_nonzero(keep) >= 3:
            current = current[keep].copy()
            xyz = _vertices_to_xyz(current)
            report["percentile"] = {
                "status": "success",
                "lower": filter_config["percentile_lower"],
                "upper": filter_config["percentile_upper"],
                "input_points": int(len(vertices)),
                "kept_points": int(len(current)),
                "removed_points": int(len(vertices) - len(current)),
                "lower_bounds": [round(float(v), 6) for v in lower_bounds],
                "upper_bounds": [round(float(v), 6) for v in upper_bounds],
            }
        else:
            report["percentile"] = {"status": "skipped_too_few_points"}

    if filter_config["cluster_enabled"]:
        before_cluster = len(current)
        voxel_size = float(filter_config["cluster_voxel_size_m"])
        minimum = xyz.min(axis=0)
        voxel_indices = np.floor((xyz - minimum) / voxel_size).astype(np.int64)
        unique_voxels, first_indices, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        representative_points = xyz[first_indices]
        labels = DBSCAN(
            eps=float(filter_config["cluster_eps_m"]),
            min_samples=int(filter_config["cluster_min_samples"]),
        ).fit_predict(representative_points)
        cluster_labels = [label for label in sorted(set(labels)) if label != -1]
        if cluster_labels:
            counts = {int(label): int(np.count_nonzero(labels == label)) for label in cluster_labels}
            main_label = max(counts, key=counts.get)
            keep = labels[inverse] == main_label
            kept_count = int(np.count_nonzero(keep))
            min_keep = max(3, int(before_cluster * float(filter_config["cluster_min_keep_ratio"])))
            if kept_count >= min_keep:
                current = current[keep].copy()
                report["cluster"] = {
                    "status": "success",
                    "input_points": int(before_cluster),
                    "kept_points": int(len(current)),
                    "removed_points": int(before_cluster - len(current)),
                    "voxel_size_m": voxel_size,
                    "eps_m": float(filter_config["cluster_eps_m"]),
                    "min_samples": int(filter_config["cluster_min_samples"]),
                    "representative_points": int(len(representative_points)),
                    "selected_label": int(main_label),
                    "cluster_counts": counts,
                    "noise_voxels": int(np.count_nonzero(labels == -1)),
                }
            else:
                report["cluster"] = {
                    "status": "skipped_min_keep_ratio",
                    "input_points": int(before_cluster),
                    "candidate_points": kept_count,
                    "min_keep_points": int(min_keep),
                    "cluster_counts": counts,
                    "noise_voxels": int(np.count_nonzero(labels == -1)),
                }
        else:
            report["cluster"] = {
                "status": "skipped_no_clusters",
                "input_points": int(before_cluster),
                "representative_points": int(len(representative_points)),
                "noise_voxels": int(np.count_nonzero(labels == -1)),
            }

    report.update({
        "status": "success" if len(current) < int(report["input_points"]) else "skipped_no_change",
        "kept_points": int(len(current)),
        "removed_points": int(int(report["input_points"]) - len(current)),
    })
    return current, report

def _write_rgb_inspection_point_cloud(vertices: np.ndarray, output_path: Path) -> str:
    colors = _vertices_to_colors(vertices)
    rgb_vertices = np.empty(
        len(vertices),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rgb_vertices["x"] = vertices["x"]
    rgb_vertices["y"] = vertices["y"]
    rgb_vertices["z"] = vertices["z"]
    rgb_vertices["red"] = colors[:, 0]
    rgb_vertices["green"] = colors[:, 1]
    rgb_vertices["blue"] = colors[:, 2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(rgb_vertices, "vertex")], text=True).write(output_path)
    return _relative_repo_path(output_path)


def _write_scaled_training_point_cloud(
    source_path: Path,
    output_path: Path,
    scale_factor: float,
    output_dirs: dict[str, Path],
    text_model_dir: Path | None,
    foreground_filter: dict[str, Any],
    denoise: dict[str, Any],
    object_filter: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    ply = PlyData.read(source_path)
    vertex_element = ply["vertex"]
    vertices = vertex_element.data
    foreground_report = None
    if foreground_filter.get("enabled"):
        if text_model_dir is None:
            raise RuntimeError("COLMAP text_model_dir is required for foreground point filtering")
        vertices, foreground_report = _filter_vertices_by_foreground_masks(vertices, output_dirs, text_model_dir, foreground_filter)
    vertices, denoise_report = _denoise_vertices(vertices, denoise)
    scaled_vertices = _scale_vertices(vertices, scale_factor)
    scaled_vertices, object_filter_report = _filter_vertices_to_main_object(scaled_vertices, object_filter)
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(scaled_vertices, "vertex"))
        else:
            elements.append(element)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(output_path)
    rgb_path = output_path.with_name("point_cloud_metric_rgb.ply")
    rgb_point_cloud = _write_rgb_inspection_point_cloud(scaled_vertices, rgb_path)
    bbox_size_m, object_center_m = _bbox_and_center(scaled_vertices)
    return {
        "bbox_size_m": bbox_size_m,
        "object_center_m": object_center_m,
        "point_count": int(len(scaled_vertices)),
        "point_cloud_metric_rgb": rgb_point_cloud,
    }, foreground_report, denoise_report, object_filter_report


def _update_metadata(output_dirs: dict[str, Path], updates: dict[str, Any]) -> None:
    metadata_path = output_dirs["object"] / "metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(updates)
    write_json(metadata_path, metadata)


def run_scale_estimation(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    scale_config = _read_scale_config(config)
    method = scale_config["method"]
    scale_factor = scale_config["scale_factor"]
    source = scale_config["source"]
    max_reprojection_error = scale_config["max_reprojection_error"]
    marker_scale_evidence = None
    foreground_filter_report = None
    denoise_report = None
    object_filter_report = None
    text_model_dir_for_scale = None

    if method in {"none", "none_or_pending", "pending"}:
        report = {
            "status": "skipped",
            "reason": "scale_estimation.method is none_or_pending",
            "scale_estimation_ran": False,
            "configured_method": method,
            "configured_source": source,
            "scale_factor": 1.0,
            "scale_method": "none_or_pending",
            "point_cloud_metric_created": False,
            "point_cloud_metric": None,
            "bbox_size_m": None,
            "object_center_m": None,
        }
        write_json(output_dirs["debug"] / "scale_report.json", report)
        return {
            "metrics": {
                "scale_factor": 1.0,
                "scale_method": "none_or_pending",
            },
            "warnings": ["metric scale estimation/export is disabled in config"],
        }

    if method in {"aruco", "aruco_marker", "marker"}:
        colmap_summary = _read_json(output_dirs["debug"] / "colmap_summary.json")
        text_model_dir_value = colmap_summary.get("text_model_dir")
        if not text_model_dir_value:
            raise RuntimeError("COLMAP text_model_dir is required for aruco marker scale estimation")
        text_model_dir_for_scale = _resolve_repo_path(text_model_dir_value)
        marker_scale_evidence = _estimate_aruco_marker_scale(
            output_dirs["debug"] / "marker_detection_report.json",
            text_model_dir_for_scale,
            scale_config["marker_id"],
            scale_config["marker_size_m"],
            scale_config["min_marker_observations"],
        )
        scale_factor = marker_scale_evidence["scale_factor"]
        method = "aruco_marker"
    elif method not in {"manual", "manual_scale"}:
        raise ValueError("scale_estimation.method must be none_or_pending, manual, manual_scale, or aruco_marker")

    output_point_cloud = output_dirs["object"] / "point_cloud_metric.ply"
    if source == "training_point_cloud":
        source_path = _load_training_point_cloud(output_dirs)
        if text_model_dir_for_scale is None:
            colmap_summary_path = output_dirs["debug"] / "colmap_summary.json"
            if colmap_summary_path.exists():
                text_model_dir_value = _read_json(colmap_summary_path).get("text_model_dir")
                text_model_dir_for_scale = _resolve_repo_path(text_model_dir_value) if text_model_dir_value else None
        exported, foreground_filter_report, denoise_report, object_filter_report = _write_scaled_training_point_cloud(
            source_path,
            output_point_cloud,
            scale_factor,
            output_dirs,
            text_model_dir_for_scale,
            scale_config["foreground_filter"],
            scale_config["denoise"],
            scale_config["object_filter"],
        )
        source_kind = "training_point_cloud"
    else:
        source_path = _load_colmap_points_path(output_dirs)
        exported = _write_colmap_sparse_point_cloud(
            source_path,
            output_point_cloud,
            scale_factor,
            max_reprojection_error,
        )
        source_kind = "colmap_sparse"

    preview_path = output_dirs["preview"] / "reconstruction_preview.png"
    reconstruction_preview = _write_reconstruction_preview(output_point_cloud, preview_path)

    metrics = {
        "scale_factor": scale_factor,
        "scale_method": method,
    }
    metadata_updates = {
        "visual_model": "point_cloud_metric.ply",
        "scale_factor": scale_factor,
        "scale_method": method,
        "bbox_size_m": exported["bbox_size_m"],
        "object_center_m": exported["object_center_m"],
    }
    _update_metadata(output_dirs, metadata_updates)

    report = {
        "status": "success",
        "scale_estimation_ran": True,
        "configured_method": method,
        "configured_source": source,
        "source_kind": source_kind,
        **metrics,
        "source_point_cloud": _relative_repo_path(source_path),
        "max_reprojection_error": max_reprojection_error,
        "marker_scale_evidence": marker_scale_evidence,
        "foreground_filter": foreground_filter_report,
        "denoise": denoise_report,
        "object_filter": object_filter_report,
        "point_cloud_metric_created": True,
        "point_cloud_metric": _relative_repo_path(output_point_cloud),
        "reconstruction_preview": reconstruction_preview,
        **exported,
    }
    write_json(output_dirs["debug"] / "scale_report.json", report)
    return {"metrics": metrics, "warnings": []}
