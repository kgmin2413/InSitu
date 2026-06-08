"""Optional ArUco marker detection for metric-scale capture support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from utils import write_json


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _get_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available")
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _detect_markers(gray: np.ndarray, dictionary_name: str):
    dictionary = _get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray)
    parameters = cv2.aruco.DetectorParameters_create()
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def _iter_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return [candidate for candidate in sorted(path.iterdir()) if candidate.suffix.lower() in _IMAGE_EXTENSIONS]


def _corner_metrics(corners: np.ndarray) -> dict[str, Any]:
    points = corners.reshape(4, 2).astype(float)
    side_lengths = [float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)]
    center = points.mean(axis=0)
    return {
        "corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "side_lengths_px": [round(length, 3) for length in side_lengths],
        "mean_side_px": round(float(np.mean(side_lengths)), 3),
        "perimeter_px": round(float(sum(side_lengths)), 3),
    }


def detect_markers_in_images(
    image_paths: Iterable[Path],
    dictionary_name: str = "DICT_4X4_50",
    marker_ids: Iterable[int] | None = None,
    marker_size_m: float | None = None,
    preview_path: Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    allowed_ids = {int(marker_id) for marker_id in marker_ids} if marker_ids else None
    paths = [Path(path) for path in image_paths]
    if max_images is not None:
        paths = paths[: max(0, int(max_images))]

    observations: list[dict[str, Any]] = []
    scanned_images = 0
    unreadable_images: list[str] = []
    preview_image = None
    preview_corners = None
    preview_ids = None

    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable_images.append(_relative_repo_path(image_path))
            continue
        scanned_images += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = _detect_markers(gray, dictionary_name)
        if ids is None or len(ids) == 0:
            continue

        kept_corners = []
        kept_ids = []
        for marker_corners, raw_marker_id in zip(corners, ids.flatten()):
            marker_id = int(raw_marker_id)
            if allowed_ids is not None and marker_id not in allowed_ids:
                continue
            metrics = _corner_metrics(marker_corners)
            observation = {
                "image": _relative_repo_path(image_path),
                "marker_id": marker_id,
                "image_size_px": [int(image.shape[1]), int(image.shape[0])],
                **metrics,
            }
            if marker_size_m is not None and marker_size_m > 0:
                observation["marker_size_m"] = float(marker_size_m)
                observation["pixels_per_meter"] = round(metrics["mean_side_px"] / float(marker_size_m), 3)
            observations.append(observation)
            kept_corners.append(marker_corners)
            kept_ids.append([marker_id])

        if kept_corners and preview_image is None:
            preview_image = image.copy()
            preview_corners = kept_corners
            preview_ids = np.array(kept_ids, dtype=np.int32)

    if preview_path is not None and preview_image is not None and preview_corners is not None and preview_ids is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.aruco.drawDetectedMarkers(preview_image, preview_corners, preview_ids)
        cv2.imwrite(str(preview_path), preview_image)

    marker_counts: dict[str, int] = {}
    for observation in observations:
        key = str(observation["marker_id"])
        marker_counts[key] = marker_counts.get(key, 0) + 1

    status = "success" if observations else "no_detections"
    return {
        "status": status,
        "marker_detection_ran": True,
        "dictionary": dictionary_name,
        "configured_marker_ids": sorted(allowed_ids) if allowed_ids is not None else None,
        "marker_size_m": marker_size_m,
        "images_requested": len(paths),
        "images_scanned": scanned_images,
        "unreadable_images": unreadable_images,
        "detections_found": bool(observations),
        "marker_counts": marker_counts,
        "observations": observations,
        "preview": _relative_repo_path(preview_path) if preview_path is not None and preview_path.exists() else None,
    }


def _read_marker_config(config: dict[str, Any]) -> dict[str, Any]:
    marker_config = config.get("marker_detection", {})
    enabled = bool(marker_config.get("enabled", False))
    dictionary_name = str(marker_config.get("dictionary", "DICT_4X4_50"))
    marker_ids_raw = marker_config.get("marker_ids")
    marker_ids = None
    if marker_ids_raw is not None:
        if not isinstance(marker_ids_raw, list):
            raise ValueError("marker_detection.marker_ids must be a list of integers")
        marker_ids = [int(marker_id) for marker_id in marker_ids_raw]
    marker_size_m_raw = marker_config.get("marker_size_m")
    marker_size_m = float(marker_size_m_raw) if marker_size_m_raw is not None else None
    if marker_size_m is not None and marker_size_m <= 0:
        raise ValueError("marker_detection.marker_size_m must be greater than 0")
    max_images = int(marker_config.get("max_images", 60))
    if max_images <= 0:
        raise ValueError("marker_detection.max_images must be greater than 0")
    require_detection = bool(marker_config.get("require_detection", False))
    return {
        "enabled": enabled,
        "dictionary": dictionary_name,
        "marker_ids": marker_ids,
        "marker_size_m": marker_size_m,
        "max_images": max_images,
        "require_detection": require_detection,
    }


def run_marker_detection(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    marker_config = _read_marker_config(config)
    report_path = output_dirs["debug"] / "marker_detection_report.json"
    if not marker_config["enabled"]:
        report = {
            "status": "skipped",
            "reason": "marker_detection.enabled is false",
            "marker_detection_ran": False,
            "detections_found": False,
            "observations": [],
            "preview": None,
        }
        write_json(report_path, report)
        return {"warnings": []}

    image_paths = _iter_image_paths(output_dirs["images_original"])
    preview_path = output_dirs["preview"] / "aruco_detection_preview.jpg"
    report = detect_markers_in_images(
        image_paths,
        dictionary_name=marker_config["dictionary"],
        marker_ids=marker_config["marker_ids"],
        marker_size_m=marker_config["marker_size_m"],
        preview_path=preview_path,
        max_images=marker_config["max_images"],
    )
    write_json(report_path, report)

    if report["status"] == "no_detections":
        message = "no configured ArUco marker was detected in selected frames"
        if marker_config["require_detection"]:
            raise RuntimeError(message)
        return {"warnings": [message]}
    return {"warnings": []}
