"""Frame extraction and deterministic frame selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import write_json


_VALID_OUTPUT_EXTS = {"png", "jpg", "jpeg"}


def _read_frame_config(config: dict[str, Any]) -> tuple[float, int, float, float, str]:
    frame_config = config.get("frame_extraction", {})
    target_fps = float(frame_config.get("target_fps", 2.0))
    max_frames = int(frame_config.get("max_frames", 120))
    blur_threshold = float(frame_config.get("blur_threshold", 50.0))
    duplicate_threshold = float(frame_config.get("duplicate_threshold", 3.0))
    output_ext = str(frame_config.get("output_ext", "png")).lower().lstrip(".")

    if target_fps <= 0:
        raise ValueError("frame_extraction.target_fps must be greater than 0")
    if max_frames <= 0:
        raise ValueError("frame_extraction.max_frames must be greater than 0")
    if blur_threshold < 0:
        raise ValueError("frame_extraction.blur_threshold must be zero or greater")
    if duplicate_threshold < 0:
        raise ValueError("frame_extraction.duplicate_threshold must be zero or greater")
    if output_ext not in _VALID_OUTPUT_EXTS:
        raise ValueError("frame_extraction.output_ext must be png, jpg, or jpeg")
    return target_fps, max_frames, blur_threshold, duplicate_threshold, output_ext


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _safe_capture_value(capture: cv2.VideoCapture, key: int) -> float:
    value = float(capture.get(key) or 0.0)
    if np.isnan(value) or value < 0:
        return 0.0
    return value


def _blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fingerprint(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)


def _duplicate_score(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    if previous is None:
        return None
    diff = cv2.absdiff(previous, current)
    return float(np.mean(diff))


def _write_input_thumbnail(frame: np.ndarray, preview_dir: Path) -> str:
    preview_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = preview_dir / "input_thumbnail.jpg"
    thumb = _resize_with_max_side(frame, 512)
    if not cv2.imwrite(str(thumbnail_path), thumb):
        raise RuntimeError(f"Failed to write input thumbnail: {thumbnail_path}")
    return _relative_repo_path(thumbnail_path)


def _resize_with_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.array(values, dtype=float)
    return {
        "min": round(float(np.min(array)), 6),
        "p25": round(float(np.percentile(array, 25)), 6),
        "median": round(float(np.median(array)), 6),
        "p75": round(float(np.percentile(array, 75)), 6),
        "max": round(float(np.max(array)), 6),
        "mean": round(float(np.mean(array)), 6),
    }


def _quality_warnings(
    selected_count: int,
    raw_frames: int,
    duration_sec: float | None,
    removed_blurry: int,
    removed_duplicates: int,
    sampled_count: int,
) -> list[str]:
    warnings: list[str] = []
    if duration_sec is not None and duration_sec < 15:
        warnings.append("input video is shorter than the recommended 15 seconds")
    if selected_count < 8:
        warnings.append("fewer than 8 frames were selected; COLMAP may be unstable")
    if sampled_count and removed_blurry / sampled_count > 0.5:
        warnings.append("more than half of sampled frames were rejected as blurry")
    if sampled_count and removed_duplicates / sampled_count > 0.5:
        warnings.append("more than half of sampled frames were rejected as duplicates")
    if raw_frames < 30:
        warnings.append("input video has fewer than 30 readable/probed frames")
    return warnings


def _write_selected_contact_sheet(selected_records: list[dict[str, Any]], preview_dir: Path) -> str | None:
    if not selected_records:
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    contact_path = preview_dir / "selected_frames_contact_sheet.jpg"
    records = selected_records[:24]
    thumbs = []
    for record in records:
        image = cv2.imread(str(_resolve_repo_path(record["image_path"])), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            Path(record["image_path"]).name,
            (6, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return None
    cols = min(6, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * 120, cols * 160, 3), 32, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // cols
        col = index % cols
        sheet[row * 120 : (row + 1) * 120, col * 160 : (col + 1) * 160] = thumb
    if not cv2.imwrite(str(contact_path), sheet):
        raise RuntimeError(f"Failed to write selected frame contact sheet: {contact_path}")
    return _relative_repo_path(contact_path)


def _build_capture_quality_report(
    input_video: Path,
    width: int,
    height: int,
    source_fps: float,
    raw_frames: int,
    selected_count: int,
    removed_blurry: int,
    removed_duplicates: int,
    sampled_frame_scores: list[dict[str, Any]],
    contact_sheet: str | None,
) -> dict[str, Any]:
    duration_sec = round(raw_frames / source_fps, 6) if source_fps > 0 and raw_frames > 0 else None
    blur_values = [float(record["blur_score"]) for record in sampled_frame_scores]
    duplicate_values = [
        float(record["duplicate_score"])
        for record in sampled_frame_scores
        if record.get("duplicate_score") is not None
    ]
    sampled_count = len(sampled_frame_scores)
    return {
        "status": "success",
        "source_video": str(input_video),
        "resolution": {
            "width": width,
            "height": height,
        },
        "source_fps": round(source_fps, 6),
        "duration_sec": duration_sec,
        "raw_frames": raw_frames,
        "sampled_frames": sampled_count,
        "selected_frames": selected_count,
        "selected_ratio": round(selected_count / sampled_count, 6) if sampled_count else 0.0,
        "removed_blurry": removed_blurry,
        "removed_duplicates": removed_duplicates,
        "blur_score_summary": _summarize_values(blur_values),
        "duplicate_score_summary": _summarize_values(duplicate_values),
        "selected_frames_contact_sheet": contact_sheet,
        "quality_warnings": _quality_warnings(
            selected_count,
            raw_frames,
            duration_sec,
            removed_blurry,
            removed_duplicates,
            sampled_count,
        ),
    }


def run_frame_extraction(input_video: Path, output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    target_fps, max_frames, blur_threshold, duplicate_threshold, output_ext = _read_frame_config(config)

    images_original_dir = output_dirs["images_original"]
    images_original_dir.mkdir(parents=True, exist_ok=True)
    output_dirs["debug"].mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {input_video}")

    source_fps = _safe_capture_value(capture, cv2.CAP_PROP_FPS) or 30.0
    probed_frame_count = int(_safe_capture_value(capture, cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(_safe_capture_value(capture, cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(_safe_capture_value(capture, cv2.CAP_PROP_FRAME_HEIGHT))
    sample_interval = max(1, round(source_fps / target_fps))

    selected_records: list[dict[str, Any]] = []
    sampled_frame_scores: list[dict[str, Any]] = []
    removed_blurry = 0
    removed_duplicates = 0
    frame_index = 0
    selected_index = 1
    previous_fingerprint: np.ndarray | None = None
    thumbnail_path: str | None = None

    try:
        while selected_index <= max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if thumbnail_path is None:
                thumbnail_path = _write_input_thumbnail(frame, output_dirs["preview"])
                if source_width <= 0 or source_height <= 0:
                    source_height, source_width = frame.shape[:2]

            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            blur = _blur_score(frame)
            current_fingerprint = _fingerprint(frame)
            duplicate = _duplicate_score(previous_fingerprint, current_fingerprint)
            sampled_frame_scores.append(
                {
                    "source_frame_index": frame_index,
                    "blur_score": round(blur, 6),
                    "duplicate_score": None if duplicate is None else round(duplicate, 6),
                }
            )

            if blur < blur_threshold:
                removed_blurry += 1
                frame_index += 1
                continue

            if duplicate is not None and duplicate < duplicate_threshold:
                removed_duplicates += 1
                frame_index += 1
                continue

            output_path = images_original_dir / f"{selected_index:06d}.{output_ext}"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write selected frame: {output_path}")

            selected_records.append(
                {
                    "source_frame_index": frame_index,
                    "image_path": _relative_repo_path(output_path),
                    "blur_score": round(blur, 6),
                    "duplicate_score": None if duplicate is None else round(duplicate, 6),
                }
            )
            previous_fingerprint = current_fingerprint
            selected_index += 1
            frame_index += 1
    finally:
        capture.release()

    raw_frames = probed_frame_count if probed_frame_count > 0 else frame_index
    if raw_frames <= 0:
        raise RuntimeError(f"No readable frames found in input video: {input_video}")
    if not selected_records:
        raise RuntimeError(
            "No frames passed selection. Lower frame_extraction.blur_threshold or duplicate_threshold, or check the input video."
        )

    selected_frames_path = output_dirs["debug"] / "selected_frames.txt"
    with selected_frames_path.open("w", encoding="utf-8") as f:
        for record in selected_records:
            f.write(f"{record['image_path']}\n")

    contact_sheet = _write_selected_contact_sheet(selected_records, output_dirs["preview"])
    quality_report = _build_capture_quality_report(
        input_video,
        source_width,
        source_height,
        source_fps,
        raw_frames,
        len(selected_records),
        removed_blurry,
        removed_duplicates,
        sampled_frame_scores,
        contact_sheet,
    )
    write_json(output_dirs["debug"] / "capture_quality_report.json", quality_report)

    report = {
        "status": "success",
        "source_video": str(input_video),
        "frame_extraction_ran": True,
        "frame_selection_ran": True,
        "source_fps": round(source_fps, 6),
        "duration_sec": quality_report["duration_sec"],
        "source_resolution": quality_report["resolution"],
        "target_fps": target_fps,
        "sample_interval": sample_interval,
        "max_frames": max_frames,
        "blur_threshold": blur_threshold,
        "duplicate_threshold": duplicate_threshold,
        "output_ext": output_ext,
        "input_thumbnail": thumbnail_path,
        "selected_frames_contact_sheet": contact_sheet,
        "images_original_dir": _relative_repo_path(images_original_dir),
        "raw_frames": raw_frames,
        "selected_frames": len(selected_records),
        "removed_blurry": removed_blurry,
        "removed_duplicates": removed_duplicates,
        "frames": selected_records,
        "sampled_frame_scores": sampled_frame_scores,
        "quality_warnings": quality_report["quality_warnings"],
    }
    write_json(output_dirs["debug"] / "frame_selection_report.json", report)

    return {
        "frame_stats": {
            "raw_frames": raw_frames,
            "selected_frames": len(selected_records),
            "removed_blurry": removed_blurry,
            "removed_duplicates": removed_duplicates,
        },
        "warnings": [],
    }
