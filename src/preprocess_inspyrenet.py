"""Background removal helpers for the incremental backend pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import write_json


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_selected_frames(debug_dir: Path) -> list[str]:
    selected_frames_path = debug_dir / "selected_frames.txt"
    if not selected_frames_path.exists():
        raise RuntimeError("selected_frames.txt is missing; run frame extraction first")
    return [
        line.strip()
        for line in selected_frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_frame_path(frame_path: str) -> Path:
    path = Path(frame_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_background_config(config: dict[str, Any]) -> tuple[str, int, float, int]:
    background_config = config.get("background_removal", {})
    backend = str(background_config.get("backend", "opencv_grabcut"))
    iterations = int(background_config.get("grabcut_iterations", 5))
    rect_margin_ratio = float(background_config.get("rect_margin_ratio", 0.08))
    max_processing_side = int(background_config.get("max_processing_side", 512))

    if backend != "opencv_grabcut":
        raise ValueError("background_removal.backend currently supports only opencv_grabcut")
    if iterations <= 0:
        raise ValueError("background_removal.grabcut_iterations must be greater than 0")
    if not 0.0 <= rect_margin_ratio < 0.5:
        raise ValueError("background_removal.rect_margin_ratio must be in [0.0, 0.5)")
    if max_processing_side < 64:
        raise ValueError("background_removal.max_processing_side must be at least 64")
    return backend, iterations, rect_margin_ratio, max_processing_side


def _resize_for_processing(image_bgr: np.ndarray, max_processing_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_bgr.shape[:2]
    max_side = max(height, width)
    if max_side <= max_processing_side:
        return image_bgr, (width, height)
    scale = max_processing_side / max_side
    resized = cv2.resize(
        image_bgr,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (width, height)


def _grabcut_alpha(
    image_bgr: np.ndarray,
    iterations: int,
    rect_margin_ratio: float,
    max_processing_side: int,
) -> np.ndarray:
    processing_image, original_size = _resize_for_processing(image_bgr, max_processing_side)
    height, width = processing_image.shape[:2]
    margin_x = max(1, int(width * rect_margin_ratio))
    margin_y = max(1, int(height * rect_margin_ratio))
    rect_width = max(1, width - margin_x * 2)
    rect_height = max(1, height - margin_y * 2)

    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        processing_image,
        mask,
        (margin_x, margin_y, rect_width, rect_height),
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")
    if (width, height) != original_size:
        alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_NEAREST)
    return alpha


def _write_masked_image(image_bgr: np.ndarray, alpha: np.ndarray, output_path: Path) -> None:
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rgba):
        raise RuntimeError(f"Failed to write masked image: {output_path}")


def _write_mask_preview(image_bgr: np.ndarray, alpha: np.ndarray, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :, 1] = 255
    foreground = alpha > 0
    overlay[foreground] = cv2.addWeighted(image_bgr, 0.65, green, 0.35, 0)[foreground]
    if not cv2.imwrite(str(preview_path), overlay):
        raise RuntimeError(f"Failed to write mask preview: {preview_path}")


def run_background_removal(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    selected_frames = _read_selected_frames(output_dirs["debug"])
    if not selected_frames:
        raise RuntimeError("No selected frames available for background removal")

    backend, iterations, rect_margin_ratio, max_processing_side = _read_background_config(config)
    images_masked_dir = output_dirs["images_masked"]
    images_masked_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, Any]] = []
    preview_path = output_dirs["preview"] / "mask_preview.jpg"
    preview_written = False

    for frame_path in selected_frames:
        input_path = _resolve_frame_path(frame_path)
        image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read selected frame for masking: {input_path}")

        alpha = _grabcut_alpha(image_bgr, iterations, rect_margin_ratio, max_processing_side)
        output_path = images_masked_dir / input_path.name
        _write_masked_image(image_bgr, alpha, output_path)

        if not preview_written:
            _write_mask_preview(image_bgr, alpha, preview_path)
            preview_written = True

        frame_records.append(
            {
                "frame": frame_path,
                "masked_image": _relative_repo_path(output_path),
                "status": "success",
                "foreground_pixel_count": int(np.count_nonzero(alpha)),
            }
        )

    manifest = {
        "status": "success",
        "background_removal_ran": True,
        "backend": backend,
        "grabcut_iterations": iterations,
        "rect_margin_ratio": rect_margin_ratio,
        "max_processing_side": max_processing_side,
        "input_frame_count": len(selected_frames),
        "mask_images_created": len(frame_records),
        "images_masked_dir": _relative_repo_path(images_masked_dir),
        "mask_preview": _relative_repo_path(preview_path),
        "frames": frame_records,
    }
    write_json(output_dirs["debug"] / "mask_manifest.json", manifest)

    return {
        "metrics": {
            "mask_images_created": len(frame_records),
        },
        "warnings": [],
    }
