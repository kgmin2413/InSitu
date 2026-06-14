"""Background removal helpers for the incremental backend pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

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


def _read_background_config(config: dict[str, Any]) -> dict[str, Any]:
    background_config = config.get("background_removal", {})
    backend = str(background_config.get("backend", "opencv_grabcut"))
    if backend not in {"opencv_grabcut", "transparent_background", "inspyrenet"}:
        raise ValueError("background_removal.backend must be opencv_grabcut, transparent_background, or inspyrenet")

    iterations = int(background_config.get("grabcut_iterations", 5))
    rect_margin_ratio = float(background_config.get("rect_margin_ratio", 0.08))
    max_processing_side = int(background_config.get("max_processing_side", 512))
    alpha_threshold = int(background_config.get("alpha_threshold", 16))
    tb_mode = str(background_config.get("transparent_mode", "base"))
    tb_device = background_config.get("device")
    tb_resize = str(background_config.get("resize", "static"))
    postprocess_config = background_config.get("postprocess", {})
    postprocess_enabled = bool(postprocess_config.get("enabled", False))
    dilate_kernel = int(postprocess_config.get("dilate_kernel", 0))
    dilate_iterations = int(postprocess_config.get("dilate_iterations", 1))

    if iterations <= 0:
        raise ValueError("background_removal.grabcut_iterations must be greater than 0")
    if not 0.0 <= rect_margin_ratio < 0.5:
        raise ValueError("background_removal.rect_margin_ratio must be in [0.0, 0.5)")
    if max_processing_side < 64:
        raise ValueError("background_removal.max_processing_side must be at least 64")
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("background_removal.alpha_threshold must be between 0 and 255")
    if dilate_kernel < 0:
        raise ValueError("background_removal.postprocess.dilate_kernel must be non-negative")
    if dilate_iterations <= 0:
        raise ValueError("background_removal.postprocess.dilate_iterations must be greater than 0")
    return {
        "backend": backend,
        "grabcut_iterations": iterations,
        "rect_margin_ratio": rect_margin_ratio,
        "max_processing_side": max_processing_side,
        "alpha_threshold": alpha_threshold,
        "transparent_mode": tb_mode,
        "device": None if tb_device is None else str(tb_device),
        "resize": tb_resize,
        "postprocess_enabled": postprocess_enabled,
        "dilate_kernel": dilate_kernel,
        "dilate_iterations": dilate_iterations,
    }


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


def _transparent_background_alpha(image_bgr: np.ndarray, remover: Any, alpha_threshold: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    rgba = remover.process(pil_image, type="rgba")
    rgba_array = np.array(rgba)
    if rgba_array.ndim != 3 or rgba_array.shape[2] != 4:
        raise RuntimeError("transparent-background did not return an RGBA image")
    alpha = rgba_array[:, :, 3].astype(np.uint8)
    if alpha_threshold > 0:
        alpha = np.where(alpha >= alpha_threshold, alpha, 0).astype(np.uint8)
    return alpha


def _inspyrenet_alpha(image_bgr: np.ndarray, remover: Any, alpha_threshold: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    mask = remover.process(pil_image, type="map")
    mask_array = np.array(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    alpha = mask_array.astype(np.uint8)
    if alpha_threshold > 0:
        alpha = np.where(alpha >= alpha_threshold, alpha, 0).astype(np.uint8)
    return alpha


def _create_transparent_background_remover(config: dict[str, Any]) -> Any:
    try:
        from transparent_background import Remover
    except ImportError as exc:
        raise RuntimeError("transparent-background is not installed") from exc
    return Remover(
        mode=config["transparent_mode"],
        device=config["device"],
        resize=config["resize"],
    )


def _postprocess_alpha(alpha: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if not config["postprocess_enabled"]:
        return alpha
    dilate_kernel = config["dilate_kernel"]
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel, dilate_kernel),
        )
        alpha = cv2.dilate(alpha, kernel, iterations=config["dilate_iterations"])
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

    background_config = _read_background_config(config)
    backend = background_config["backend"]
    images_masked_dir = output_dirs["images_masked"]
    images_masked_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, Any]] = []
    preview_path = output_dirs["preview"] / "mask_preview.jpg"
    preview_written = False
    remover = None
    if backend in {"transparent_background", "inspyrenet"}:
        remover = _create_transparent_background_remover(background_config)

    for frame_path in selected_frames:
        input_path = _resolve_frame_path(frame_path)
        image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read selected frame for masking: {input_path}")

        if backend == "opencv_grabcut":
            alpha = _grabcut_alpha(
                image_bgr,
                background_config["grabcut_iterations"],
                background_config["rect_margin_ratio"],
                background_config["max_processing_side"],
            )
        elif backend == "transparent_background":
            alpha = _transparent_background_alpha(
                image_bgr,
                remover,
                background_config["alpha_threshold"],
            )
        else:
            alpha = _inspyrenet_alpha(
                image_bgr,
                remover,
                background_config["alpha_threshold"],
            )
        alpha = _postprocess_alpha(alpha, background_config)
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
        "grabcut_iterations": background_config["grabcut_iterations"],
        "rect_margin_ratio": background_config["rect_margin_ratio"],
        "max_processing_side": background_config["max_processing_side"],
        "alpha_threshold": background_config["alpha_threshold"],
        "transparent_mode": background_config["transparent_mode"] if backend in {"transparent_background", "inspyrenet"} else None,
        "device": background_config["device"] if backend in {"transparent_background", "inspyrenet"} else None,
        "resize": background_config["resize"] if backend in {"transparent_background", "inspyrenet"} else None,
        "postprocess": {
            "enabled": background_config["postprocess_enabled"],
            "dilate_kernel": background_config["dilate_kernel"],
            "dilate_iterations": background_config["dilate_iterations"],
        },
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
