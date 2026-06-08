#!/usr/bin/env python3
"""Generate a printable A4 ArUco scale marker sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
MM_PER_INCH = 25.4


def _get_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available")
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / MM_PER_INCH * dpi))


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, center_x: int, y: int, fill=(0, 0, 0)) -> None:
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0]
    draw.text((center_x - width // 2, y), text, fill=fill)


def generate_sheet(
    output_path: Path,
    metadata_path: Path,
    dictionary_name: str,
    marker_id: int,
    marker_size_mm: float,
    dpi: int,
) -> dict[str, Any]:
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be greater than 0")
    if marker_size_mm > min(A4_WIDTH_MM, A4_HEIGHT_MM) - 40:
        raise ValueError("marker_size_mm is too large for A4 with margins")
    if dpi <= 0:
        raise ValueError("dpi must be greater than 0")

    dictionary = _get_dictionary(dictionary_name)
    page_width_px = _mm_to_px(A4_WIDTH_MM, dpi)
    page_height_px = _mm_to_px(A4_HEIGHT_MM, dpi)
    marker_size_px = _mm_to_px(marker_size_mm, dpi)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)

    page = Image.new("RGB", (page_width_px, page_height_px), "white")
    marker_image = Image.fromarray(marker_rgb)
    marker_x = (page_width_px - marker_size_px) // 2
    marker_y = _mm_to_px(55.0, dpi)
    page.paste(marker_image, (marker_x, marker_y))

    draw = ImageDraw.Draw(page)
    center_x = page_width_px // 2
    title_y = _mm_to_px(18.0, dpi)
    _draw_centered_text(draw, "InSitu Scale Marker", center_x, title_y)
    _draw_centered_text(draw, f"{dictionary_name} ID {marker_id}", center_x, title_y + _mm_to_px(8.0, dpi))
    _draw_centered_text(draw, f"Marker side: {marker_size_mm:.1f} mm", center_x, marker_y + marker_size_px + _mm_to_px(10.0, dpi))
    _draw_centered_text(draw, "Print on A4 at 100% scale. Do not fit to page.", center_x, marker_y + marker_size_px + _mm_to_px(20.0, dpi))

    border_margin = _mm_to_px(10.0, dpi)
    draw.rectangle(
        [border_margin, border_margin, page_width_px - border_margin, page_height_px - border_margin],
        outline=(0, 0, 0),
        width=max(1, _mm_to_px(0.3, dpi)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path)

    metadata = {
        "schema_version": 1,
        "page": "A4",
        "page_size_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
        "dpi": dpi,
        "dictionary": dictionary_name,
        "marker_id": marker_id,
        "marker_size_mm": marker_size_mm,
        "marker_size_m": marker_size_mm / 1000.0,
        "image_path": str(output_path),
        "print_instructions": "Print at 100% scale on A4. Disable fit-to-page or scaling.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a printable A4 ArUco marker sheet for InSitu scale capture.")
    parser.add_argument("--output", type=Path, default=Path("data/markers/insitu_aruco_a4.png"))
    parser.add_argument("--metadata", type=Path, default=Path("data/markers/insitu_aruco_a4.json"))
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker_id", type=int, default=23)
    parser.add_argument("--marker_size_mm", type=float, default=120.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    metadata = generate_sheet(
        args.output,
        args.metadata,
        args.dictionary,
        args.marker_id,
        args.marker_size_mm,
        args.dpi,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
