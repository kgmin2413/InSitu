#!/usr/bin/env python3
"""Detect ArUco scale markers in an image file or image directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marker_detection import _iter_image_paths, detect_markers_in_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect InSitu ArUco scale markers in captured frames.")
    parser.add_argument("input", type=Path, help="Image file or directory of images to scan.")
    parser.add_argument("--output_json", type=Path, default=Path("marker_detection_report.json"))
    parser.add_argument("--preview", type=Path, default=None, help="Optional output image with detections drawn.")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker_id", type=int, action="append", dest="marker_ids")
    parser.add_argument("--marker_size_m", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    image_paths = _iter_image_paths(args.input)
    report = detect_markers_in_images(
        image_paths,
        dictionary_name=args.dictionary,
        marker_ids=args.marker_ids,
        marker_size_m=args.marker_size_m,
        preview_path=args.preview,
        max_images=args.max_images,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
