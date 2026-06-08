#!/usr/bin/env python3
"""Create a Unity-loadable zip package from an InSitu output folder."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_output


INCLUDE_ALWAYS = {
    "metadata.json",
    "processing_log.json",
    "package_manifest.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_files(object_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for name in INCLUDE_ALWAYS:
        path = object_dir / name
        if path.exists() and path.is_file():
            files.add(path)
    for folder in ("preview", "debug"):
        base = object_dir / folder
        if base.exists():
            files.update(path for path in base.rglob("*") if path.is_file())
    for name in ("point_cloud_metric.ply", "point_cloud_metric_rgb.ply", "proxy_collider.obj"):
        path = object_dir / name
        if path.exists() and path.is_file():
            files.add(path)
    return sorted(files)


def _validate(object_dir: Path, mode: str) -> None:
    errors: list[str] = []
    if mode == "default_safe":
        validate_output._validate_default_safe(object_dir, errors)
    elif mode == "sparse_proxy":
        validate_output._validate_sparse_proxy(object_dir, errors)
    else:
        raise ValueError(f"unsupported package mode: {mode}")
    if errors:
        raise RuntimeError("output validation failed before packaging:\n" + "\n".join(errors))


def _write_package_summary(object_dir: Path, package_path: Path, included_files: list[Path]) -> Path:
    manifest = _read_json(object_dir / "package_manifest.json")
    summary = {
        "schema_version": 1,
        "object_id": manifest.get("object_id"),
        "source_output": str(object_dir),
        "package_path": str(package_path),
        "file_count": len(included_files),
        "files": [path.relative_to(object_dir).as_posix() for path in included_files],
    }
    summary_path = object_dir / "debug" / "unity_package_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def create_package(object_dir: Path, package_path: Path, mode: str) -> Path:
    _validate(object_dir, mode)
    files = _collect_files(object_dir)
    summary_path = _write_package_summary(object_dir, package_path, files)
    files = sorted(set(files + [summary_path]))
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(object_dir).as_posix())
    return package_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Unity package zip from an InSitu output folder.")
    parser.add_argument("object_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("default_safe", "sparse_proxy"),
        default="sparse_proxy",
        help="Validation contract to apply before packaging.",
    )
    parser.add_argument(
        "--package_path",
        type=Path,
        default=None,
        help="Output zip path. Defaults to <object_dir>/<object_id>_unity_package.zip.",
    )
    args = parser.parse_args()

    manifest = _read_json(args.object_dir / "package_manifest.json")
    object_id = manifest.get("object_id") or args.object_dir.name
    package_path = args.package_path or (args.object_dir / f"{object_id}_unity_package.zip")
    created = create_package(args.object_dir, package_path, args.mode)
    print(f"OK: wrote Unity package {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
