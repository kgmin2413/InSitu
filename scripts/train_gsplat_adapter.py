#!/usr/bin/env python3
"""Adapter that runs gsplat's simple_trainer against an InSitu COLMAP output."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ensure_colmap_sparse_model(sparse_model_dir: Path) -> None:
    missing = [name for name in REQUIRED_COLMAP_FILES if not (sparse_model_dir / name).exists()]
    if missing:
        raise RuntimeError(f"COLMAP sparse model is missing required binary files: {missing}")


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy_dir(source: Path, target: Path, mode: str) -> str:
    if not source.exists():
        raise RuntimeError(f"source directory does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _replace_path(target)
    if mode == "symlink":
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
            return "symlink"
        except OSError:
            shutil.copytree(source, target)
            return "copy_fallback"
    if mode == "copy":
        shutil.copytree(source, target)
        return "copy"
    raise ValueError("link_mode must be symlink or copy")


def _parse_background_color(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("mask_background_color must be R,G,B")
    color = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("mask_background_color channels must be in [0, 255]")
    return color


def _stage_masked_images(masked_images_dir: Path, target_dir: Path, background_color: str) -> int:
    if target_dir.exists() or target_dir.is_symlink():
        _replace_path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rgb_background = np.array(_parse_background_color(background_color), dtype=np.float32)
    count = 0
    for source in sorted(masked_images_dir.iterdir()):
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3 and image.shape[2] == 4:
            bgr = image[:, :, :3].astype(np.float32)
            alpha = (image[:, :, 3:4].astype(np.float32) / 255.0)
            background_bgr = rgb_background[::-1].reshape(1, 1, 3)
            composited = bgr * alpha + background_bgr * (1.0 - alpha)
            output = np.clip(composited, 0, 255).astype(np.uint8)
        elif image.ndim == 3:
            output = image[:, :, :3]
        else:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if not cv2.imwrite(str(target_dir / source.name), output):
            raise RuntimeError(f"Failed to write staged masked image: {target_dir / source.name}")
        count += 1
    if count == 0:
        raise RuntimeError(f"No masked images were staged from {masked_images_dir}")
    return count


def stage_gsplat_dataset(
    images_dir: Path,
    sparse_model_dir: Path,
    dataset_dir: Path,
    link_mode: str,
    masked_images_dir: Path | None = None,
    mask_background_color: str = "255,255,255",
) -> dict[str, Any]:
    _ensure_colmap_sparse_model(sparse_model_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if masked_images_dir is not None:
        staged_count = _stage_masked_images(masked_images_dir, dataset_dir / "images", mask_background_color)
        images_mode = "masked_composite"
    else:
        staged_count = None
        images_mode = _link_or_copy_dir(images_dir, dataset_dir / "images", link_mode)
    sparse_mode = _link_or_copy_dir(sparse_model_dir, dataset_dir / "sparse" / "0", link_mode)
    return {
        "dataset_dir": _relative_repo_path(dataset_dir),
        "images": _relative_repo_path(dataset_dir / "images"),
        "sparse_model": _relative_repo_path(dataset_dir / "sparse" / "0"),
        "images_stage_mode": images_mode,
        "masked_images_dir": None if masked_images_dir is None else _relative_repo_path(masked_images_dir),
        "masked_images_staged": staged_count,
        "mask_background_color": None if masked_images_dir is None else mask_background_color,
        "sparse_stage_mode": sparse_mode,
    }


def _find_trainer_path(gsplat_repo: Path | None, trainer_path: Path | None) -> Path:
    if trainer_path is not None:
        resolved = trainer_path
    elif gsplat_repo is not None:
        resolved = gsplat_repo / "examples" / "simple_trainer.py"
    else:
        raise RuntimeError("Either --gsplat_repo or --trainer_path is required")
    resolved = resolved.resolve()
    if not resolved.exists():
        raise RuntimeError(
            f"gsplat simple_trainer.py was not found: {resolved}. "
            "Clone https://github.com/nerfstudio-project/gsplat or pass --trainer_path."
        )
    return resolved


def _default_gsplat_args(iterations: int, data_factor: int, dataset_dir: Path, result_dir: Path) -> list[str]:
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0 for gsplat training")
    if data_factor <= 0:
        raise ValueError("data_factor must be greater than 0")
    return [
        "default",
        "--data-dir",
        str(dataset_dir),
        "--data-factor",
        str(data_factor),
        "--result-dir",
        str(result_dir),
        "--max-steps",
        str(iterations),
        "--save-ply",
        "--ply-steps",
        str(iterations),
        "--save-steps",
        str(iterations),
        "--eval-steps",
        str(iterations),
        "--disable-viewer",
        "--disable-video",
    ]


def build_gsplat_command(
    python_executable: str,
    trainer_path: Path,
    iterations: int,
    data_factor: int,
    dataset_dir: Path,
    result_dir: Path,
    extra_args: list[str],
) -> list[str]:
    return [
        python_executable,
        str(trainer_path.resolve()),
        *_default_gsplat_args(iterations, data_factor, dataset_dir.resolve(), result_dir.resolve()),
        *extra_args,
    ]


def _candidate_ply_paths(result_dir: Path, iterations: int) -> list[Path]:
    candidates = [
        result_dir / "ply" / f"point_cloud_{iterations}.ply",
        result_dir / "ply" / f"point_cloud_{max(0, iterations - 1)}.ply",
        result_dir / f"point_cloud_{iterations}.ply",
        result_dir / f"point_cloud_{max(0, iterations - 1)}.ply",
    ]
    ply_dir = result_dir / "ply"
    if ply_dir.exists():
        candidates.extend(sorted(ply_dir.glob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True))
    candidates.extend(sorted(result_dir.glob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen:
            deduped.append(candidate)
            seen.add(resolved)
    return deduped


def export_expected_point_cloud(result_dir: Path, model_dir: Path, iterations: int) -> dict[str, Any]:
    source = next((path for path in _candidate_ply_paths(result_dir, iterations) if path.exists()), None)
    if source is None:
        checked = [_relative_repo_path(path) for path in _candidate_ply_paths(result_dir, iterations)]
        raise RuntimeError(f"gsplat did not create a PLY point cloud. Checked: {checked}")
    target = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source_point_cloud": _relative_repo_path(source),
        "expected_point_cloud": _relative_repo_path(target),
        "size_bytes": target.stat().st_size,
    }


def run_gsplat_adapter(
    images_dir: Path,
    sparse_model_dir: Path,
    model_dir: Path,
    iterations: int,
    gsplat_repo: Path | None = None,
    trainer_path: Path | None = None,
    masked_images_dir: Path | None = None,
    mask_background_color: str = "255,255,255",
    data_factor: int = 1,
    link_mode: str = "symlink",
    python_executable: str = sys.executable,
    timeout_sec: int | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    extra_args = extra_args or []
    trainer = _find_trainer_path(gsplat_repo, trainer_path)
    dataset_dir = model_dir / "gsplat_dataset"
    result_dir = model_dir / "gsplat_results"
    report_path = model_dir / "gsplat_adapter_report.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "running",
        "trainer_path": _relative_repo_path(trainer),
        "model_dir": _relative_repo_path(model_dir),
        "iterations": iterations,
        "data_factor": data_factor,
        "link_mode": link_mode,
    }
    try:
        report["staged_dataset"] = stage_gsplat_dataset(
            images_dir,
            sparse_model_dir,
            dataset_dir,
            link_mode,
            masked_images_dir=masked_images_dir,
            mask_background_color=mask_background_color,
        )
        command = build_gsplat_command(
            python_executable,
            trainer,
            iterations,
            data_factor,
            dataset_dir,
            result_dir,
            extra_args,
        )
        report["command"] = " ".join(command)
        _write_json(report_path, report)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        completed = subprocess.run(
            command,
            cwd=str(trainer.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        report.update(
            {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout.strip()[-4000:],
                "stderr_tail": completed.stderr.strip()[-4000:],
            }
        )
        if completed.returncode != 0:
            detail = report["stderr_tail"] or report["stdout_tail"] or "unknown gsplat error"
            raise RuntimeError(f"gsplat trainer failed: {detail}")

        report.update(export_expected_point_cloud(result_dir, model_dir, iterations))
        report["status"] = "success"
        _write_json(report_path, report)
        return report
    except subprocess.TimeoutExpired as exc:
        report.update({"status": "failed", "error": f"gsplat trainer timed out: {exc}"})
        _write_json(report_path, report)
        raise RuntimeError(report["error"]) from exc
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc)})
        _write_json(report_path, report)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gsplat simple_trainer for an InSitu COLMAP reconstruction.")
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--sparse_model_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gsplat_repo", type=Path, default=Path("third_party/gsplat"))
    parser.add_argument("--trainer_path", type=Path, default=None)
    parser.add_argument("--masked_images_dir", type=Path, default=None)
    parser.add_argument("--mask_background_color", default="255,255,255")
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--link_mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout_sec", type=int, default=None)
    parser.add_argument("--extra_arg", action="append", default=[], help="Extra argument appended to simple_trainer; repeat for each token.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_gsplat_adapter(
        images_dir=args.images_dir,
        sparse_model_dir=args.sparse_model_dir,
        model_dir=args.model_dir,
        iterations=args.iterations,
        gsplat_repo=args.gsplat_repo,
        trainer_path=args.trainer_path,
        masked_images_dir=args.masked_images_dir,
        mask_background_color=args.mask_background_color,
        data_factor=args.data_factor,
        link_mode=args.link_mode,
        python_executable=args.python,
        timeout_sec=args.timeout_sec,
        extra_args=args.extra_arg,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
