#!/usr/bin/env python3
"""Adapter that runs the InSitu-A4 Gaussian Splatting trainer inside this pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _ensure_sparse_model(path: Path) -> None:
    missing = [name for name in REQUIRED_COLMAP_FILES if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"COLMAP sparse model is missing required files: {missing}")


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _run(command: list[str], timeout_sec: int, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    record = {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip()[-4000:],
        "stderr_tail": result.stderr.strip()[-4000:],
    }
    if result.returncode != 0:
        detail = record["stderr_tail"] or record["stdout_tail"] or "unknown command error"
        raise RuntimeError(f"command failed: {record['command']}\n{detail}")
    return record


def _normalize_undistorted_sparse(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse"
    target_dir = sparse_dir / "0"
    if all((target_dir / name).exists() for name in REQUIRED_COLMAP_FILES):
        return
    loose_files = [sparse_dir / name for name in REQUIRED_COLMAP_FILES]
    if all(path.exists() for path in loose_files):
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in loose_files:
            shutil.move(str(path), str(target_dir / path.name))
        return
    raise RuntimeError(f"Undistorted sparse model was not created under {sparse_dir}")


def _build_env(a4_repo: Path, extra_pythonpath: str | None) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(a4_repo / "scripts")]
    if extra_pythonpath:
        paths.append(extra_pythonpath)
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_a4_training(
    a4_repo: Path,
    images_dir: Path,
    sparse_model_dir: Path,
    model_dir: Path,
    iterations: int,
    colmap_binary: str,
    python_executable: str,
    timeout_sec: int,
    extra_pythonpath: str | None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")
    a4_repo = a4_repo.resolve()
    images_dir = images_dir.resolve()
    sparse_model_dir = sparse_model_dir.resolve()
    model_dir = model_dir.resolve()
    if not a4_repo.exists():
        raise RuntimeError(f"InSitu-A4 repo does not exist: {a4_repo}")
    trainer = (a4_repo / "scripts" / "train.py").resolve()
    if not trainer.exists():
        raise RuntimeError(f"A4 trainer was not found: {trainer}")
    if not images_dir.exists():
        raise RuntimeError(f"images_dir does not exist: {images_dir}")
    _ensure_sparse_model(sparse_model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = model_dir / "a4_dataset"
    report_path = model_dir / "a4_adapter_report.json"
    if dataset_dir.exists():
        _replace_path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "running",
        "a4_repo": _relative_repo_path(a4_repo),
        "trainer": _relative_repo_path(trainer),
        "images_dir": _relative_repo_path(images_dir),
        "sparse_model_dir": _relative_repo_path(sparse_model_dir),
        "dataset_dir": _relative_repo_path(dataset_dir),
        "model_dir": _relative_repo_path(model_dir),
        "iterations": iterations,
        "commands": [],
    }
    _write_json(report_path, report)

    undistort_command = [
        colmap_binary,
        "image_undistorter",
        "--image_path",
        str(images_dir),
        "--input_path",
        str(sparse_model_dir),
        "--output_path",
        str(dataset_dir),
        "--output_type",
        "COLMAP",
    ]
    report["commands"].append(_run(undistort_command, timeout_sec))
    _normalize_undistorted_sparse(dataset_dir)

    env = _build_env(a4_repo, extra_pythonpath)
    train_command = [
        python_executable,
        str(trainer),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--quiet",
    ]
    report["commands"].append(_run(train_command, timeout_sec, cwd=a4_repo, env=env))

    point_cloud = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not point_cloud.exists():
        raise RuntimeError(f"A4 trainer did not create expected PLY: {point_cloud}")

    report.update(
        {
            "status": "success",
            "point_cloud": _relative_repo_path(point_cloud),
            "size_bytes": point_cloud.stat().st_size,
        }
    )
    _write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InSitu-A4 training against an existing COLMAP sparse model.")
    parser.add_argument("--a4_repo", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--sparse_model_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--colmap_binary", default="colmap")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--timeout_sec", type=int, default=21600)
    parser.add_argument("--extra_pythonpath", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_a4_training(
            a4_repo=Path(args.a4_repo),
            images_dir=Path(args.images_dir),
            sparse_model_dir=Path(args.sparse_model_dir),
            model_dir=Path(args.model_dir),
            iterations=args.iterations,
            colmap_binary=args.colmap_binary,
            python_executable=args.python_executable,
            timeout_sec=args.timeout_sec,
            extra_pythonpath=args.extra_pythonpath,
        )
    except Exception as exc:
        model_dir = Path(args.model_dir)
        _write_json(
            model_dir / "a4_adapter_report.json",
            {
                "status": "failed",
                "error": str(exc),
            },
        )
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
