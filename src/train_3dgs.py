"""Controlled 3D Gaussian Splatting training wrapper."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from utils import write_json


_DEFAULT_TRAINING_STATS = {
    "iterations": 0,
    "final_gaussian_count": 0,
}


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_training_config(config: dict[str, Any]) -> dict[str, Any]:
    training_config = config.get("training", {})
    iterations = int(training_config.get("iterations", 0))
    timeout_sec = int(training_config.get("timeout_sec", 7200))
    if iterations < 0:
        raise ValueError("training.iterations must be zero or greater")
    if timeout_sec <= 0:
        raise ValueError("training.timeout_sec must be greater than 0")
    return {
        "enabled": bool(training_config.get("enabled", False)),
        "iterations": iterations,
        "timeout_sec": timeout_sec,
        "command": training_config.get("command", []),
        "working_dir": training_config.get("working_dir"),
        "model_dir_name": str(training_config.get("model_dir_name", "3dgs")),
        "expected_point_cloud": str(
            training_config.get(
                "expected_point_cloud",
                "point_cloud/iteration_{iterations}/point_cloud.ply",
            )
        ),
    }


def _normalize_command(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return command
    if command in (None, []):
        return []
    raise ValueError("training.command must be a shell-style string or a list of strings")


def _format_command(command: list[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in command]


def _command_for_summary(command: list[str]) -> str:
    return " ".join(command)


def _count_ply_vertices(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def _write_report(output_dirs: dict[str, Path], report: dict[str, Any]) -> None:
    write_json(output_dirs["debug"] / "training_report.json", report)


def run_training(output_dirs: dict[str, Path], config: dict[str, Any]) -> dict:
    training_config = _read_training_config(config)
    stats = dict(_DEFAULT_TRAINING_STATS)
    model_dir = output_dirs["intermediate"] / training_config["model_dir_name"]

    if not training_config["enabled"]:
        report = {
            "status": "skipped",
            "reason": "training.enabled is false",
            "training_ran": False,
            "requested_iterations": training_config["iterations"],
            "model_dir": _relative_repo_path(model_dir),
            "model_created": False,
            "point_cloud": None,
            **stats,
        }
        _write_report(output_dirs, report)
        return {
            "training_stats": stats,
            "warnings": ["3DGS training execution is disabled in config"],
        }

    colmap_summary_path = output_dirs["debug"] / "colmap_summary.json"
    if not colmap_summary_path.exists():
        report = {
            "status": "failed",
            "reason": "colmap_summary.json is missing; run COLMAP before training",
            "training_ran": False,
            "requested_iterations": training_config["iterations"],
            "model_dir": _relative_repo_path(model_dir),
            "model_created": False,
            "point_cloud": None,
            **stats,
        }
        _write_report(output_dirs, report)
        raise RuntimeError(report["reason"])

    colmap_summary = _read_json(colmap_summary_path)
    if colmap_summary.get("status") != "success" or not colmap_summary.get("sparse_model_created"):
        report = {
            "status": "failed",
            "reason": "successful COLMAP sparse model is required before 3DGS training",
            "training_ran": False,
            "requested_iterations": training_config["iterations"],
            "colmap_status": colmap_summary.get("status"),
            "model_dir": _relative_repo_path(model_dir),
            "model_created": False,
            "point_cloud": None,
            **stats,
        }
        _write_report(output_dirs, report)
        raise RuntimeError(report["reason"])

    command_template = _normalize_command(training_config["command"])
    if not command_template:
        report = {
            "status": "failed",
            "reason": "training.command is required when training.enabled is true",
            "training_ran": False,
            "requested_iterations": training_config["iterations"],
            "model_dir": _relative_repo_path(model_dir),
            "model_created": False,
            "point_cloud": None,
            **stats,
        }
        _write_report(output_dirs, report)
        raise RuntimeError(report["reason"])

    model_dir.mkdir(parents=True, exist_ok=True)
    sparse_model_dir = Path(str(colmap_summary["sparse_model_dir"]))
    if not sparse_model_dir.is_absolute():
        sparse_model_dir = Path.cwd() / sparse_model_dir

    values = {
        "intermediate_dir": str(output_dirs["intermediate"]),
        "images_dir": str(output_dirs["images_original"]),
        "images_masked_dir": str(output_dirs["images_masked"]),
        "sparse_model_dir": str(sparse_model_dir),
        "model_dir": str(model_dir),
        "iterations": str(training_config["iterations"]),
    }
    command = _format_command(command_template, values)
    working_dir = training_config["working_dir"]
    cwd = Path(working_dir) if working_dir else None

    report = {
        "status": "running",
        "training_ran": True,
        "requested_iterations": training_config["iterations"],
        "model_dir": _relative_repo_path(model_dir),
        "command": _command_for_summary(command),
        "working_dir": None if cwd is None else _relative_repo_path(cwd),
        "model_created": False,
        "point_cloud": None,
        **stats,
    }
    _write_report(output_dirs, report)

    try:
        result = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=training_config["timeout_sec"],
        )
        report.update(
            {
                "returncode": result.returncode,
                "stdout_tail": result.stdout.strip()[-4000:],
                "stderr_tail": result.stderr.strip()[-4000:],
            }
        )
        if result.returncode != 0:
            detail = report["stderr_tail"] or report["stdout_tail"] or "unknown training error"
            raise RuntimeError(f"3DGS training command failed: {detail}")

        expected_relative = training_config["expected_point_cloud"].format(
            iterations=training_config["iterations"]
        )
        point_cloud_path = model_dir / expected_relative
        if not point_cloud_path.exists():
            raise RuntimeError(f"Expected 3DGS point cloud was not created: {point_cloud_path}")

        gaussian_count = _count_ply_vertices(point_cloud_path)
        stats = {
            "iterations": training_config["iterations"],
            "final_gaussian_count": gaussian_count,
        }
        report.update(
            {
                "status": "success",
                "model_created": True,
                "point_cloud": _relative_repo_path(point_cloud_path),
                **stats,
            }
        )
        _write_report(output_dirs, report)
        return {"training_stats": stats, "warnings": []}
    except subprocess.TimeoutExpired as exc:
        report.update(
            {
                "status": "failed",
                "error": f"3DGS training command timed out after {training_config['timeout_sec']} seconds: {_command_for_summary(exc.cmd)}",
                "model_created": False,
            }
        )
        _write_report(output_dirs, report)
        raise RuntimeError(report["error"]) from exc
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": str(exc),
                "model_created": False,
            }
        )
        _write_report(output_dirs, report)
        raise
