#!/usr/bin/env python3
"""Run the reproducible baseline acceptance check for the current repository."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AcceptanceCommand:
    name: str
    argv: list[str]


def _repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _python_files_for_compile() -> list[str]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "src").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").glob("test_*.py")))
    return [_repo_path(path) for path in paths]


def build_acceptance_commands(
    run_id: str,
    output_dir: Path,
    skip_sparse: bool,
    include_gsplat_smoke: bool = False,
) -> list[AcceptanceCommand]:
    acceptance_dir = output_dir / f"acceptance_{run_id}"
    marker_png = acceptance_dir / "markers" / "insitu_aruco_a4.png"
    marker_json = acceptance_dir / "markers" / "insitu_aruco_a4.json"
    marker_report = acceptance_dir / "aruco_sheet_detection.json"
    marker_preview = acceptance_dir / "aruco_sheet_detection_preview.jpg"
    default_object_id = f"acceptance_default_{run_id}"
    sparse_object_id = f"acceptance_sparse_{run_id}"
    gsplat_object_id = f"acceptance_gsplat_{run_id}"
    package_path = output_dir / sparse_object_id / f"{sparse_object_id}_unity_package.zip"
    gsplat_package_path = output_dir / gsplat_object_id / f"{gsplat_object_id}_unity_package.zip"

    commands = [
        AcceptanceCommand(
            "py_compile",
            [sys.executable, "-m", "py_compile", *_python_files_for_compile()],
        ),
        AcceptanceCommand(
            "unit_tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        ),
        AcceptanceCommand(
            "generate_aruco_sheet",
            [
                sys.executable,
                "scripts/generate_aruco_sheet.py",
                "--output",
                _repo_path(marker_png),
                "--metadata",
                _repo_path(marker_json),
            ],
        ),
        AcceptanceCommand(
            "detect_aruco_sheet",
            [
                sys.executable,
                "scripts/detect_aruco_marker.py",
                _repo_path(marker_png),
                "--output_json",
                _repo_path(marker_report),
                "--preview",
                _repo_path(marker_preview),
                "--dictionary",
                "DICT_4X4_50",
                "--marker_id",
                "23",
                "--marker_size_m",
                "0.12",
            ],
        ),
        AcceptanceCommand(
            "default_pipeline",
            [
                sys.executable,
                "src/auto_pipeline.py",
                "--input_video",
                "data/input/input.mp4",
                "--object_id",
                default_object_id,
                "--preset",
                "fast",
                "--config",
                "configs/default.yaml",
                "--output_dir",
                _repo_path(output_dir),
            ],
        ),
        AcceptanceCommand(
            "validate_default_safe",
            [
                sys.executable,
                "scripts/validate_output.py",
                _repo_path(output_dir / default_object_id),
                "--mode",
                "default_safe",
            ],
        ),
    ]

    if not skip_sparse:
        commands.extend(
            [
                AcceptanceCommand(
                    "sparse_proxy_pipeline",
                    [
                        sys.executable,
                        "src/auto_pipeline.py",
                        "--input_video",
                        "data/input/input.mp4",
                        "--object_id",
                        sparse_object_id,
                        "--preset",
                        "fast",
                        "--config",
                        "configs/colmap_sparse_proxy.yaml",
                        "--output_dir",
                        _repo_path(output_dir),
                    ],
                ),
                AcceptanceCommand(
                    "validate_sparse_proxy",
                    [
                        sys.executable,
                        "scripts/validate_output.py",
                        _repo_path(output_dir / sparse_object_id),
                        "--mode",
                        "sparse_proxy",
                    ],
                ),
                AcceptanceCommand(
                    "create_sparse_unity_package",
                    [
                        sys.executable,
                        "scripts/create_unity_package.py",
                        _repo_path(output_dir / sparse_object_id),
                        "--mode",
                        "sparse_proxy",
                        "--package_path",
                        _repo_path(package_path),
                    ],
                ),
            ]
        )


    if include_gsplat_smoke:
        commands.extend(
            [
                AcceptanceCommand(
                    "gsplat_smoke_pipeline",
                    [
                        sys.executable,
                        "src/auto_pipeline.py",
                        "--input_video",
                        "data/input/input.mp4",
                        "--object_id",
                        gsplat_object_id,
                        "--preset",
                        "gsplat_smoke",
                        "--config",
                        "configs/gsplat_smoke.yaml",
                        "--output_dir",
                        _repo_path(output_dir),
                    ],
                ),
                AcceptanceCommand(
                    "validate_gsplat_smoke",
                    [
                        sys.executable,
                        "scripts/validate_output.py",
                        _repo_path(output_dir / gsplat_object_id),
                        "--mode",
                        "sparse_proxy",
                    ],
                ),
                AcceptanceCommand(
                    "create_gsplat_unity_package",
                    [
                        sys.executable,
                        "scripts/create_unity_package.py",
                        _repo_path(output_dir / gsplat_object_id),
                        "--mode",
                        "sparse_proxy",
                        "--package_path",
                        _repo_path(gsplat_package_path),
                    ],
                ),
            ]
        )
    return commands


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_command(command: AcceptanceCommand) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command.argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished = datetime.now(timezone.utc).isoformat()
    return {
        "name": command.name,
        "argv": command.argv,
        "started_at": started,
        "finished_at": finished,
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run InSitu baseline acceptance checks.")
    parser.add_argument("--run_id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--skip_sparse", action="store_true", help="Skip COLMAP sparse proxy smoke validation.")
    parser.add_argument(
        "--include_gsplat_smoke",
        action="store_true",
        help="Also run the gsplat smoke trainer path. Requires gsplat setup in this environment.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    summary_path = output_dir / f"acceptance_{args.run_id}" / "debug" / "baseline_acceptance_summary.json"
    commands = build_acceptance_commands(args.run_id, output_dir, args.skip_sparse, args.include_gsplat_smoke)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "repository": str(ROOT),
        "output_dir": _repo_path(output_dir),
        "skip_sparse": bool(args.skip_sparse),
        "include_gsplat_smoke": bool(args.include_gsplat_smoke),
        "commands": [],
    }

    for command in commands:
        print(f"RUN {command.name}: {' '.join(command.argv)}", flush=True)
        result = run_command(command)
        summary["commands"].append(result)
        write_summary(summary_path, summary)
        if result["returncode"] != 0:
            summary["status"] = "failed"
            summary["failed_command"] = command.name
            write_summary(summary_path, summary)
            print(f"FAILED {command.name}; see {summary_path}", file=sys.stderr)
            return result["returncode"] or 1
        print(f"OK {command.name}", flush=True)

    summary["status"] = "success"
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(summary_path, summary)
    print(f"OK: baseline acceptance passed; summary written to {_repo_path(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
