"""CLI entry point for the InSitu Stage 1 backend scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap_runner import run_colmap
from frame_selection import run_frame_extraction
from generate_proxy import run_proxy_generation
from marker_detection import run_marker_detection
from pipeline_logging import PipelineLogger
from preprocess_inspyrenet import run_background_removal
from scale_estimation import run_scale_estimation
from train_3dgs import run_training
from utils import build_metadata, ensure_output_dirs, load_config, validate_object_id, write_json, write_package_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the InSitu Stage 1 backend scaffold."
    )
    parser.add_argument("--input_video", required=True, help="Path to the source video.")
    parser.add_argument(
        "--object_id",
        required=True,
        type=validate_object_id,
        help="Stable object output ID.",
    )
    parser.add_argument("--preset", default="balanced", help="Pipeline preset name.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Base output directory for generated object folders.",
    )
    return parser.parse_args()


def collect_warnings(logger: PipelineLogger, stage_name: str, result: dict) -> None:
    for warning in result.get("warnings", []):
        logger.add_warning(stage_name, warning)


def run_pipeline(args: argparse.Namespace) -> int:
    object_id = validate_object_id(args.object_id)
    input_video = Path(args.input_video)
    config_path = Path(args.config)
    base_output_dir = Path(args.output_dir)

    logger = PipelineLogger(object_id, args.preset)
    output_dirs = ensure_output_dirs(base_output_dir, object_id, create_intermediate=False)
    metadata_path = output_dirs["object"] / "metadata.json"
    log_path = output_dirs["object"] / "processing_log.json"

    try:
        config = load_config(config_path)
        output_dirs = ensure_output_dirs(base_output_dir, object_id, config)
        metadata_path = output_dirs["object"] / "metadata.json"
        log_path = output_dirs["object"] / "processing_log.json"
        metadata = build_metadata(object_id)
        write_json(metadata_path, metadata)

        if not input_video.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_video}")

        stages = (
            ("frame_extraction", run_frame_extraction),
            ("background_removal", run_background_removal),
            ("marker_detection", run_marker_detection),
            ("colmap", run_colmap),
            ("training", run_training),
            ("scale_estimation", run_scale_estimation),
            ("proxy_generation", run_proxy_generation),
        )

        for stage_name, stage_func in stages:
            with logger.stage_timer(stage_name):
                if stage_name == "frame_extraction":
                    result = stage_func(input_video, output_dirs, config)
                else:
                    result = stage_func(output_dirs, config)

            if "frame_stats" in result:
                logger.update_metrics("frame_stats", result["frame_stats"])
            if "colmap_stats" in result:
                logger.update_metrics("colmap_stats", result["colmap_stats"])
            if "training_stats" in result:
                logger.update_metrics("training_stats", result["training_stats"])
            collect_warnings(logger, stage_name, result)

        return_code = 0
    except Exception as exc:
        message = str(exc)
        if not any(error.endswith(f": {message}") for error in logger.log["errors"]):
            logger.add_error("pipeline", message)
        else:
            logger.mark_failed()
        return_code = 1
    finally:
        if not metadata_path.exists():
            write_json(metadata_path, build_metadata(object_id))
        logger.write(log_path)
        write_package_manifest(output_dirs)

    if return_code != 0:
        print(f"Pipeline failed. See log: {log_path}", file=sys.stderr)
    else:
        print(f"Pipeline scaffold completed: {output_dirs['object']}")
    return return_code


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
