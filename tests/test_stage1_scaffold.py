from __future__ import annotations

import json
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from auto_pipeline import run_pipeline
from create_unity_package import create_package
from generate_aruco_sheet import generate_sheet
from marker_detection import run_marker_detection
from run_baseline_acceptance import build_acceptance_commands
from colmap_runner import _mean_reprojection_error, _quality_warnings, _registered_image_count, _sparse_point_count, run_colmap
from generate_proxy import run_proxy_generation
from scale_estimation import _denoise_vertices, _estimate_aruco_marker_scale, _filter_vertices_to_main_object, _parse_colmap_points, _vertices_to_colors, run_scale_estimation
from train_3dgs import _count_ply_vertices, run_training
from train_gsplat_adapter import build_gsplat_command, run_gsplat_adapter, stage_gsplat_dataset
from utils import build_metadata, ensure_output_dirs, validate_object_id


EXPECTED_METADATA_KEYS = {
    "object_id",
    "visual_model",
    "proxy_model",
    "unit",
    "scale_factor",
    "scale_method",
    "bbox_size_m",
    "object_center_m",
    "proxy_center_m",
    "coordinate_system",
    "up_axis",
    "created_at",
}

EXPECTED_LOG_KEYS = {
    "object_id",
    "preset",
    "timing_sec",
    "frame_stats",
    "colmap_stats",
    "training_stats",
    "status",
    "warnings",
    "errors",
}


def make_args(input_video: Path, object_id: str, output_dir: Path, config: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        input_video=str(input_video),
        object_id=object_id,
        preset="balanced",
        config=str(config or ROOT / "configs" / "default.yaml"),
        output_dir=str(output_dir),
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_test_config(path: Path, intermediate_dir: Path) -> None:
    path.write_text(
        "preset: balanced\n"
        f"intermediate_dir: {intermediate_dir}\n"
        "frame_extraction:\n"
        "  target_fps: 2\n"
        "  max_frames: 5\n"
        "  blur_threshold: 0.0\n"
        "  duplicate_threshold: 0.0\n"
        "  output_ext: png\n"
        "background_removal:\n"
        "  backend: opencv_grabcut\n"
        "  grabcut_iterations: 1\n"
        "  rect_margin_ratio: 0.12\n"
        "marker_detection:\n"
        "  enabled: false\n"
        "colmap:\n"
        "  enabled: false\n"
        "  matcher: exhaustive\n"
        "training:\n"
        "  iterations: 0\n"
        "scale_estimation:\n"
        "  method: none_or_pending\n"
        "proxy_generation:\n"
        "  method: none_or_pending\n",
        encoding="utf-8",
    )


def create_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 72),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create test video")
    try:
        for index in range(20):
            frame = np.full((72, 96, 3), 210, dtype=np.uint8)
            cv2.rectangle(frame, (18 + index % 4, 14), (76, 58), (30, 80, 190), -1)
            cv2.circle(frame, (30 + index * 2, 36), 6, (240, 240, 30), -1)
            cv2.putText(frame, str(index), (5, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            writer.write(frame)
    finally:
        writer.release()


class Stage1ScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        run_id = f"{self._testMethodName}_{uuid4().hex}"
        self.tmp_dir = ROOT / ".test_tmp" / run_id
        self.tmp_dir.mkdir(parents=True)

    def test_metadata_schema_matches_stage1_contract(self) -> None:
        metadata = build_metadata("chair_001")

        self.assertEqual(set(metadata), EXPECTED_METADATA_KEYS)
        self.assertEqual(metadata["object_id"], "chair_001")
        self.assertEqual(metadata["visual_model"], "point_cloud_metric.ply")
        self.assertEqual(metadata["proxy_model"], "proxy_collider.obj")
        self.assertEqual(metadata["unit"], "meter")
        self.assertEqual(metadata["scale_factor"], 1.0)
        self.assertEqual(metadata["scale_method"], "none_or_pending")
        self.assertIsNone(metadata["bbox_size_m"])
        self.assertEqual(metadata["up_axis"], "Y")

    def test_pipeline_extracts_frames_and_masks_without_fake_models(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        input_video = self.tmp_dir / "input.mp4"
        create_test_video(input_video)
        output_dir = self.tmp_dir / "output"
        config_path = self.tmp_dir / "default.yaml"
        intermediate_dir = self.tmp_dir / "intermediate"
        write_test_config(config_path, intermediate_dir)

        result = run_pipeline(make_args(input_video, object_id, output_dir, config_path))

        object_dir = output_dir / object_id
        self.assertEqual(result, 0)
        self.assertTrue((object_dir / "metadata.json").exists())
        self.assertTrue((object_dir / "processing_log.json").exists())
        self.assertTrue((object_dir / "package_manifest.json").exists())
        self.assertTrue((object_dir / "package_manifest.json").exists())
        self.assertTrue((object_dir / "preview").is_dir())
        self.assertTrue((object_dir / "debug").is_dir())
        self.assertFalse((object_dir / "point_cloud_metric.ply").exists())
        self.assertFalse((object_dir / "proxy_collider.obj").exists())

        metadata = read_json(object_dir / "metadata.json")
        log = read_json(object_dir / "processing_log.json")
        manifest = read_json(object_dir / "package_manifest.json")

        self.assertEqual(set(metadata), EXPECTED_METADATA_KEYS)
        self.assertEqual(set(log), EXPECTED_LOG_KEYS)
        self.assertEqual(log["status"], "success")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["object_id"], object_id)
        self.assertTrue(manifest["unity_contract"]["visual_and_physics_are_separate"])
        self.assertEqual(log["errors"], [])
        self.assertEqual(len(log["warnings"]), 4)
        self.assertGreater(log["frame_stats"]["raw_frames"], 0)
        self.assertGreater(log["frame_stats"]["selected_frames"], 0)
        self.assertEqual(log["frame_stats"]["removed_blurry"], 0)
        self.assertEqual(log["colmap_stats"]["registered_images"], 0)
        self.assertEqual(log["training_stats"]["iterations"], 0)

        selected_frames_path = object_dir / "debug" / "selected_frames.txt"
        self.assertTrue(selected_frames_path.exists())
        selected_frame_lines = selected_frames_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(selected_frame_lines), log["frame_stats"]["selected_frames"])

        frame_report = read_json(object_dir / "debug" / "frame_selection_report.json")
        self.assertEqual(frame_report["status"], "success")
        self.assertTrue(frame_report["frame_extraction_ran"])
        self.assertTrue(frame_report["frame_selection_ran"])
        self.assertTrue((object_dir / "preview" / "input_thumbnail.jpg").exists())
        self.assertTrue((object_dir / "preview" / "selected_frames_contact_sheet.jpg").exists())
        first_selected_frame = ROOT / selected_frame_lines[0]
        self.assertTrue(first_selected_frame.exists())
        self.assertEqual(first_selected_frame.parent.name, "images_original")

        quality_report = read_json(object_dir / "debug" / "capture_quality_report.json")
        self.assertEqual(quality_report["status"], "success")
        self.assertGreater(quality_report["duration_sec"], 0)
        self.assertEqual(quality_report["selected_frames"], len(selected_frame_lines))

        mask_manifest = read_json(object_dir / "debug" / "mask_manifest.json")
        self.assertEqual(mask_manifest["status"], "success")
        self.assertTrue(mask_manifest["background_removal_ran"])
        self.assertEqual(mask_manifest["mask_images_created"], len(selected_frame_lines))
        self.assertTrue((object_dir / "preview" / "mask_preview.jpg").exists())
        first_masked_frame = ROOT / mask_manifest["frames"][0]["masked_image"]
        self.assertTrue(first_masked_frame.exists())
        self.assertEqual(first_masked_frame.parent.name, "images_masked")
        self.assertGreaterEqual(mask_manifest["frames"][0]["foreground_pixel_count"], 0)

        marker_report = read_json(object_dir / "debug" / "marker_detection_report.json")
        colmap_summary = read_json(object_dir / "debug" / "colmap_summary.json")
        training_report = read_json(object_dir / "debug" / "training_report.json")
        scale_report = read_json(object_dir / "debug" / "scale_report.json")
        proxy_report = read_json(object_dir / "debug" / "proxy_report.json")

        self.assertEqual(marker_report["status"], "skipped")
        self.assertFalse(marker_report["marker_detection_ran"])

        self.assertEqual(colmap_summary["status"], "skipped")
        self.assertFalse(colmap_summary["colmap_ran"])
        self.assertFalse(colmap_summary["sparse_model_created"])

        self.assertEqual(training_report["status"], "skipped")
        self.assertFalse(training_report["training_ran"])
        self.assertFalse(training_report["model_created"])

        self.assertEqual(scale_report["status"], "skipped")
        self.assertFalse(scale_report["scale_estimation_ran"])
        self.assertEqual(scale_report["scale_method"], "none_or_pending")

        self.assertEqual(proxy_report["status"], "skipped")
        self.assertFalse(proxy_report["proxy_generation_ran"])
        self.assertFalse(proxy_report["proxy_model_created"])

    def test_colmap_enabled_missing_binary_fails_with_summary(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        selected_frame = output_dirs["images_original"] / "000001.png"
        selected_frame.write_bytes(b"not an image for this unit test")
        (output_dirs["debug"] / "selected_frames.txt").write_text(
            f"{selected_frame}\n",
            encoding="utf-8",
        )

        with patch("colmap_runner.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                run_colmap(
                    output_dirs,
                    {
                        "colmap": {
                            "enabled": True,
                            "binary": "missing-colmap",
                            "matcher": "exhaustive",
                        }
                    },
                )

        summary = read_json(output_dirs["debug"] / "colmap_summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["colmap_ran"])
        self.assertFalse(summary["sparse_model_created"])
        self.assertIn("COLMAP binary not found", summary["reason"])

    def test_colmap_text_metric_parsers(self) -> None:
        images_txt = self.tmp_dir / "images.txt"
        images_txt.write_text(
            "# header\n"
            "1 1 0 0 0 0 0 0 1 000001.png\n"
            "0 0 -1\n"
            "2 1 0 0 0 0 0 0 1 000002.png\n"
            "0 0 -1\n",
            encoding="utf-8",
        )
        points_txt = self.tmp_dir / "points3D.txt"
        points_txt.write_text(
            "# header\n"
            "1 0 0 0 255 255 255 0.5 1 0\n"
            "2 0 0 0 255 255 255 1.5 2 0\n",
            encoding="utf-8",
        )

        self.assertEqual(_registered_image_count(images_txt), 2)
        self.assertEqual(_mean_reprojection_error(points_txt), 1.0)
        self.assertEqual(_sparse_point_count(points_txt), 2)

    def test_colmap_quality_warnings_compare_configured_thresholds(self) -> None:
        warnings = _quality_warnings(
            {
                "registered_images": 4,
                "registered_ratio": 0.2,
                "reprojection_error": 2.5,
                "sparse_points": 50,
            },
            {
                "min_registered_images": 8,
                "min_registered_ratio": 0.3,
                "min_sparse_points": 100,
                "max_reprojection_error": 2.0,
                "fail_on_low_quality": False,
            },
        )

        self.assertEqual(len(warnings), 4)
        self.assertIn("registered image count", warnings[0])
        self.assertIn("registered image ratio", warnings[1])
        self.assertIn("sparse point count", warnings[2])
        self.assertIn("reprojection error", warnings[3])

    def test_training_enabled_requires_command(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        sparse_dir = output_dirs["intermediate"] / "colmap" / "run_test" / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        (output_dirs["debug"] / "colmap_summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "sparse_model_created": True,
                    "sparse_model_dir": str(sparse_dir),
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            run_training(
                output_dirs,
                {
                    "training": {
                        "enabled": True,
                        "iterations": 3,
                    }
                },
            )

        report = read_json(output_dirs["debug"] / "training_report.json")
        self.assertEqual(report["status"], "failed")
        self.assertIn("training.command is required", report["reason"])

    def test_training_wrapper_success_validates_point_cloud_artifact(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        sparse_dir = output_dirs["intermediate"] / "colmap" / "run_test" / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        (output_dirs["debug"] / "colmap_summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "sparse_model_created": True,
                    "sparse_model_dir": str(sparse_dir),
                }
            ),
            encoding="utf-8",
        )
        script = self.tmp_dir / "write_point_cloud.py"
        script.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "model_dir = Path(sys.argv[1])\n"
            "iterations = sys.argv[2]\n"
            "path = model_dir / 'point_cloud' / f'iteration_{iterations}' / 'point_cloud.ply'\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "path.write_text('ply\\nformat ascii 1.0\\nelement vertex 2\\nproperty float x\\nproperty float y\\nproperty float z\\nend_header\\n0 0 0\\n1 1 1\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = run_training(
            output_dirs,
            {
                "training": {
                    "enabled": True,
                    "iterations": 3,
                    "command": ["python", str(script), "{model_dir}", "{iterations}"],
                    "timeout_sec": 30,
                }
            },
        )

        report = read_json(output_dirs["debug"] / "training_report.json")
        self.assertEqual(report["status"], "success")
        self.assertTrue(report["model_created"])
        self.assertEqual(result["training_stats"]["iterations"], 3)
        self.assertEqual(result["training_stats"]["final_gaussian_count"], 2)
        self.assertEqual(_count_ply_vertices(Path(report["point_cloud"])), 2)

    def test_manual_scale_exports_metric_point_cloud_and_updates_metadata(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        metadata_path = output_dirs["object"] / "metadata.json"
        metadata_path.write_text(json.dumps(build_metadata(object_id)), encoding="utf-8")
        source_cloud = self.tmp_dir / "source.ply"
        source_cloud.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "0 0 0\n"
            "1 2 3\n",
            encoding="utf-8",
        )
        (output_dirs["debug"] / "training_report.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "model_created": True,
                    "point_cloud": str(source_cloud),
                }
            ),
            encoding="utf-8",
        )

        result = run_scale_estimation(
            output_dirs,
            {
                "scale_estimation": {
                    "method": "manual",
                    "scale_factor": 2.0,
                }
            },
        )

        metric_cloud = output_dirs["object"] / "point_cloud_metric.ply"
        report = read_json(output_dirs["debug"] / "scale_report.json")
        metadata = read_json(metadata_path)
        self.assertTrue(metric_cloud.exists())
        self.assertTrue((output_dirs["preview"] / "reconstruction_preview.png").exists())
        self.assertEqual(report["status"], "success")
        self.assertTrue(report["point_cloud_metric_created"])
        self.assertEqual(report["bbox_size_m"], [2.0, 4.0, 6.0])
        self.assertEqual(report["object_center_m"], [1.0, 2.0, 3.0])
        self.assertEqual(result["metrics"]["scale_factor"], 2.0)
        self.assertEqual(metadata["scale_factor"], 2.0)
        self.assertEqual(metadata["scale_method"], "manual")
        self.assertEqual(metadata["bbox_size_m"], [2.0, 4.0, 6.0])
        self.assertEqual(metadata["object_center_m"], [1.0, 2.0, 3.0])


    def test_object_filter_keeps_largest_metric_cluster(self) -> None:
        vertices = np.array(
            [
                (0.00, 0.00, 0.00),
                (0.02, 0.00, 0.00),
                (0.00, 0.02, 0.00),
                (0.02, 0.02, 0.00),
                (1.00, 1.00, 1.00),
                (1.02, 1.00, 1.00),
                (1.00, 1.02, 1.00),
            ],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )

        filtered, report = _filter_vertices_to_main_object(
            vertices,
            {
                "enabled": True,
                "percentile_enabled": False,
                "percentile_lower": 1.0,
                "percentile_upper": 99.0,
                "cluster_enabled": True,
                "cluster_voxel_size_m": 0.01,
                "cluster_eps_m": 0.05,
                "cluster_min_samples": 2,
                "cluster_min_keep_ratio": 0.1,
            },
        )

        self.assertEqual(len(filtered), 4)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["cluster"]["status"], "success")
        self.assertEqual(report["cluster"]["removed_points"], 3)
        self.assertTrue(np.all(_vertices_to_colors(filtered).shape == (4, 3)))

    def test_manual_scale_exports_colmap_sparse_points(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        metadata_path = output_dirs["object"] / "metadata.json"
        metadata_path.write_text(json.dumps(build_metadata(object_id)), encoding="utf-8")
        text_model_dir = self.tmp_dir / "sparse_txt"
        text_model_dir.mkdir()
        points_txt = text_model_dir / "points3D.txt"
        points_txt.write_text(
            "# header\n"
            "1 0 0 0 255 0 0 0.25 1 1\n"
            "2 1 2 3 0 255 0 0.50 1 2\n"
            "3 5 5 5 0 0 255 2.00 1 3\n",
            encoding="utf-8",
        )
        (output_dirs["debug"] / "colmap_summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "sparse_model_created": True,
                    "text_model_dir": str(text_model_dir),
                }
            ),
            encoding="utf-8",
        )

        parsed = _parse_colmap_points(points_txt, 2.0, 1.0)
        self.assertEqual(len(parsed), 2)
        result = run_scale_estimation(
            output_dirs,
            {
                "scale_estimation": {
                    "method": "manual",
                    "source": "colmap_sparse",
                    "scale_factor": 2.0,
                    "max_reprojection_error": 1.0,
                }
            },
        )

        metric_cloud = output_dirs["object"] / "point_cloud_metric.ply"
        report = read_json(output_dirs["debug"] / "scale_report.json")
        metadata = read_json(metadata_path)
        self.assertTrue(metric_cloud.exists())
        self.assertTrue((output_dirs["preview"] / "reconstruction_preview.png").exists())
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["source_kind"], "colmap_sparse")
        self.assertEqual(report["point_count"], 2)
        self.assertEqual(report["bbox_size_m"], [2.0, 4.0, 6.0])
        self.assertEqual(report["object_center_m"], [1.0, 2.0, 3.0])
        self.assertEqual(result["metrics"]["scale_factor"], 2.0)
        self.assertEqual(metadata["bbox_size_m"], [2.0, 4.0, 6.0])

    def test_denoise_vertices_filters_low_opacity_and_isolated_points(self) -> None:
        vertices = np.array(
            [
                (0.00, 0.00, 0.00, 2.0),
                (0.01, 0.00, 0.00, 2.0),
                (0.00, 0.01, 0.00, 2.0),
                (0.00, 0.00, 0.01, 2.0),
                (0.01, 0.01, 0.00, 2.0),
                (0.00, 0.01, 0.01, 2.0),
                (8.00, 8.00, 8.00, 2.0),
                (0.02, 0.00, 0.00, -5.0),
            ],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")],
        )

        filtered, report = _denoise_vertices(
            vertices,
            {
                "enabled": True,
                "opacity_threshold": 0.05,
                "statistical_enabled": True,
                "statistical_neighbors": 2,
                "statistical_std_ratio": 1.0,
            },
        )

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["after_opacity_points"], 7)
        self.assertEqual(report["kept_points"], 6)
        self.assertEqual(len(filtered), 6)
        self.assertLess(float(filtered["x"].max()), 1.0)

    def test_vertices_to_colors_uses_gaussian_dc_features(self) -> None:
        vertices = np.zeros(1, dtype=[("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")])
        colors = _vertices_to_colors(vertices)

        self.assertEqual(colors.shape, (1, 3))
        self.assertEqual(colors[0].tolist(), [127, 127, 127])

    def test_aruco_marker_scale_estimation_from_colmap_camera_poses(self) -> None:
        text_model_dir = self.tmp_dir / "sparse_txt"
        text_model_dir.mkdir()
        marker_report_path = self.tmp_dir / "marker_detection_report.json"
        f, cx, cy = 1000.0, 500.0, 500.0
        corners_world = np.array(
            [
                [-1.0, -1.0, 5.0],
                [1.0, -1.0, 5.0],
                [1.0, 1.0, 5.0],
                [-1.0, 1.0, 5.0],
            ],
            dtype=float,
        )

        def project(point: np.ndarray, camera_center_x: float) -> list[float]:
            camera_point = point - np.array([camera_center_x, 0.0, 0.0])
            return [float(f * camera_point[0] / camera_point[2] + cx), float(f * camera_point[1] / camera_point[2] + cy)]

        (text_model_dir / "cameras.txt").write_text(
            "# cameras\n"
            "1 PINHOLE 1000 1000 1000 1000 500 500\n",
            encoding="utf-8",
        )
        (text_model_dir / "images.txt").write_text(
            "# images\n"
            "1 1 0 0 0 0 0 0 1 000001.png\n"
            "0 0 -1\n"
            "2 1 0 0 0 -1 0 0 1 000002.png\n"
            "0 0 -1\n",
            encoding="utf-8",
        )
        marker_report_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "detections_found": True,
                    "marker_size_m": 0.12,
                    "observations": [
                        {
                            "image": "data/intermediate/test/images_original/000001.png",
                            "marker_id": 7,
                            "corners_px": [project(point, 0.0) for point in corners_world],
                        },
                        {
                            "image": "data/intermediate/test/images_original/000002.png",
                            "marker_id": 7,
                            "corners_px": [project(point, 1.0) for point in corners_world],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        estimate = _estimate_aruco_marker_scale(
            marker_report_path,
            text_model_dir,
            marker_id=7,
            marker_size_m=0.12,
            min_observations=2,
        )

        self.assertEqual(estimate["marker_id"], 7)
        self.assertEqual(estimate["observations_used"], 2)
        self.assertEqual(estimate["rays_per_corner"], [2, 2, 2, 2])
        self.assertAlmostEqual(estimate["reconstructed_marker_size_colmap"], 2.0, places=5)
        self.assertAlmostEqual(estimate["scale_factor"], 0.06, places=6)

    def test_obb_proxy_generation_from_metric_point_cloud_updates_metadata(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        metadata_path = output_dirs["object"] / "metadata.json"
        metadata_path.write_text(json.dumps(build_metadata(object_id)), encoding="utf-8")
        metric_cloud = output_dirs["object"] / "point_cloud_metric.ply"
        metric_cloud.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 8\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "0 0 0\n"
            "2 0 0\n"
            "0 4 0\n"
            "2 4 0\n"
            "0 0 6\n"
            "2 0 6\n"
            "0 4 6\n"
            "2 4 6\n",
            encoding="utf-8",
        )

        result = run_proxy_generation(
            output_dirs,
            {
                "proxy_generation": {
                    "method": "obb",
                }
            },
        )

        proxy_path = output_dirs["object"] / "proxy_collider.obj"
        report = read_json(output_dirs["debug"] / "proxy_report.json")
        metadata = read_json(metadata_path)
        self.assertTrue(proxy_path.exists())
        self.assertEqual(report["status"], "success")
        self.assertTrue(report["proxy_model_created"])
        self.assertEqual(report["point_count"], 8)
        self.assertTrue(result["metrics"]["proxy_model_created"])
        self.assertEqual(metadata["proxy_model"], "proxy_collider.obj")
        self.assertIsNotNone(metadata["proxy_center_m"])
        obj_lines = proxy_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len([line for line in obj_lines if line.startswith("v ")]), 8)
        self.assertEqual(len([line for line in obj_lines if line.startswith("f ")]), 6)

    def test_proxy_generation_filters_percentile_outlier_before_obb(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        metadata_path = output_dirs["object"] / "metadata.json"
        metadata_path.write_text(json.dumps(build_metadata(object_id)), encoding="utf-8")
        metric_cloud = output_dirs["object"] / "point_cloud_metric.ply"
        metric_cloud.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 9\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "0 0 0\n"
            "2 0 0\n"
            "0 4 0\n"
            "2 4 0\n"
            "0 0 6\n"
            "2 0 6\n"
            "0 4 6\n"
            "2 4 6\n"
            "100 100 100\n",
            encoding="utf-8",
        )

        run_proxy_generation(
            output_dirs,
            {
                "proxy_generation": {
                    "method": "obb",
                    "percentile_filter": {
                        "enabled": True,
                        "lower": 0.0,
                        "upper": 95.0,
                    },
                }
            },
        )

        report = read_json(output_dirs["debug"] / "proxy_report.json")
        self.assertEqual(report["input_point_count"], 9)
        self.assertEqual(report["point_count"], 8)
        self.assertEqual(report["preprocessing"][0]["status"], "success")
        self.assertEqual(report["preprocessing"][0]["removed_points"], 1)

    def test_generate_aruco_sheet_writes_printable_a4_marker(self) -> None:
        output = self.tmp_dir / "marker.png"
        metadata_path = self.tmp_dir / "marker.json"

        metadata = generate_sheet(
            output,
            metadata_path,
            "DICT_4X4_50",
            7,
            100.0,
            150,
        )

        self.assertTrue(output.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(metadata["page"], "A4")
        self.assertEqual(metadata["marker_id"], 7)
        self.assertEqual(metadata["marker_size_m"], 0.1)
        from PIL import Image

        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (1240, 1754))

    def test_marker_detection_finds_generated_sheet_marker(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        output_dirs = ensure_output_dirs(
            self.tmp_dir / "output",
            object_id,
            {"intermediate_dir": str(self.tmp_dir / "intermediate")},
        )
        marker_image = output_dirs["images_original"] / "000001.png"
        metadata_path = self.tmp_dir / "marker.json"
        generate_sheet(
            marker_image,
            metadata_path,
            "DICT_4X4_50",
            7,
            100.0,
            150,
        )

        result = run_marker_detection(
            output_dirs,
            {
                "marker_detection": {
                    "enabled": True,
                    "dictionary": "DICT_4X4_50",
                    "marker_ids": [7],
                    "marker_size_m": 0.1,
                    "max_images": 5,
                }
            },
        )

        report = read_json(output_dirs["debug"] / "marker_detection_report.json")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(report["status"], "success")
        self.assertTrue(report["detections_found"])
        self.assertEqual(report["marker_counts"], {"7": 1})
        self.assertEqual(report["observations"][0]["marker_size_m"], 0.1)
        self.assertGreater(report["observations"][0]["pixels_per_meter"], 0)
        self.assertTrue((output_dirs["preview"] / "aruco_detection_preview.jpg").exists())

    def test_create_unity_package_zip_from_default_safe_output(self) -> None:
        object_id = f"chair_{uuid4().hex[:8]}"
        object_dir = self.tmp_dir / "output" / object_id
        preview_dir = object_dir / "preview"
        debug_dir = object_dir / "debug"
        preview_dir.mkdir(parents=True)
        debug_dir.mkdir(parents=True)
        (preview_dir / "input_thumbnail.jpg").write_bytes(b"thumbnail")
        (preview_dir / "mask_preview.jpg").write_bytes(b"mask")
        (preview_dir / "selected_frames_contact_sheet.jpg").write_bytes(b"sheet")
        for report_name in [
            "frame_selection_report.json",
            "capture_quality_report.json",
            "mask_manifest.json",
            "marker_detection_report.json",
            "colmap_summary.json",
            "training_report.json",
            "scale_report.json",
            "proxy_report.json",
        ]:
            (debug_dir / report_name).write_text(json.dumps({"status": "skipped"}), encoding="utf-8")
        (debug_dir / "selected_frames.txt").write_text("", encoding="utf-8")
        metadata = build_metadata(object_id)
        (object_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        log = {
            "object_id": object_id,
            "preset": "balanced",
            "timing_sec": {
                "frame_extraction": 0.0,
                "background_removal": 0.0,
                "marker_detection": 0.0,
                "colmap": 0.0,
                "training": 0.0,
                "scale_estimation": 0.0,
                "proxy_generation": 0.0,
                "total": 0.0,
            },
            "frame_stats": {
                "raw_frames": 1,
                "selected_frames": 1,
                "removed_blurry": 0,
                "removed_duplicates": 0,
            },
            "colmap_stats": {
                "registered_images": 0,
                "registered_ratio": 0.0,
                "reprojection_error": 0.0,
            },
            "training_stats": {
                "iterations": 0,
                "final_gaussian_count": 0,
            },
            "status": "success",
            "warnings": [],
            "errors": [],
        }
        (object_dir / "processing_log.json").write_text(json.dumps(log), encoding="utf-8")
        from utils import write_package_manifest

        write_package_manifest(
            {
                "object": object_dir,
                "preview": preview_dir,
                "debug": debug_dir,
            }
        )

        package_path = object_dir / "package.zip"
        create_package(object_dir, package_path, "default_safe")

        self.assertTrue(package_path.exists())
        with zipfile.ZipFile(package_path) as zf:
            names = set(zf.namelist())
        self.assertIn("metadata.json", names)
        self.assertIn("processing_log.json", names)
        self.assertIn("package_manifest.json", names)
        self.assertIn("debug/unity_package_summary.json", names)



    def test_gsplat_adapter_stages_colmap_dataset_and_normalizes_ply(self) -> None:
        images_dir = self.tmp_dir / "images_original"
        sparse_model_dir = self.tmp_dir / "sparse" / "0"
        images_dir.mkdir(parents=True)
        sparse_model_dir.mkdir(parents=True)
        (images_dir / "000001.png").write_bytes(b"fake image")
        for name in ["cameras.bin", "images.bin", "points3D.bin"]:
            (sparse_model_dir / name).write_bytes(b"fake colmap binary")

        trainer = self.tmp_dir / "simple_trainer.py"
        trainer.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "result_dir = Path(args[args.index('--result-dir') + 1])\n"
            "max_steps = args[args.index('--max-steps') + 1]\n"
            "ply = result_dir / 'ply' / f'point_cloud_{max_steps}.ply'\n"
            "ply.parent.mkdir(parents=True, exist_ok=True)\n"
            "ply.write_text('ply\\nformat ascii 1.0\\nelement vertex 1\\nproperty float x\\nproperty float y\\nproperty float z\\nend_header\\n0 0 0\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )

        report = run_gsplat_adapter(
            images_dir=images_dir,
            sparse_model_dir=sparse_model_dir,
            model_dir=self.tmp_dir / "model",
            iterations=3,
            trainer_path=trainer,
            link_mode="copy",
        )

        expected_cloud = self.tmp_dir / "model" / "point_cloud" / "iteration_3" / "point_cloud.ply"
        adapter_report = read_json(self.tmp_dir / "model" / "gsplat_adapter_report.json")
        self.assertEqual(report["status"], "success")
        self.assertTrue(expected_cloud.exists())
        self.assertEqual(_count_ply_vertices(expected_cloud), 1)
        self.assertEqual(adapter_report["status"], "success")
        self.assertEqual(adapter_report["staged_dataset"]["images_stage_mode"], "copy")


    def test_gsplat_adapter_can_stage_masked_images_with_flat_background(self) -> None:
        images_dir = self.tmp_dir / "images_original"
        masked_dir = self.tmp_dir / "images_masked"
        sparse_model_dir = self.tmp_dir / "sparse" / "0"
        images_dir.mkdir(parents=True)
        masked_dir.mkdir(parents=True)
        sparse_model_dir.mkdir(parents=True)
        (images_dir / "000001.png").write_bytes(b"unused original")
        for name in ["cameras.bin", "images.bin", "points3D.bin"]:
            (sparse_model_dir / name).write_bytes(b"fake colmap binary")
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[:, :, :3] = [10, 20, 30]
        rgba[0, 0, 3] = 255
        cv2.imwrite(str(masked_dir / "000001.png"), rgba)

        report = stage_gsplat_dataset(
            images_dir,
            sparse_model_dir,
            self.tmp_dir / "dataset",
            "copy",
            masked_images_dir=masked_dir,
            mask_background_color="255,255,255",
        )

        staged = cv2.imread(str(self.tmp_dir / "dataset" / "images" / "000001.png"), cv2.IMREAD_COLOR)
        self.assertEqual(report["images_stage_mode"], "masked_composite")
        self.assertEqual(report["masked_images_staged"], 1)
        self.assertEqual(staged[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(staged[1, 1].tolist(), [255, 255, 255])

    def test_gsplat_command_uses_simple_trainer_colmap_arguments(self) -> None:
        command = build_gsplat_command(
            "python",
            Path("third_party/gsplat/examples/simple_trainer.py"),
            10,
            2,
            Path("dataset"),
            Path("results"),
            ["--random-bkgd"],
        )

        self.assertIn("default", command)
        self.assertIn("--data-dir", command)
        self.assertTrue(any(value.endswith("/dataset") or value == "dataset" for value in command))
        self.assertIn("--result-dir", command)
        self.assertTrue(any(value.endswith("/results") or value == "results" for value in command))
        self.assertIn("--max-steps", command)
        self.assertIn("10", command)
        self.assertIn("--save-ply", command)
        self.assertIn("--random-bkgd", command)

    def test_baseline_acceptance_command_plan_includes_core_gates(self) -> None:
        commands = build_acceptance_commands("test_run", ROOT / "output", skip_sparse=False)
        names = [command.name for command in commands]

        self.assertEqual(names[0], "py_compile")
        self.assertIn("unit_tests", names)
        self.assertIn("generate_aruco_sheet", names)
        self.assertIn("detect_aruco_sheet", names)
        self.assertIn("default_pipeline", names)
        self.assertIn("validate_default_safe", names)
        self.assertIn("sparse_proxy_pipeline", names)
        self.assertIn("validate_sparse_proxy", names)
        self.assertIn("create_sparse_unity_package", names)

    def test_baseline_acceptance_command_plan_can_skip_sparse_smoke(self) -> None:
        commands = build_acceptance_commands("test_run", ROOT / "output", skip_sparse=True)
        names = [command.name for command in commands]

        self.assertIn("default_pipeline", names)
        self.assertNotIn("sparse_proxy_pipeline", names)
        self.assertNotIn("create_sparse_unity_package", names)

    def test_baseline_acceptance_command_plan_can_include_gsplat_smoke(self) -> None:
        commands = build_acceptance_commands(
            "test_run",
            ROOT / "output",
            skip_sparse=True,
            include_gsplat_smoke=True,
        )
        names = [command.name for command in commands]

        self.assertIn("gsplat_smoke_pipeline", names)
        self.assertIn("validate_gsplat_smoke", names)
        self.assertIn("create_gsplat_unity_package", names)

    def test_missing_video_fails_but_writes_processing_log_and_metadata(self) -> None:
        output_dir = self.tmp_dir / "output"
        missing_video = self.tmp_dir / "missing.mp4"

        result = run_pipeline(make_args(missing_video, "chair_001", output_dir))

        object_dir = output_dir / "chair_001"
        log_path = object_dir / "processing_log.json"
        metadata_path = object_dir / "metadata.json"
        manifest_path = object_dir / "package_manifest.json"
        self.assertEqual(result, 1)
        self.assertTrue(log_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue((object_dir / "preview").is_dir())
        self.assertTrue((object_dir / "debug").is_dir())

        log = read_json(log_path)
        manifest = read_json(manifest_path)
        self.assertEqual(log["status"], "failed")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(log["warnings"], [])
        self.assertIn("Input video does not exist", log["errors"][0])

    def test_validate_object_id_accepts_safe_values(self) -> None:
        for object_id in ["chair_001", "chair-001", "A1", "object_2026"]:
            with self.subTest(object_id=object_id):
                self.assertEqual(validate_object_id(object_id), object_id)

    def test_validate_object_id_rejects_unsafe_values(self) -> None:
        unsafe_values = [
            "",
            "../escape",
            "bad/id",
            "bad id",
            ".hidden",
            "-dash",
            "under/slash",
        ]
        for object_id in unsafe_values:
            with self.subTest(object_id=object_id):
                with self.assertRaises(ValueError):
                    validate_object_id(object_id)


if __name__ == "__main__":
    unittest.main()
