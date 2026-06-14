# InSitu Backend Project Specification

## Project
InSitu: 3D Gaussian Splatting-Based Real-Scale Furniture Deployment AR System

## Backend Role
The backend receives a user-captured furniture video and generates a Unity AR-ready 3D asset package.

## Final Goal
Given one furniture video, automatically produce:
1. A metric-scale 3DGS visual model.
2. A lightweight proxy collider mesh.
3. Metadata and processing logs.
4. A predictable output folder that Unity can load.

## Current Baseline Status

The current implemented baseline uses InSitu-A4 for 3DGS reconstruction, ArUco marker observations for metric scale, mask-vote pruning plus DBSCAN for background cleanup, and a final Unity/AR alignment step that places the bottom center at `(0,0,0)` with `Y-up` and configurable front axis.

The backend scope is asset generation and packaging. Unity/client runtime loading is intentionally outside this backend scope.

See `docs/furniture_a4_metric_pipeline.md` for the generic furniture workflow and `docs/chair_a4_metric_aligned_baseline.md` for the known-good chair reference output.

## Pipeline
Input video
→ frame extraction
→ frame selection
→ background removal / object masking
→ COLMAP camera pose estimation
→ 3D Gaussian Splatting training
→ ArUco metric scale estimation
→ metric 3DGS PLY export
→ DBSCAN foreground cluster cleanup
→ bottom-center pivot / front-axis alignment
→ proxy collider generation
→ client-ready output package

## Output Contract

output/<object_id>/
├── point_cloud_metric.ply
├── proxy_collider.obj
├── metadata.json
├── processing_log.json
├── preview/
│   ├── input_thumbnail.jpg
│   ├── mask_preview.jpg
│   └── reconstruction_preview.png
└── debug/
    ├── selected_frames.txt
    ├── colmap_summary.json
    ├── scale_report.json
    └── proxy_report.json

## Important Design Rule
Separate visual representation and physical representation.

- Visual representation:
  - point_cloud_metric.ply
  - 3D Gaussian Splatting model
  - Used for realistic rendering in Unity

- Physical representation:
  - proxy_collider.obj
  - Lightweight OBB-based mesh
  - Used for placement, collision, and interaction in Unity

## First Development Stage
Do not implement the full reconstruction system immediately.

Start with Stage 1: Baseline Stabilization.

Required first implementation:
1. auto_pipeline.py
2. pipeline_logging.py
3. config loading
4. output folder creation
5. processing_log.json generation
6. metadata.json generation
7. placeholder stage functions
8. clear TODO points for actual COLMAP, 3DGS, scale estimation, and proxy generation

## Required processing_log.json

{
  "object_id": "chair_001",
  "preset": "balanced",
  "timing_sec": {
    "frame_extraction": 0.0,
    "background_removal": 0.0,
    "colmap": 0.0,
    "training": 0.0,
    "scale_estimation": 0.0,
    "proxy_generation": 0.0,
    "total": 0.0
  },
  "frame_stats": {
    "raw_frames": 0,
    "selected_frames": 0,
    "removed_blurry": 0,
    "removed_duplicates": 0
  },
  "colmap_stats": {
    "registered_images": 0,
    "registered_ratio": 0.0,
    "reprojection_error": 0.0
  },
  "training_stats": {
    "iterations": 0,
    "final_gaussian_count": 0
  },
  "status": "success",
  "warnings": [],
  "errors": []
}

## Required metadata.json

{
  "object_id": "chair_001",
  "visual_model": "point_cloud_metric.ply",
  "proxy_model": "proxy_collider.obj",
  "unit": "meter",
  "scale_factor": 1.0,
  "scale_method": "none_or_pending",
  "bbox_size_m": null,
  "object_center_m": null,
  "proxy_center_m": null,
  "coordinate_system": "backend_defined",
  "up_axis": "Y",
  "created_at": "YYYY-MM-DD HH:MM:SS"
}

## Development Rules
- Do not modify the host environment.
- All code must run inside Docker.
- Use Python 3.10.
- Use pathlib.Path for paths.
- Use json.dump(..., indent=2) for JSON outputs.
- Use time.perf_counter() for timing.
- Keep comments concise and in English.
- Do not fake unavailable metrics.
- If a stage is not implemented yet, make it explicit and log a warning.
- Do not implement heavy 3DGS training logic in Stage 1.
- Do not implement odometry scale estimation in Stage 1.
- Do not rewrite the architecture without justification.
