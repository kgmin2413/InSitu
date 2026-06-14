# InSitu Backend

Backend pipeline for generating metric-scale furniture assets from user-captured video.

The current baseline is:

```text
InSitu-A4 3DGS -> mask-vote pruning -> ArUco metric scale -> DBSCAN cleanup -> Unity/AR alignment
```

Final backend output is an asset folder with a metric Gaussian-splat PLY for visuals and a lightweight OBJ proxy for placement/collision.

## Current Baseline

Use this path for furniture-like objects such as chairs, stools, tables, shelves, cabinets, and sofas:

```bash
OBJECT_SLUG=table \
A4_DATASET=third_party/InSitu-A4/data/table_a4pose_fgmask \
A4_MODEL=third_party/InSitu-A4/output/table_a4pose_fgmask_result \
A4_MARKER_IMAGE_DATASET=third_party/InSitu-A4/data/table_a4true \
SOURCE_VIDEO=data/input/table.mp4 \
MARKER_ID=23 \
MARKER_SIZE_M=0.12 \
CURRENT_FRONT=-Z \
TARGET_FRONT=+Z \
bash scripts/run_a4_metric_aligned_baseline.sh
```

The wrapper expects the InSitu-A4 foreground-mask dataset/model to already exist locally. Large captures, A4 workspaces, trained models, and generated outputs are intentionally ignored by git.

For the known chair baseline:

```bash
bash scripts/run_chair_a4_metric_aligned_baseline.sh
```

Exact chair output reproduced by the wrapper:

```text
output/chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611
```

## Metric Scale

Metric scale is ArUco-marker based. It is not a manual PLY resize.

For the chair run:

```text
marker_id: 23
marker_size_m: 0.12
scale_factor: 0.22297524225757243
```

The export path applies the metric factor to both Gaussian centers and 3DGS Gaussian log-radii (`scale_0..2`).

## Output Contract

A successful asset folder contains:

```text
output/<object_id>/
+-- point_cloud_metric.ply      # metric 3DGS visual asset
+-- point_cloud_metric_rgb.ply  # RGB inspection point cloud
+-- proxy_collider.obj          # lightweight placement/collision proxy
+-- metadata.json               # unit, scale, bbox, pivot, coordinate convention
+-- package_manifest.json       # client-facing asset manifest
+-- processing_log.json
+-- preview/
+-- debug/
```

Runtime asset split:

```text
point_cloud_metric.ply  # visual Gaussian splat
proxy_collider.obj      # physics / placement
metadata.json           # meter scale, pivot, bbox
```

GLB export is available as a preview path, but it does not preserve thin furniture quality as well as Gaussian-splat PLY rendering.

## Documentation

Start here:

- [docs/furniture_a4_metric_pipeline.md](docs/furniture_a4_metric_pipeline.md): generalized furniture baseline.
- [docs/chair_a4_metric_aligned_baseline.md](docs/chair_a4_metric_aligned_baseline.md): exact chair run values and reproduction notes.
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md): backend contract and project-level requirements.
- [docs/a4_chair_pipeline.md](docs/a4_chair_pipeline.md): historical chair runbook with A4-specific context.

## Key Scripts

```text
scripts/run_a4_metric_aligned_baseline.sh      # generic furniture wrapper
scripts/run_chair_a4_metric_aligned_baseline.sh
scripts/prune_gaussians_by_mask_votes.py
scripts/package_a4_metric_aruco.py
scripts/filter_metric_point_cloud_dbscan.py
scripts/align_point_cloud_for_unity.py
scripts/export_splatmesh_glb.py               # optional preview export
scripts/validate_output.py
```

## Validation

Lightweight checks:

```bash
python -m py_compile \
  scripts/package_a4_metric_aruco.py \
  scripts/prune_gaussians_by_mask_votes.py \
  scripts/filter_metric_point_cloud_dbscan.py \
  scripts/align_point_cloud_for_unity.py

bash -n scripts/run_a4_metric_aligned_baseline.sh \
  scripts/run_chair_a4_metric_aligned_baseline.sh

python scripts/validate_output.py \
  output/chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611 \
  --mode sparse_proxy
```

Full unit tests require `pytest` or `unittest` dependencies in the active Python environment.

## Legacy / Auxiliary Paths

The original scaffold still includes default-safe, COLMAP sparse-proxy, and gsplat-adapter paths. They are useful for smoke tests and package-contract validation, but the recommended reconstruction baseline is the InSitu-A4 furniture pipeline above.
