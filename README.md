# InSitu Backend Pipeline

Default safe run:

```bash
python src/auto_pipeline.py \
  --input_video data/input/input.mp4 \
  --object_id chair_001 \
  --preset fast \
  --config configs/default.yaml \
  --output_dir output
```

Implemented stages:

- OpenCV video frame extraction.
- Deterministic blur and duplicate frame selection.
- `preview/input_thumbnail.jpg` generation.
- Capture quality report and `preview/selected_frames_contact_sheet.jpg` generation.
- OpenCV GrabCut masking into `data/intermediate/<object_id>/images_masked/`.
- Optional ArUco scale-marker detection with `debug/marker_detection_report.json` and `preview/aruco_detection_preview.jpg`.
- Controlled COLMAP wrapper with sparse-point metrics and configurable quality warnings/gates, disabled by default.
- Controlled 3DGS training command wrapper, disabled by default.
- Manual-scale metric point cloud export from either a 3DGS training point cloud or COLMAP sparse points.
- `preview/reconstruction_preview.png` generation from metric point clouds.
- PCA OBB proxy collider OBJ generation from `point_cloud_metric.ply`, with optional point-cloud cleanup before collider fitting.
- Unity-facing `package_manifest.json` generation.
- Output contract validation via `scripts/validate_output.py`.
- Unity package zip creation via `scripts/create_unity_package.py`.
- Printable A4 ArUco marker sheet generation via `scripts/generate_aruco_sheet.py`.

Default output includes `metadata.json`, `processing_log.json`, `package_manifest.json`,
`preview/`, and `debug/`. The default config does not create `point_cloud_metric.ply` or
`proxy_collider.obj`, because COLMAP, training, scale export, and proxy export
are disabled until their prerequisites are explicitly configured.

COLMAP sparse proxy smoke run:

```bash
python src/auto_pipeline.py \
  --input_video data/input/input.mp4 \
  --object_id sparse_proxy_001 \
  --preset fast \
  --config configs/colmap_sparse_proxy.yaml \
  --output_dir output
```

This smoke config creates `point_cloud_metric.ply`,
`preview/reconstruction_preview.png`, and `proxy_collider.obj` from real COLMAP
sparse points. It is useful for validating the Unity package shape before the
real 3DGS trainer command is configured. The visual model is still a sparse
COLMAP point cloud in this mode, not a trained 3DGS model.

Scale marker tools:

```bash
python scripts/generate_aruco_sheet.py \
  --output data/markers/insitu_aruco_a4.png \
  --metadata data/markers/insitu_aruco_a4.json

python scripts/detect_aruco_marker.py data/markers/insitu_aruco_a4.png \
  --output_json output/aruco_sheet_detection.json \
  --preview output/aruco_sheet_detection_preview.jpg \
  --dictionary DICT_4X4_50 \
  --marker_id 23 \
  --marker_size_m 0.12
```

The marker detector is available as an optional pipeline stage. Marker observations
can also drive ArUco-based metric scaling when `scale_estimation.method` is
`aruco_marker`.

Chair marker gsplat run:

```bash
python src/auto_pipeline.py \
  --input_video data/input/chair.mp4 \
  --object_id chair_marker_gsplat_filtered_001 \
  --preset chair_marker_gsplat \
  --config configs/chair_marker_gsplat_training.yaml \
  --output_dir output

python scripts/validate_output.py output/chair_marker_gsplat_filtered_001 --mode sparse_proxy
python scripts/create_unity_package.py output/chair_marker_gsplat_filtered_001 \
  --mode sparse_proxy \
  --package_path output/chair_marker_gsplat_filtered_001/chair_marker_gsplat_filtered_unity_package.zip
```

This path detects the printed ArUco marker, estimates metric scale from COLMAP
camera poses and marker corners, trains gsplat, filters the exported point cloud
through foreground masks, generates an OBB proxy, and packages the result.

gsplat trainer integration:

```bash
# one-time inside the container
pip install gsplat
git clone https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
pip install -r third_party/gsplat/examples/requirements.txt --no-build-isolation

# quick integration smoke after setup
python src/auto_pipeline.py \
  --input_video data/input/input.mp4 \
  --object_id gsplat_smoke_001 \
  --preset gsplat_smoke \
  --config configs/gsplat_smoke.yaml \
  --output_dir output

# longer gsplat path after setup
python src/auto_pipeline.py \
  --input_video data/input/input.mp4 \
  --object_id gsplat_001 \
  --preset gsplat \
  --config configs/gsplat_training.yaml \
  --output_dir output
```

`configs/gsplat_training.yaml` uses `scripts/train_gsplat_adapter.py` to stage the
COLMAP output as a gsplat COLMAP dataset, run `examples/simple_trainer.py`, and
copy the exported PLY to the pipeline's expected `point_cloud/iteration_<N>/point_cloud.ply`.

Baseline acceptance command:

```bash
python scripts/run_baseline_acceptance.py
python scripts/run_baseline_acceptance.py --include_gsplat_smoke
```

This runs compile checks, unit tests, marker sheet generation/detection, default-safe
pipeline validation, sparse-proxy smoke validation, and sparse Unity package creation.
With `--include_gsplat_smoke`, it also runs the 10-step gsplat trainer smoke path and
packages that output. A machine-readable summary is written under
`output/acceptance_<run_id>/debug/`.

Validation commands:

```bash
python -m py_compile scripts/generate_aruco_sheet.py scripts/detect_aruco_marker.py scripts/run_baseline_acceptance.py scripts/train_gsplat_adapter.py scripts/validate_output.py src/*.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_output.py output/marker_default_001 --mode default_safe
python scripts/validate_output.py output/marker_sparse_001 --mode sparse_proxy
python scripts/create_unity_package.py output/marker_sparse_001 --mode sparse_proxy --package_path output/marker_sparse_001/marker_sparse_unity_package.zip
```
