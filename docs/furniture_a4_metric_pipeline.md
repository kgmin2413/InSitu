# Generic Furniture A4 Metric Pipeline

This is the generalized baseline for floor-standing furniture:

```text
InSitu-A4 3DGS -> mask-vote pruning -> ArUco metric scale -> DBSCAN cleanup -> Unity/AR alignment
```

Metric scale is always computed from an ArUco marker. It is not a manual PLY scale edit. The scale-estimation export also applies the metric factor to 3DGS Gaussian radii (`scale_0..2`), not only to point centers.

## When It Generalizes Well

This pipeline is meant for furniture-like objects that:

- sit on the floor,
- are captured with an ArUco marker visible in enough frames,
- can be segmented by the A4/rembg mask pipeline,
- form a dominant 3DGS cluster after background removal.

Examples: chairs, stools, small tables, side tables, cabinets, shelves, sofas. Smaller objects or non-floor-mounted objects may need different pivot and front-axis settings.

## Required Inputs

The generic wrapper starts after the InSitu-A4 foreground-mask dataset/model exists:

```text
A4_DATASET=<path containing input/, sparse_txt/, mask_undistorted/images/>
A4_MODEL=<path containing point_cloud/iteration_30000/point_cloud.ply>
OBJECT_SLUG=<chair|table|sofa|...>
```

If the marker is more visible in the unmasked/full-pose A4 dataset, pass it separately:

```text
A4_MARKER_IMAGE_DATASET=<path containing input/ images with marker visible>
```

## Generic Command

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

The wrapper writes:

```text
output/${OBJECT_SLUG}_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco/point_cloud_metric.ply
output/${OBJECT_SLUG}_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco/proxy_collider.obj
```

## Tunable Parameters

```text
RATIO=0.3                 # mask-vote foreground ratio
MIN_FG_OBS=5              # minimum mask foreground observations
DBSCAN_EPS_M=0.10         # object/background cluster separation in meters
DBSCAN_MIN_SAMPLES=8
CURRENT_FRONT=-Z          # object front before final alignment
TARGET_FRONT=+Z           # Unity-facing local front axis
MARKER_ID=23
MARKER_SIZE_M=0.12
```

For larger furniture, `DBSCAN_EPS_M` may need to increase. For small furniture or dense scans, it may need to decrease.

## Step-By-Step Commands

Mask-vote prune the A4 raw PLY:

```bash
python scripts/prune_gaussians_by_mask_votes.py \
  --input-ply "$A4_MODEL/point_cloud/iteration_30000/point_cloud.ply" \
  --output-ply "$A4_MODEL/point_cloud/iteration_30000/point_cloud_maskvote_pruned_r03.ply" \
  --colmap-text-dir "$A4_DATASET/sparse_txt" \
  --mask-dir "$A4_DATASET/mask_undistorted/images" \
  --min-fg-observations 5 \
  --min-foreground-ratio 0.3
```

Package the pruned PLY with ArUco metric scale:

```bash
python scripts/package_a4_metric_aruco.py \
  --object-id table_a4pose_fgmask_pruned_r03_30000_aruco \
  --a4-dataset "$A4_DATASET" \
  --a4-marker-image-dataset "$A4_MARKER_IMAGE_DATASET" \
  --a4-model "$A4_MODEL" \
  --raw-ply "$A4_MODEL/point_cloud/iteration_30000/point_cloud_maskvote_pruned_r03.ply" \
  --marker-id 23 \
  --marker-size-m 0.12
```

Then DBSCAN and align:

```bash
python scripts/filter_metric_point_cloud_dbscan.py \
  --source-dir output/table_a4pose_fgmask_pruned_r03_30000_aruco \
  --object-id table_a4pose_fgmask_pruned_r03_dbscan_main_30000_aruco \
  --eps-m 0.10 \
  --min-samples 8 \
  --keep-mode largest

python scripts/align_point_cloud_for_unity.py \
  --source-dir output/table_a4pose_fgmask_pruned_r03_dbscan_main_30000_aruco \
  --object-id table_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco \
  --current-front=-Z \
  --target-front=+Z
```

## Runtime Asset Recommendation

Use the aligned 3DGS PLY for visuals and the OBJ proxy for physics:

```text
point_cloud_metric.ply  # visual Gaussian splat
proxy_collider.obj      # collision / placement
metadata.json           # meter scale, bbox, pivot, coordinate convention
```

GLB conversion is optional for previews, but it loses the 3DGS visual quality on thin furniture.
