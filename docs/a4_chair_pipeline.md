# A4 Chair Reconstruction Pipeline

This runbook captures the reproducible pipeline used for the current chair result:

1. Run the public `kgmin2413/InSitu-A4` pipeline to get strong COLMAP poses and a 3DGS model.
2. Keep full A4 poses, but train with foreground masks and a foreground sparse seed.
3. Prune Gaussians by projecting them into undistorted foreground masks.
4. Package the pruned PLY with ArUco metric scale.
5. Remove disconnected background clusters with DBSCAN.
6. Align the final metric PLY for Unity/AR placement.
7. Optionally export a GLB splatmesh preview; for best visual quality, use the 3DGS PLY directly and keep OBJ for collision.

Generated captures, trained models, and output folders are intentionally not tracked in git.

## External Inputs

Place or clone these locally:

```bash
git clone https://github.com/kgmin2413/InSitu-A4 third_party/InSitu-A4
# Put the capture at data/input/chair.mp4 or point the A4 commands at another video.
```

The full rebuild requires CUDA, COLMAP, the A4 Python environment, and background-removal dependencies such as `rembg`.

## Successful Chair Settings

The current best chair artifact was produced from:

```text
A4 dataset: third_party/InSitu-A4/data/chair_insitu_a4true_20260611
Foreground-mask dataset: third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611
A4 model: third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611
Iterations: 30000
Mask-vote threshold: min_fg_observations=5, min_foreground_ratio=0.3
DBSCAN: eps=0.10m, min_samples=8, keep largest cluster
ArUco marker: id=23, marker_size_m=0.12
Metric scale factor from ArUco: 0.22297524225757243
Unity alignment: Y-up, front=+Z, bottom-center pivot at origin
```

## Command Outline

Run the A4 preprocessing/training side first. The public A4 repo includes its own scripts; locally we used that repo under `third_party/InSitu-A4` and kept large A4 data out of this repository.

After A4 has produced the full-pose dataset and foreground-mask training result, prune the raw 3DGS PLY by mask votes:

```bash
python scripts/prune_gaussians_by_mask_votes.py \
  --input-ply third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611/point_cloud/iteration_30000/point_cloud.ply \
  --output-ply third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611/point_cloud/iteration_30000/point_cloud_maskvote_pruned_r03.ply \
  --colmap-text-dir third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611/sparse_txt \
  --mask-dir third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611/mask_undistorted/images \
  --min-fg-observations 5 \
  --min-foreground-ratio 0.3 \
  --votes-npz data/intermediate/chair_a4pose_fgmask_votes_r03.npz
```

Package the mask-pruned PLY with ArUco metric scale:

```bash
python scripts/package_a4pose_fgmask_pruned_r03_aruco.py
```

The strict background-cleaned output used for Unity placement was then created by DBSCAN largest-cluster filtering from the packaged metric PLY. If repeating manually, keep DBSCAN conservative until the chair backrest is verified.

Align the final strict PLY for Unity/AR:

```bash
python scripts/align_point_cloud_for_unity.py \
  --source-dir output/chair_a4pose_fgmask_pruned_r03_dbscan_main_30000_aruco_20260611 \
  --object-id chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611 \
  --current-front -Z \
  --target-front +Z
```

Optional GLB preview export:

```bash
python scripts/export_splatmesh_glb.py \
  --source-dir output/chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611 \
  --object-id chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_splatmesh_glb_20260611
```

## Recommended Runtime Asset Split

For Unity/AR, keep visual and collision separate:

```text
point_cloud_metric.ply  # visual Gaussian splat, best quality
proxy_collider.obj      # lightweight collision / placement proxy
metadata.json           # meter scale, bbox, pivot, coordinate convention
```

GLB mesh conversion is useful for quick preview and generic web viewers, but it does not preserve the original 3DGS quality as well as a Gaussian-splat renderer.
