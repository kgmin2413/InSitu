# Chair A4 Metric Aligned Baseline

This is the baseline to use for the Unity/AR chair asset:

```text
output/chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611
```

The intended final visual asset is the aligned 3DGS PLY:

```text
point_cloud_metric.ply
```

The collider remains separate:

```text
proxy_collider.obj
```

## Contract

- 3DGS training backend: InSitu-A4
- Metric scale: ArUco marker scale, not manual scale editing
- Marker: id `23`, physical size `0.12m`
- Scale factor measured for the chair run: `0.22297524225757243`
- Background cleanup: mask-vote pruning plus DBSCAN largest metric cluster
- Unity placement: `Y-up`, local front `+Z`, bottom-center pivot at `(0,0,0)`
- Output unit: meter

## Exact Final Output Values

From the generated aligned folder:

```text
points: 40785
bbox: 0.506830 x 0.998843 x 0.791196 m
bounds min: [-0.253415, 0.0, -0.395598]
bounds max: [ 0.253415, 0.998843, 0.395598]
object center: [0.0, 0.499422, 0.0]
```

The alignment transform was:

```text
rotation: yaw 180 degrees about Y
translation after rotation: [0.035813227, 0.324176639, 0.425215472] m
pivot: bottom center of rotated Gaussian-center bbox
```

The transform is applied to Gaussian centers, normals, and `rot_0..3` quaternions. `scale_0..2` already include the ArUco metric scale through the scale-estimation export path.

## Reproduction Commands

The script below starts after the InSitu-A4 foreground-mask training result exists:

```bash
bash scripts/run_chair_a4_metric_aligned_baseline.sh
```

It performs:

1. Mask-vote pruning of the raw A4 3DGS PLY:

```text
min_fg_observations=5
min_foreground_ratio=0.3
```

2. ArUco metric packaging:

```text
marker_id=23
marker_size_m=0.12
scale_factor=0.22297524225757243
```

3. DBSCAN strict cleanup on metric Gaussian centers:

```text
eps_m=0.10
min_samples=8
keep_mode=largest_cluster_only
input_points=46405
kept_points=40785
removed_points=5620
```

4. Unity/AR alignment:

```text
current_front=-Z
target_front=+Z
bottom center -> origin
```

## Required Local A4 Inputs

These large generated folders are intentionally not tracked in git:

```text
third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611
third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611
```

They are produced by the InSitu-A4-based reconstruction step. Keep the heavy A4 repo and outputs local; this repo tracks only the reproducible wrapper scripts and metadata logic.

## Why PLY, Not GLB

For this baseline, the visual quality target is the 3DGS result itself. GLB mesh conversion is only a fallback preview path and loses too much Gaussian-splat quality for the chair. Runtime should render `point_cloud_metric.ply` with a Gaussian-splat renderer and use `proxy_collider.obj` only for collision/placement.
