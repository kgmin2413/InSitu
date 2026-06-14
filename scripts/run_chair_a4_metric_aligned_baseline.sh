#!/usr/bin/env bash
# Chair baseline wrapper around the generic furniture A4 metric alignment pipeline.
set -euo pipefail

export OBJECT_SLUG="${OBJECT_SLUG:-chair}"
export A4_MODEL="${A4_MODEL:-third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611}"
export A4_DATASET="${A4_DATASET:-third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611}"
export A4_MARKER_IMAGE_DATASET="${A4_MARKER_IMAGE_DATASET:-third_party/InSitu-A4/data/chair_insitu_a4true_20260611}"
export SOURCE_VIDEO="${SOURCE_VIDEO:-data/input/chair.mp4}"
export ITERATIONS="${ITERATIONS:-30000}"
export RATIO="${RATIO:-0.3}"
export MIN_FG_OBS="${MIN_FG_OBS:-5}"
export MARKER_ID="${MARKER_ID:-23}"
export MARKER_SIZE_M="${MARKER_SIZE_M:-0.12}"
export DBSCAN_EPS_M="${DBSCAN_EPS_M:-0.10}"
export DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-8}"
export CURRENT_FRONT="${CURRENT_FRONT:--Z}"
export TARGET_FRONT="${TARGET_FRONT:-+Z}"
export RAW_PLY="${RAW_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply}"
export PRUNED_PLY="${PRUNED_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud_maskvote_pruned_r03.ply}"
export PACKAGED_ID="${PACKAGED_ID:-chair_a4pose_fgmask_pruned_r03_30000_aruco_20260611}"
export DBSCAN_ID="${DBSCAN_ID:-chair_a4pose_fgmask_pruned_r03_dbscan_main_30000_aruco_20260611}"
export ALIGNED_ID="${ALIGNED_ID:-chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611}"

exec "$(dirname "${BASH_SOURCE[0]}")/run_a4_metric_aligned_baseline.sh"
