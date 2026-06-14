#!/usr/bin/env bash
# Reproduce the final Unity/AR-ready metric PLY used for the chair baseline.
# This script assumes the InSitu-A4 foreground-mask training model already exists.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ITERATIONS="${ITERATIONS:-30000}"
RATIO="${RATIO:-0.3}"
MIN_FG_OBS="${MIN_FG_OBS:-5}"
MARKER_ID="${MARKER_ID:-23}"
MARKER_SIZE_M="${MARKER_SIZE_M:-0.12}"
DBSCAN_EPS_M="${DBSCAN_EPS_M:-0.10}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-8}"

A4_MODEL="${A4_MODEL:-third_party/InSitu-A4/output/chair_insitu_a4pose_fgmask_result_20260611}"
A4_DATASET="${A4_DATASET:-third_party/InSitu-A4/data/chair_insitu_a4pose_fgmask_20260611}"
RAW_PLY="${RAW_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply}"
PRUNED_PLY="${PRUNED_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud_maskvote_pruned_r03.ply}"
PACKAGED_ID="${PACKAGED_ID:-chair_a4pose_fgmask_pruned_r03_30000_aruco_20260611}"
DBSCAN_ID="${DBSCAN_ID:-chair_a4pose_fgmask_pruned_r03_dbscan_main_30000_aruco_20260611}"
ALIGNED_ID="${ALIGNED_ID:-chair_a4pose_fgmask_pruned_r03_dbscan_main_aligned_30000_aruco_20260611}"

python scripts/prune_gaussians_by_mask_votes.py \
  --input-ply "${RAW_PLY}" \
  --output-ply "${PRUNED_PLY}" \
  --colmap-text-dir "${A4_DATASET}/sparse_txt" \
  --mask-dir "${A4_DATASET}/mask_undistorted/images" \
  --min-fg-observations "${MIN_FG_OBS}" \
  --min-foreground-ratio "${RATIO}" \
  --votes-npz "data/intermediate/chair_a4pose_fgmask_votes_r03.npz"

# This packaging script records the ArUco marker scale and writes output/${PACKAGED_ID}.
# For the current chair run: scale_factor=0.22297524225757243 from marker id 23, size 0.12m.
python scripts/package_a4pose_fgmask_pruned_r03_aruco.py

python scripts/filter_metric_point_cloud_dbscan.py \
  --source-dir "output/${PACKAGED_ID}" \
  --object-id "${DBSCAN_ID}" \
  --eps-m "${DBSCAN_EPS_M}" \
  --min-samples "${DBSCAN_MIN_SAMPLES}" \
  --keep-mode largest

python scripts/align_point_cloud_for_unity.py \
  --source-dir "output/${DBSCAN_ID}" \
  --object-id "${ALIGNED_ID}" \
  --current-front=-Z \
  --target-front=+Z

python scripts/validate_output.py "output/${ALIGNED_ID}" --mode sparse_proxy

echo "Final aligned metric PLY: output/${ALIGNED_ID}/point_cloud_metric.ply"
