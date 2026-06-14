#!/usr/bin/env bash
# Generic furniture baseline: InSitu-A4 3DGS -> mask-vote pruning -> ArUco metric scale -> DBSCAN -> Unity alignment.
# Required: an InSitu-A4 foreground-mask dataset/model already exists locally.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OBJECT_SLUG="${OBJECT_SLUG:?Set OBJECT_SLUG, e.g. chair, table, sofa}"
A4_DATASET="${A4_DATASET:?Set A4_DATASET, e.g. third_party/InSitu-A4/data/table_a4pose_fgmask}"
A4_MODEL="${A4_MODEL:?Set A4_MODEL, e.g. third_party/InSitu-A4/output/table_a4pose_fgmask_result}"
A4_MARKER_IMAGE_DATASET="${A4_MARKER_IMAGE_DATASET:-${A4_DATASET}}"
SOURCE_VIDEO="${SOURCE_VIDEO:-data/input/${OBJECT_SLUG}.mp4}"

ITERATIONS="${ITERATIONS:-30000}"
RATIO="${RATIO:-0.3}"
MIN_FG_OBS="${MIN_FG_OBS:-5}"
MARKER_ID="${MARKER_ID:-23}"
MARKER_SIZE_M="${MARKER_SIZE_M:-0.12}"
DBSCAN_EPS_M="${DBSCAN_EPS_M:-0.10}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-8}"
CURRENT_FRONT="${CURRENT_FRONT:--Z}"
TARGET_FRONT="${TARGET_FRONT:-+Z}"

RAW_PLY="${RAW_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply}"
PRUNED_PLY="${PRUNED_PLY:-${A4_MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud_maskvote_pruned_r03.ply}"
PACKAGED_ID="${PACKAGED_ID:-${OBJECT_SLUG}_a4pose_fgmask_pruned_r03_${ITERATIONS}_aruco}"
DBSCAN_ID="${DBSCAN_ID:-${OBJECT_SLUG}_a4pose_fgmask_pruned_r03_dbscan_main_${ITERATIONS}_aruco}"
ALIGNED_ID="${ALIGNED_ID:-${OBJECT_SLUG}_a4pose_fgmask_pruned_r03_dbscan_main_aligned_${ITERATIONS}_aruco}"
VOTES_NPZ="${VOTES_NPZ:-data/intermediate/${OBJECT_SLUG}_a4pose_fgmask_votes_r03.npz}"

python scripts/prune_gaussians_by_mask_votes.py \
  --input-ply "${RAW_PLY}" \
  --output-ply "${PRUNED_PLY}" \
  --colmap-text-dir "${A4_DATASET}/sparse_txt" \
  --mask-dir "${A4_DATASET}/mask_undistorted/images" \
  --min-fg-observations "${MIN_FG_OBS}" \
  --min-foreground-ratio "${RATIO}" \
  --votes-npz "${VOTES_NPZ}"

python scripts/package_a4_metric_aruco.py \
  --object-id "${PACKAGED_ID}" \
  --a4-dataset "${A4_DATASET}" \
  --a4-marker-image-dataset "${A4_MARKER_IMAGE_DATASET}" \
  --a4-model "${A4_MODEL}" \
  --raw-ply "${PRUNED_PLY}" \
  --iterations "${ITERATIONS}" \
  --source-video "${SOURCE_VIDEO}" \
  --marker-id "${MARKER_ID}" \
  --marker-size-m "${MARKER_SIZE_M}"

python scripts/filter_metric_point_cloud_dbscan.py \
  --source-dir "output/${PACKAGED_ID}" \
  --object-id "${DBSCAN_ID}" \
  --eps-m "${DBSCAN_EPS_M}" \
  --min-samples "${DBSCAN_MIN_SAMPLES}" \
  --keep-mode largest

python scripts/align_point_cloud_for_unity.py \
  --source-dir "output/${DBSCAN_ID}" \
  --object-id "${ALIGNED_ID}" \
  --current-front="${CURRENT_FRONT}" \
  --target-front="${TARGET_FRONT}"

python scripts/validate_output.py "output/${ALIGNED_ID}" --mode sparse_proxy

echo "Final aligned metric PLY: output/${ALIGNED_ID}/point_cloud_metric.ply"
