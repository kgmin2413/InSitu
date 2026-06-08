# TODO

## Next Implementation Steps

1. Install/clone gsplat inside the container and run `configs/gsplat_training.yaml` with `training.enabled: true`.
2. Validate `scripts/train_gsplat_adapter.py` against a real gsplat `examples/simple_trainer.py` run and tune iteration/data-factor defaults.
3. Run full enabled path with `colmap.enabled: true`, `training.enabled: true`, manual scale, and `proxy_generation.method: obb` after a trainer command is configured.
4. Validate Unity import expectations for `package_manifest.json`, `point_cloud_metric.ply`, `proxy_collider.obj`, `metadata.json`, and preview assets.
5. Use `debug/capture_quality_report.json` to tune capture guidance thresholds across more real phone videos.
6. Validate the real COLMAP quality thresholds on larger selected real-video frame sets with CPU SIFT.
7. Harden real frame extraction on more phone videos, including variable FPS and rotation metadata.
8. Tune deterministic frame selection thresholds against real furniture captures.
9. Improve background removal beyond rectangle-seeded GrabCut, preferably with a stronger segmentation backend.
10. Tune proxy cleanup defaults, especially DBSCAN/flat-bottom settings, against more furniture captures.
11. Add ARCore odometry + COLMAP trajectory scale estimation as the non-marker scale path.
12. Add integration tests with tiny fixtures for each real stage as it lands.
