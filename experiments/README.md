# Experiments

- `piper_pickplace_60`: whole-object tray pick-and-place; primarily exposes the scale/jaw mismatch.
- `piper_local_grasp_60`: edge/corner pinch and active preload attempts; exposes local load-transfer and IK limits.
- `planar_transport_pilot`: Piper/Franka push and drag trials; separates reachability from contact mechanics.

`work/*.json` and `fold_work/*.json` are not trajectories. They retain
configuration, IK diagnostics, phase-end particle statistics, and success
gates. Compiler caches and failed-trial videos are excluded.
