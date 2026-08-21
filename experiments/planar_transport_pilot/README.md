# Planar soft-body transport pilot

This pilot compares fixed-base AgileX Piper and Franka Panda on the same
original-scale Genesis 1.1.2 `MPM.Elastic` assets. It tests three actions:

- `push`: use the closed gripper as a low pusher and move the object laterally
  along the table into a green region, preserving Piper's natural wrist angle.
- `press_drag`: press the closed gripper into the far top surface and drag the
  object back toward the robot.
- `edge_drag`: pinch a locally narrow far-side edge, keep the object on the
  table, and drag it back toward the robot.

The green goal is visual only (`collision=False`). Success requires at least
140 mm directed COM progress, final COM within 85 mm of the goal center, at
least 85% of particles inside the 400 x 400 mm goal, a
finite constant-particle state, table contact, and final RMS speed below
0.10 m/s. Failed trials retain JSON and diagnostic PNGs, while MP4 is encoded
only for successful trials. No trajectories are recorded.

The original workspace also contained a stock/fork source audit. That separate
report was not present in the file snapshot used to build this repository.
This pilot deliberately runs the installed stock Genesis 1.1.2 environment;
the fork's `MPM.ElastoPlastic`-specific stress feedback is not treated as an
`MPM.Elastic` capability.

Run the reachability scan from this directory:

```bash
${GENESIS_PYTHON:-python} scripts/scan_reachability.py \
  --robot piper --result reports/reachability_piper.json
```

Run one dynamics trial:

```bash
${GENESIS_PYTHON:-python} scripts/attempt.py \
  --robot piper --action push \
  --asset-id 0001_stage3_100k_qwen2_v1_solid_053842 \
  --result work/piper_push_0001.json \
  --video videos/piper_push_0001.mp4
```
