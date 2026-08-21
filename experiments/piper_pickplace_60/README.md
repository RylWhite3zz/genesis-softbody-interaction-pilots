# Piper + 60 MPM Elastic assets: tray pick-and-place

This folder is an isolated attempt to pick and place the 30 `solid` and 30 `shell` packaged assets with the fixed-base AgileX Piper in stock Genesis 1.1.2.

The batch obeys three hard constraints:

- Every asset stays at packaged scale `1.0`; only 90-degree axis-aligned rotations are allowed.
- An asset whose minimum original-scale extent exceeds the Piper jaw opening (`0.066 m`) is skipped before Genesis is started.
- No trajectory is recorded. An MP4 is retained in `videos/` only after the object is lifted, released, fully contained in the shallow tray, and stable enough to pass the gates in `pickplace60/protocol.py`.

The scene contains the fixed-base Piper, a coupled rigid workbench, and a coupled rigid shallow tray with a `0.44 m x 0.44 m` inner area. The soft bodies use their packaged `MPM.Elastic` parameters and PBS particle sizes; the runner does not substitute `MPM.ElastoPlastic`.

To avoid spending the tiny jaw clearance during a free-space descent, the episode reset places the open jaws around a grasp line `40%` of the settled object height above its center before the first physics step. This lets the body hang below the grasp instead of balancing around a center pinch. The measured interaction still includes gravity, a 10 N per-finger force limit, lift, transfer, release into the tray, and gripper retreat. Robot-MPM coupling friction is `6.0`, the upper end of the range already used by gentle_manip's soft-object domain-randomization preset. This is a permissive pre-grasp initialization, not evidence that an autonomous approach planner can acquire the same object.

Run the deterministic screen without a GPU:

```bash
${GENESIS_PYTHON:-python} scripts/run_batch.py --screen-only
```

Run the dependency-free logic tests:

```bash
${GENESIS_PYTHON:-python} -m unittest discover -s tests -v
```

Run every geometry-eligible asset in a fresh Genesis process:

```bash
${GENESIS_PYTHON:-python} scripts/run_batch.py
```

Outputs are `reports/batch_results.json`, `reports/batch_results.csv`, and successful videos under `videos/`. Lightweight per-attempt status files under `work/` are diagnostics, not trajectories.

The completed 2026-08-14 run produced no successful MP4: 58 assets were skipped by static size, 0090 was skipped after its settled particle width exceeded the jaw tolerance, and 0079 failed transport retention. See `reports/RESULTS.md` for the measured results and limitations.
