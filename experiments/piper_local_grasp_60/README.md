# Piper local edge/corner MPM pick-place

This experiment keeps all 60 `solid`/`shell` assets at their packaged scale and
tries local edge or corner pinches with the fixed-base AgileX Piper parallel
gripper. The object settles while the robot stays clear. Grasp candidates are
then generated from active MPM particles under the 30 x 30 mm finger-pad footprint.

Failed attempts retain scalar JSON diagnostics only. An MP4 is encoded only when
the object clears the table, remains lifted through transfer, is released inside
the shallow tray, and reaches the final stability gate. No robot or particle
trajectory files are recorded.

Run one asset with the Genesis 1.1.2 environment:

```bash
${GENESIS_PYTHON:-python} scripts/attempt_asset.py \
  --asset-id 0079_stage3_100k_qwen2_v1_shell_025403 \
  --result work/0079_stage3_100k_qwen2_v1_shell_025403.json \
  --video videos/0079_stage3_100k_qwen2_v1_shell_025403.mp4
```

Final audited outputs are under `reports/`. `RESULTS.md` summarizes the
60-asset edge/corner batch and the two active wrinkle-preload controls. The
`videos/` directories intentionally remain empty when no attempt passes every
lift, transfer, release, containment, and stability gate.
