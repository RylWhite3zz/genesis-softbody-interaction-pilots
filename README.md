# Genesis soft-body robot interaction pilots

This repository archives three attempts to make robot arms interact with the
`stage3_showcase_150_genesis_mpm_20260724` soft-body assets in Genesis 1.1.2.
The immediate goal was to record credible simulation demos—not to train a
policy or collect per-step robot or particle trajectories.

## Bottom line

Passing import and MPM drop validation does not make an asset suitable for
parallel-jaw pick-and-place. The 150 assets were normalized to a longest axis
near 0.32 m, and their median shortest axis is 0.146 m. Piper opens to only
0.066 m, while its roughly 30 x 30 mm local finger pads cannot reliably
transfer the weight of these large objects away from the table.

| Experiment | Scope | Outcome |
| --- | --- | --- |
| `piper_pickplace_60` | 30 solid + 30 shell assets; whole-object tray pick-and-place | 58/60 failed the static jaw-width screen, 0090 became too wide after settling, and 0079 lifted briefly but fell during transfer; zero successes |
| `piper_local_grasp_60` | Original-scale local edge/corner pinches | 237 candidates, 67 executed close/lift probes, zero passes; best COM rise was 9.46 mm |
| `planar_transport_pilot` | Piper and Franka push, press-drag, and edge-drag | One Piper push of 0079 succeeded; Franka improved IK reachability but did not solve global load transfer against table friction |

The only trial to pass every gate was
[Piper pushing 0079 Inflatable Mattress](media/piper_push_0079.mp4): directed
COM progress was 234.9 mm and 99.84% of particles ended inside the goal. This
supports changing the task or contact tool instead of continuing to force
standard two-finger pick-and-place.

## Repository contents

- `experiments/`: key code, tests, aggregate reports, and per-attempt scalar diagnostics.
- `docs/ASSET_CHARACTERISTICS.md`: geometry, scale, material parameters, and representative assets.
- `docs/FAILURE_ANALYSIS.md`: measured failures, causal boundaries, and next-step priorities.
- `assets/representative/`: six lightweight asset packs with collision meshes, physical properties, validation reports, and previews.
- `assets/catalog/delivery_manifest.jsonl`: the full 150-item delivery manifest.
- `source_reports/`: original generation and validation summaries for all 150 assets.
- `media/`: the successful video and selected terminal-state images.

The complete source asset collection is approximately 8.2 GB and is not
duplicated here.

## Environment and reproducibility boundary

The original runs used stock `genesis-world==1.1.2`, CUDA, and
`MPM.Elastic`. Python 3.10–3.13, `numpy`, `trimesh`, and `imageio-ffmpeg` are
required; dynamics runs also require a working NVIDIA GPU.

```bash
python -m pip install "genesis-world==1.1.2" numpy trimesh "imageio[ffmpeg]"
```

Paths are configurable rather than tied to the original workstation:

```bash
export SOFTBODY_ASSET_ROOT=/path/to/stage3_showcase_150_genesis_mpm_20260724/assets
export PIPER_URDF=/path/to/piper_with_gripper.urdf
export GENESIS_PYTHON=/path/to/genesis/python
```

The included representative assets are sufficient for code inspection and
small-sample checks. Re-running the original 60-item batches requires the full
asset root. Piper runs also require the exact generated URDF used by the
experiments, with `tcp_link` and two MPM contact pads per finger. That generated
file was absent from the available snapshot, so this repository does not
substitute a merely similar robot model. The Franka path uses Genesis's bundled
Panda URDF.

Run the logic and geometry tests from each experiment directory:

```bash
cd experiments/planar_transport_pilot && python -m unittest discover -s tests -v
cd ../piper_local_grasp_60 && python -m unittest discover -s tests -v
cd ../piper_pickplace_60 && python -m unittest discover -s tests -v
```

## Interpretation boundary

These results do not establish that soft-body grasping is generally
impossible. They show that, at original packaged scale, under the current
fixed-base robot, parallel gripper, table, and stock Genesis 1.1.2
`MPM.Elastic` contact settings, more wrist-angle search or a swap from Piper to
Franka is insufficient for reliable pick-and-place.

Higher-priority demo tasks include pushing, sweeping, indentation, rolling,
spreading, edge lifting, scooping, suction, and bimanual manipulation. See the
failure analysis for the evidence behind that ranking.

## License status

No open-source license is currently granted. Redistribution rights for the
generated assets and their upstream content still need to be confirmed. Robot
URDF and mesh files are not redistributed in this repository.
