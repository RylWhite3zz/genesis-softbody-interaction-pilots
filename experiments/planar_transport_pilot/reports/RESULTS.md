# Original-scale planar soft-body transport pilot

## Outcome

Three representative assets were screened with continuous IK. Dynamics trials
used 0001 solid and 0079 shell; 0074 shell remained at the reachability stage.
The 400 x 400 mm green target was non-colliding, and every asset retained its
packaged scale and `MPM.Elastic` parameters.

Two findings should be separated:

1. **Franka materially improved reachability.** It passed 6/6 representative continuous paths, versus 4/6 for Piper. Piper far-side press-drag errors were about 60.5 mm on 0001 and 77.9 mm on 0074; Franka stayed below 0.7 mm on the same paths.
2. **Franka did not solve whole-object transport.** Its thick-solid push moved 39.3 mm, far-side press-drag 3.3 mm, and diagonal far-edge drag 32.4 mm. Small contacts displaced local material while the main body remained table-constrained or stretched locally.

Piper's lateral push of thin shell 0079 was the only trial to pass every gate:
234.9 mm directed COM progress, 33.7 mm final goal error, and 99.84% particle
occupancy. Piper pushing thick solid 0001 reached 79.4 mm; its widened-contact
press-drag of 0079 reached 79.1 mm. “Success” refers to this pilot's goal,
stability, finite-state, and occupancy gates—not validated real-robot quality.

This was an iterative pilot. Each `work/*.json` retains an
`implementation_revision`; the successful video is revision v5, while later
versions added occupancy and edge-candidate gates. The files are not repeated
samples under an identical configuration.

## Dynamics results

| Robot | Action | Asset | Directed COM progress | Goal occupancy | Result |
| --- | --- | --- | ---: | ---: | --- |
| Piper | push | 0079 Inflatable Mattress | 234.9 mm | 99.84% | success |
| Piper | push | 0001 Chiffon Cake | 79.4 mm | below gate | failed |
| Piper | press-drag | 0079 Inflatable Mattress | 79.1 mm | below gate | failed |
| Franka | push | 0001 Chiffon Cake | 39.3 mm | 42.63% | failed |
| Franka | press-drag | 0001 Chiffon Cake | 3.3 mm | 34.49% | failed |
| Franka | press-drag | 0079 Inflatable Mattress | 7.3 mm | 23.38% | failed |
| Franka | diagonal edge-drag | 0001 Chiffon Cake | 32.4 mm | 54.52% | failed |

The successful 640 x 480, 24 FPS, 76-frame MP4 is retained in the repository.
Failed trials retain JSON and selected diagnostic PNGs, not MP4 or `.npy/.npz`
trajectories.

## Why Franka was still insufficient

DLO-Lab's Panda ROD example uses a `rod_solver`-specific fingertip coupling
registration. It does not transfer directly to the table support and friction
problem of an MPM object. This pilot therefore used the bundled Panda URDF and
standard rigid–MPM coupling.

Franka's extra degree of freedom and workspace remain useful for far-side
contacts. Moving a large body globally still requires a different load path:
wide/enveloping contacts, scoop or roller support, suction, or a calibrated
lower-friction fixture. More wrist-angle search cannot compensate for
insufficient local load transfer. Pure elasticity also causes pressed wrinkles
to rebound; fold-dependent tasks should separately evaluate material modeling.

## Artifacts

- `experiment_results.json/csv`: dynamics and IK summary.
- `reachability_piper.json` and `reachability_franka.json`: continuous IK scans.
- `work/*_plan*.json`: retained and rejected edge/corner candidates.
- `work/*.json`: phase-end scalars, IK errors, and success gates.
- `media/piper_push_0079.mp4`: the only trial that passed every gate.
