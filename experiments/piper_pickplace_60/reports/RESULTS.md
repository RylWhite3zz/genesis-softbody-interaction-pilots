# Piper original-scale MPM.Elastic tray pick-and-place results

Run date: 2026-08-14

## Outcome

No asset passed the complete pick-and-place protocol, so no MP4 was retained.

- 58/60 assets had a shortest original-scale axis above Piper's 0.066 m opening and were skipped by the static screen.
- 0090 Steam Mop Pad measured 0.064462 m statically, but its settled PBS particle body reached 0.073592 m along the jaw axis and exceeded the 0.070 m simulation tolerance.
- 0079 Inflatable Mattress lifted briefly but was not retained through the end of transfer.
- Successful videos: zero.

This result applies to original scale, the existing small Piper pads, and a
10 N per-finger force limit. It does not establish that stock Genesis 1.1.2
`MPM.Elastic` rigid–soft interaction is generally infeasible.

## Protocol constraints

- Stock Genesis 1.1.2 on CUDA with an NVIDIA A800 80 GB PCIe GPU.
- Packaged `MPM.Elastic` E, nu, density, particle size, and grid density.
- Scale fixed at 1.0; only axis-aligned 90-degree rotations.
- Fixed-base AgileX Piper with existing finger collision pads.
- Coupled rigid workbench and a shallow tray with a 0.44 x 0.44 m interior.
- Open jaws initialized on a grasp line 40% of object height above COM before the first physics step.
- 10 N per-finger limit and robot–MPM coupling friction 6.0.
- No per-step trajectory recording; phase-end scalars only.
- MP4 retained only after every lift, transfer, containment, release, and stability gate passed.

## Dynamic candidates

| Asset | Original dimensions (m) | Rotated dimensions (m) | Mass | Result |
| --- | --- | --- | ---: | --- |
| 0079 Inflatable Mattress | 0.157327 x 0.322368 x 0.053658 | 0.322368 x 0.053658 x 0.157327 | 0.676478 kg | `failed_gates` |
| 0090 Steam Mop Pad | 0.324037 x 0.172402 x 0.064462 | 0.324037 x 0.064462 x 0.172402 | 0.817585 kg | `skipped_post_settle_size` |

0079 reached a maximum COM rise of 0.057969 m. At transfer end its COM was
0.037455 m below the initial value and its lowest particle was at z=0.055639 m,
back on the z=0.055 m table. Earlier iterations also showed rotation around the
fingers and lower-friction slip after lift.

0090 had only 1.538 mm of static clearance inside the 0.066 m opening. The
settled particle width exceeded the opening, so the runner did not force a
closure by increasing force.

## Artifacts

- `batch_results.json/csv`: full 60-asset statuses and screening reasons.
- `screening.json/csv`: geometry-only screening results.
- `../work/*.json`: phase-end scalar diagnostics for 0079 and 0090; not trajectories.
- `../videos/`: empty because no trial passed all gates.

The next attempt should change the task or hardware assumption—larger opening,
larger contact surface, scoop/support, suction, or explicitly justified asset
rescaling—rather than rely on still higher coupling friction.
