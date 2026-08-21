# Piper local pinching of 60 original-scale soft-body assets

## Outcome

No asset completed pick-and-place under this protocol, so no successful MP4
was retained.

- All 60 solid/shell assets remained at scale 1.0 with stock Genesis 1.1.2 `MPM.Elastic`.
- All local candidates were unreachable during the grasp segment for 23 assets.
- 37 assets executed at least one edge/corner contact, for 67 lift probes total and zero passes.
- The eight-GPU batch took approximately 43.8 minutes of wall time.
- No trajectories or failed-trial videos were retained.

This was not a result of screening everything out by size. The planner generated
and checked 237 local candidates across 0/-90/-45/45-degree closing-axis wrist
orientations; 67 candidates entered MPM contact, closing, and lifting stages.

## Quantitative summary

| Metric | Result |
| --- | ---: |
| solid: probe failure / all grasp IK failure | 20 / 10 |
| shell: probe failure / all grasp IK failure | 17 / 13 |
| local candidates | 237 |
| executed lift probes | 67 |
| probes with positive COM rise | 29 |
| lift passes | 0 |
| successful MP4s | 0 |

The best COM rise was 9.46 mm on 0011 Mango, below the required 45 mm. Its
lowest particle was still at 56.72 mm while the table top was 55 mm. The best
minimum-height candidate was 0074 Sofa Armrest Padding at 62.34 mm, still below
the 63 mm clearance gate, with only 0.63 mm COM rise.

## Active wrinkle-preload controls

0079 Inflatable Mattress and 0090 Steam Mop Pad were pressed tangentially
25 mm inward with open jaws before closing. Both moved laterally but did not
form load-bearing wrinkles.

- 0079: best post-preload COM change -3.85 mm; minimum height 54.65 mm.
- 0090: best post-preload COM change -0.74 mm; minimum height 57.42 mm.

## Interpretation

The current parallel gripper is not a reliable original-scale local
pick-and-place tool for these approximately 0.32 m normalized assets. The two
main bottlenecks are table-edge grasp reachability and insufficient load
transfer through a roughly 30 mm local pad. The best lift is too far below the
gate for a minor threshold adjustment to resolve the problem.

The result does not make the assets unusable for soft-body manipulation. They
remain candidates for pushing, dragging, indentation, spreading, folding,
bimanual work, or supported lift. Complete pick-and-place should first test a
wider/enveloping gripper, suction, or scoop/support tool.

The active-preload result also has a modeling boundary: pure `MPM.Elastic`
does not retain plastic strain. Tool pitch/roll, single-finger hooking, and
multi-stage closed-loop acquisition were not systematically searched.

## Artifacts

- `batch_results.json/csv`: audited 60-asset summary.
- `fold_preload_results.json`: 0079/0090 active-preload controls.
- `../work/*.json`: candidate, IK, phase-end particle scalars, and failure reasons.
- `../fold_work/*.json`: active-preload phase-end scalars.
