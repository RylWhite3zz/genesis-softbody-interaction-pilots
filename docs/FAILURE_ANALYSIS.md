# Failure analysis

## Executive finding

The evidence points to a mismatch among task, normalized object scale, and end
effector—not a blanket inability of Genesis to simulate rigid–soft contact.
Franka removed most far-side IK failures but did not change the mechanical
conflict among small local contacts, table friction, and large object loads.
The clearest evidence is asset 0079: Piper could push it into the target region,
but neither whole-object nor local pinching could retain it through transfer.

## 1. Whole-object two-finger pick-and-place

Experiment: `experiments/piper_pickplace_60/`

The protocol used packaged scale 1.0, axis-aligned 90-degree rotations, a
66 mm Piper opening, a 10 N limit per finger, a fixed base, and a shallow tray.
An MP4 was retained only if every gate passed.

- 58/60 assets had a shortest axis above 66 mm and were skipped before Genesis started.
- 0090 Steam Mop Pad measured 64.46 mm statically, but the settled PBS particle body measured 73.59 mm along the jaw axis, above the 70 mm simulation tolerance.
- 0079 Inflatable Mattress reached a maximum COM rise of 57.97 mm, but its COM was 37.46 mm below the initial value at transfer end and its lowest particle had returned to the table.
- Successes: zero. Retained videos: zero.

This experiment primarily exposed whole-object geometry mismatch rather than
an MPM API failure. Raising coupling friction cannot create missing jaw opening
and would further weaken engineering credibility.

## 2. Local edge and corner pinching

Experiment: `experiments/piper_local_grasp_60/`

The second attempt avoided the whole-object fit requirement. It generated local
edge and corner candidates from active particles under a 30 x 30 mm pad area.

- 237 candidates were generated across 60 assets.
- All grasp-path IK candidates failed for 23 assets.
- 37 assets executed at least one contact, close, and lift sequence: 67 probes total.
- 29 probes produced positive COM motion, but none lifted the complete object clear of the table.
- The best COM rise was 9.46 mm on 0011 Mango, far below the 45 mm gate; particles remained table-supported.
- 0074 Sofa Armrest Padding came closest in minimum height at 62.34 mm, still below the 63 mm gate, while COM rose only 0.63 mm.

The zero-success result is not a small threshold miss. The best global lift was
several times below the required value while substantial material remained
supported by the table.

### Active wrinkle preload

0079 and 0090 were also pressed inward by 25 mm with open jaws before closing.
Both moved laterally but did not form load-bearing wrinkles. Their best COM
changes were -3.85 mm and -0.74 mm respectively. Elastic rebound is consistent
with this observation, although tool pitch/roll, single-finger hooking, and
closed-loop acquisition were not exhaustively tested.

## 3. Planar push and drag

Experiment: `experiments/planar_transport_pilot/`

| Robot | Action | Asset | Directed COM progress | Goal occupancy | Status |
| --- | --- | --- | ---: | ---: | --- |
| Piper | push | 0079 Inflatable Mattress | 234.9 mm | 99.84% | success |
| Piper | push | 0001 Chiffon Cake | 79.4 mm | below gate | failed |
| Piper | press-drag | 0079 | 79.1 mm | below gate | failed |
| Franka | push | 0001 | 39.3 mm | 42.63% | failed |
| Franka | press-drag | 0001 | 3.3 mm | 34.49% | failed |
| Franka | press-drag | 0079 | 7.3 mm | 23.38% | failed |
| Franka | diagonal edge-drag | 0001 | 32.4 mm | 54.52% | failed |

Franka passed 6/6 representative continuous-IK paths while Piper passed 4/6.
That reachability gain is real, but it removed only the kinematic bottleneck.
In dynamics, the small tool contacts displaced local material while the object
body remained constrained by table friction, or motion became local stretch
and elastic recovery.

The successful Piper push demonstrates that the current rigid–MPM coupling can
produce a stable recordable interaction. Task selection matters more than an
arm-only substitution.

## Ranked causes

1. **Task–scale–tool mismatch.** Approximately 32 cm normalized assets do not match a 66 mm jaw opening; local pads cannot carry the full objects.
2. **Insufficient contact area and load path.** Pushing retains table support, whereas pick-and-place routes the entire load through a small pinch.
3. **Table friction and local deformation.** Dragging stretches or indents the contacted region before the full body moves.
4. **Reachability.** Piper fails some far-side paths; Franka improves this layer but not contact mechanics.
5. **Uncalibrated manipulation physics.** Rule-assigned properties and drop validation do not calibrate grasp or sliding behavior; pure elasticity also discourages persistent folds.
6. **Limited protocol coverage.** Wide-area tools, scoops, suction, rollers, bimanual manipulation, tilted fixtures, and closed-loop control remain untested.

## Conclusions not supported by the data

- The data do not show that Genesis 1.1.2 rigid–MPM coupling is generally broken.
- They do not show that Franka is useless; it materially improves reachability.
- They do not show that all 150 assets are unsuitable; cloth, garment, and bag families were not systematically run in the three dynamics batches.
- They do not support fixing local grasping merely by relaxing success thresholds.
- They do not establish that the rule-assigned parameters are real-object calibrations.

## Near-term engineering priorities

1. Replace pick-and-place with interaction-showcase tasks: pushing, sweeping, indentation, rolling, spreading, edge lifting, or scooping.
2. Use 0031, 0079, 0090, and 0001 as a four-level pilot before expanding the asset count.
3. If complete lift is mandatory, change the tool first: wide soft fingers, an enveloping gripper, scoop/tray, suction, or bimanual support.
4. Keep reachability screening separate from dynamics validation; IK success is only a prerequisite.
5. Run friction, mass, Young's modulus, and particle-size sensitivity checks on 2–3 assets and report stable parameter windows.
