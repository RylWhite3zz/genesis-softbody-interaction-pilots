# Characteristics of the 150 soft-body assets

## Dataset organization

`stage3_showcase_150_genesis_mpm_20260724` contains 150 items: 30 each from
`solid`, `cloth_sheet`, `shell`, `garment`, and `bag`. Each original delivery
pack contains a textured visual mesh, a watertight collision proxy, URDF,
caption, physical properties, an MPM drop preview, and validation reports.

All 150 items passed the prescribed Genesis MPM drop test, and all 150 passed
URDF import and texture rendering. These checks establish loadability and
stability for the drop protocol—not robot reachability, jaw fit, contact load
transfer, or manipulation success.

## Geometry and scale

The generation pipeline normalized the longest axis to roughly 0.32 m. Across
the full set, the longest-axis range is 0.3217–0.3309 m with a 0.3237 m median.
Consequently, an object that is small in the real world, such as an earplug or
berry, can be nearly 32 cm long in simulation. This is a showcase-normalized
scale rather than a claim about real object dimensions.

| Family | Shortest-axis range / median | Mass range / median | Estimated particles range / median |
| --- | --- | --- | --- |
| solid | 93.5–320.5 / 195.5 mm | 0.08–4.50 / 2.90 kg | 422–49,388 / 17,008 |
| cloth_sheet | 10.7–25.4 / 19.3 mm | 0.09–0.67 / 0.38 kg | 7,825–17,736 / 10,827 |
| shell | 53.7–303.2 / 159.4 mm | 0.13–4.47 / 1.22 kg | 1,005–52,228 / 8,366 |
| garment | 86.0–269.0 / 158.3 mm | 0.19–1.60 / 0.80 kg | 1,884–43,701 / 7,682 |
| bag | 39.7–233.4 / 124.8 mm | 0.31–2.00 / 1.86 kg | 747–39,309 / 8,371 |

The full-set median shortest axis is 146.3 mm, and 115/150 assets have a
shortest axis above 66 mm. In the tested 30-solid + 30-shell subset, only 0079
and 0090 had a static shortest axis no greater than Piper's 66 mm opening.

## Physical representation

- Solver: stock Genesis 1.1.2 `MPM.Elastic`, corotation model.
- Collision proxy: closed watertight mesh produced by alpha wrapping, then PBS sampled into MPM particles.
- Young's modulus: 31.5–180 kPa; median 60 kPa.
- Density: 180–1080 kg/m3; median 460 kg/m3.
- Particle size: 4/6/7/8/9 mm; median 7 mm.
- Mass: 0.08–4.5 kg; median 1.04 kg.

Physical properties were assigned by the policy “object rule > material
profile > geometry family.” They create stable and visibly differentiated drop
behavior, but the reports contain no physical calibration for robot
manipulation. Friction, contact stiffness, mass caps, effective sheet
thickness, and rebound should therefore be treated as modeling assumptions.

Because the material is purely elastic, it has no plastic residual strain.
Wrinkles made by preloading tend to recover. Tasks that depend on persistent
creases or permanent deformation require a separate evaluation of
`MPM.ElastoPlastic` or another material formulation.

## Representative samples

| ID | Family | Dimensions (m) | Mass | E / nu / rho | Interaction role |
| --- | --- | --- | --- | --- | --- |
| 0001 Chiffon Cake | solid | 0.322 x 0.323 x 0.116 | 0.663 kg | 65 kPa / 0.36 / 600 | Thick solid that resisted global Piper and Franka transport |
| 0031 Tarpaulin | cloth_sheet | 0.329 x 0.223 x 0.016 | 0.294 kg | 120 kPa / 0.34 / 460 | Thin candidate for spreading, edge lifting, and sweeping |
| 0079 Inflatable Mattress | shell | 0.157 x 0.322 x 0.054 | 0.676 kg | 74.34 kPa / 0.40 / 533 | Only successful planar push; two-finger transport still failed |
| 0090 Steam Mop Pad | shell | 0.324 x 0.172 x 0.064 | 0.818 kg | 60 kPa / 0.34 / 460 | Static jaw fit that became too wide after particle settling |
| 0091 Medical Glove | garment | 0.167 x 0.324 x 0.113 | 1.601 kg | 120 kPa / 0.46 / 950 | Garment/rubber example with a relatively heavy volumetric proxy |
| 0121 Backpack Fabric Body | bag | 0.225 x 0.324 x 0.192 | 1.200 kg | 48.6 kPa / 0.34 / 460 | Bag/pre-deformation example with a 1.2 kg mass cap |
