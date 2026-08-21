# Assets

`catalog/delivery_manifest.jsonl` is the original 150-item delivery manifest.
`representative/` contains six lightweight samples selected to explain the
geometry families and the observed manipulation failures.

| ID | Family | Object | Why it is included |
| --- | --- | --- | --- |
| 0001 | solid | Chiffon Cake | Thick solid used in failed Piper/Franka push, press-drag, and edge-drag trials |
| 0031 | cloth_sheet | Tarpaulin | A 16 mm sheet representative of spreading, folding, and sweeping tasks |
| 0079 | shell | Inflatable Mattress | The only planar-transport success and a dynamic candidate in both grasping studies |
| 0090 | shell | Steam Mop Pad | A jaw-width boundary case: 64.46 mm statically and 73.59 mm after particle settling |
| 0091 | garment | Medical Glove | Representative garment and high-Poisson-ratio rubber profile |
| 0121 | bag | Backpack Fabric Body | Representative bag, pre-deformation rule, and capped-mass case |

Each lightweight pack contains:

```text
collision/collision.obj
caption.txt
captions.json
metadata.json
physical_properties.json
preview.gif
validation/genesis_mpm_report.json
validation/urdf_import_report.json
validation/urdf_import.png
```

The full visual OBJ, UV texture, and `asset.urdf` are intentionally omitted.
The original URDF references both visual and collision files, so retaining it
without its large visual dependencies would create a broken partial package.
The interaction code samples `collision.obj` directly into MPM particles, so
the relevant dynamics input remains available.
