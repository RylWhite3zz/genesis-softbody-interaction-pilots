"""Discover the 30 solid and 30 shell MPM.Elastic assets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import ASSET_ROOT


ALLOWED_FAMILIES = frozenset({"solid", "shell"})


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    object_type: str
    canonical_id: str
    geometry_family: str
    collision_mesh: str
    extent_m: tuple[float, float, float]
    mass_kg: float
    youngs_modulus_pa: float
    poisson_ratio: float
    density_kg_m3: float
    particle_size_m: float
    grid_density_per_m: float
    source_dt_s: float
    source_substeps: int
    estimated_particles: int
    source_solver: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def inspect_asset(asset_dir: Path) -> AssetSpec | None:
    metadata_path = asset_dir / "metadata.json"
    properties_path = asset_dir / "physical_properties.json"
    if not metadata_path.is_file() or not properties_path.is_file():
        return None
    metadata = _read_json(metadata_path)
    family = str(metadata.get("geometry_family", ""))
    if family not in ALLOWED_FAMILIES:
        return None
    properties = _read_json(properties_path)
    mpm = properties["mpm"]
    urdf = properties["urdf"]
    mesh = (asset_dir / metadata.get("paths", {}).get("collision_mesh", "collision/collision.obj")).resolve()
    report_path = asset_dir / "validation" / "genesis_mpm_report.json"
    report = _read_json(report_path) if report_path.is_file() else {}
    extents = tuple(float(value) for value in urdf["extents"])
    if len(extents) != 3 or any(value <= 0.0 or not math.isfinite(value) for value in extents):
        raise ValueError(f"invalid extents in {properties_path}")
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    return AssetSpec(
        asset_id=asset_dir.name,
        object_type=str(metadata.get("object_type", asset_dir.name)),
        canonical_id=str(metadata.get("canonical_id", "")),
        geometry_family=family,
        collision_mesh=str(mesh),
        extent_m=extents,
        mass_kg=float(urdf["mass"]),
        youngs_modulus_pa=float(mpm["youngs_modulus"]),
        poisson_ratio=float(mpm["poisson_ratio"]),
        density_kg_m3=float(mpm["density"]),
        particle_size_m=float(mpm["particle_size"]),
        grid_density_per_m=float(mpm["grid_density"]),
        source_dt_s=float(mpm["dt"]),
        source_substeps=int(mpm["substeps"]),
        estimated_particles=int(report.get("metrics", {}).get("mesh", {}).get("estimated_particles", 0)),
        source_solver=str(properties.get("solver", "")),
    )


def discover_assets() -> list[AssetSpec]:
    if not ASSET_ROOT.is_dir():
        raise FileNotFoundError(ASSET_ROOT)
    assets = [asset for path in sorted(ASSET_ROOT.iterdir()) if (asset := inspect_asset(path)) is not None]
    if not assets:
        raise RuntimeError(f"no solid/shell assets found under {ASSET_ROOT}")
    return assets
