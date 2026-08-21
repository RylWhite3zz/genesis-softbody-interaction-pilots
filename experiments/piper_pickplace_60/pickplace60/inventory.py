"""Discover, screen, and orient the packaged solid/shell collision meshes."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paths import ASSET_ROOT


OPEN_WIDTH_M = 0.066
MAX_UPRIGHT_HEIGHT_M = 0.220
ALLOWED_FAMILIES = frozenset({"solid", "shell"})


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    asset_dir: str
    object_type: str
    canonical_id: str
    geometry_family: str
    collision_mesh: str
    extent_m: tuple[float, float, float]
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
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    mpm = properties.get("mpm", {})
    urdf = properties.get("urdf", {})
    relative_mesh = metadata.get("paths", {}).get("collision_mesh", "collision/collision.obj")
    collision_mesh = (asset_dir / relative_mesh).resolve()
    report_path = asset_dir / "validation" / "genesis_mpm_report.json"
    report = _read_json(report_path) if report_path.is_file() else {}
    estimated_particles = int(report.get("metrics", {}).get("mesh", {}).get("estimated_particles", 0))
    extents = tuple(float(value) for value in urdf.get("extents", ()))
    if len(extents) != 3 or any(not math.isfinite(value) or value <= 0 for value in extents):
        raise ValueError(f"invalid extents in {properties_path}: {extents}")
    if not collision_mesh.is_file():
        raise FileNotFoundError(collision_mesh)
    return AssetSpec(
        asset_id=asset_dir.name,
        asset_dir=str(asset_dir.resolve()),
        object_type=str(metadata.get("object_type", asset_dir.name)),
        canonical_id=str(metadata.get("canonical_id", "")),
        geometry_family=family,
        collision_mesh=str(collision_mesh),
        extent_m=extents,
        youngs_modulus_pa=float(mpm["youngs_modulus"]),
        poisson_ratio=float(mpm["poisson_ratio"]),
        density_kg_m3=float(mpm["density"]),
        particle_size_m=float(mpm["particle_size"]),
        grid_density_per_m=float(mpm["grid_density"]),
        source_dt_s=float(mpm["dt"]),
        source_substeps=int(mpm["substeps"]),
        estimated_particles=estimated_particles,
        source_solver=str(properties.get("solver", "")),
    )


def discover_assets(asset_root: Path = ASSET_ROOT) -> list[AssetSpec]:
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)
    assets = [result for path in sorted(asset_root.iterdir()) if (result := inspect_asset(path)) is not None]
    if not assets:
        raise RuntimeError(f"no solid/shell assets found under {asset_root}")
    return assets


def screen_asset(asset: AssetSpec) -> dict[str, Any]:
    minimum_extent = min(asset.extent_m)
    eligible = bool(minimum_extent <= OPEN_WIDTH_M + 1.0e-12)
    return {
        "eligible": eligible,
        "status": "eligible_geometry" if eligible else "skipped_size",
        "reason": (
            "at least one original mesh axis fits the 66 mm jaw opening"
            if eligible
            else f"minimum original-scale extent {minimum_extent:.6f} m exceeds 0.066000 m jaw opening"
        ),
        "minimum_extent_m": float(minimum_extent),
        "jaw_open_width_m": OPEN_WIDTH_M,
        "scale": 1.0,
    }


def euler_to_matrix(euler_deg: tuple[int, int, int]) -> np.ndarray:
    """Match Genesis/SciPy extrinsic x-y-z Euler convention."""

    roll, pitch, yaw = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cr, cp, cy = np.cos((roll, pitch, yaw))
    sr, sp, sy = np.sin((roll, pitch, yaw))
    return np.asarray(
        [
            [cp * cy, -cr * sy + sr * sp * cy, sr * sy + cr * sp * cy],
            [cp * sy, cr * cy + sr * sp * sy, -sr * cy + cr * sp * sy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )


def _load_vertices(mesh_path: str | Path) -> np.ndarray:
    mesh_path = Path(mesh_path)
    try:
        import trimesh
    except ModuleNotFoundError:
        if mesh_path.suffix.lower() != ".obj":
            raise
        # Geometry screening only needs OBJ vertices. Keeping this tiny fallback
        # makes the no-GPU inventory tests usable before the full sim stack is
        # installed; dynamics still uses trimesh through Genesis-side tooling.
        parsed_vertices: list[tuple[float, float, float]] = []
        with mesh_path.open(encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if not line.startswith("v "):
                    continue
                fields = line.split()
                if len(fields) >= 4:
                    parsed_vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        vertices = np.asarray(parsed_vertices, dtype=np.float64)
    else:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
        raise ValueError(f"mesh has no valid vertices: {mesh_path}")
    return vertices


def select_axis_aligned_orientation(asset: AssetSpec) -> dict[str, Any]:
    """Choose a 90-degree rotation with jaw-fit y and the lowest possible z extent."""

    vertices = _load_vertices(asset.collision_mesh)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for euler in itertools.product((0, 90, 180, 270), repeat=3):
        rotation = euler_to_matrix(euler)
        key = tuple(np.round(rotation, 8).reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        transformed = vertices @ rotation.T
        lower = transformed.min(axis=0)
        upper = transformed.max(axis=0)
        extents = upper - lower
        if extents[1] > OPEN_WIDTH_M + 1.0e-9 or extents[2] > MAX_UPRIGHT_HEIGHT_M:
            continue
        candidates.append(
            {
                "euler_deg": [int(value) for value in euler],
                "rotation_matrix": rotation.tolist(),
                "bounds_before_translation_m": [lower.tolist(), upper.tolist()],
                "extent_after_rotation_m": extents.tolist(),
                "jaw_clearance_m": float(OPEN_WIDTH_M - extents[1]),
            }
        )
    if not candidates:
        raise ValueError(f"no unscaled axis-aligned grasp orientation for {asset.asset_id}")
    candidates.sort(
        key=lambda item: (
            round(float(item["extent_after_rotation_m"][2]), 9),
            -round(float(item["extent_after_rotation_m"][0]), 9),
            -round(float(item["jaw_clearance_m"]), 9),
            tuple(item["euler_deg"]),
        )
    )
    selected = candidates[0]
    selected["scale"] = 1.0
    selected["candidate_orientation_count"] = len(candidates)
    return selected
