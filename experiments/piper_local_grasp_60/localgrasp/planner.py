"""Generate local edge/corner pinches from the settled active MPM particles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import OPEN_WIDTH_M, PAD_GROUND_CLEARANCE_M, PAD_SIZE_XZ_M, TABLE_TOP_Z_M


def _candidate_for_patch(
    positions: np.ndarray,
    close_unit_xy: np.ndarray,
    tangent_unit_xy: np.ndarray,
    wrist_yaw_deg: float,
    tangent_center: float,
    z_center: float,
    particle_size_m: float,
) -> dict[str, Any] | None:
    tangent_half = PAD_SIZE_XZ_M[0] / 2.0
    vertical_half = PAD_SIZE_XZ_M[1] / 2.0
    close_coordinates = positions[:, :2] @ close_unit_xy
    tangent_coordinates = positions[:, :2] @ tangent_unit_xy
    mask = (
        (np.abs(tangent_coordinates - tangent_center) <= tangent_half)
        & (np.abs(positions[:, 2] - z_center) <= vertical_half)
    )
    patch = positions[mask]
    patch_close = close_coordinates[mask]
    minimum_count = max(8, int(round(0.35 * PAD_SIZE_XZ_M[0] * PAD_SIZE_XZ_M[1] / particle_size_m**2)))
    if len(patch) < minimum_count:
        return None
    low = float(np.min(patch_close))
    high = float(np.max(patch_close))
    width = high - low
    # Coarse MPM sampling stair-steps a curved edge: the outermost occupied
    # chord can be 1-3 particles wider than the source surface. Permit limited
    # force-capped entry compression, while still rejecting a thick interior.
    maximum_entry_width = OPEN_WIDTH_M + max(0.018, 2.5 * particle_size_m)
    if not 1.25 * particle_size_m <= width <= maximum_entry_width + 1.0e-9:
        return None
    surface_band = max(1.6 * particle_size_m, 0.006)
    low_support = int(np.count_nonzero(patch_close <= low + surface_band))
    high_support = int(np.count_nonzero(patch_close >= high - surface_band))
    if min(low_support, high_support) < 3:
        return None
    center = np.asarray(
        [*((low + high) / 2.0 * close_unit_xy + tangent_center * tangent_unit_xy), z_center],
        dtype=np.float64,
    )
    close_angle = math.degrees(math.atan2(close_unit_xy[1], close_unit_xy[0]))
    return {
        "grasp_center_m": center.tolist(),
        "close_axis": f"xy_{close_angle:.0f}deg",
        "close_axis_unit_xy": close_unit_xy.tolist(),
        "tangent_axis_unit_xy": tangent_unit_xy.tolist(),
        "wrist_yaw_deg": float(wrist_yaw_deg),
        "local_width_m": width,
        "commanded_close_width_m": max(
            0.012,
            min(0.040, width - max(0.018, 2.5 * particle_size_m)),
        ),
        "requires_entry_compression": bool(width > OPEN_WIDTH_M),
        "entry_interference_m": max(0.0, width - OPEN_WIDTH_M),
        "patch_particle_count": int(len(patch)),
        "low_surface_support_count": low_support,
        "high_surface_support_count": high_support,
        "patch_tangent_center_m": float(tangent_center),
        "patch_z_center_m": float(z_center),
    }


def generate_local_grasps(
    positions: np.ndarray,
    particle_size_m: float,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must be non-empty (N, 3)")
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    com = positions.mean(axis=0)
    min_center_z = TABLE_TOP_Z_M + PAD_GROUND_CLEARANCE_M + PAD_SIZE_XZ_M[1] / 2.0
    max_center_z = upper[2] - min(0.006, particle_size_m)
    if max_center_z < min_center_z:
        return []
    z_values = np.unique(
        np.clip(
            np.asarray(
                [
                    min_center_z,
                    lower[2] + 0.42 * (upper[2] - lower[2]),
                    lower[2] + 0.62 * (upper[2] - lower[2]),
                    lower[2] + 0.78 * (upper[2] - lower[2]),
                ]
            ),
            min_center_z,
            max_center_z,
        ).round(6)
    )
    candidates: list[dict[str, Any]] = []
    for wrist_yaw_deg in (0.0, -90.0, -45.0, 45.0):
        yaw = np.deg2rad(wrist_yaw_deg)
        tangent_unit = np.asarray((np.cos(yaw), np.sin(yaw)), dtype=np.float64)
        close_unit = np.asarray((-np.sin(yaw), np.cos(yaw)), dtype=np.float64)
        tangent_coordinates = positions[:, :2] @ tangent_unit
        close_coordinates = positions[:, :2] @ close_unit
        com_tangent = float(com[:2] @ tangent_unit)
        com_close = float(com[:2] @ close_unit)
        tangent_lower = float(np.min(tangent_coordinates))
        tangent_upper = float(np.max(tangent_coordinates))
        close_lower = float(np.min(close_coordinates))
        close_upper = float(np.max(close_coordinates))
        # Let most of a pad overhang the perimeter. A local edge pinch generally
        # contacts only the inner strip of the 30 mm pad; centering the full pad
        # inside a large round object would incorrectly measure the thick interior.
        overhang = PAD_SIZE_XZ_M[0] * 0.50
        inward = PAD_SIZE_XZ_M[0] * 0.75
        tangent_values = np.unique(
            np.concatenate(
                (
                    np.linspace(tangent_lower - overhang, tangent_lower + inward, 11),
                    np.linspace(tangent_upper - inward, tangent_upper + overhang, 11),
                )
            ).round(6)
        )
        for tangent_center in tangent_values:
            for z_center in z_values:
                candidate = _candidate_for_patch(
                    positions,
                    close_unit,
                    tangent_unit,
                    wrist_yaw_deg,
                    float(tangent_center),
                    float(z_center),
                    particle_size_m,
                )
                if candidate is None:
                    continue
                center = np.asarray(candidate["grasp_center_m"])
                center_tangent = float(center[:2] @ tangent_unit)
                center_close = float(center[:2] @ close_unit)
                tangent_distance = abs(center_tangent - com_tangent)
                tangent_radius = max(tangent_upper - tangent_lower, 1.0e-6) / 2.0
                edge_fraction = min(1.0, tangent_distance / tangent_radius)
                close_center_offset = abs(center_close - com_close)
                close_radius = max(close_upper - close_lower, 1.0e-6) / 2.0
                corner_fraction = min(1.0, close_center_offset / close_radius)
                candidate["grasp_type"] = "corner" if corner_fraction >= 0.45 else "edge"
                candidate["edge_fraction"] = float(edge_fraction)
                candidate["corner_fraction"] = float(corner_fraction)
                # Favor supported, moderately deep edge pinches near the upper half.
                candidate["score"] = float(
                    2.0 * min(candidate["low_surface_support_count"], candidate["high_surface_support_count"])
                    + 0.15 * candidate["patch_particle_count"]
                    + 30.0 * edge_fraction
                    - 180.0 * abs(candidate["local_width_m"] - 0.038)
                    - 12.0 * corner_fraction
                    - 70.0 * center[0]
                )
                candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["score"], item["local_width_m"], item["wrist_yaw_deg"]))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        center = np.asarray(candidate["grasp_center_m"])
        if any(
            candidate["close_axis"] == previous["close_axis"]
            and np.linalg.norm(center - np.asarray(previous["grasp_center_m"])) < 0.035
            for previous in selected
        ):
            continue
        candidate["rank"] = len(selected)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected
