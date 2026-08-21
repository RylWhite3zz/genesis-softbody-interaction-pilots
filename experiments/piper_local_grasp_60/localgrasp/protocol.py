"""Geometry, timing, and hard success gates for local-pinching pick-place."""

from __future__ import annotations

from typing import Any

import numpy as np


OPEN_WIDTH_M = 0.066
PAD_SIZE_XZ_M = (0.030, 0.030)
PAD_GROUND_CLEARANCE_M = 0.006
TABLE_TOP_Z_M = 0.055
TABLE_CENTER_M = (0.47, -0.015, 0.025)
TABLE_SIZE_M = (0.80, 1.01, 0.060)
PICK_CENTER_XY_M = (0.47, -0.235)
TRAY_CENTER_XY_M = (0.47, 0.205)
TRAY_INNER_SIZE_XY_M = (0.44, 0.44)
TRAY_WALL_THICKNESS_M = 0.010
TRAY_BOTTOM_THICKNESS_M = 0.008
TRAY_WALL_HEIGHT_M = 0.035
TRAY_FLOOR_Z_M = TABLE_TOP_Z_M + TRAY_BOTTOM_THICKNESS_M
SPAWN_GAP_M = 0.004
LIFT_HEIGHT_M = 0.220
ROBOT_COUP_FRICTION = 6.0
FRAME_PERIOD_S = 1.0 / 24.0
VIDEO_FPS = 24
HOME_ARM_Q = np.asarray([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
HOME_GRIP_CENTER_M = np.asarray([0.4765879444, 0.0, 0.3683466240], dtype=np.float64)

STAGE_DURATION_S = {
    "settle_clear": 0.35,
    "approach": 0.28,
    "descend": 0.38,
    "preload": 0.40,
    "close": 0.45,
    "grip_hold": 0.18,
    "probe_lift": 0.70,
    "probe_hold": 0.24,
    "transfer": 1.05,
    "lower_into_tray": 0.65,
    "open": 0.35,
    "retreat": 0.45,
    "settle_in_tray": 0.65,
}


def stage_steps(dt_s: float) -> dict[str, int]:
    return {name: max(1, int(round(duration / dt_s))) for name, duration in STAGE_DURATION_S.items()}


def smoothstep01(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def tray_boxes() -> list[dict[str, tuple[float, float, float]]]:
    cx, cy = TRAY_CENTER_XY_M
    inner_x, inner_y = TRAY_INNER_SIZE_XY_M
    wall = TRAY_WALL_THICKNESS_M
    outer_x, outer_y = inner_x + 2.0 * wall, inner_y + 2.0 * wall
    bottom_z = TABLE_TOP_Z_M + TRAY_BOTTOM_THICKNESS_M / 2.0
    wall_z = TRAY_FLOOR_Z_M + TRAY_WALL_HEIGHT_M / 2.0
    return [
        {"pos": (cx, cy, bottom_z), "size": (outer_x, outer_y, TRAY_BOTTOM_THICKNESS_M)},
        {"pos": (cx - (inner_x + wall) / 2.0, cy, wall_z), "size": (wall, outer_y, TRAY_WALL_HEIGHT_M)},
        {"pos": (cx + (inner_x + wall) / 2.0, cy, wall_z), "size": (wall, outer_y, TRAY_WALL_HEIGHT_M)},
        {"pos": (cx, cy - (inner_y + wall) / 2.0, wall_z), "size": (inner_x, wall, TRAY_WALL_HEIGHT_M)},
        {"pos": (cx, cy + (inner_y + wall) / 2.0, wall_z), "size": (inner_x, wall, TRAY_WALL_HEIGHT_M)},
    ]


def assess_probe(reference: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    com_delta = float(probe["com_m"][2] - reference["com_m"][2])
    min_z = float(probe["aabb_min_m"][2])
    passed = bool(
        reference["active_count"] == probe["active_count"]
        and reference["finite"]
        and probe["finite"]
        and com_delta >= 0.045
        and min_z >= TABLE_TOP_Z_M + 0.008
    )
    return {
        "passed": passed,
        "com_lift_delta_z_m": com_delta,
        "aabb_min_z_m": min_z,
        "required_com_lift_m": 0.045,
        "required_aabb_min_z_m": TABLE_TOP_Z_M + 0.008,
    }


def assess_success(metrics: dict[str, Any]) -> dict[str, Any]:
    aabb_min = np.asarray(metrics["final_aabb_min_m"], dtype=np.float64)
    aabb_max = np.asarray(metrics["final_aabb_max_m"], dtype=np.float64)
    cx, cy = TRAY_CENTER_XY_M
    inner_x, inner_y = TRAY_INNER_SIZE_XY_M
    tolerance = 0.008
    extents = aabb_max - aabb_min
    finite_values = np.asarray(
        [*aabb_min, *aabb_max, metrics["max_lift_delta_z_m"], metrics["final_speed_rms_m_s"]],
        dtype=np.float64,
    )
    gates = {
        "finite_state": bool(np.isfinite(finite_values).all() and metrics["finite_state"]),
        "active_particle_count_constant": bool(metrics["active_particle_count_constant"]),
        "probe_lift_passed": bool(metrics["probe_lift_passed"]),
        "object_still_lifted_at_transfer_end": bool(
            metrics["transfer_end_com_delta_z_m"] >= 0.045
            and metrics["transfer_end_aabb_min_z_m"] >= TABLE_TOP_Z_M + 0.008
        ),
        "all_particles_horizontally_inside_tray": bool(
            aabb_min[0] >= cx - inner_x / 2.0 - tolerance
            and aabb_max[0] <= cx + inner_x / 2.0 + tolerance
            and aabb_min[1] >= cy - inner_y / 2.0 - tolerance
            and aabb_max[1] <= cy + inner_y / 2.0 + tolerance
        ),
        "object_resting_on_tray_floor": bool(
            TRAY_FLOOR_Z_M - 0.012 <= aabb_min[2] <= TRAY_FLOOR_Z_M + 0.035
        ),
        "object_released_from_local_grasp": bool(metrics["final_grip_to_object_aabb_distance_m"] >= 0.075),
        "final_particle_rms_speed_under_0_12m_s": bool(metrics["final_speed_rms_m_s"] <= 0.12),
        "object_extent_not_exploded": bool(np.all(extents <= np.asarray((0.55, 0.55, 0.50)))),
        "video_has_at_least_60_nonblank_frames": bool(
            metrics["frame_count"] >= 60 and metrics["minimum_frame_std"] >= 1.0
        ),
    }
    return {"status": "success" if all(gates.values()) else "failed_gates", "gates": gates}
