"""Pure configuration and success gates for the tray pick-and-place attempt."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DT_S = 0.002
SUBSTEPS = 64
OPEN_WIDTH_M = 0.066
TABLE_TOP_Z_M = 0.055
TABLE_CENTER_M = (0.47, 0.0, 0.025)
TABLE_SIZE_M = (0.78, 0.92, 0.060)
PICK_CENTER_XY_M = (0.48, -0.14)
TRAY_CENTER_XY_M = (0.48, 0.16)
TRAY_INNER_SIZE_XY_M = (0.44, 0.44)
TRAY_WALL_THICKNESS_M = 0.010
TRAY_BOTTOM_THICKNESS_M = 0.008
TRAY_WALL_HEIGHT_M = 0.035
TRAY_FLOOR_Z_M = TABLE_TOP_Z_M + TRAY_BOTTOM_THICKNESS_M
SPAWN_GAP_M = 0.004
LIFT_HEIGHT_M = 0.145
GRASP_HEIGHT_FRACTION = 0.40
ROBOT_COUP_FRICTION = 6.0
FRAME_EVERY_STEPS = 40
VIDEO_FPS = 24
HOME_ARM_Q = np.asarray([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
HOME_GRIP_CENTER_M = np.asarray([0.4765879444, 0.0, 0.3683466240], dtype=np.float64)

STAGE_STEPS = {
    "initial_hold": 60,
    "approach": 40,
    "descend": 100,
    "close": 170,
    "grip_hold": 80,
    "lift": 300,
    "lifted_hold": 150,
    "transfer": 800,
    "lower_into_tray": 400,
    "open": 250,
    "retreat": 300,
    "settle_in_tray": 400,
}


def smoothstep01(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


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


def assess_success(metrics: dict[str, Any]) -> dict[str, Any]:
    aabb_min = np.asarray(metrics["final_aabb_min_m"], dtype=np.float64)
    aabb_max = np.asarray(metrics["final_aabb_max_m"], dtype=np.float64)
    cx, cy = TRAY_CENTER_XY_M
    inner_x, inner_y = TRAY_INNER_SIZE_XY_M
    tolerance = 0.008
    horizontal_inside = bool(
        aabb_min[0] >= cx - inner_x / 2.0 - tolerance
        and aabb_max[0] <= cx + inner_x / 2.0 + tolerance
        and aabb_min[1] >= cy - inner_y / 2.0 - tolerance
        and aabb_max[1] <= cy + inner_y / 2.0 + tolerance
    )
    resting_height = bool(TRAY_FLOOR_Z_M - 0.012 <= aabb_min[2] <= TRAY_FLOOR_Z_M + 0.035)
    extents = aabb_max - aabb_min
    finite_values = np.asarray(
        [
            *aabb_min,
            *aabb_max,
            float(metrics["max_lift_delta_z_m"]),
            float(metrics["final_speed_rms_m_s"]),
            float(metrics["final_grip_object_distance_m"]),
        ],
        dtype=np.float64,
    )
    gates = {
        "finite_state": bool(np.isfinite(finite_values).all() and metrics.get("finite_state", False)),
        "active_particle_count_constant": bool(metrics.get("active_particle_count_constant", False)),
        "object_lifted_over_50mm": float(metrics["max_lift_delta_z_m"]) >= 0.050,
        "object_still_lifted_at_transfer_end": bool(
            float(metrics["transfer_end_com_delta_z_m"]) >= 0.050
            and float(metrics["transfer_end_aabb_min_z_m"]) >= TABLE_TOP_Z_M + 0.010
        ),
        "all_particles_horizontally_inside_tray": horizontal_inside,
        "object_resting_on_tray_floor": resting_height,
        "object_released_from_gripper": float(metrics["final_grip_object_distance_m"]) >= 0.075,
        "final_particle_rms_speed_under_0_12m_s": float(metrics["final_speed_rms_m_s"]) <= 0.12,
        "object_extent_not_exploded": bool(np.all(extents <= np.asarray((0.55, 0.40, 0.40)))),
        "video_has_at_least_60_nonblank_frames": bool(
            int(metrics.get("frame_count", 0)) >= 60 and float(metrics.get("minimum_frame_std", 0.0)) >= 1.0
        ),
    }
    return {
        "status": "success" if all(gates.values()) else "failed_gates",
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "tray_horizontal_tolerance_m": tolerance,
    }


def validate_protocol() -> None:
    if not math.isclose(TABLE_CENTER_M[2] + TABLE_SIZE_M[2] / 2.0, TABLE_TOP_Z_M, abs_tol=1e-12):
        raise ValueError("table top geometry is inconsistent")
    if DT_S / SUBSTEPS > 0.02 / 160.0:
        raise ValueError("substep dt exceeds Genesis MPM guidance for the densest candidate")
    if any(steps <= 0 for steps in STAGE_STEPS.values()):
        raise ValueError("all stages need positive step counts")
