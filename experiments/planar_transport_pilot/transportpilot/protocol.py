"""Shared geometry, timing, and hard gates for planar transport."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_PUSH = "push"
ACTION_PRESS_DRAG = "press_drag"
ACTION_EDGE_DRAG = "edge_drag"
ROBOT_PIPER = "piper"
ROBOT_FRANKA = "franka"
ACTIONS = (ACTION_PUSH, ACTION_PRESS_DRAG, ACTION_EDGE_DRAG)
ROBOTS = (ROBOT_PIPER, ROBOT_FRANKA)

TABLE_TOP_Z_M = 0.055
TABLE_CENTER_M = (0.47, 0.0, 0.025)
TABLE_SIZE_M = (0.80, 0.86, 0.060)
PUSH_START_CENTER_XY_M = (0.47, -0.15)
PUSH_GOAL_CENTER_XY_M = (0.47, 0.10)
DRAG_START_CENTER_XY_M = (0.62, 0.0)
DRAG_GOAL_CENTER_XY_M = (0.37, 0.0)
GOAL_SIZE_XY_M = (0.40, 0.40)
SPAWN_GAP_M = 0.004
ROBOT_COUP_FRICTION = 6.0
TABLE_COUP_FRICTION = 1.2
FRAME_PERIOD_S = 1.0 / 24.0
VIDEO_FPS = 24
TRANSPORT_DISTANCE_M = 0.25
MIN_PROGRESS_M = 0.14
FINAL_COM_TOLERANCE_M = 0.085
GOAL_CONTAINMENT_TOLERANCE_M = 0.012

PIPER_HOME_ARM_Q = np.asarray([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
FRANKA_HOME_ARM_Q = np.asarray(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398], dtype=np.float64
)

STAGE_DURATION_S = {
    "settle_clear": 0.35,
    "approach": 0.24,
    "descend": 0.30,
    "preload_hold": 0.18,
    "close": 0.35,
    "transport": 1.30,
    "transport_hold": 0.18,
    "open": 0.25,
    "retreat": 0.30,
    "settle_goal": 0.65,
}


def action_centers(action: str) -> tuple[np.ndarray, np.ndarray]:
    if action == ACTION_PUSH:
        return np.asarray(PUSH_START_CENTER_XY_M), np.asarray(PUSH_GOAL_CENTER_XY_M)
    if action in (ACTION_PRESS_DRAG, ACTION_EDGE_DRAG):
        return np.asarray(DRAG_START_CENTER_XY_M), np.asarray(DRAG_GOAL_CENTER_XY_M)
    raise ValueError(f"unknown action: {action}")


def stage_steps(dt_s: float) -> dict[str, int]:
    return {name: max(1, int(round(duration / dt_s))) for name, duration in STAGE_DURATION_S.items()}


def smoothstep01(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def assess_transport(
    action: str,
    reference: dict[str, Any],
    final: dict[str, Any],
    *,
    active_count_constant: bool,
    goal_particle_fraction: float,
) -> dict[str, Any]:
    start_xy, goal_xy = action_centers(action)
    direction = goal_xy - start_xy
    direction /= np.linalg.norm(direction)
    initial_com = np.asarray(reference["com_m"], dtype=np.float64)
    final_com = np.asarray(final["com_m"], dtype=np.float64)
    aabb_min = np.asarray(final["aabb_min_m"], dtype=np.float64)
    aabb_max = np.asarray(final["aabb_max_m"], dtype=np.float64)
    progress = float(np.dot(final_com[:2] - initial_com[:2], direction))
    goal_error = float(np.linalg.norm(final_com[:2] - goal_xy))
    half_goal = np.asarray(GOAL_SIZE_XY_M, dtype=np.float64) / 2.0
    tolerance = GOAL_CONTAINMENT_TOLERANCE_M
    extent = aabb_max - aabb_min
    finite_values = np.asarray(
        [*final_com, *aabb_min, *aabb_max, final["speed_rms_m_s"], progress, goal_error],
        dtype=np.float64,
    )
    gates = {
        "finite_state": bool(reference["finite"] and final["finite"] and np.isfinite(finite_values).all()),
        "active_particle_count_constant": bool(active_count_constant),
        "planar_progress_at_least_0_14m": bool(progress >= MIN_PROGRESS_M),
        "final_com_within_0_085m_of_goal": bool(goal_error <= FINAL_COM_TOLERANCE_M),
        "at_least_85pct_particles_inside_colored_goal_xy": bool(goal_particle_fraction >= 0.85),
        "object_remains_on_table": bool(
            TABLE_TOP_Z_M - 0.006 <= aabb_min[2] <= TABLE_TOP_Z_M + 0.020
        ),
        "final_particle_rms_speed_under_0_10m_s": bool(final["speed_rms_m_s"] <= 0.10),
        "object_extent_not_exploded": bool(np.all(extent <= np.asarray((0.50, 0.50, 0.45)))),
    }
    return {
        "status": "success" if all(gates.values()) else "failed_gates",
        "gates": gates,
        "planar_progress_m": progress,
        "final_goal_error_m": goal_error,
        "goal_particle_fraction": float(goal_particle_fraction),
        "full_aabb_inside_goal_with_tolerance": bool(
            np.all(aabb_min[:2] >= goal_xy - half_goal - tolerance)
            and np.all(aabb_max[:2] <= goal_xy + half_goal + tolerance)
        ),
        "initial_com_m": initial_com.tolist(),
        "final_com_m": final_com.tolist(),
    }
