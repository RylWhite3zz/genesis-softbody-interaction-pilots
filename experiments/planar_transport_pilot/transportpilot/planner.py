"""Generate nominal straight-line push and press-drag tool paths."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import ACTION_EDGE_DRAG, ACTION_PRESS_DRAG, ACTION_PUSH, TABLE_TOP_Z_M, action_centers


def nominal_reference(extent_m: tuple[float, float, float], action: str) -> dict[str, Any]:
    start_xy, _ = action_centers(action)
    extent = np.asarray(extent_m, dtype=np.float64)
    minimum = np.asarray(
        [start_xy[0] - extent[0] / 2.0, start_xy[1] - extent[1] / 2.0, TABLE_TOP_Z_M]
    )
    maximum = minimum + extent
    return {
        "com_m": (minimum + maximum) / 2.0,
        "aabb_min_m": minimum,
        "aabb_max_m": maximum,
        "active_count": 1,
        "finite": True,
    }


def make_tool_path(
    action: str,
    reference: dict[str, Any],
    tool_vertical_offsets_m: tuple[float, float],
    *,
    path_samples: int = 13,
) -> dict[str, Any]:
    if path_samples < 3:
        raise ValueError("path_samples must be at least 3")
    start_xy, goal_xy = action_centers(action)
    direction_xy = goal_xy - start_xy
    direction_xy /= np.linalg.norm(direction_xy)
    com = np.asarray(reference["com_m"], dtype=np.float64)
    lower = np.asarray(reference["aabb_min_m"], dtype=np.float64)
    upper = np.asarray(reference["aabb_max_m"], dtype=np.float64)
    lower_tool_offset, upper_tool_offset = tool_vertical_offsets_m
    tool_mid_offset = 0.5 * (lower_tool_offset + upper_tool_offset)
    height = float(upper[2] - lower[2])

    if action == ACTION_PUSH:
        desired_contact_z = lower[2] + min(max(0.42 * height, 0.018), 0.040)
        tool_z = desired_contact_z - tool_mid_offset
        half_extent_xy = 0.5 * (upper[:2] - lower[:2])
        support_radius = float(np.dot(np.abs(direction_xy), half_extent_xy))
        contact_xy = com[:2] - (support_radius + 0.018) * direction_xy
        start_tool = np.asarray((*contact_xy, tool_z), dtype=np.float64)
        end_tool = start_tool.copy()
        end_tool[:2] += 0.340 * direction_xy
        close_axis_yaw = math.degrees(math.atan2(direction_xy[1], direction_xy[0])) - 90.0
        yaw_candidates_deg = (close_axis_yaw, close_axis_yaw + 180.0)
        preload_m = 0.0
    elif action == ACTION_PRESS_DRAG:
        press_xy = np.asarray((upper[0] - 0.025, com[1]), dtype=np.float64)
        # The particle surface and position-controlled fingertip both have a
        # few millimeters of discretization/tracking slack. Keep a meaningful
        # preload while leaving ample clearance above the table.
        penetration_m = min(0.026, max(0.020, 0.45 * height))
        tool_z = upper[2] - penetration_m - lower_tool_offset
        start_tool = np.asarray((*press_xy, tool_z), dtype=np.float64)
        end_tool = start_tool.copy()
        end_tool[:2] += 0.25 * direction_xy
        yaw_candidates_deg = (0.0, -90.0, 90.0, 180.0)
        preload_m = penetration_m
    else:
        raise ValueError(f"unknown action: {action}")

    transport = np.linspace(start_tool, end_tool, path_samples)
    return {
        "action": action,
        "start_object_center_xy_m": start_xy.tolist(),
        "goal_center_xy_m": goal_xy.tolist(),
        "direction_xy": direction_xy.tolist(),
        "approach_grip_center_m": (start_tool + np.asarray((0.0, 0.0, 0.10))).tolist(),
        "contact_grip_center_m": start_tool.tolist(),
        "transport_grip_centers_m": transport.tolist(),
        "retreat_grip_center_m": (end_tool + np.asarray((0.0, 0.0, 0.12))).tolist(),
        "yaw_candidates_deg": list(yaw_candidates_deg),
        "surface_preload_m": preload_m,
        "tool_vertical_offsets_m": [lower_tool_offset, upper_tool_offset],
    }


def generate_far_edge_grasp(
    positions: np.ndarray,
    particle_size_m: float,
    open_width_m: float,
    tool_vertical_offsets_m: tuple[float, float],
) -> dict[str, Any] | None:
    """Find a reachable axial or diagonal pinch on the far (+x) perimeter."""
    positions = np.asarray(positions, dtype=np.float64)
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    com = positions.mean(axis=0)
    tool_mid_offset = 0.5 * sum(tool_vertical_offsets_m)
    pad_half_tangent = 0.015
    pad_half_z = 0.015
    min_pad_z = TABLE_TOP_Z_M + 0.006 + pad_half_z
    max_pad_z = upper[2] - min(0.005, particle_size_m)
    if max_pad_z < min_pad_z:
        return None
    z_centers = np.unique(
        np.clip(
            np.asarray((min_pad_z, lower[2] + 0.50 * (upper[2] - lower[2]), lower[2] + 0.72 * (upper[2] - lower[2]))),
            min_pad_z,
            max_pad_z,
        ).round(6)
    )
    candidates: list[dict[str, Any]] = []
    for yaw_deg in (0.0, -45.0, 45.0, -90.0):
        yaw = math.radians(yaw_deg)
        tangent = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
        close_axis = np.asarray((-math.sin(yaw), math.cos(yaw)), dtype=np.float64)
        tangent_coordinates = positions[:, :2] @ tangent
        close_coordinates = positions[:, :2] @ close_axis
        tangent_min = float(np.min(tangent_coordinates))
        tangent_max = float(np.max(tangent_coordinates))
        tangent_centers = np.unique(
            np.concatenate(
                (
                    np.linspace(tangent_min - 0.015, tangent_min + 0.025, 9),
                    np.linspace(tangent_max - 0.025, tangent_max + 0.015, 9),
                )
            ).round(6)
        )
        for tangent_center in tangent_centers:
            for pad_z in z_centers:
                mask = (
                    (np.abs(tangent_coordinates - tangent_center) <= pad_half_tangent)
                    & (np.abs(positions[:, 2] - pad_z) <= pad_half_z)
                )
                patch = positions[mask]
                patch_close = close_coordinates[mask]
                if len(patch) < 6:
                    continue
                low_close = float(np.min(patch_close))
                high_close = float(np.max(patch_close))
                width = high_close - low_close
                entry_allowance = max(0.020, 2.5 * particle_size_m)
                if not 1.25 * particle_size_m <= width <= open_width_m + entry_allowance:
                    continue
                surface_band = max(0.006, 1.6 * particle_size_m)
                low_support = int(np.count_nonzero(patch_close <= low_close + surface_band))
                high_support = int(np.count_nonzero(patch_close >= high_close - surface_band))
                if min(low_support, high_support) < 2:
                    continue
                center_xy = 0.5 * (low_close + high_close) * close_axis + tangent_center * tangent
                radial_reach = float(np.linalg.norm(center_xy))
                if center_xy[0] < com[0] + 0.055 or center_xy[0] > 0.795 or radial_reach > 0.81:
                    continue
                grip_z = float(
                    max(
                        pad_z - tool_mid_offset,
                        TABLE_TOP_Z_M + 0.006 - tool_vertical_offsets_m[0],
                    )
                )
                candidates.append(
                    {
                        "grasp_center_m": [float(center_xy[0]), float(center_xy[1]), grip_z],
                        "desired_pad_center_z_m": float(pad_z),
                        "wrist_yaw_deg": yaw_deg,
                        "local_width_m": width,
                        "commanded_close_width_m": max(
                            0.012,
                            min(0.050, width - max(0.014, 2.0 * particle_size_m)),
                        ),
                        "patch_particle_count": int(len(patch)),
                        "low_surface_support_count": low_support,
                        "high_surface_support_count": high_support,
                        "radial_reach_m": radial_reach,
                        "score": float(
                            4.0 * min(low_support, high_support)
                            + 0.25 * len(patch)
                            - 180.0 * abs(width - 0.045)
                            + 45.0 * (center_xy[0] - com[0])
                            - 18.0 * abs(center_xy[1] - com[1])
                        ),
                    }
                )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate["score"])


def make_edge_drag_tool_path(candidate: dict[str, Any], reference: dict[str, Any], *, path_samples: int = 13) -> dict[str, Any]:
    start_xy, goal_xy = action_centers(ACTION_EDGE_DRAG)
    direction_xy = goal_xy - start_xy
    direction_xy /= np.linalg.norm(direction_xy)
    start_tool = np.asarray(candidate["grasp_center_m"], dtype=np.float64)
    end_tool = start_tool.copy()
    end_tool[:2] += 0.25 * direction_xy
    return {
        "action": ACTION_EDGE_DRAG,
        "start_object_center_xy_m": start_xy.tolist(),
        "goal_center_xy_m": goal_xy.tolist(),
        "direction_xy": direction_xy.tolist(),
        "approach_grip_center_m": (start_tool + np.asarray((0.0, 0.0, 0.10))).tolist(),
        "contact_grip_center_m": start_tool.tolist(),
        "transport_grip_centers_m": np.linspace(start_tool, end_tool, path_samples).tolist(),
        "retreat_grip_center_m": (end_tool + np.asarray((0.0, 0.0, 0.12))).tolist(),
        "yaw_candidates_deg": [candidate["wrist_yaw_deg"], candidate["wrist_yaw_deg"] + 180.0],
        "surface_preload_m": 0.0,
        "candidate": candidate,
        "commanded_close_width_m": candidate["commanded_close_width_m"],
        "reference_com_m": np.asarray(reference["com_m"]).tolist(),
    }
