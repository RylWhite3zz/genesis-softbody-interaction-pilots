from __future__ import annotations

import unittest

import numpy as np

from transportpilot.assets import discover_assets
from transportpilot.planner import generate_far_edge_grasp, make_tool_path, nominal_reference
from transportpilot.protocol import (
    ACTION_PRESS_DRAG,
    ACTION_PUSH,
    PUSH_GOAL_CENTER_XY_M,
    PUSH_START_CENTER_XY_M,
    TABLE_TOP_Z_M,
    assess_transport,
)


class ProtocolPlannerTests(unittest.TestCase):
    def test_representative_inventory_contains_a_solid_and_shell(self) -> None:
        assets = discover_assets()
        self.assertGreaterEqual(len(assets), 3)
        self.assertIn("solid", {asset.geometry_family for asset in assets})
        self.assertIn("shell", {asset.geometry_family for asset in assets})
        self.assertIn(
            "0079_stage3_100k_qwen2_v1_shell_025403",
            {asset.asset_id for asset in assets},
        )

    def test_push_and_drag_have_opposite_transport_directions(self) -> None:
        extent = (0.32, 0.20, 0.10)
        push = make_tool_path(ACTION_PUSH, nominal_reference(extent, ACTION_PUSH), (-0.015, 0.015))
        drag = make_tool_path(
            ACTION_PRESS_DRAG,
            nominal_reference(extent, ACTION_PRESS_DRAG),
            (-0.015, 0.015),
        )
        push_path = np.asarray(push["transport_grip_centers_m"])
        drag_path = np.asarray(drag["transport_grip_centers_m"])
        self.assertGreater(push_path[-1, 1], push_path[0, 1])
        self.assertLess(drag_path[-1, 0], drag_path[0, 0])
        self.assertAlmostEqual(push_path[-1, 0], push_path[0, 0])
        self.assertAlmostEqual(drag_path[-1, 1], drag_path[0, 1])

    def test_success_requires_real_progress_and_goal_containment(self) -> None:
        reference = {
            "com_m": np.asarray((*PUSH_START_CENTER_XY_M, 0.10)),
            "aabb_min_m": np.asarray((0.31, -0.25, TABLE_TOP_Z_M)),
            "aabb_max_m": np.asarray((0.63, -0.05, 0.15)),
            "speed_rms_m_s": 0.0,
            "active_count": 100,
            "finite": True,
        }
        final = {
            "com_m": np.asarray((*PUSH_GOAL_CENTER_XY_M, 0.10)),
            "aabb_min_m": np.asarray((0.31, 0.00, TABLE_TOP_Z_M)),
            "aabb_max_m": np.asarray((0.63, 0.20, 0.15)),
            "speed_rms_m_s": 0.01,
            "active_count": 100,
            "finite": True,
        }
        passed = assess_transport(
            ACTION_PUSH,
            reference,
            final,
            active_count_constant=True,
            goal_particle_fraction=1.0,
        )
        self.assertEqual(passed["status"], "success")
        failed = assess_transport(
            ACTION_PUSH,
            reference,
            reference,
            active_count_constant=True,
            goal_particle_fraction=0.0,
        )
        self.assertEqual(failed["status"], "failed_gates")
        self.assertFalse(failed["gates"]["planar_progress_at_least_0_14m"])

    def test_far_edge_grasp_finds_narrow_ellipsoid_chord(self) -> None:
        xs = np.arange(0.46, 0.781, 0.007)
        ys = np.arange(-0.16, 0.161, 0.007)
        zs = np.arange(TABLE_TOP_Z_M, 0.156, 0.007)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        normalized = (
            ((grid[:, 0] - 0.62) / 0.16) ** 2
            + (grid[:, 1] / 0.16) ** 2
            + ((grid[:, 2] - 0.105) / 0.05) ** 2
        )
        points = grid[normalized <= 1.0]
        candidate = generate_far_edge_grasp(points, 0.007, 0.078, (-0.007, 0.046))
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertGreater(candidate["grasp_center_m"][0], 0.70)
        self.assertLess(candidate["local_width_m"], 0.095)


if __name__ == "__main__":
    unittest.main()
