from __future__ import annotations

import unittest

import numpy as np

from pickplace60.inventory import OPEN_WIDTH_M, discover_assets, screen_asset, select_axis_aligned_orientation
from pickplace60.protocol import TRAY_CENTER_XY_M, TRAY_FLOOR_Z_M, assess_success, tray_boxes, validate_protocol


EXPECTED_CANDIDATES = {
    "0079_stage3_100k_qwen2_v1_shell_025403",
    "0090_stage3_100k_qwen2_v1_shell_020115",
}


class InventoryProtocolTests(unittest.TestCase):
    def test_representative_inventory_has_expected_geometry_candidates(self) -> None:
        assets = discover_assets()
        self.assertGreaterEqual(len(assets), 3)
        self.assertEqual({asset.geometry_family for asset in assets}, {"solid", "shell"})
        candidates = {asset.asset_id for asset in assets if screen_asset(asset)["eligible"]}
        self.assertEqual(candidates, EXPECTED_CANDIDATES)

    def test_candidate_orientations_fit_without_scaling(self) -> None:
        candidates = [asset for asset in discover_assets() if screen_asset(asset)["eligible"]]
        for asset in candidates:
            orientation = select_axis_aligned_orientation(asset)
            extents = np.asarray(orientation["extent_after_rotation_m"])
            rotation = np.asarray(orientation["rotation_matrix"])
            self.assertEqual(orientation["scale"], 1.0)
            self.assertLessEqual(extents[1], OPEN_WIDTH_M + 1e-9)
            self.assertLessEqual(extents[2], 0.22)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            self.assertGreater(np.linalg.det(rotation), 0.999999)

    def test_tray_has_bottom_and_four_walls(self) -> None:
        validate_protocol()
        boxes = tray_boxes()
        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(all(value > 0 for value in box["size"]) for box in boxes))

    def test_success_gate_requires_containment_lift_release_and_video(self) -> None:
        cx, cy = TRAY_CENTER_XY_M
        passing = {
            "final_aabb_min_m": [cx - 0.16, cy - 0.04, TRAY_FLOOR_Z_M],
            "final_aabb_max_m": [cx + 0.16, cy + 0.04, TRAY_FLOOR_Z_M + 0.17],
            "max_lift_delta_z_m": 0.12,
            "transfer_end_com_delta_z_m": 0.11,
            "transfer_end_aabb_min_z_m": 0.10,
            "final_speed_rms_m_s": 0.03,
            "final_grip_object_distance_m": 0.14,
            "frame_count": 100,
            "minimum_frame_std": 12.0,
            "finite_state": True,
            "active_particle_count_constant": True,
        }
        assessment = assess_success(passing)
        self.assertEqual(assessment["status"], "success")
        outside = dict(passing)
        outside["final_aabb_max_m"] = [cx + 0.30, cy + 0.04, TRAY_FLOOR_Z_M + 0.17]
        failed = assess_success(outside)
        self.assertEqual(failed["status"], "failed_gates")
        self.assertFalse(failed["gates"]["all_particles_horizontally_inside_tray"])
        dragged = dict(passing)
        dragged["transfer_end_aabb_min_z_m"] = 0.056
        failed_drag = assess_success(dragged)
        self.assertFalse(failed_drag["gates"]["object_still_lifted_at_transfer_end"])


if __name__ == "__main__":
    unittest.main()
