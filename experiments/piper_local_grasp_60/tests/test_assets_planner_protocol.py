from __future__ import annotations

import unittest

import numpy as np

from localgrasp.assets import discover_assets
from localgrasp.planner import generate_local_grasps
from localgrasp.protocol import TABLE_TOP_Z_M, assess_probe, stage_steps, tray_boxes


class LocalGraspTests(unittest.TestCase):
    def test_representative_inventory_contains_a_solid_and_shell(self) -> None:
        assets = discover_assets()
        self.assertGreaterEqual(len(assets), 3)
        self.assertIn("solid", {asset.geometry_family for asset in assets})
        self.assertIn("shell", {asset.geometry_family for asset in assets})
        self.assertTrue(all(asset.source_solver == "genesis_mpm_elastic" for asset in assets))

    def test_planner_finds_local_edge_on_oversize_disc(self) -> None:
        spacing = 0.007
        xs = np.arange(-0.16, 0.161, spacing)
        ys = np.arange(-0.16, 0.161, spacing)
        zs = np.arange(TABLE_TOP_Z_M + 0.004, TABLE_TOP_Z_M + 0.075, spacing)
        points = []
        for x in xs:
            for y in ys:
                if x * x + y * y > 0.16**2:
                    continue
                for z in zs:
                    points.append((0.47 + x, -0.22 + y, z))
        candidates = generate_local_grasps(np.asarray(points), spacing, limit=6)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate["local_width_m"] <= 0.085 for candidate in candidates))
        self.assertTrue(any(candidate["edge_fraction"] > 0.55 for candidate in candidates))

    def test_probe_requires_whole_object_clear_of_table(self) -> None:
        reference = {
            "com_m": np.asarray((0.4, 0.0, 0.10)),
            "aabb_min_m": np.asarray((0.3, -0.1, TABLE_TOP_Z_M)),
            "active_count": 100,
            "finite": True,
        }
        probe = {
            "com_m": np.asarray((0.4, 0.0, 0.16)),
            "aabb_min_m": np.asarray((0.3, -0.1, TABLE_TOP_Z_M + 0.001)),
            "active_count": 100,
            "finite": True,
        }
        self.assertFalse(assess_probe(reference, probe)["passed"])
        probe["aabb_min_m"][2] = TABLE_TOP_Z_M + 0.012
        self.assertTrue(assess_probe(reference, probe)["passed"])

    def test_stage_timings_and_tray(self) -> None:
        self.assertTrue(all(value > 0 for value in stage_steps(0.001).values()))
        self.assertEqual(len(tray_boxes()), 5)


if __name__ == "__main__":
    unittest.main()
