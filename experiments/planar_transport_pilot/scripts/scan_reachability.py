#!/usr/bin/env python3
"""Compare continuous push/drag IK paths without running MPM dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transportpilot.assets import get_asset  # noqa: E402
from transportpilot.paths import require_within_project  # noqa: E402
from transportpilot.protocol import ROBOTS  # noqa: E402
from transportpilot.runtime import atomic_write_json, run_reachability  # noqa: E402


DEFAULT_ASSETS = (
    "0001_stage3_100k_qwen2_v1_solid_053842",
    "0079_stage3_100k_qwen2_v1_shell_025403",
    "0074_stage3_100k_qwen2_v1_shell_016589",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=ROBOTS, required=True)
    parser.add_argument("--asset-id", action="append", dest="asset_ids")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    asset_ids = tuple(args.asset_ids or DEFAULT_ASSETS)
    payload = run_reachability(args.robot, [get_asset(asset_id) for asset_id in asset_ids])
    result_path = require_within_project(args.result)
    atomic_write_json(result_path, payload)
    for case in payload["cases"]:
        motion = case["motion_plan"]
        print(
            case["asset_id"][:4],
            case["action"],
            "valid=" + str(motion["valid"]),
            f"pos={motion['selected_max_position_error_m']:.5f}",
            f"rot={motion['selected_max_rotation_error_rad']:.5f}",
            f"yaw={motion['selected_yaw_deg']:.0f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
