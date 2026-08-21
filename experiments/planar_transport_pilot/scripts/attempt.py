#!/usr/bin/env python3
"""Run one original-scale planar transport trial."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transportpilot.assets import get_asset  # noqa: E402
from transportpilot.paths import require_within_project  # noqa: E402
from transportpilot.protocol import ACTIONS, ROBOTS  # noqa: E402
from transportpilot.runtime import atomic_write_json, run_trial  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--robot", choices=ROBOTS, required=True)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result_path = require_within_project(args.result)
    video_path = require_within_project(args.video)
    try:
        result = run_trial(
            get_asset(args.asset_id),
            args.robot,
            args.action,
            video_path,
            plan_only=args.plan_only,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "exception",
            "asset_id": args.asset_id,
            "robot": args.robot,
            "action": args.action,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "video": None,
        }
        atomic_write_json(result_path, result)
        raise
    atomic_write_json(result_path, result)
    print(
        {
            "status": result["status"],
            "robot": args.robot,
            "action": args.action,
            "asset": args.asset_id,
            "video": result.get("video"),
            "assessment": result.get("assessment"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
