#!/usr/bin/env python3
"""Run one asset in a fresh process so Genesis GPU state cannot leak between attempts."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pickplace60.inventory import discover_assets, select_axis_aligned_orientation  # noqa: E402
from pickplace60.paths import require_within_project  # noqa: E402
from pickplace60.runtime import run_attempt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    result_path = require_within_project(args.result)
    video_path = require_within_project(args.video)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        matches = [asset for asset in discover_assets() if asset.asset_id == args.asset_id]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one asset named {args.asset_id}, found {len(matches)}")
        asset = matches[0]
        orientation = select_axis_aligned_orientation(asset)
        result = run_attempt(asset, orientation, video_path)
        exit_code = 0
    except Exception as exc:
        result = {
            "schema_version": 1,
            "asset": {"asset_id": args.asset_id},
            "status": "exception",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "video": None,
        }
        exit_code = 2
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"asset_id": args.asset_id, "status": result["status"], "reason": result["reason"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
