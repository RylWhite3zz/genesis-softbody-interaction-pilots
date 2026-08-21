#!/usr/bin/env python3
"""Run one asset in an isolated Genesis process."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from localgrasp.assets import discover_assets  # noqa: E402
from localgrasp.paths import require_within_project  # noqa: E402
from localgrasp.runtime import (  # noqa: E402
    FOLD_IMPLEMENTATION_REVISION,
    IMPLEMENTATION_REVISION,
    run_attempt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--preload-m", type=float, default=0.0)
    args = parser.parse_args()
    result_path = require_within_project(args.result)
    video_path = require_within_project(args.video)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        matches = [asset for asset in discover_assets() if asset.asset_id == args.asset_id]
        if len(matches) != 1:
            raise ValueError(f"expected one asset named {args.asset_id}, found {len(matches)}")
        result = run_attempt(
            matches[0],
            video_path,
            max_candidates=args.max_candidates,
            preload_distance_m=args.preload_m,
        )
        exit_code = 0
    except Exception as exc:
        result = {
            "schema_version": 2,
            "implementation_revision": (
                FOLD_IMPLEMENTATION_REVISION if args.preload_m > 0.0 else IMPLEMENTATION_REVISION
            ),
            "asset": {"asset_id": args.asset_id},
            "status": "exception",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "video": None,
        }
        exit_code = 2
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps({"asset_id": args.asset_id, "status": result["status"], "reason": result["reason"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
