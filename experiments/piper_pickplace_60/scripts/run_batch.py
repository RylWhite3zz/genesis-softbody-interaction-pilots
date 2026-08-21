#!/usr/bin/env python3
"""Screen all 60 assets, then run only original-scale geometric candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pickplace60.inventory import discover_assets, screen_asset, select_axis_aligned_orientation  # noqa: E402
from pickplace60.paths import (  # noqa: E402
    GENESIS_PYTHON,
    REPORT_ROOT,
    VIDEO_ROOT,
    WORK_ROOT,
    require_within_project,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "asset_id",
        "object_type",
        "geometry_family",
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
        "minimum_extent_m",
        "screen_status",
        "final_status",
        "reason",
        "attempted",
        "scale",
        "video",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            asset = entry["asset"]
            screen = entry["screen"]
            attempt = entry.get("attempt") or {}
            extents = asset["extent_m"]
            writer.writerow(
                {
                    "asset_id": asset["asset_id"],
                    "object_type": asset["object_type"],
                    "geometry_family": asset["geometry_family"],
                    "extent_x_m": extents[0],
                    "extent_y_m": extents[1],
                    "extent_z_m": extents[2],
                    "minimum_extent_m": screen["minimum_extent_m"],
                    "screen_status": screen["status"],
                    "final_status": entry["final_status"],
                    "reason": entry["reason"],
                    "attempted": bool(entry.get("attempted", False)),
                    "scale": 1.0,
                    "video": (attempt.get("video") or {}).get("path", ""),
                }
            )
    temporary.replace(path)


def _payload(entries: list[dict[str, Any]], started: float, mode: str) -> dict[str, Any]:
    statuses = Counter(entry["final_status"] for entry in entries)
    successful_videos = [
        entry["attempt"]["video"]
        for entry in entries
        if entry.get("attempt") and entry["attempt"].get("video") is not None
    ]
    return {
        "schema_version": 1,
        "mode": mode,
        "scope": ["solid", "shell"],
        "asset_count": len(entries),
        "constraints": {
            "original_scale_only": True,
            "scale": 1.0,
            "jaw_open_width_m": 0.066,
            "wrong_size_policy": "skip without simulation",
            "trajectory_recording": False,
            "video_policy": "retain MP4 only when every success gate passes",
            "robot_base_fixed": True,
            "robot_reset": "open jaws centered 40% of object height above COM before the first physics step",
            "robot_mpm_coupling_friction": 6.0,
            "place_target": "rigid shallow tray on workbench",
        },
        "counts": dict(sorted(statuses.items())),
        "attempted_count": sum(bool(entry.get("attempted")) for entry in entries),
        "successful_video_count": len(successful_videos),
        "successful_videos": successful_videos,
        "elapsed_seconds": time.perf_counter() - started,
        "assets": entries,
    }


def _write_reports(entries: list[dict[str, Any]], started: float, mode: str) -> dict[str, Any]:
    name = "screening" if mode == "screen_only" else "batch_results"
    payload = _payload(entries, started, mode)
    _atomic_json(REPORT_ROOT / f"{name}.json", payload)
    _write_csv(REPORT_ROOT / f"{name}.csv", entries)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--asset-id", help="run only one eligible asset while keeping all 60 rows in the report")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--force", action="store_true", help="rerun an asset even if a successful result already exists")
    args = parser.parse_args()
    if not GENESIS_PYTHON.is_file():
        raise FileNotFoundError(GENESIS_PYTHON)
    started = time.perf_counter()
    assets = discover_assets()
    entries: list[dict[str, Any]] = []
    for asset in assets:
        screen = screen_asset(asset)
        entry: dict[str, Any] = {
            "asset": asset.to_dict(),
            "screen": screen,
            "attempted": False,
            "attempt": None,
            "final_status": screen["status"],
            "reason": screen["reason"],
        }
        if screen["eligible"]:
            entry["orientation"] = select_axis_aligned_orientation(asset)
        entries.append(entry)
    if len(entries) != 60:
        raise RuntimeError(f"expected exactly 60 solid/shell assets, found {len(entries)}")
    _write_reports(entries, started, "screen_only" if args.screen_only else "batch")
    if args.screen_only:
        payload = _write_reports(entries, started, "screen_only")
        print(json.dumps({"counts": payload["counts"], "attempted_count": 0}, indent=2))
        return 0

    attempt_script = PROJECT_ROOT / "scripts" / "attempt_asset.py"
    for entry in entries:
        if not entry["screen"]["eligible"]:
            continue
        asset_id = entry["asset"]["asset_id"]
        result_path = WORK_ROOT / f"{asset_id}.json"
        video_path = VIDEO_ROOT / f"{asset_id}.mp4"
        if args.asset_id and asset_id != args.asset_id:
            if result_path.is_file():
                attempt = json.loads(result_path.read_text(encoding="utf-8"))
                entry["attempted"] = True
                entry["attempt"] = attempt
                entry["final_status"] = str(attempt.get("status", "unknown"))
                entry["reason"] = str(attempt.get("reason", "missing reason"))
            else:
                entry["final_status"] = "not_run_filter"
                entry["reason"] = f"not selected by --asset-id={args.asset_id}"
            _write_reports(entries, started, "batch")
            continue
        previous = None
        if result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not args.force
            and previous
            and (previous.get("status") != "success" or video_path.is_file())
        ):
            attempt = previous
        else:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(GENESIS_PYTHON),
                str(attempt_script),
                "--asset-id",
                asset_id,
                "--result",
                str(result_path),
                "--video",
                str(video_path),
            ]
            print(f"attempting {asset_id}", flush=True)
            try:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, timeout=args.timeout_s, check=False)
                if result_path.is_file():
                    attempt = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    attempt = {
                        "status": "exception",
                        "reason": f"child exited {completed.returncode} without a result file",
                        "video": None,
                    }
            except subprocess.TimeoutExpired:
                attempt = {
                    "status": "timeout",
                    "reason": f"attempt exceeded {args.timeout_s:.1f} seconds",
                    "video": None,
                }
                _atomic_json(result_path, attempt)
        entry["attempted"] = True
        entry["attempt"] = attempt
        entry["final_status"] = str(attempt.get("status", "unknown"))
        entry["reason"] = str(attempt.get("reason", "missing reason"))
        _write_reports(entries, started, "batch")

    payload = _write_reports(entries, started, "batch")
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "attempted_count": payload["attempted_count"],
                "successful_video_count": payload["successful_video_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
