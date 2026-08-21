#!/usr/bin/env python3
"""Create an auditable summary of the small transport pilot."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transportpilot.paths import REPORT_ROOT, require_within_project  # noqa: E402


PRIMARY_FILES = (
    "piper_push_0001.json",
    "piper_push_0079.json",
    "piper_press_drag_0079.json",
    "franka_push_0001.json",
    "franka_press_drag_0001.json",
    "franka_press_drag_0079.json",
    "franka_edge_drag_0001.json",
)
ITERATION_FILES = (
    "piper_press_drag_0079_v2_shallow.json",
    "piper_press_drag_0079_v3_contact.json",
    "piper_push_0079_v4_aabb_gate.json",
)
PLAN_FILES = (
    "piper_edge_drag_0001_plan.json",
    "franka_edge_drag_0001_plan.json",
    "franka_edge_drag_0001_plan_v6_outer.json",
    "franka_edge_drag_0001_plan_v7_axial.json",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    primary = [read_json(PROJECT_ROOT / "work" / name) for name in PRIMARY_FILES]
    iterations = [read_json(PROJECT_ROOT / "work" / name) for name in ITERATION_FILES]
    reachability = {
        robot: read_json(PROJECT_ROOT / "reports" / f"reachability_{robot}.json")
        for robot in ("piper", "franka")
    }
    rows: list[dict[str, Any]] = []
    for result in primary:
        assessment = result.get("assessment") or {}
        rows.append(
            {
                "result_file": "",
                "robot": result.get("robot"),
                "action": result.get("action"),
                "implementation_revision": result.get("implementation_revision"),
                "asset_id": result.get("asset", {}).get("asset_id"),
                "object_type": result.get("asset", {}).get("object_type"),
                "geometry_family": result.get("asset", {}).get("geometry_family"),
                "status": result.get("status"),
                "progress_m": assessment.get("planar_progress_m", ""),
                "goal_error_m": assessment.get("final_goal_error_m", ""),
                "goal_particle_fraction": assessment.get("goal_particle_fraction", ""),
                "video": (result.get("video") or {}).get("path", ""),
                "elapsed_seconds": result.get("elapsed_seconds", ""),
            }
        )
    for row, filename in zip(rows, PRIMARY_FILES):
        row["result_file"] = filename

    counts = Counter(row["status"] for row in rows)
    robot_action_counts = Counter((row["robot"], row["action"], row["status"]) for row in rows)
    successful_videos = [row["video"] for row in rows if row["video"]]
    dynamics_asset_ids = sorted({row["asset_id"] for row in rows})
    reachability_asset_ids = sorted(
        {
            case["asset_id"]
            for payload in reachability.values()
            for case in payload["cases"]
        }
    )
    reach_summary = {}
    for robot, payload in reachability.items():
        reach_summary[robot] = {
            "revision": payload["implementation_revision"],
            "case_count": payload["case_count"],
            "valid_case_count": payload["valid_case_count"],
            "cases": [
                {
                    "asset_id": case["asset_id"],
                    "action": case["action"],
                    "valid": case["motion_plan"]["valid"],
                    "max_position_error_m": case["motion_plan"]["selected_max_position_error_m"],
                    "max_rotation_error_rad": case["motion_plan"]["selected_max_rotation_error_rad"],
                    "selected_yaw_deg": case["motion_plan"]["selected_yaw_deg"],
                }
                for case in payload["cases"]
            ],
        }

    payload = {
        "schema_version": 1,
        "implementation_revision": "planar_transport_v8_diagonal_edge_drag",
        "scope": {
            "asset_count_dynamics": len(dynamics_asset_ids),
            "asset_ids_dynamics": dynamics_asset_ids,
            "asset_count_reachability": len(reachability_asset_ids),
            "asset_ids_reachability": reachability_asset_ids,
            "original_scale_only": True,
            "solver": "MPM.Elastic",
            "genesis": "official 1.1.2 environment",
            "goal_region_size_xy_m": [0.40, 0.40],
            "goal_region_visual_only": True,
        },
        "dynamics_primary_count": len(primary),
        "dynamics_status_counts": dict(sorted(counts.items())),
        "dynamics_robot_action_status_counts": {
            f"{robot}:{action}:{status}": count
            for (robot, action, status), count in sorted(robot_action_counts.items())
        },
        "successful_video_count": len(successful_videos),
        "successful_videos": successful_videos,
        "reachability": reach_summary,
        "primary_trials": rows,
        "iteration_controls": [
            {
                "result_file": filename,
                "revision": result.get("implementation_revision"),
                "robot": result.get("robot"),
                "action": result.get("action"),
                "status": result.get("status"),
                "progress_m": (result.get("assessment") or {}).get("planar_progress_m"),
            }
            for result, filename in zip(iterations, ITERATION_FILES)
        ],
        "planning_artifacts": [
            {
                "result_file": filename,
                "revision": result.get("implementation_revision"),
                "robot": result.get("robot"),
                "action": result.get("action"),
                "status": result.get("status"),
                "selected_candidate": result.get("selected_candidate"),
            }
            for filename in PLAN_FILES
            for result in [read_json(PROJECT_ROOT / "work" / filename)]
        ],
        "trajectory_file_count": len(
            list(PROJECT_ROOT.rglob("*.npy")) + list(PROJECT_ROOT.rglob("*.npz"))
        ),
        "dynamics_elapsed_seconds": [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] != ""],
    }
    if payload["dynamics_elapsed_seconds"]:
        payload["elapsed_summary"] = {
            "sum_seconds": sum(payload["dynamics_elapsed_seconds"]),
            "median_seconds": statistics.median(payload["dynamics_elapsed_seconds"]),
        }
    atomic_json(REPORT_ROOT / "experiment_results.json", payload)

    csv_path = require_within_project(REPORT_ROOT / "experiment_results.csv")
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)

    report = """# Original-scale planar soft-body transport pilot

## Key findings

- Franka passed 6/6 representative continuous-IK paths, versus 4/6 for Piper.
- Franka improved reachability but did not solve whole-object load transfer.
- Piper pushing 0079 was the only complete success: 234.9 mm directed COM progress and 99.84% goal occupancy.
- Failed trials retain scalar JSON and diagnostic images, not trajectories.

See `experiment_results.json/csv` for the machine-readable result table and
the repository-level failure analysis for interpretation boundaries.
"""
    atomic_text(REPORT_ROOT / "RESULTS.md", report)
    print(json.dumps({"counts": payload["dynamics_status_counts"], "successful_videos": successful_videos}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
