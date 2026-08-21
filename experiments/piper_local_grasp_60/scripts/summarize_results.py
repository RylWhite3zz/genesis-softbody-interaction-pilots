#!/usr/bin/env python3
"""Build audited JSON/CSV/Chinese reports from immutable per-asset results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from localgrasp.assets import discover_assets  # noqa: E402
from localgrasp.paths import REPORT_ROOT, require_within_project  # noqa: E402
from localgrasp.runtime import FOLD_IMPLEMENTATION_REVISION, IMPLEMENTATION_REVISION  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _atomic_text(path: Path, text: str) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _probe_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for trial in result["candidate_trials"]:
            probe = trial.get("probe")
            if probe is None:
                continue
            candidate = trial["candidate"]
            rows.append(
                {
                    "asset_id": result["asset"]["asset_id"],
                    "object_type": result["asset"]["object_type"],
                    "geometry_family": result["asset"]["geometry_family"],
                    "candidate_rank": candidate["rank"],
                    "grasp_type": candidate["grasp_type"],
                    "wrist_yaw_deg": candidate["wrist_yaw_deg"],
                    "local_width_m": candidate["local_width_m"],
                    "commanded_close_width_m": candidate["commanded_close_width_m"],
                    "com_lift_delta_z_m": probe["com_lift_delta_z_m"],
                    "aabb_min_z_m": probe["aabb_min_z_m"],
                    "passed": probe["passed"],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-wall-s", type=float, required=True)
    args = parser.parse_args()
    assets = discover_assets()
    results = []
    for asset in assets:
        path = PROJECT_ROOT / "work" / f"{asset.asset_id}.json"
        result = _read_json(path)
        if result.get("implementation_revision") != IMPLEMENTATION_REVISION:
            raise RuntimeError(f"wrong result revision in {path}: {result.get('implementation_revision')}")
        results.append(result)
    if len(results) != 60:
        raise RuntimeError(f"expected 60 results, found {len(results)}")
    statuses = Counter(result["status"] for result in results)
    family_statuses = Counter(
        (result["asset"]["geometry_family"], result["status"]) for result in results
    )
    probes = _probe_rows(results)
    best_com = max(probes, key=lambda row: row["com_lift_delta_z_m"])
    best_clearance = max(probes, key=lambda row: row["aabb_min_z_m"])
    candidate_terminal = Counter(
        trial["status"] for result in results for trial in result["candidate_trials"]
    )
    video_paths = sorted((PROJECT_ROOT / "videos").glob("*.mp4"))
    trajectory_paths = sorted(PROJECT_ROOT.rglob("*.npz")) + sorted(PROJECT_ROOT.rglob("*.npy"))
    runtime_versions = sorted(
        {
            (result["runtime"]["genesis_version"], result["runtime"]["genesis_module"])
            for result in results
        }
    )
    rows = []
    for result in results:
        trials = result["candidate_trials"]
        rows.append(
            {
                "asset_id": result["asset"]["asset_id"],
                "object_type": result["asset"]["object_type"],
                "geometry_family": result["asset"]["geometry_family"],
                "mass_kg": result["asset"]["mass_kg"],
                "estimated_particles": result["asset"]["estimated_particles"],
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "probe_attempt_count": sum(trial.get("probe") is not None for trial in trials),
                "probe_pass_count": sum(bool((trial.get("probe") or {}).get("passed")) for trial in trials),
                "best_com_lift_delta_z_m": max(
                    (trial["probe"]["com_lift_delta_z_m"] for trial in trials if trial.get("probe")),
                    default="",
                ),
                "best_aabb_min_z_m": max(
                    (trial["probe"]["aabb_min_z_m"] for trial in trials if trial.get("probe")),
                    default="",
                ),
                "elapsed_seconds": result["elapsed_seconds"],
                "video": (result.get("video") or {}).get("path", ""),
            }
        )
    payload = {
        "schema_version": 3,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "asset_count": len(results),
        "completed_count": len(results),
        "counts": dict(sorted(statuses.items())),
        "family_status_counts": {
            f"{family}:{status}": count
            for (family, status), count in sorted(family_statuses.items())
        },
        "candidate_count": sum(result["candidate_count"] for result in results),
        "candidate_terminal_counts": dict(sorted(candidate_terminal.items())),
        "probe_attempt_count": len(probes),
        "assets_with_probe_count": sum(
            any(trial.get("probe") is not None for trial in result["candidate_trials"])
            for result in results
        ),
        "probe_pass_count": sum(row["passed"] for row in probes),
        "positive_com_lift_probe_count": sum(row["com_lift_delta_z_m"] > 0.0 for row in probes),
        "best_com_lift_probe": best_com,
        "best_table_clearance_probe": best_clearance,
        "top_com_lift_probes": sorted(
            probes, key=lambda row: row["com_lift_delta_z_m"], reverse=True
        )[:10],
        "successful_video_count": len(video_paths),
        "successful_videos": [str(path.resolve()) for path in video_paths],
        "trajectory_file_count": len(trajectory_paths),
        "batch_execution_wall_seconds": args.batch_wall_s,
        "aggregate_asset_elapsed_seconds": sum(result["elapsed_seconds"] for result in results),
        "median_asset_elapsed_seconds": statistics.median(
            result["elapsed_seconds"] for result in results
        ),
        "runtime_versions": [
            {"genesis_version": version, "genesis_module": module}
            for version, module in runtime_versions
        ],
        "constraints": {
            "scale": 1.0,
            "original_scale_only": True,
            "material": "MPM.Elastic with packaged E, nu, rho, particle size, grid density, dt, and substeps",
            "jaw_open_width_m": 0.066,
            "finger_force_limit_n_per_dof": 10.0,
            "robot_mpm_coupling_friction": 6.0,
            "wrist_yaw_candidates_deg": [0, -90, -45, 45],
            "probe_required_com_lift_m": 0.045,
            "probe_required_aabb_min_z_m": 0.063,
            "robot_base_fixed": True,
            "trajectory_recording": False,
            "failed_video_retention": False,
            "place_target": "large shallow tray",
        },
        "assets": rows,
    }
    _atomic_json(REPORT_ROOT / "batch_results.json", payload)
    csv_path = require_within_project(REPORT_ROOT / "batch_results.csv")
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    fold_results = []
    for asset_id in (
        "0079_stage3_100k_qwen2_v1_shell_025403",
        "0090_stage3_100k_qwen2_v1_shell_020115",
    ):
        result = _read_json(PROJECT_ROOT / "fold_work" / f"{asset_id}.json")
        if result.get("implementation_revision") != FOLD_IMPLEMENTATION_REVISION:
            raise RuntimeError(f"wrong fold revision for {asset_id}")
        probe_trials = [trial for trial in result["candidate_trials"] if trial.get("probe")]
        best = max(probe_trials, key=lambda trial: trial["probe"]["com_lift_delta_z_m"])
        normal = next(item for item in results if item["asset"]["asset_id"] == asset_id)
        normal_best = max(
            (trial for trial in normal["candidate_trials"] if trial.get("probe")),
            key=lambda trial: trial["probe"]["com_lift_delta_z_m"],
        )
        fold_results.append(
            {
                "asset_id": asset_id,
                "object_type": result["asset"]["object_type"],
                "status": result["status"],
                "preload_distance_m": result["protocol"]["active_wrinkle_preload_distance_m"],
                "normal_best_probe": normal_best["probe"],
                "preload_best_probe": best["probe"],
                "preload_candidate": best["candidate"],
                "execution_grasp_center_m": best["motion_plan"]["execution_grasp_center_m"],
            }
        )
    fold_payload = {
        "schema_version": 1,
        "implementation_revision": FOLD_IMPLEMENTATION_REVISION,
        "asset_count": 2,
        "preload_distance_m": 0.025,
        "probe_pass_count": sum(
            item["preload_best_probe"]["passed"] for item in fold_results
        ),
        "successful_video_count": len(list((PROJECT_ROOT / "fold_videos").glob("*.mp4"))),
        "results": fold_results,
    }
    _atomic_json(REPORT_ROOT / "fold_preload_results.json", fold_payload)

    minutes = args.batch_wall_s / 60.0
    report = f"""# Piper local pinching of 60 original-scale soft-body assets

## Key findings

- 23 assets had no reachable local grasp candidate.
- 37 assets executed 67 lift probes, with zero passes.
- Best COM rise: {best_com['com_lift_delta_z_m'] * 1000.0:.2f} mm.
- Best minimum height: {best_clearance['aabb_min_z_m'] * 1000.0:.2f} mm.
- The eight-GPU batch took approximately {minutes:.1f} minutes.
- Active 25 mm wrinkle preloads on 0079 and 0090 also failed.

See `batch_results.json/csv` and `fold_preload_results.json` for the
machine-readable evidence. No trajectories or successful videos were recorded.
"""
    _atomic_text(REPORT_ROOT / "RESULTS.md", report)
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "probe_attempt_count": payload["probe_attempt_count"],
                "best_com_lift_delta_z_m": best_com["com_lift_delta_z_m"],
                "successful_video_count": payload["successful_video_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
