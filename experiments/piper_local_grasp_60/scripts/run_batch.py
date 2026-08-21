#!/usr/bin/env python3
"""Run all 60 local-grasp attempts, one isolated process per GPU worker."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from localgrasp.assets import AssetSpec, discover_assets  # noqa: E402
from localgrasp.paths import GENESIS_PYTHON, REPORT_ROOT, VIDEO_ROOT, WORK_ROOT, require_within_project  # noqa: E402
from localgrasp.runtime import IMPLEMENTATION_REVISION  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize(asset: AssetSpec, attempt: dict[str, Any] | None) -> dict[str, Any]:
    attempt = attempt or {}
    trials = attempt.get("candidate_trials") or []
    return {
        "asset_id": asset.asset_id,
        "object_type": asset.object_type,
        "geometry_family": asset.geometry_family,
        "mass_kg": asset.mass_kg,
        "estimated_particles": asset.estimated_particles,
        "status": str(attempt.get("status", "pending")),
        "reason": str(attempt.get("reason", "not run")),
        "candidate_count": int(attempt.get("candidate_count", 0)),
        "candidate_trials": len(trials),
        "probe_pass_count": sum(bool((trial.get("probe") or {}).get("passed")) for trial in trials),
        "elapsed_seconds": attempt.get("elapsed_seconds"),
        "video": (attempt.get("video") or {}).get("path", ""),
    }


def _write_reports(assets: list[AssetSpec], attempts: dict[str, dict[str, Any]], started: float) -> dict[str, Any]:
    rows = [_summarize(asset, attempts.get(asset.asset_id)) for asset in assets]
    counts = Counter(row["status"] for row in rows)
    videos = [row["video"] for row in rows if row["status"] == "success" and row["video"]]
    payload = {
        "schema_version": 2,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "scope": ["solid", "shell"],
        "asset_count": len(assets),
        "completed_count": sum(row["status"] != "pending" for row in rows),
        "counts": dict(sorted(counts.items())),
        "successful_video_count": len(videos),
        "successful_videos": videos,
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "scale": 1.0,
            "original_scale_only": True,
            "local_edge_corner_grasps": True,
            "wrist_yaw_candidates_deg": [0, -90, -45, 45],
            "probe_requires_entire_object_clear_of_table": True,
            "trajectory_recording": False,
            "failed_video_retention": False,
            "success_video_format": "mp4",
            "place_target": "large shallow tray",
        },
        "assets": rows,
    }
    _atomic_json(REPORT_ROOT / "batch_results.json", payload)
    csv_path = require_within_project(REPORT_ROOT / "batch_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return payload


def _load_reusable(path: Path, video_path: Path, force: bool) -> dict[str, Any] | None:
    if force or not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("implementation_revision") != IMPLEMENTATION_REVISION:
        return None
    if result.get("status") == "success" and not video_path.is_file():
        return None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--asset-id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids is empty")
    worker_count = min(args.workers, len(gpu_ids))
    if worker_count <= 0:
        raise ValueError("--workers must be positive")
    if not GENESIS_PYTHON.is_file():
        raise FileNotFoundError(GENESIS_PYTHON)

    started = time.perf_counter()
    assets = discover_assets()
    selected = [asset for asset in assets if args.asset_id is None or asset.asset_id == args.asset_id]
    if args.asset_id and not selected:
        raise ValueError(f"unknown --asset-id={args.asset_id}")
    selected.sort(key=lambda asset: (asset.estimated_particles, asset.asset_id))
    attempts: dict[str, dict[str, Any]] = {}
    pending: list[AssetSpec] = []
    for asset in selected:
        result_path = WORK_ROOT / f"{asset.asset_id}.json"
        video_path = VIDEO_ROOT / f"{asset.asset_id}.mp4"
        previous = _load_reusable(result_path, video_path, args.force)
        if previous is None:
            pending.append(asset)
        else:
            attempts[asset.asset_id] = previous
    report_lock = threading.Lock()
    print(
        json.dumps(
            {
                "revision": IMPLEMENTATION_REVISION,
                "selected": len(selected),
                "pending": len(pending),
                "workers": worker_count,
                "gpu_ids": gpu_ids[:worker_count],
            }
        ),
        flush=True,
    )
    _write_reports(selected, attempts, started)
    attempt_script = PROJECT_ROOT / "scripts" / "attempt_asset.py"
    assignments = [pending[index::worker_count] for index in range(worker_count)]

    def worker(gpu_id: str, queue: list[AssetSpec]) -> None:
        for asset in queue:
            result_path = WORK_ROOT / f"{asset.asset_id}.json"
            video_path = VIDEO_ROOT / f"{asset.asset_id}.mp4"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(GENESIS_PYTHON),
                str(attempt_script),
                "--asset-id",
                asset.asset_id,
                "--result",
                str(result_path),
                "--video",
                str(video_path),
                "--max-candidates",
                str(args.max_candidates),
            ]
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"[gpu {gpu_id}] start {asset.asset_id} particles={asset.estimated_particles}", flush=True)
            try:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    timeout=args.timeout_s,
                    check=False,
                )
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = {
                        "schema_version": 2,
                        "implementation_revision": IMPLEMENTATION_REVISION,
                        "status": "exception",
                        "reason": f"child exited {completed.returncode} without result JSON",
                        "video": None,
                    }
            except subprocess.TimeoutExpired:
                result = {
                    "schema_version": 2,
                    "implementation_revision": IMPLEMENTATION_REVISION,
                    "asset": asset.to_dict(),
                    "status": "timeout",
                    "reason": f"attempt exceeded {args.timeout_s:.1f} seconds",
                    "video": None,
                }
                _atomic_json(result_path, result)
            with report_lock:
                attempts[asset.asset_id] = result
                payload = _write_reports(selected, attempts, started)
                print(
                    f"[gpu {gpu_id}] done {asset.asset_id} status={result.get('status')} "
                    f"completed={payload['completed_count']}/{len(selected)}",
                    flush=True,
                )

    threads = [
        threading.Thread(target=worker, args=(gpu_ids[index], assignments[index]), daemon=False)
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    payload = _write_reports(selected, attempts, started)
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "successful_video_count": payload["successful_video_count"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
