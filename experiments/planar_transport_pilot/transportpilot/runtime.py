"""Genesis 1.1.2 planar push and press-drag experiments."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .assets import AssetSpec
from .paths import (
    DIAGNOSTIC_ROOT,
    configure_writable_runtime,
    isolate_genesis_particle_sampler,
    require_within_project,
)
from .planner import generate_far_edge_grasp, make_edge_drag_tool_path, make_tool_path, nominal_reference
from .protocol import (
    ACTION_EDGE_DRAG,
    ACTION_PRESS_DRAG,
    FRAME_PERIOD_S,
    GOAL_SIZE_XY_M,
    ROBOT_COUP_FRICTION,
    SPAWN_GAP_M,
    TABLE_CENTER_M,
    TABLE_COUP_FRICTION,
    TABLE_SIZE_M,
    TABLE_TOP_Z_M,
    VIDEO_FPS,
    action_centers,
    assess_transport,
    smoothstep01,
    stage_steps,
)
from .robots import (
    RobotHandle,
    add_robot,
    configure_robot,
    flat,
    numpy,
    serial_motion_plan,
    solve_tool_path,
)


IMPLEMENTATION_REVISION = "planar_transport_v8_diagonal_edge_drag"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_snapshot(soft: Any, *, include_positions: bool = False) -> dict[str, Any]:
    state = soft.get_state()
    positions = numpy(state.pos)[0].astype(np.float64)
    velocities = numpy(state.vel)[0].astype(np.float64)
    active = numpy(state.active)[0].astype(bool)
    soft._queried_states.discard(state)
    positions = positions[active]
    velocities = velocities[active]
    if positions.size == 0:
        raise RuntimeError("MPM entity has no active particles")
    speeds = np.linalg.norm(velocities, axis=1)
    result = {
        "com_m": positions.mean(axis=0),
        "aabb_min_m": positions.min(axis=0),
        "aabb_max_m": positions.max(axis=0),
        "speed_rms_m_s": float(np.sqrt(np.mean(speeds * speeds))),
        "speed_max_m_s": float(np.max(speeds)),
        "active_count": int(len(positions)),
        "finite": bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
    }
    if include_positions:
        result["positions_m"] = positions
    return result


def _goal_particle_fraction(soft: Any, goal_xy: np.ndarray) -> float:
    state = soft.get_state()
    positions = numpy(state.pos)[0].astype(np.float64)
    active = numpy(state.active)[0].astype(bool)
    soft._queried_states.discard(state)
    positions = positions[active]
    half_goal = np.asarray(GOAL_SIZE_XY_M, dtype=np.float64) / 2.0
    inside = np.all(np.abs(positions[:, :2] - goal_xy) <= half_goal, axis=1)
    return float(np.mean(inside))


def _serial_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in snapshot.items()
        if key != "positions_m"
    }


def _render_rgb(camera: Any) -> np.ndarray:
    rgb, _, _, _ = camera.render(rgb=True, force_render=True)
    frame = np.asarray(rgb)
    if frame.dtype != np.uint8:
        multiplier = 255.0 if np.nanmax(frame) <= 1.0 else 1.0
        frame = np.clip(frame * multiplier, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"unexpected camera frame shape: {frame.shape}")
    return np.ascontiguousarray(frame[..., :3])


def _write_png(frame: np.ndarray, path: Path) -> str:
    import imageio.v2 as iio

    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    iio.imwrite(temporary, frame, format="png")
    temporary.replace(path)
    return str(path)


def _encode_video(frames: list[np.ndarray], output_path: Path) -> dict[str, Any]:
    import imageio.v2 as iio

    output_path = require_within_project(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    try:
        with iio.get_writer(
            temporary,
            fps=VIDEO_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_log_level="warning",
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        if temporary.stat().st_size < 10_000:
            raise RuntimeError("encoded MP4 is unexpectedly small")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path),
        "size_bytes": int(output_path.stat().st_size),
        "sha256": _sha256(output_path),
        "fps": VIDEO_FPS,
        "frame_count": len(frames),
        "resolution_wh": [int(frames[0].shape[1]), int(frames[0].shape[0])],
    }


def _fixture_material(gs: Any) -> Any:
    return gs.materials.Rigid(
        needs_coup=True,
        friction=TABLE_COUP_FRICTION,
        coup_friction=TABLE_COUP_FRICTION,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.004,
        sdf_min_res=24,
        sdf_max_res=64,
        gravity_compensation=1.0,
    )


def _asset_color(asset: AssetSpec) -> tuple[float, float, float, float]:
    return (0.88, 0.24, 0.12, 1.0) if asset.geometry_family == "solid" else (0.10, 0.55, 0.78, 1.0)


def _mesh_spawn(asset: AssetSpec, action: str) -> tuple[float, float, float]:
    import trimesh

    mesh = trimesh.load(asset.collision_mesh, force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
    start_xy, _ = action_centers(action)
    return (
        float(start_xy[0] - center_xy[0]),
        float(start_xy[1] - center_xy[1]),
        float(TABLE_TOP_Z_M + SPAWN_GAP_M - bounds[0, 2]),
    )


def _build_scene(gs: Any, asset: AssetSpec, robot_type: str, action: str) -> tuple[Any, Any, Any, Any, tuple[float, float, float]]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=asset.source_dt_s,
            substeps=asset.source_substeps,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=False,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=False,
            use_contact_island=False,
        ),
        mpm_options=gs.options.MPMOptions(
            particle_size=asset.particle_size_m,
            grid_density=asset.grid_density_per_m,
            lower_bound=(0.02, -0.46, 0.015),
            upper_bound=(0.92, 0.46, 0.72),
            enable_CPIC=False,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_mpm=True),
        show_viewer=False,
    )
    robot = add_robot(gs, scene, robot_type)
    scene.add_entity(
        morph=gs.morphs.Box(pos=TABLE_CENTER_M, size=TABLE_SIZE_M, fixed=True),
        material=_fixture_material(gs),
        surface=gs.surfaces.Rough(color=(0.31, 0.34, 0.36, 1.0)),
        name="workbench",
    )
    _, goal_xy = action_centers(action)
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(float(goal_xy[0]), float(goal_xy[1]), TABLE_TOP_Z_M + 0.0005),
            size=(*GOAL_SIZE_XY_M, 0.001),
            fixed=True,
            collision=False,
        ),
        material=gs.materials.Rigid(needs_coup=False, gravity_compensation=1.0),
        surface=gs.surfaces.Rough(color=(0.16, 0.72, 0.28, 1.0)),
        name="colored_goal_region",
    )
    spawn_pos = _mesh_spawn(asset, action)
    soft = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=asset.collision_mesh,
            pos=spawn_pos,
            euler=(0.0, 0.0, 0.0),
            scale=1.0,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.MPM.Elastic(
            E=asset.youngs_modulus_pa,
            nu=asset.poisson_ratio,
            rho=asset.density_kg_m3,
            model="corotation",
            sampler="pbs-32",
        ),
        surface=gs.surfaces.Default(color=_asset_color(asset), vis_mode="particle"),
        name="soft_object",
    )
    camera = scene.add_camera(
        res=(640, 480),
        pos=(1.04, -0.88, 0.72),
        lookat=(0.46, 0.0, 0.16),
        fov=49,
        GUI=False,
        near=0.05,
        far=2.2,
    )
    scene.build()
    return scene, robot, soft, camera, spawn_pos


def _build_reachability_scene(gs: Any, robot_type: str) -> tuple[Any, RobotHandle]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, substeps=2, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = add_robot(gs, scene, robot_type)
    scene.add_entity(
        morph=gs.morphs.Box(pos=TABLE_CENTER_M, size=TABLE_SIZE_M, fixed=True),
        material=gs.materials.Rigid(gravity_compensation=1.0),
        surface=gs.surfaces.Rough(color=(0.31, 0.34, 0.36, 1.0)),
    )
    scene.build()
    return scene, configure_robot(robot, robot_type)


def run_reachability(robot_type: str, assets: Iterable[AssetSpec]) -> dict[str, Any]:
    run_cache = configure_writable_runtime(f"reachability_{robot_type}")
    import genesis as gs
    import torch

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    scene = None
    started = time.perf_counter()
    try:
        if gs.backend != gs.cuda:
            raise RuntimeError(f"GPU requested, Genesis selected {gs.backend}")
        scene, handle = _build_reachability_scene(gs, robot_type)
        cases: list[dict[str, Any]] = []
        for asset in assets:
            for action in ("push", "press_drag"):
                reference = nominal_reference(asset.extent_m, action)
                tool_path = make_tool_path(action, reference, handle.tool_vertical_offsets_m)
                motion = solve_tool_path(handle, tool_path)
                cases.append(
                    {
                        "asset_id": asset.asset_id,
                        "object_type": asset.object_type,
                        "geometry_family": asset.geometry_family,
                        "extent_m": list(asset.extent_m),
                        "action": action,
                        "tool_path": tool_path,
                        "motion_plan": serial_motion_plan(motion),
                    }
                )
        return {
            "schema_version": 1,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "robot": robot_type,
            "runtime": {
                "backend": str(gs.backend),
                "genesis_version": getattr(gs, "__version__", "unknown"),
                "genesis_module": str(Path(gs.__file__).resolve()),
                "cuda_device": torch.cuda.get_device_name(),
                "run_cache": str(run_cache),
            },
            "robot_geometry": {
                "home_grip_center_m": handle.home_grip_center_m.tolist(),
                "tool_vertical_offsets_m": list(handle.tool_vertical_offsets_m),
                "open_width_m": handle.open_width_m,
                "closed_width_m": handle.closed_width_m,
                "arm_dof_count": len(handle.arm_dofs),
            },
            "case_count": len(cases),
            "valid_case_count": sum(case["motion_plan"]["valid"] for case in cases),
            "cases": cases,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if scene is not None:
            del scene
        gs.destroy()


def run_trial(
    asset: AssetSpec,
    robot_type: str,
    action: str,
    output_video: Path,
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    output_video = require_within_project(output_video)
    run_key = f"{robot_type}_{action}_{asset.asset_id}"
    run_cache = configure_writable_runtime(run_key)
    import genesis as gs
    import torch

    scene = None
    started = time.perf_counter()
    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
        write_isolation = isolate_genesis_particle_sampler(run_cache)
        if gs.backend != gs.cuda:
            raise RuntimeError(f"GPU requested, Genesis selected {gs.backend}")
        torch.cuda.reset_peak_memory_stats()
        build_started = time.perf_counter()
        scene, robot, soft, camera, spawn_pos = _build_scene(gs, asset, robot_type, action)
        build_seconds = time.perf_counter() - build_started
        handle = configure_robot(robot, robot_type)
        steps = stage_steps(asset.source_dt_s)
        for _ in range(steps["settle_clear"]):
            robot.control_dofs_position(handle.home_arm_q, handle.arm_dofs)
            robot.control_dofs_position(handle.finger_target(handle.open_width_m), handle.finger_dofs)
            scene.step(update_visualizer=False)
        reference = _particle_snapshot(soft, include_positions=action == ACTION_EDGE_DRAG)
        if action == ACTION_EDGE_DRAG:
            candidate = generate_far_edge_grasp(
                reference["positions_m"],
                asset.particle_size_m,
                handle.open_width_m,
                handle.tool_vertical_offsets_m,
            )
            tool_path = (
                make_edge_drag_tool_path(candidate, reference) if candidate is not None else None
            )
        else:
            candidate = None
            tool_path = make_tool_path(action, reference, handle.tool_vertical_offsets_m)
        motion = solve_tool_path(handle, tool_path) if tool_path is not None else None
        base_result: dict[str, Any] = {
            "schema_version": 1,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "asset": asset.to_dict(),
            "robot": robot_type,
            "action": action,
            "scale": 1.0,
            "spawn_position_m": list(spawn_pos),
            "settled_reference": _serial_snapshot(reference),
            "tool_path": tool_path,
            "motion_plan": serial_motion_plan(motion) if motion is not None else None,
            "robot_geometry": {
                "home_grip_center_m": handle.home_grip_center_m.tolist(),
                "base_grip_offset_m": handle.base_grip_offset_m.tolist(),
                "tool_vertical_offsets_m": list(handle.tool_vertical_offsets_m),
                "open_width_m": handle.open_width_m,
                "closed_width_m": handle.closed_width_m,
                "arm_dof_count": len(handle.arm_dofs),
            },
            "protocol": {
                "original_scale_only": True,
                "material": "MPM.Elastic",
                "dt_s": asset.source_dt_s,
                "substeps": asset.source_substeps,
                "stage_steps": steps,
                "robot_mpm_coupling_friction": ROBOT_COUP_FRICTION,
                "colored_goal_region_size_xy_m": list(GOAL_SIZE_XY_M),
                "trajectory_recording": False,
                "failed_video_retention": False,
            },
            "runtime": {
                "backend": str(gs.backend),
                "genesis_version": getattr(gs, "__version__", "unknown"),
                "genesis_module": str(Path(gs.__file__).resolve()),
                "cuda_device": torch.cuda.get_device_name(),
                "build_seconds": build_seconds,
                "peak_cuda_bytes": None,
                "write_isolation": write_isolation,
            },
            "phase_end": [],
            "assessment": None,
            "video": None,
            "diagnostic_images": {},
        }
        start_frame = _render_rgb(camera)
        diagnostic_prefix = DIAGNOSTIC_ROOT / f"{robot_type}_{action}_{asset.asset_id}"
        base_result["diagnostic_images"]["settled"] = _write_png(
            start_frame, diagnostic_prefix.with_name(diagnostic_prefix.name + "_settled.png")
        )
        if motion is None:
            base_result["status"] = "no_far_edge_candidate"
            base_result["plan_only"] = plan_only
            base_result["runtime"]["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
            base_result["elapsed_seconds"] = time.perf_counter() - started
            return base_result
        if plan_only:
            base_result["status"] = "plan_valid" if motion["valid"] else "failed_ik"
            base_result["plan_only"] = True
            base_result["runtime"]["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
            base_result["elapsed_seconds"] = time.perf_counter() - started
            return base_result
        if not motion["valid"]:
            base_result["status"] = "failed_ik"
            base_result["plan_only"] = False
            base_result["runtime"]["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
            base_result["elapsed_seconds"] = time.perf_counter() - started
            return base_result

        frames: list[np.ndarray] = [start_frame]
        frame_stds = [float(np.std(start_frame.astype(np.float32)))]
        current_q = handle.home_arm_q.copy()
        current_width = handle.open_width_m
        sim_time = 0.0
        next_frame_time = FRAME_PERIOD_S
        active_count_constant = True

        finger_link_names = (
            ("gripper_link1", "gripper_link2")
            if robot_type == "piper"
            else ("panda_leftfinger", "panda_rightfinger")
        )

        def actual_tool_state() -> dict[str, Any]:
            vertices = np.concatenate(
                [
                    numpy(geom.get_verts()).reshape(-1, 3)
                    for link_name in finger_link_names
                    for geom in robot.get_link(link_name).geoms
                ],
                axis=0,
            ).astype(np.float64, copy=False)
            tcp_pos = flat(handle.tcp.get_pos(relative=False))
            grip_center = tcp_pos + motion["grip_offset_m"]
            return {
                "actual_tcp_m": tcp_pos.tolist(),
                "actual_grip_center_m": grip_center.tolist(),
                "actual_finger_aabb_min_m": vertices.min(axis=0).tolist(),
                "actual_finger_aabb_max_m": vertices.max(axis=0).tolist(),
            }

        def record_phase(name: str) -> dict[str, Any]:
            nonlocal active_count_constant
            snapshot = _particle_snapshot(soft)
            active_count_constant = active_count_constant and snapshot["active_count"] == reference["active_count"]
            serial = {"stage": name, **_serial_snapshot(snapshot), **actual_tool_state()}
            base_result["phase_end"].append(serial)
            return snapshot

        def step_controls(arm_q: np.ndarray, width_m: float) -> None:
            nonlocal sim_time, next_frame_time
            robot.control_dofs_position(arm_q, handle.arm_dofs)
            robot.control_dofs_position(handle.finger_target(width_m), handle.finger_dofs)
            scene.step(update_visualizer=False)
            sim_time += asset.source_dt_s
            if sim_time + 1.0e-12 >= next_frame_time:
                frame = _render_rgb(camera)
                frames.append(frame)
                frame_stds.append(float(np.std(frame.astype(np.float32))))
                next_frame_time += FRAME_PERIOD_S

        def stage(name: str, target_q: np.ndarray, target_width: float) -> dict[str, Any]:
            nonlocal current_q, current_width
            source_q = current_q.copy()
            source_width = float(current_width)
            count = steps[name]
            for index in range(count):
                alpha = smoothstep01((index + 1) / count)
                q = (1.0 - alpha) * source_q + alpha * target_q
                width = (1.0 - alpha) * source_width + alpha * target_width
                step_controls(q, width)
            current_q = target_q.copy()
            current_width = float(target_width)
            return record_phase(name)

        if action == ACTION_PRESS_DRAG:
            contact_width = min(0.050, handle.open_width_m)
        elif action == ACTION_EDGE_DRAG:
            contact_width = float(tool_path["commanded_close_width_m"])
        else:
            contact_width = handle.closed_width_m
        approach_width = handle.open_width_m if action == ACTION_EDGE_DRAG else contact_width
        stage("approach", motion["solutions"]["approach"], approach_width)
        stage("descend", motion["solutions"]["contact"], approach_width)
        if action == ACTION_EDGE_DRAG:
            stage("close", motion["solutions"]["contact"], contact_width)
        stage("preload_hold", motion["solutions"]["contact"], contact_width)

        transport_names = sorted(name for name in motion["solutions"] if name.startswith("transport_"))
        segment_steps = max(1, steps["transport"] // len(transport_names))
        for name in transport_names:
            source_q = current_q.copy()
            target_q = motion["solutions"][name]
            for index in range(segment_steps):
                alpha = smoothstep01((index + 1) / segment_steps)
                step_controls((1.0 - alpha) * source_q + alpha * target_q, contact_width)
            current_q = target_q.copy()
        after_transport = record_phase("transport")
        transport_frame = _render_rgb(camera)
        base_result["diagnostic_images"]["after_transport"] = _write_png(
            transport_frame, diagnostic_prefix.with_name(diagnostic_prefix.name + "_after_transport.png")
        )
        stage("transport_hold", current_q, contact_width)
        retreat_width = contact_width
        if action == ACTION_EDGE_DRAG:
            stage("open", current_q, handle.open_width_m)
            retreat_width = handle.open_width_m
        stage("retreat", motion["solutions"]["retreat"], retreat_width)
        final = stage("settle_goal", motion["solutions"]["retreat"], retreat_width)
        final_frame = _render_rgb(camera)
        base_result["diagnostic_images"]["final"] = _write_png(
            final_frame, diagnostic_prefix.with_name(diagnostic_prefix.name + "_final.png")
        )
        _, goal_xy = action_centers(action)
        goal_particle_fraction = _goal_particle_fraction(soft, goal_xy)
        assessment = assess_transport(
            action,
            reference,
            final,
            active_count_constant=active_count_constant,
            goal_particle_fraction=goal_particle_fraction,
        )
        assessment["after_transport_com_m"] = after_transport["com_m"].tolist()
        assessment["simulated_time_s"] = sim_time
        assessment["frame_count"] = len(frames)
        assessment["minimum_frame_std"] = float(min(frame_stds))
        assessment["median_frame_std"] = float(np.median(frame_stds))
        base_result["assessment"] = assessment
        base_result["status"] = assessment["status"]
        if assessment["status"] == "success":
            base_result["video"] = _encode_video(frames, output_video)
        base_result["plan_only"] = False
        base_result["runtime"]["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
        base_result["elapsed_seconds"] = time.perf_counter() - started
        return base_result
    finally:
        if scene is not None:
            del scene
        gs.destroy()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
