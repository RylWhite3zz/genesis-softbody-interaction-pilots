"""One isolated Genesis 1.1.2 pick-and-place attempt with no trajectory recorder."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import AssetSpec, OPEN_WIDTH_M
from .paths import ROBOT_URDF, configure_writable_runtime, isolate_genesis_particle_sampler, require_within_project
from .protocol import (
    DT_S,
    FRAME_EVERY_STEPS,
    GRASP_HEIGHT_FRACTION,
    HOME_ARM_Q,
    HOME_GRIP_CENTER_M,
    LIFT_HEIGHT_M,
    OPEN_WIDTH_M as PROTOCOL_OPEN_WIDTH_M,
    PICK_CENTER_XY_M,
    SPAWN_GAP_M,
    STAGE_STEPS,
    SUBSTEPS,
    ROBOT_COUP_FRICTION,
    TABLE_CENTER_M,
    TABLE_SIZE_M,
    TABLE_TOP_Z_M,
    TRAY_CENTER_XY_M,
    TRAY_FLOOR_Z_M,
    VIDEO_FPS,
    assess_success,
    smoothstep01,
    tray_boxes,
    validate_protocol,
)


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _flat(tensor: Any) -> np.ndarray:
    return _numpy(tensor).reshape(-1).astype(np.float64, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_snapshot(soft: Any) -> dict[str, Any]:
    state = soft.get_state()
    positions = _numpy(state.pos)[0].astype(np.float64)
    velocities = _numpy(state.vel)[0].astype(np.float64)
    active = _numpy(state.active)[0].astype(bool)
    soft._queried_states.discard(state)
    positions = positions[active]
    velocities = velocities[active]
    if positions.size == 0:
        raise RuntimeError("MPM entity has no active particles")
    speed = np.linalg.norm(velocities, axis=1)
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    return {
        "com_m": positions.mean(axis=0),
        "aabb_min_m": positions.min(axis=0),
        "aabb_max_m": positions.max(axis=0),
        "speed_rms_m_s": float(np.sqrt(np.mean(speed * speed))),
        "speed_max_m_s": float(np.max(speed)),
        "active_count": int(np.count_nonzero(active)),
        "finite": finite,
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


def _asset_color(asset: AssetSpec) -> tuple[float, float, float, float]:
    if "inflatable_mattress" in asset.canonical_id:
        return (0.86, 0.58, 0.10, 1.0)
    if "steam_mop_pad" in asset.canonical_id:
        return (0.78, 0.16, 0.20, 1.0)
    return (0.16, 0.62, 0.58, 1.0)


def _spawn_pose(orientation: dict[str, Any]) -> tuple[float, float, float]:
    lower, upper = np.asarray(orientation["bounds_before_translation_m"], dtype=np.float64)
    center = (lower + upper) / 2.0
    return (
        float(PICK_CENTER_XY_M[0] - center[0]),
        float(PICK_CENTER_XY_M[1] - center[1]),
        float(TABLE_TOP_Z_M + SPAWN_GAP_M - lower[2]),
    )


def _fixture_material(gs: Any) -> Any:
    return gs.materials.Rigid(
        needs_coup=True,
        friction=1.2,
        coup_friction=1.2,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.004,
        sdf_min_res=24,
        sdf_max_res=64,
        gravity_compensation=1.0,
    )


def _build_scene(gs: Any, asset: AssetSpec, orientation: dict[str, Any]) -> tuple[Any, Any, Any, Any, tuple[float, float, float]]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT_S,
            substeps=SUBSTEPS,
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
            lower_bound=(0.10, -0.28, 0.015),
            upper_bound=(0.86, 0.43, 0.50),
            enable_CPIC=False,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_mpm=True),
        show_viewer=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(ROBOT_URDF),
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=("tcp_link",),
            requires_jac_and_IK=True,
            collision=True,
            convexify=True,
            decimate=True,
            recompute_inertia=False,
            default_armature=0.005,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            friction=1.0,
            coup_friction=ROBOT_COUP_FRICTION,
            coup_softness=0.002,
            coup_restitution=0.0,
            sdf_cell_size=0.003,
            sdf_min_res=24,
            sdf_max_res=96,
            gravity_compensation=1.0,
        ),
        name="fixed_piper",
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=TABLE_CENTER_M, size=TABLE_SIZE_M, fixed=True),
        material=_fixture_material(gs),
        surface=gs.surfaces.Rough(color=(0.34, 0.37, 0.39, 1.0)),
        name="workbench",
    )
    tray_surfaces = (
        (0.18, 0.42, 0.58, 1.0),
        (0.14, 0.35, 0.49, 1.0),
        (0.14, 0.35, 0.49, 1.0),
        (0.14, 0.35, 0.49, 1.0),
        (0.14, 0.35, 0.49, 1.0),
    )
    for index, (box, color) in enumerate(zip(tray_boxes(), tray_surfaces, strict=True)):
        scene.add_entity(
            morph=gs.morphs.Box(pos=box["pos"], size=box["size"], fixed=True),
            material=_fixture_material(gs),
            surface=gs.surfaces.Rough(color=color),
            name=f"tray_{index}",
        )
    spawn_pos = _spawn_pose(orientation)
    soft = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=asset.collision_mesh,
            pos=spawn_pos,
            euler=tuple(orientation["euler_deg"]),
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
        pos=(0.95, -0.73, 0.60),
        lookat=(0.42, 0.015, 0.18),
        fov=48,
        GUI=False,
        near=0.05,
        far=2.0,
    )
    scene.build()
    return scene, robot, soft, camera, spawn_pos


def _configure_robot(robot: Any) -> tuple[list[int], list[int], Any]:
    arm_dofs = [int(robot.get_joint(f"joint{i}").dofs_idx_local[0]) for i in range(1, 7)]
    finger_dofs = [
        int(robot.get_joint("gripper_joint1").dofs_idx_local[0]),
        int(robot.get_joint("gripper_joint2").dofs_idx_local[0]),
    ]
    robot.set_dofs_kp([2000, 2000, 1600, 800, 400, 400], arm_dofs)
    robot.set_dofs_kv([100, 100, 80, 40, 20, 20], arm_dofs)
    robot.set_dofs_force_range([-100] * 6, [100] * 6, arm_dofs)
    robot.set_dofs_kp([40, 40], finger_dofs)
    robot.set_dofs_kv([5, 5], finger_dofs)
    robot.set_dofs_force_range([-10, -10], [10, 10], finger_dofs)
    robot.set_dofs_position(HOME_ARM_Q, arm_dofs, zero_velocity=True)
    robot.set_dofs_position([OPEN_WIDTH_M / 2.0, -OPEN_WIDTH_M / 2.0], finger_dofs, zero_velocity=True)
    robot.control_dofs_position(HOME_ARM_Q, arm_dofs)
    robot.control_dofs_position([OPEN_WIDTH_M / 2.0, -OPEN_WIDTH_M / 2.0], finger_dofs)
    return arm_dofs, finger_dofs, robot.get_link("tcp_link")


def _solve_waypoint(
    robot: Any,
    tcp: Any,
    arm_dofs: list[int],
    target_grip_center: np.ndarray,
    grip_minus_tcp: np.ndarray,
    target_quat: np.ndarray,
    init_arm_q: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_tcp = target_grip_center - grip_minus_tcp
    init_q = _flat(robot.get_qpos()).copy()
    init_q[:6] = init_arm_q
    qpos, error = robot.inverse_kinematics(
        link=tcp,
        pos=target_tcp,
        quat=target_quat,
        init_qpos=init_q,
        dofs_idx_local=arm_dofs,
        return_error=True,
        max_samples=8,
        max_solver_iters=150,
        damping=0.03,
        max_step_size=0.08,
        pos_tol=7.5e-4,
        rot_tol=7.5e-3,
    )
    arm_q = _flat(qpos)[:6]
    error6 = _flat(error)
    report = {
        "target_grip_center_m": target_grip_center.tolist(),
        "target_tcp_m": target_tcp.tolist(),
        "arm_q_rad": arm_q.tolist(),
        "reported_error": error6.tolist(),
        "position_error_norm_m": float(np.linalg.norm(error6[:3])),
        "rotation_error_norm_rad": float(np.linalg.norm(error6[3:])),
    }
    return arm_q, report


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
        size_bytes = temporary.stat().st_size
        if size_bytes < 10_000:
            raise RuntimeError(f"encoded MP4 is unexpectedly small: {size_bytes} bytes")
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


def run_attempt(asset: AssetSpec, orientation: dict[str, Any], output_video: Path) -> dict[str, Any]:
    validate_protocol()
    if not math.isclose(OPEN_WIDTH_M, PROTOCOL_OPEN_WIDTH_M, abs_tol=0.0):
        raise RuntimeError("inventory and protocol jaw widths differ")
    if not ROBOT_URDF.is_file():
        raise FileNotFoundError(ROBOT_URDF)
    output_video = require_within_project(output_video)
    configure_writable_runtime()
    import genesis as gs
    import torch

    scene = None
    started = time.perf_counter()
    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
        write_isolation = isolate_genesis_particle_sampler()
        if gs.backend != gs.cuda:
            raise RuntimeError(f"GPU requested, but Genesis selected {gs.backend}")
        torch.cuda.reset_peak_memory_stats()
        build_started = time.perf_counter()
        scene, robot, soft, camera, spawn_pos = _build_scene(gs, asset, orientation)
        build_seconds = time.perf_counter() - build_started
        arm_dofs, finger_dofs, tcp = _configure_robot(robot)
        home_tcp = _flat(tcp.get_pos(relative=False))
        home_quat = _flat(tcp.get_quat(relative=False))
        grip_minus_tcp = HOME_GRIP_CENTER_M - home_tcp
        rotated_extent = np.asarray(orientation["extent_after_rotation_m"], dtype=np.float64)
        nominal_object_center = np.asarray(
            (
                PICK_CENTER_XY_M[0],
                PICK_CENTER_XY_M[1],
                TABLE_TOP_Z_M + SPAWN_GAP_M + rotated_extent[2] / 2.0,
            ),
            dtype=np.float64,
        )
        nominal_grasp_center = nominal_object_center + np.asarray(
            (0.0, 0.0, GRASP_HEIGHT_FRACTION * rotated_extent[2]), dtype=np.float64
        )
        preposition_q, preposition_ik = _solve_waypoint(
            robot,
            tcp,
            arm_dofs,
            nominal_grasp_center,
            grip_minus_tcp,
            home_quat,
            HOME_ARM_Q,
        )
        if preposition_ik["position_error_norm_m"] > 0.002 or preposition_ik["rotation_error_norm_rad"] > 0.015:
            raise RuntimeError("nominal centered open-jaw reset pose is not reachable")
        robot.set_dofs_position(preposition_q, arm_dofs, zero_velocity=True)
        robot.control_dofs_position(preposition_q, arm_dofs)
        frames: list[np.ndarray] = [_render_rgb(camera)]
        frame_stds: list[float] = [float(np.std(frames[-1].astype(np.float32)))]
        step = 0
        current_q = preposition_q.copy()
        current_width = OPEN_WIDTH_M
        reference: dict[str, Any] | None = None
        max_lift_com_z = -math.inf
        active_count_constant = True
        finite_state = True
        phase_end_metrics: list[dict[str, Any]] = []
        ik_reports: dict[str, Any] = {"reset_upper_quarter_open_jaw": preposition_ik}
        finger_links = (robot.get_link("gripper_link1"), robot.get_link("gripper_link2"))

        def capture(*, track_object: bool) -> None:
            nonlocal max_lift_com_z, active_count_constant, finite_state
            frame = _render_rgb(camera)
            frames.append(frame)
            frame_stds.append(float(np.std(frame.astype(np.float32))))
            if track_object:
                snapshot = _particle_snapshot(soft)
                finite_state = finite_state and bool(snapshot["finite"])
                if reference is not None:
                    active_count_constant = active_count_constant and (
                        snapshot["active_count"] == reference["active_count"]
                    )
                    max_lift_com_z = max(max_lift_com_z, float(snapshot["com_m"][2]))

        def record_phase_end(name: str) -> None:
            nonlocal max_lift_com_z, active_count_constant, finite_state
            snapshot = _particle_snapshot(soft)
            qpos = _flat(robot.get_dofs_position())
            tcp_position = _flat(tcp.get_pos(relative=False))
            finger_aabbs = [_flat(link.get_AABB()).reshape(2, 3) for link in finger_links]
            finger_aabbs.sort(key=lambda aabb: float(np.mean(aabb[:, 1])))
            pad_gap_y = float(finger_aabbs[1][0, 1] - finger_aabbs[0][1, 1])
            finite_state = finite_state and bool(snapshot["finite"])
            if reference is not None:
                active_count_constant = active_count_constant and (
                    snapshot["active_count"] == reference["active_count"]
                )
                max_lift_com_z = max(max_lift_com_z, float(snapshot["com_m"][2]))
            phase_end_metrics.append(
                {
                    "stage": name,
                    "step": step,
                    "com_m": snapshot["com_m"].tolist(),
                    "aabb_min_m": snapshot["aabb_min_m"].tolist(),
                    "aabb_max_m": snapshot["aabb_max_m"].tolist(),
                    "speed_rms_m_s": snapshot["speed_rms_m_s"],
                    "active_count": snapshot["active_count"],
                    "actual_jaw_width_m": float(qpos[6] - qpos[7]),
                    "collision_pad_gap_y_m": pad_gap_y,
                    "grip_center_m": (tcp_position + grip_minus_tcp).tolist(),
                }
            )

        def stage(
            name: str,
            target_q: np.ndarray,
            target_width: float,
            *,
            track_object: bool = False,
        ) -> None:
            nonlocal step, current_q, current_width
            source_q = current_q.copy()
            source_width = float(current_width)
            count = STAGE_STEPS[name]
            for index in range(count):
                alpha = smoothstep01((index + 1) / count)
                arm_q = (1.0 - alpha) * source_q + alpha * target_q
                width = (1.0 - alpha) * source_width + alpha * target_width
                robot.control_dofs_position(arm_q, arm_dofs)
                robot.control_dofs_position([width / 2.0, -width / 2.0], finger_dofs)
                scene.step(update_visualizer=False)
                step += 1
                if step % FRAME_EVERY_STEPS == 0 or index + 1 == count:
                    capture(track_object=track_object)
            current_q = target_q.copy()
            current_width = float(target_width)
            record_phase_end(name)

        stage("initial_hold", current_q, OPEN_WIDTH_M)
        reference = _particle_snapshot(soft)
        finite_state = finite_state and bool(reference["finite"])
        max_lift_com_z = float(reference["com_m"][2])
        settled_width = float(reference["aabb_max_m"][1] - reference["aabb_min_m"][1])
        settled_height = float(reference["aabb_max_m"][2] - reference["aabb_min_m"][2])
        grasp_z_offset = GRASP_HEIGHT_FRACTION * settled_height
        initial_metrics = {
            "settled_com_m": reference["com_m"].tolist(),
            "settled_aabb_min_m": reference["aabb_min_m"].tolist(),
            "settled_aabb_max_m": reference["aabb_max_m"].tolist(),
            "settled_jaw_axis_extent_m": settled_width,
            "settled_height_m": settled_height,
            "grasp_z_offset_from_com_m": grasp_z_offset,
            "active_particle_count": reference["active_count"],
        }
        if settled_width > OPEN_WIDTH_M + 0.004:
            return {
                "schema_version": 1,
                "asset": asset.to_dict(),
                "status": "skipped_post_settle_size",
                "reason": f"settled jaw-axis extent {settled_width:.6f} m exceeds 70 mm tolerance",
                "orientation": orientation,
                "scale": 1.0,
                "initial_metrics": initial_metrics,
                "phase_end_metrics": phase_end_metrics,
                "ik": ik_reports,
                "video": None,
                "elapsed_seconds": time.perf_counter() - started,
            }

        reference_com = np.asarray(reference["com_m"], dtype=np.float64)
        grasp_center = reference_com + np.asarray((0.0, 0.0, grasp_z_offset), dtype=np.float64)
        tray_delta = np.asarray(
            [
                TRAY_CENTER_XY_M[0] - PICK_CENTER_XY_M[0],
                TRAY_CENTER_XY_M[1] - PICK_CENTER_XY_M[1],
                TRAY_FLOOR_Z_M - TABLE_TOP_Z_M,
            ],
            dtype=np.float64,
        )
        grip_targets = {
            "approach": grasp_center.copy(),
            "descend": grasp_center.copy(),
            "lift": grasp_center + np.asarray((0.0, 0.0, LIFT_HEIGHT_M)),
            "transfer": grasp_center + tray_delta + np.asarray((0.0, 0.0, LIFT_HEIGHT_M)),
            "lower_into_tray": grasp_center + tray_delta,
            "retreat": grasp_center + tray_delta + np.asarray((0.0, 0.0, LIFT_HEIGHT_M)),
        }
        waypoint_q: dict[str, np.ndarray] = {}
        init_arm_q = current_q.copy()
        for name in ("approach", "descend", "lift", "transfer", "lower_into_tray", "retreat"):
            solution, report = _solve_waypoint(
                robot,
                tcp,
                arm_dofs,
                grip_targets[name],
                grip_minus_tcp,
                home_quat,
                init_arm_q,
            )
            waypoint_q[name] = solution
            ik_reports[name] = report
            init_arm_q = solution
        ik_valid = all(
            report["position_error_norm_m"] <= 0.002 and report["rotation_error_norm_rad"] <= 0.015
            for report in ik_reports.values()
        )
        if not ik_valid:
            return {
                "schema_version": 1,
                "asset": asset.to_dict(),
                "status": "failed_ik",
                "reason": "one or more fixed-orientation waypoints exceed IK tolerances",
                "orientation": orientation,
                "scale": 1.0,
                "initial_metrics": initial_metrics,
                "ik": ik_reports,
                "phase_end_metrics": phase_end_metrics,
                "video": None,
                "elapsed_seconds": time.perf_counter() - started,
            }

        close_width = max(0.024, settled_width - 0.014)
        stage("approach", waypoint_q["approach"], OPEN_WIDTH_M)
        stage("descend", waypoint_q["descend"], OPEN_WIDTH_M)
        stage("close", waypoint_q["descend"], close_width, track_object=True)
        stage("grip_hold", waypoint_q["descend"], close_width, track_object=True)
        stage("lift", waypoint_q["lift"], close_width, track_object=True)
        stage("lifted_hold", waypoint_q["lift"], close_width, track_object=True)
        stage("transfer", waypoint_q["transfer"], close_width, track_object=True)
        stage("lower_into_tray", waypoint_q["lower_into_tray"], close_width, track_object=True)
        stage("open", waypoint_q["lower_into_tray"], OPEN_WIDTH_M, track_object=True)
        stage("retreat", waypoint_q["retreat"], OPEN_WIDTH_M, track_object=True)
        stage("settle_in_tray", waypoint_q["retreat"], OPEN_WIDTH_M, track_object=True)

        final = _particle_snapshot(soft)
        finite_state = finite_state and bool(final["finite"])
        active_count_constant = active_count_constant and final["active_count"] == reference["active_count"]
        final_tcp = _flat(tcp.get_pos(relative=False))
        final_grip_center = final_tcp + grip_minus_tcp
        max_lift_delta = float(max_lift_com_z - reference_com[2])
        transfer_end = next(row for row in phase_end_metrics if row["stage"] == "transfer")
        transfer_end_com_delta_z = float(transfer_end["com_m"][2] - reference_com[2])
        metrics = {
            **initial_metrics,
            "particle_count": int(soft.n_particles),
            "final_active_particle_count": final["active_count"],
            "active_particle_count_constant": active_count_constant,
            "finite_state": finite_state,
            "max_lift_delta_z_m": max_lift_delta,
            "transfer_end_com_delta_z_m": transfer_end_com_delta_z,
            "transfer_end_aabb_min_z_m": float(transfer_end["aabb_min_m"][2]),
            "final_com_m": final["com_m"].tolist(),
            "final_aabb_min_m": final["aabb_min_m"].tolist(),
            "final_aabb_max_m": final["aabb_max_m"].tolist(),
            "final_speed_rms_m_s": final["speed_rms_m_s"],
            "final_speed_max_m_s": final["speed_max_m_s"],
            "final_grip_center_m": final_grip_center.tolist(),
            "final_grip_object_distance_m": float(np.linalg.norm(final_grip_center - final["com_m"])),
            "commanded_close_width_m": close_width,
            "frame_count": len(frames),
            "minimum_frame_std": float(min(frame_stds)),
            "median_frame_std": float(np.median(frame_stds)),
            "simulated_time_s": float(step * DT_S),
            "physics_step_count": step,
            "build_seconds": build_seconds,
            "total_seconds_before_encoding": time.perf_counter() - started,
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        assessment = assess_success(metrics)
        video = None
        status = assessment["status"]
        reason = "all pick, lift, release, tray-containment, stability, and video gates passed"
        if assessment["all_gates_pass"]:
            video = _encode_video(frames, output_video)
            status = "success"
        else:
            failed = [name for name, passed in assessment["gates"].items() if not passed]
            reason = "failed gates: " + ", ".join(failed)
        return {
            "schema_version": 1,
            "asset": asset.to_dict(),
            "status": status,
            "reason": reason,
            "orientation": orientation,
            "scale": 1.0,
            "spawn_position_m": list(spawn_pos),
            "runtime": {
                "backend": str(gs.backend),
                "genesis_module": str(Path(gs.__file__).resolve()),
                "genesis_version": getattr(gs, "__version__", "unknown"),
                "cuda_device": torch.cuda.get_device_name(),
                "write_isolation": write_isolation,
            },
            "protocol": {
                "dt_s": DT_S,
                "substeps": SUBSTEPS,
                "stage_steps": STAGE_STEPS,
                "table_top_z_m": TABLE_TOP_Z_M,
                "tray_floor_z_m": TRAY_FLOOR_Z_M,
                "grasp_height_fraction_above_com": GRASP_HEIGHT_FRACTION,
                "robot_mpm_coupling_friction": ROBOT_COUP_FRICTION,
                "scale": 1.0,
            },
            "ik": ik_reports,
            "metrics": metrics,
            "phase_end_metrics": phase_end_metrics,
            "assessment": assessment,
            "video": video,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if scene is not None:
            scene.destroy()
        gs.destroy()
