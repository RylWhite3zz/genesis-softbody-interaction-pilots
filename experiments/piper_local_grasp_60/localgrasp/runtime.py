"""Genesis 1.1.2 local edge/corner pinch trials with probe-first execution."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetSpec
from .paths import (
    ROBOT_URDF,
    configure_writable_runtime,
    isolate_genesis_particle_sampler,
    require_within_project,
)
from .planner import generate_local_grasps
from .protocol import (
    FRAME_PERIOD_S,
    HOME_ARM_Q,
    HOME_GRIP_CENTER_M,
    LIFT_HEIGHT_M,
    OPEN_WIDTH_M,
    PICK_CENTER_XY_M,
    ROBOT_COUP_FRICTION,
    SPAWN_GAP_M,
    TABLE_CENTER_M,
    TABLE_SIZE_M,
    TABLE_TOP_Z_M,
    TRAY_CENTER_XY_M,
    TRAY_FLOOR_Z_M,
    VIDEO_FPS,
    assess_probe,
    assess_success,
    smoothstep01,
    stage_steps,
    tray_boxes,
)


IMPLEMENTATION_REVISION = "local_edge_corner_v6_deferred_place_ik"
FOLD_IMPLEMENTATION_REVISION = "active_wrinkle_preload_v1"


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


def _particle_snapshot(soft: Any, *, include_positions: bool = False) -> dict[str, Any]:
    state = soft.get_state()
    positions = _numpy(state.pos)[0].astype(np.float64)
    velocities = _numpy(state.vel)[0].astype(np.float64)
    active = _numpy(state.active)[0].astype(bool)
    soft._queried_states.discard(state)
    positions = positions[active]
    velocities = velocities[active]
    if positions.size == 0:
        raise RuntimeError("MPM entity has no active particles")
    speeds = np.linalg.norm(velocities, axis=1)
    result: dict[str, Any] = {
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


def _asset_color(asset: AssetSpec) -> tuple[float, float, float, float]:
    return (0.83, 0.24, 0.16, 1.0) if asset.geometry_family == "solid" else (0.13, 0.58, 0.68, 1.0)


def _mesh_spawn(asset: AssetSpec) -> tuple[float, float, float]:
    import trimesh

    mesh = trimesh.load(asset.collision_mesh, force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
    return (
        float(PICK_CENTER_XY_M[0] - center_xy[0]),
        float(PICK_CENTER_XY_M[1] - center_xy[1]),
        float(TABLE_TOP_Z_M + SPAWN_GAP_M - bounds[0, 2]),
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


def _build_scene(gs: Any, asset: AssetSpec) -> tuple[Any, Any, Any, Any, tuple[float, float, float]]:
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
            lower_bound=(0.04, -0.48, 0.015),
            upper_bound=(0.90, 0.48, 0.72),
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
    for index, box in enumerate(tray_boxes()):
        scene.add_entity(
            morph=gs.morphs.Box(pos=box["pos"], size=box["size"], fixed=True),
            material=_fixture_material(gs),
            surface=gs.surfaces.Rough(color=(0.18, 0.40, 0.56, 1.0)),
            name=f"tray_{index}",
        )
    spawn_pos = _mesh_spawn(asset)
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
        pos=(1.08, -0.92, 0.70),
        lookat=(0.46, -0.01, 0.20),
        fov=50,
        GUI=False,
        near=0.05,
        far=2.2,
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


def _pad_centers(robot: Any, geom_index: int | None = None) -> np.ndarray:
    centers: list[np.ndarray] = []
    for link_name in ("gripper_link1", "gripper_link2"):
        link = robot.get_link(link_name)
        if len(link.geoms) != 2:
            raise RuntimeError(f"expected two MPM pad geoms on {link_name}, found {len(link.geoms)}")
        geoms = link.geoms if geom_index is None else (link.geoms[geom_index],)
        centers.extend(_flat(geom.get_pos(relative=False)) for geom in geoms)
    return np.asarray(centers, dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _yawed_tool(home_quat: np.ndarray, base_grip_offset: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    qz = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    return _quat_multiply(qz, home_quat), rotation @ base_grip_offset


def _solve_waypoint(
    robot: Any,
    tcp: Any,
    arm_dofs: list[int],
    grip_center: np.ndarray,
    grip_offset: np.ndarray,
    quat: np.ndarray,
    init_arm_q: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_tcp = grip_center - grip_offset
    init_q = _flat(robot.get_qpos()).copy()
    init_q[:6] = init_arm_q
    qpos, error = robot.inverse_kinematics(
        link=tcp,
        pos=target_tcp,
        quat=quat,
        init_qpos=init_q,
        dofs_idx_local=arm_dofs,
        return_error=True,
        max_samples=24,
        max_solver_iters=150,
        damping=0.03,
        max_step_size=0.08,
        pos_tol=7.5e-4,
        rot_tol=7.5e-3,
    )
    arm_q = _flat(qpos)[:6]
    error6 = _flat(error)
    return arm_q, {
        "target_grip_center_m": grip_center.tolist(),
        "arm_q_rad": arm_q.tolist(),
        "position_error_norm_m": float(np.linalg.norm(error6[:3])),
        "rotation_error_norm_rad": float(np.linalg.norm(error6[3:])),
    }


def _candidate_grasp_waypoints(
    robot: Any,
    tcp: Any,
    arm_dofs: list[int],
    candidate: dict[str, Any],
    reference: dict[str, Any],
    home_quat: np.ndarray,
    base_grip_offset: np.ndarray,
    preload_distance_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    grasp = np.asarray(candidate["grasp_center_m"], dtype=np.float64)
    execution_grasp = grasp.copy()
    if preload_distance_m > 0.0:
        tangent = np.asarray(candidate["tangent_axis_unit_xy"], dtype=np.float64)
        to_com = np.asarray(reference["com_m"], dtype=np.float64)[:2] - grasp[:2]
        direction = 1.0 if float(np.dot(to_com, tangent)) >= 0.0 else -1.0
        execution_grasp[:2] += preload_distance_m * direction * tangent
    grasp_quat, grasp_offset = _yawed_tool(
        home_quat, base_grip_offset, candidate["wrist_yaw_deg"]
    )
    centers = {
        "approach": grasp + np.asarray((0.0, 0.0, 0.075)),
        "descend": grasp,
        "preload": execution_grasp,
        "probe_lift": execution_grasp + np.asarray((0.0, 0.0, LIFT_HEIGHT_M)),
    }
    reports: dict[str, Any] = {}
    solutions: dict[str, np.ndarray] = {}
    init_q = HOME_ARM_Q.copy()
    names = ["approach", "descend"]
    if preload_distance_m > 0.0:
        names.append("preload")
    names.append("probe_lift")
    for name in names:
        solution, report = _solve_waypoint(
            robot, tcp, arm_dofs, centers[name], grasp_offset, grasp_quat, init_q
        )
        solutions[name] = solution
        reports[name] = report
        init_q = solution
    return solutions, reports, grasp_quat, grasp_offset, execution_grasp


def _candidate_place_waypoints(
    robot: Any,
    tcp: Any,
    arm_dofs: list[int],
    candidate: dict[str, Any],
    reference: dict[str, Any],
    home_quat: np.ndarray,
    base_grip_offset: np.ndarray,
    lifted_arm_q: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    grasp = np.asarray(candidate["grasp_center_m"], dtype=np.float64)
    reference_com = np.asarray(reference["com_m"], dtype=np.float64)
    transfer_delta = np.asarray(
        (TRAY_CENTER_XY_M[0] - reference_com[0], TRAY_CENTER_XY_M[1] - reference_com[1], 0.0)
    )
    release_z = TRAY_FLOOR_Z_M + max(0.025, grasp[2] - float(reference["aabb_min_m"][2]))
    centers = {
        "transfer": grasp + transfer_delta + np.asarray((0.0, 0.0, LIFT_HEIGHT_M)),
        "lower_into_tray": np.asarray((grasp + transfer_delta)[0:2].tolist() + [release_z]),
        "retreat": np.asarray((grasp + transfer_delta)[0:2].tolist() + [release_z + 0.16]),
    }
    place_trials: list[dict[str, Any]] = []
    place_yaws = list(dict.fromkeys((0.0, -90.0, -45.0, 45.0, float(candidate["wrist_yaw_deg"]))))
    for place_yaw in place_yaws:
        place_quat, place_offset = _yawed_tool(home_quat, base_grip_offset, place_yaw)
        place_solutions: dict[str, np.ndarray] = {}
        place_reports: dict[str, Any] = {}
        place_init_q = lifted_arm_q.copy()
        for name in ("transfer", "lower_into_tray", "retreat"):
            solution, report = _solve_waypoint(
                robot, tcp, arm_dofs, centers[name], place_offset, place_quat, place_init_q
            )
            place_solutions[name] = solution
            place_reports[name] = report
            place_init_q = solution
        score = max(
            report["position_error_norm_m"] / 0.012
            + report["rotation_error_norm_rad"] / 0.060
            for report in place_reports.values()
        )
        place_trials.append(
            {
                "wrist_yaw_deg": place_yaw,
                "score": float(score),
                "solutions": place_solutions,
                "reports": place_reports,
                "quat": place_quat,
                "offset": place_offset,
            }
        )
    selected_place = min(place_trials, key=lambda trial: trial["score"])
    plan = {
        "grasp_wrist_yaw_deg": float(candidate["wrist_yaw_deg"]),
        "place_wrist_yaw_deg": float(selected_place["wrist_yaw_deg"]),
        "place_ik_candidate_scores": [
            {"wrist_yaw_deg": trial["wrist_yaw_deg"], "score": trial["score"]}
            for trial in place_trials
        ],
    }
    return (
        selected_place["solutions"],
        selected_place["reports"],
        selected_place["quat"],
        selected_place["offset"],
        plan,
    )


def run_attempt(
    asset: AssetSpec,
    output_video: Path,
    *,
    max_candidates: int = 4,
    preload_distance_m: float = 0.0,
) -> dict[str, Any]:
    if preload_distance_m < 0.0 or preload_distance_m > 0.050:
        raise ValueError("preload_distance_m must be in [0, 0.050]")
    if not ROBOT_URDF.is_file():
        raise FileNotFoundError(ROBOT_URDF)
    output_video = require_within_project(output_video)
    run_cache = configure_writable_runtime(asset.asset_id)
    import genesis as gs
    import torch

    scene = None
    started = time.perf_counter()
    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
        write_isolation = isolate_genesis_particle_sampler(run_cache)
        if gs.backend != gs.cuda:
            raise RuntimeError(f"GPU requested, but Genesis selected {gs.backend}")
        torch.cuda.reset_peak_memory_stats()
        build_started = time.perf_counter()
        scene, robot, soft, camera, spawn_pos = _build_scene(gs, asset)
        build_seconds = time.perf_counter() - build_started
        arm_dofs, finger_dofs, tcp = _configure_robot(robot)
        home_tcp = _flat(tcp.get_pos(relative=False))
        home_quat = _flat(tcp.get_quat(relative=False))
        home_pad_centers = _pad_centers(robot)
        home_far_pad_centers = _pad_centers(robot, geom_index=1)
        calibrated_home_grip_center = home_far_pad_centers.mean(axis=0)
        base_grip_offset = calibrated_home_grip_center - home_tcp
        steps_by_stage = stage_steps(asset.source_dt_s)
        for _ in range(steps_by_stage["settle_clear"]):
            robot.control_dofs_position(HOME_ARM_Q, arm_dofs)
            robot.control_dofs_position([OPEN_WIDTH_M / 2.0, -OPEN_WIDTH_M / 2.0], finger_dofs)
            scene.step(update_visualizer=False)
        reference = _particle_snapshot(soft, include_positions=True)
        settled_state = scene.get_state()
        scene.reset(settled_state)
        candidates = generate_local_grasps(
            reference["positions_m"], asset.particle_size_m, limit=max_candidates
        )
        base_result: dict[str, Any] = {
            "schema_version": 2,
            "implementation_revision": (
                FOLD_IMPLEMENTATION_REVISION if preload_distance_m > 0.0 else IMPLEMENTATION_REVISION
            ),
            "asset": asset.to_dict(),
            "scale": 1.0,
            "spawn_position_m": list(spawn_pos),
            "settled": _serial_snapshot(reference),
            "candidate_count": len(candidates),
            "candidate_trials": [],
            "runtime": {
                "backend": str(gs.backend),
                "genesis_module": str(Path(gs.__file__).resolve()),
                "genesis_version": getattr(gs, "__version__", "unknown"),
                "cuda_device": torch.cuda.get_device_name(),
                "write_isolation": write_isolation,
                "pad_center_calibration": {
                    "four_home_pad_centers_m": home_pad_centers.tolist(),
                    "two_far_home_pad_centers_m": home_far_pad_centers.tolist(),
                    "calibrated_home_grip_center_m": calibrated_home_grip_center.tolist(),
                    "legacy_approximate_home_grip_center_m": HOME_GRIP_CENTER_M.tolist(),
                    "legacy_calibration_error_m": float(
                        np.linalg.norm(calibrated_home_grip_center - HOME_GRIP_CENTER_M)
                    ),
                },
            },
            "protocol": {
                "dt_s": asset.source_dt_s,
                "substeps": asset.source_substeps,
                "stage_steps": steps_by_stage,
                "local_grasp_only": True,
                "original_scale_only": True,
                "robot_mpm_coupling_friction": ROBOT_COUP_FRICTION,
                "trajectory_recording": False,
                "failed_video_retention": False,
                "active_wrinkle_preload_distance_m": preload_distance_m,
            },
            "video": None,
        }
        if not candidates:
            return {
                **base_result,
                "status": "no_local_candidate",
                "reason": "settled MPM particle cloud has no supported 30x30 mm edge/corner patch within the jaw opening",
                "elapsed_seconds": time.perf_counter() - started,
            }

        successful_payload: dict[str, Any] | None = None
        for candidate in candidates:
            scene.reset()
            trial_started = time.perf_counter()
            solutions, ik_reports, grasp_quat, grasp_offset, execution_grasp = _candidate_grasp_waypoints(
                robot,
                tcp,
                arm_dofs,
                candidate,
                reference,
                home_quat,
                base_grip_offset,
                preload_distance_m,
            )
            grasp_names = ["approach", "descend"]
            if preload_distance_m > 0.0:
                grasp_names.append("preload")
            grasp_names.append("probe_lift")
            grasp_ik_valid = all(
                ik_reports[name]["position_error_norm_m"] <= 0.006
                and ik_reports[name]["rotation_error_norm_rad"] <= 0.030
                for name in grasp_names
            )
            trial: dict[str, Any] = {
                "candidate": candidate,
                "motion_plan": {
                    "grasp_wrist_yaw_deg": float(candidate["wrist_yaw_deg"]),
                    "active_wrinkle_preload_distance_m": preload_distance_m,
                    "execution_grasp_center_m": execution_grasp.tolist(),
                },
                "ik": ik_reports,
                "grasp_ik_valid": grasp_ik_valid,
                "place_ik_valid": None,
                "phase_end": [],
                "probe": None,
                "assessment": None,
                "elapsed_seconds": None,
            }
            base_result["candidate_trials"].append(trial)
            if not grasp_ik_valid:
                trial["status"] = "failed_grasp_ik"
                trial["elapsed_seconds"] = time.perf_counter() - trial_started
                continue

            frames: list[np.ndarray] = [_render_rgb(camera)]
            frame_stds = [float(np.std(frames[-1].astype(np.float32)))]
            current_q = HOME_ARM_Q.copy()
            current_width = OPEN_WIDTH_M
            sim_time = 0.0
            next_frame_time = FRAME_PERIOD_S
            max_lift_z = float(reference["com_m"][2])
            finite_state = bool(reference["finite"])
            count_constant = True

            def record_phase(name: str) -> dict[str, Any]:
                nonlocal max_lift_z, finite_state, count_constant
                snapshot = _particle_snapshot(soft)
                max_lift_z = max(max_lift_z, float(snapshot["com_m"][2]))
                finite_state = finite_state and bool(snapshot["finite"])
                count_constant = count_constant and snapshot["active_count"] == reference["active_count"]
                finger_q = _flat(robot.get_dofs_position(finger_dofs))
                pads = _pad_centers(robot)
                far_pads = _pad_centers(robot, geom_index=1)
                actual_close = far_pads[1] - far_pads[0]
                actual_close /= np.linalg.norm(actual_close)
                planned_close = np.asarray(candidate["close_axis_unit_xy"] + [0.0], dtype=np.float64)
                serial = {
                    "stage": name,
                    **_serial_snapshot(snapshot),
                    "actual_jaw_width_m": float(finger_q[0] - finger_q[1]),
                    "actual_pad_centroid_m": pads.mean(axis=0).tolist(),
                    "actual_far_pad_centroid_m": far_pads.mean(axis=0).tolist(),
                    "actual_close_axis_unit_xyz": actual_close.tolist(),
                    "planned_close_axis_alignment_abs": float(abs(np.dot(actual_close, planned_close))),
                    "target_grip_center_error_m": float(
                        np.linalg.norm(far_pads.mean(axis=0) - ik_reports[name]["target_grip_center_m"])
                    )
                    if name in ik_reports
                    else None,
                }
                trial["phase_end"].append(serial)
                return snapshot

            def stage(name: str, target_q: np.ndarray, target_width: float) -> dict[str, Any]:
                nonlocal current_q, current_width, sim_time, next_frame_time
                source_q = current_q.copy()
                source_width = float(current_width)
                count = steps_by_stage[name]
                for index in range(count):
                    alpha = smoothstep01((index + 1) / count)
                    arm_q = (1.0 - alpha) * source_q + alpha * target_q
                    width = (1.0 - alpha) * source_width + alpha * target_width
                    robot.control_dofs_position(arm_q, arm_dofs)
                    robot.control_dofs_position([width / 2.0, -width / 2.0], finger_dofs)
                    scene.step(update_visualizer=False)
                    sim_time += asset.source_dt_s
                    if sim_time + 1.0e-12 >= next_frame_time:
                        frame = _render_rgb(camera)
                        frames.append(frame)
                        frame_stds.append(float(np.std(frame.astype(np.float32))))
                        next_frame_time += FRAME_PERIOD_S
                current_q = target_q.copy()
                current_width = float(target_width)
                return record_phase(name)

            close_width = float(candidate["commanded_close_width_m"])
            stage("approach", solutions["approach"], OPEN_WIDTH_M)
            stage("descend", solutions["descend"], OPEN_WIDTH_M)
            close_q_name = "descend"
            if preload_distance_m > 0.0:
                stage("preload", solutions["preload"], OPEN_WIDTH_M)
                close_q_name = "preload"
            stage("close", solutions[close_q_name], close_width)
            stage("grip_hold", solutions[close_q_name], close_width)
            stage("probe_lift", solutions["probe_lift"], close_width)
            probe_snapshot = stage("probe_hold", solutions["probe_lift"], close_width)
            probe = assess_probe(reference, probe_snapshot)
            trial["probe"] = probe
            if not probe["passed"]:
                trial["status"] = "failed_probe"
                trial["elapsed_seconds"] = time.perf_counter() - trial_started
                continue

            place_candidate = dict(candidate)
            place_candidate["grasp_center_m"] = execution_grasp.tolist()
            place_solutions, place_reports, place_quat, place_offset, place_plan = _candidate_place_waypoints(
                robot,
                tcp,
                arm_dofs,
                place_candidate,
                reference,
                home_quat,
                base_grip_offset,
                solutions["probe_lift"],
            )
            solutions.update(place_solutions)
            ik_reports.update(place_reports)
            trial["motion_plan"].update(place_plan)
            place_ik_valid = all(
                ik_reports[name]["position_error_norm_m"] <= 0.012
                and ik_reports[name]["rotation_error_norm_rad"] <= 0.060
                for name in ("transfer", "lower_into_tray", "retreat")
            )
            trial["place_ik_valid"] = place_ik_valid
            if not place_ik_valid:
                trial["status"] = "failed_place_ik_after_probe"
                trial["elapsed_seconds"] = time.perf_counter() - trial_started
                continue

            transfer = stage("transfer", solutions["transfer"], close_width)
            stage("lower_into_tray", solutions["lower_into_tray"], close_width)
            stage("open", solutions["lower_into_tray"], OPEN_WIDTH_M)
            stage("retreat", solutions["retreat"], OPEN_WIDTH_M)
            final = stage("settle_in_tray", solutions["retreat"], OPEN_WIDTH_M)
            final_tcp = _flat(tcp.get_pos(relative=False))
            final_grip = final_tcp + place_offset
            final_aabb_min = np.asarray(final["aabb_min_m"], dtype=np.float64)
            final_aabb_max = np.asarray(final["aabb_max_m"], dtype=np.float64)
            grip_to_aabb = np.maximum(np.maximum(final_aabb_min - final_grip, final_grip - final_aabb_max), 0.0)
            metrics = {
                "finite_state": finite_state,
                "active_particle_count_constant": count_constant,
                "probe_lift_passed": probe["passed"],
                "max_lift_delta_z_m": float(max_lift_z - reference["com_m"][2]),
                "transfer_end_com_delta_z_m": float(transfer["com_m"][2] - reference["com_m"][2]),
                "transfer_end_aabb_min_z_m": float(transfer["aabb_min_m"][2]),
                "final_com_m": final["com_m"].tolist(),
                "final_aabb_min_m": final["aabb_min_m"].tolist(),
                "final_aabb_max_m": final["aabb_max_m"].tolist(),
                "final_speed_rms_m_s": final["speed_rms_m_s"],
                "final_speed_max_m_s": final["speed_max_m_s"],
                "final_grip_center_m": final_grip.tolist(),
                "final_grip_to_object_aabb_distance_m": float(np.linalg.norm(grip_to_aabb)),
                "frame_count": len(frames),
                "minimum_frame_std": float(min(frame_stds)),
                "median_frame_std": float(np.median(frame_stds)),
                "simulated_time_s": sim_time,
            }
            assessment = assess_success(metrics)
            trial["metrics"] = metrics
            trial["assessment"] = assessment
            trial["status"] = assessment["status"]
            trial["elapsed_seconds"] = time.perf_counter() - trial_started
            if assessment["status"] == "success":
                successful_payload = {
                    "candidate": candidate,
                    "metrics": metrics,
                    "assessment": assessment,
                    "grasp_tool_quat_wxyz": grasp_quat.tolist(),
                    "place_tool_quat_wxyz": place_quat.tolist(),
                    "frames": frames,
                }
                break

        if successful_payload is not None:
            video = _encode_video(successful_payload.pop("frames"), output_video)
            return {
                **base_result,
                "status": "success",
                "reason": "local edge/corner pinch passed lift, transfer, tray release, containment, and stability gates",
                "successful_trial": successful_payload,
                "video": video,
                "build_seconds": build_seconds,
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        statuses = [trial["status"] for trial in base_result["candidate_trials"]]
        status = "failed_probe_all" if "failed_probe" in statuses and "failed_gates" not in statuses else "failed_place_all"
        if all(value == "failed_grasp_ik" for value in statuses):
            status = "failed_grasp_ik_all"
        return {
            **base_result,
            "status": status,
            "reason": f"no candidate completed pick-place; candidate terminal statuses: {statuses}",
            "build_seconds": build_seconds,
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if scene is not None:
            scene.destroy()
        gs.destroy()
