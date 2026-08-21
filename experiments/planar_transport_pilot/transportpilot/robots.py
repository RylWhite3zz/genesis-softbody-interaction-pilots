"""Robot adapters and continuous Cartesian-path IK for Piper and Franka."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .paths import PIPER_URDF
from .protocol import (
    FRANKA_HOME_ARM_Q,
    PIPER_HOME_ARM_Q,
    ROBOT_COUP_FRICTION,
    ROBOT_FRANKA,
    ROBOT_PIPER,
)


def numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def flat(tensor: Any) -> np.ndarray:
    return numpy(tensor).reshape(-1).astype(np.float64, copy=False)


@dataclass
class RobotHandle:
    robot_type: str
    entity: Any
    tcp: Any
    arm_dofs: list[int]
    finger_dofs: list[int]
    home_arm_q: np.ndarray
    open_width_m: float
    closed_width_m: float
    base_tool_quat_wxyz: np.ndarray
    base_grip_offset_m: np.ndarray
    tool_vertical_offsets_m: tuple[float, float]
    home_grip_center_m: np.ndarray

    def finger_target(self, width_m: float) -> np.ndarray:
        if self.robot_type == ROBOT_PIPER:
            return np.asarray((width_m / 2.0, -width_m / 2.0), dtype=np.float64)
        return np.asarray((width_m / 2.0, width_m / 2.0), dtype=np.float64)


def robot_material(gs: Any) -> Any:
    return gs.materials.Rigid(
        needs_coup=True,
        friction=1.0,
        coup_friction=ROBOT_COUP_FRICTION,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.003,
        sdf_min_res=24,
        sdf_max_res=96,
        gravity_compensation=1.0,
    )


def add_robot(gs: Any, scene: Any, robot_type: str) -> Any:
    if robot_type == ROBOT_PIPER:
        if not PIPER_URDF.is_file():
            raise FileNotFoundError(PIPER_URDF)
        morph = gs.morphs.URDF(
            file=str(PIPER_URDF),
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=("tcp_link",),
            requires_jac_and_IK=True,
            collision=True,
            convexify=True,
            decimate=True,
            recompute_inertia=False,
            default_armature=0.005,
        )
    elif robot_type == ROBOT_FRANKA:
        # This is the same bundled Panda URDF and retained TCP used by DLO-Lab.
        morph = gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=("panda_grasptarget",),
            requires_jac_and_IK=True,
            collision=True,
            convexify=True,
            decimate=True,
            recompute_inertia=False,
            default_armature=0.005,
        )
    else:
        raise ValueError(f"unknown robot: {robot_type}")
    return scene.add_entity(
        morph=morph,
        material=robot_material(gs),
        surface=gs.surfaces.Smooth(),
        name=f"fixed_{robot_type}",
    )


def _finger_vertices(robot: Any, link_names: tuple[str, str]) -> np.ndarray:
    vertices: list[np.ndarray] = []
    for link_name in link_names:
        link = robot.get_link(link_name)
        for geom in link.geoms:
            verts = numpy(geom.get_verts()).reshape(-1, 3).astype(np.float64, copy=False)
            vertices.append(verts)
    if not vertices:
        raise RuntimeError("robot has no finger collision vertices")
    return np.concatenate(vertices, axis=0)


def _piper_far_pad_centers(robot: Any) -> np.ndarray:
    centers: list[np.ndarray] = []
    for link_name in ("gripper_link1", "gripper_link2"):
        link = robot.get_link(link_name)
        if len(link.geoms) != 2:
            raise RuntimeError(f"expected two pad geoms on {link_name}, found {len(link.geoms)}")
        centers.append(flat(link.geoms[1].get_pos(relative=False)))
    return np.asarray(centers, dtype=np.float64)


def configure_robot(robot: Any, robot_type: str) -> RobotHandle:
    if robot_type == ROBOT_PIPER:
        arm_dofs = [int(robot.get_joint(f"joint{i}").dofs_idx_local[0]) for i in range(1, 7)]
        finger_dofs = [
            int(robot.get_joint("gripper_joint1").dofs_idx_local[0]),
            int(robot.get_joint("gripper_joint2").dofs_idx_local[0]),
        ]
        home_arm_q = PIPER_HOME_ARM_Q.copy()
        open_width, closed_width = 0.066, 0.012
        tcp = robot.get_link("tcp_link")
        robot.set_dofs_kp([2000, 2000, 1600, 800, 400, 400], arm_dofs)
        robot.set_dofs_kv([100, 100, 80, 40, 20, 20], arm_dofs)
        robot.set_dofs_force_range([-100] * 6, [100] * 6, arm_dofs)
        robot.set_dofs_kp([40, 40], finger_dofs)
        robot.set_dofs_kv([5, 5], finger_dofs)
        robot.set_dofs_force_range([-10, -10], [10, 10], finger_dofs)
        finger_target = np.asarray((open_width / 2.0, -open_width / 2.0))
    elif robot_type == ROBOT_FRANKA:
        arm_dofs = [int(robot.get_joint(f"panda_joint{i}").dofs_idx_local[0]) for i in range(1, 8)]
        finger_dofs = [
            int(robot.get_joint("panda_finger_joint1").dofs_idx_local[0]),
            int(robot.get_joint("panda_finger_joint2").dofs_idx_local[0]),
        ]
        home_arm_q = FRANKA_HOME_ARM_Q.copy()
        open_width, closed_width = 0.078, 0.012
        tcp = robot.get_link("panda_grasptarget")
        robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], arm_dofs)
        robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], arm_dofs)
        robot.set_dofs_force_range([-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12], arm_dofs)
        robot.set_dofs_kp([80, 80], finger_dofs)
        robot.set_dofs_kv([20, 20], finger_dofs)
        robot.set_dofs_force_range([-30, -30], [30, 30], finger_dofs)
        finger_target = np.asarray((open_width / 2.0, open_width / 2.0))
    else:
        raise ValueError(f"unknown robot: {robot_type}")

    robot.set_dofs_position(home_arm_q, arm_dofs, zero_velocity=True)
    robot.set_dofs_position(finger_target, finger_dofs, zero_velocity=True)
    robot.control_dofs_position(home_arm_q, arm_dofs)
    robot.control_dofs_position(finger_target, finger_dofs)

    tcp_pos = flat(tcp.get_pos(relative=False))
    tcp_quat = flat(tcp.get_quat(relative=False))
    if robot_type == ROBOT_PIPER:
        home_grip = _piper_far_pad_centers(robot).mean(axis=0)
        link_names = ("gripper_link1", "gripper_link2")
    else:
        home_grip = tcp_pos.copy()
        link_names = ("panda_leftfinger", "panda_rightfinger")
    vertices = _finger_vertices(robot, link_names)
    grip_offset = home_grip - tcp_pos
    vertical_offsets = (
        float(np.min(vertices[:, 2]) - home_grip[2]),
        float(np.max(vertices[:, 2]) - home_grip[2]),
    )
    return RobotHandle(
        robot_type=robot_type,
        entity=robot,
        tcp=tcp,
        arm_dofs=arm_dofs,
        finger_dofs=finger_dofs,
        home_arm_q=home_arm_q,
        open_width_m=open_width,
        closed_width_m=closed_width,
        base_tool_quat_wxyz=tcp_quat,
        base_grip_offset_m=grip_offset,
        tool_vertical_offsets_m=vertical_offsets,
        home_grip_center_m=home_grip,
    )


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


def yawed_tool(handle: RobotHandle, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    qz = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _quat_multiply(qz, handle.base_tool_quat_wxyz), rotation @ handle.base_grip_offset_m


def _solve_waypoint(
    handle: RobotHandle,
    grip_center_m: np.ndarray,
    quat_wxyz: np.ndarray,
    grip_offset_m: np.ndarray,
    init_arm_q: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_tcp = grip_center_m - grip_offset_m
    init_q = flat(handle.entity.get_qpos()).copy()
    init_q[handle.arm_dofs] = init_arm_q
    qpos, error = handle.entity.inverse_kinematics(
        link=handle.tcp,
        pos=target_tcp,
        quat=quat_wxyz,
        init_qpos=init_q,
        dofs_idx_local=handle.arm_dofs,
        return_error=True,
        max_samples=32,
        max_solver_iters=180,
        damping=0.03,
        max_step_size=0.08,
        pos_tol=7.5e-4,
        rot_tol=7.5e-3,
    )
    q_array = flat(qpos)
    arm_q = q_array if len(q_array) == len(handle.arm_dofs) else q_array[handle.arm_dofs]
    error6 = flat(error)
    return arm_q, {
        "target_grip_center_m": grip_center_m.tolist(),
        "target_tcp_m": target_tcp.tolist(),
        "arm_q_rad": arm_q.tolist(),
        "position_error_norm_m": float(np.linalg.norm(error6[:3])),
        "rotation_error_norm_rad": float(np.linalg.norm(error6[3:])),
    }


def solve_tool_path(handle: RobotHandle, tool_path: dict[str, Any]) -> dict[str, Any]:
    named_centers: list[tuple[str, np.ndarray]] = [
        ("approach", np.asarray(tool_path["approach_grip_center_m"], dtype=np.float64)),
        ("contact", np.asarray(tool_path["contact_grip_center_m"], dtype=np.float64)),
    ]
    transport = np.asarray(tool_path["transport_grip_centers_m"], dtype=np.float64)
    named_centers.extend((f"transport_{index:02d}", center) for index, center in enumerate(transport[1:], 1))
    named_centers.append(("retreat", np.asarray(tool_path["retreat_grip_center_m"], dtype=np.float64)))

    yaw_trials: list[dict[str, Any]] = []
    for yaw_deg in tool_path["yaw_candidates_deg"]:
        quat, grip_offset = yawed_tool(handle, float(yaw_deg))
        init_arm = handle.home_arm_q.copy()
        solutions: dict[str, np.ndarray] = {}
        reports: dict[str, dict[str, Any]] = {}
        for name, center in named_centers:
            solution, report = _solve_waypoint(handle, center, quat, grip_offset, init_arm)
            solutions[name] = solution
            reports[name] = report
            init_arm = solution
        max_position_error = max(report["position_error_norm_m"] for report in reports.values())
        max_rotation_error = max(report["rotation_error_norm_rad"] for report in reports.values())
        valid = bool(max_position_error <= 0.008 and max_rotation_error <= 0.050)
        score = max_position_error / 0.008 + max_rotation_error / 0.050
        yaw_trials.append(
            {
                "yaw_deg": float(yaw_deg),
                "valid": valid,
                "score": float(score),
                "max_position_error_m": float(max_position_error),
                "max_rotation_error_rad": float(max_rotation_error),
                "quat_wxyz": quat,
                "grip_offset_m": grip_offset,
                "solutions": solutions,
                "reports": reports,
            }
        )
    selected = min(yaw_trials, key=lambda trial: (not trial["valid"], trial["score"]))
    return {
        "valid": selected["valid"],
        "selected_yaw_deg": selected["yaw_deg"],
        "selected_score": selected["score"],
        "selected_max_position_error_m": selected["max_position_error_m"],
        "selected_max_rotation_error_rad": selected["max_rotation_error_rad"],
        "quat_wxyz": selected["quat_wxyz"],
        "grip_offset_m": selected["grip_offset_m"],
        "solutions": selected["solutions"],
        "reports": selected["reports"],
        "yaw_trials": [
            {
                key: value
                for key, value in trial.items()
                if key not in {"quat_wxyz", "grip_offset_m", "solutions", "reports"}
            }
            for trial in yaw_trials
        ],
    }


def serial_motion_plan(motion: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": motion["valid"],
        "selected_yaw_deg": motion["selected_yaw_deg"],
        "selected_score": motion["selected_score"],
        "selected_max_position_error_m": motion["selected_max_position_error_m"],
        "selected_max_rotation_error_rad": motion["selected_max_rotation_error_rad"],
        "quat_wxyz": motion["quat_wxyz"].tolist(),
        "grip_offset_m": motion["grip_offset_m"].tolist(),
        "reports": motion["reports"],
        "yaw_trials": motion["yaw_trials"],
    }
