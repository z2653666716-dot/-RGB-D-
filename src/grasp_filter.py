"""基于 IK 和隔离仿真状态筛选几何抓取候选的可达性与路径风险。 

每个候选要分别求预抓取、抓取、抬升末端位姿，并在独立 MjData 中做关节空间
插值检查。规划状态只复制机械臂/夹爪关节；目标方块位置使用视觉几何估计重设，
不会读取当前仿真数据里的目标真值，也不会改变正式场景状态。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from geometry import PlanarObjectGeometry
from grasp_planner import CandidateGenerationResult, GraspCandidate
from inverse_kinematics import IkResult, solve_fr3_ik


@dataclass(frozen=True)
class CandidateAssessment:
    """一个候选的 IK、路径检查结果及出错阶段。"""

    candidate_id: str
    feasible: bool
    reason: str
    failed_phase: str | None
    ik_results: tuple[IkResult, ...]
    checked_path_samples: int


@dataclass(frozen=True)
class CandidateFilterResult:
    """所有候选的检查记录和可行候选；保留失败项用于后续统计。"""

    success: bool
    reason: str
    feasible_candidates: tuple[GraspCandidate, ...]
    assessments: tuple[CandidateAssessment, ...]


def _joint_state_indices(model: mujoco.MjModel):
    """按名称取得 FR3 七轴与夹爪双指的 qpos 地址，避免混淆执行器编号。"""
    arm = []
    fingers = []
    for number in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"fr3_joint{number}")
        if joint_id < 0:
            raise ValueError(f"模型缺少 FR3 关节 fr3_joint{number}")
        arm.append(int(model.jnt_qposadr[joint_id]))
    for number in range(1, 3):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"gripper_finger_joint{number}")
        if joint_id < 0:
            raise ValueError(f"模型缺少夹爪关节 gripper_finger_joint{number}")
        fingers.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(arm), np.asarray(fingers)


def _make_planning_data(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    geometry: PlanarObjectGeometry,
    arm_addresses: np.ndarray,
    finger_addresses: np.ndarray,
) -> mujoco.MjData:
    """新建规划副本，只复制机器人关节，并以视觉估计设置目标方块位姿。"""
    planning_data = mujoco.MjData(model)
    # 明确只复制 FR3 和夹爪状态；不复制 current_data 中的目标自由关节真值。
    planning_data.qpos[arm_addresses] = current_data.qpos[arm_addresses]
    planning_data.qpos[finger_addresses] = current_data.qpos[finger_addresses]
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      "target_cube_free")
    if cube_joint_id < 0:
        raise ValueError("模型缺少 target_cube_free 自由关节")
    cube_qpos = int(model.jnt_qposadr[cube_joint_id])
    center = np.asarray(geometry.object_center_world_m, dtype=np.float64)
    planning_data.qpos[cube_qpos:cube_qpos + 3] = center
    yaw = geometry.yaw_world_rad if geometry.orientation_confident else 0.0
    yaw = 0.0 if yaw is None else float(yaw)
    planning_data.qpos[cube_qpos + 3:cube_qpos + 7] = [
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0),
    ]
    planning_data.qvel[:] = 0.0
    mujoco.mj_forward(model, planning_data)
    return planning_data


def _check_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    allow_finger_cube_contact: bool,
) -> str | None:
    """把机械臂碰桌、非指尖撞目标等接触转换为明确的风险分类。"""
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  "target_cube_collision")
    table_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        .startswith("table_")
    }
    finger_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_finger"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_finger"),
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geoms = (int(contact.geom1), int(contact.geom2))
        bodies = (int(model.geom_bodyid[geoms[0]]), int(model.geom_bodyid[geoms[1]]))
        body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                           for body in bodies)
        if any(geom in table_geoms for geom in geoms):
            if any(name.startswith(("fr3_", "gripper_")) for name in body_names):
                return "robot_table_collision"
        if cube_geom in geoms:
            cube_side = 0 if geoms[0] == cube_geom else 1
            other_body = bodies[1 - cube_side]
            other_name = body_names[1 - cube_side]
            if (allow_finger_cube_contact and other_body in finger_bodies):
                continue
            if other_name.startswith(("fr3_", "gripper_")):
                return "robot_target_collision"
    return None


def _interpolate_path_segment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_addresses: np.ndarray,
    finger_addresses: np.ndarray,
    start_angles: np.ndarray,
    goal_angles: np.ndarray,
    requested_opening_width_m: float,
    sample_count: int,
    allow_finger_contact: bool,
) -> tuple[str | None, int]:
    """用五次时间曲线采样关节路径，返回首个碰撞分类和实际检查点数。"""
    maximum_finger_position = min(0.04, requested_opening_width_m / 2.0)
    data.qpos[finger_addresses] = maximum_finger_position
    for sample_index in range(1, sample_count + 1):
        t = sample_index / sample_count
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        data.qpos[arm_addresses] = start_angles + blend * (goal_angles - start_angles)
        mujoco.mj_forward(model, data)
        collision = _check_contacts(model, data, allow_finger_contact)
        if collision is not None:
            return collision, sample_index
    return None, sample_count


def filter_candidates_by_ik_and_path(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    geometry: PlanarObjectGeometry,
    generated: CandidateGenerationResult,
    filter_config: dict,
    end_effector_site: str = "attachment_site",
) -> CandidateFilterResult:
    """求候选关键位姿 IK 并检查预抓取/下降/抬升路径，输出可行子集和原因。"""
    if not generated.success or not generated.candidates:
        return CandidateFilterResult(False, "no_generated_candidates", (), ())
    if not geometry.valid or geometry.object_center_world_m is None:
        return CandidateFilterResult(False, "geometry_invalid", (), ())
    sample_count = int(filter_config["path_samples_per_segment"])
    lift_distance = float(filter_config["lift_distance_m"])
    if sample_count <= 0 or lift_distance < 0:
        raise ValueError("路径采样数须为正，抬升距离不能为负")

    arm_addresses, finger_addresses = _joint_state_indices(model)
    current_angles = current_data.qpos[arm_addresses].copy()
    lift_allowed_contact = filter_config[
        "allowed_fingertip_object_contact_during_descent_and_lift"]
    feasible: list[GraspCandidate] = []
    assessments: list[CandidateAssessment] = []

    for candidate in generated.candidates:
        planning_data = _make_planning_data(
            model, current_data, geometry, arm_addresses, finger_addresses,
        )
        lift_position = np.asarray(candidate.tool_site_position_world_m, dtype=np.float64)
        lift_position[2] += lift_distance
        targets = (
            ("pregrasp", np.asarray(candidate.pregrasp_site_position_world_m)),
            ("grasp", np.asarray(candidate.tool_site_position_world_m)),
            ("lift", lift_position),
        )
        ik_results: list[IkResult] = []
        joint_solutions: list[np.ndarray] = []
        ik_failure: CandidateAssessment | None = None
        for phase, target_position in targets:
            ik = solve_fr3_ik(
                model, planning_data, target_position,
                np.asarray(candidate.tool_orientation_wxyz),
                end_effector_site=end_effector_site,
            )
            ik_results.append(ik)
            if not ik.success or ik.joint_positions_rad is None:
                ik_failure = CandidateAssessment(
                    candidate.candidate_id, False, f"ik_failed:{ik.reason}",
                    phase, tuple(ik_results), 0,
                )
                break
            solution = np.asarray(ik.joint_positions_rad, dtype=np.float64)
            joint_solutions.append(solution)
            planning_data.qpos[arm_addresses] = solution
            mujoco.mj_forward(model, planning_data)
        if ik_failure is not None:
            assessments.append(ik_failure)
            continue

        # 每条路径都从真实机器人当前关节角起步，但在规划副本中执行前向运动学。
        path_data = _make_planning_data(
            model, current_data, geometry, arm_addresses, finger_addresses,
        )
        segments = (
            ("approach", current_angles, joint_solutions[0], False),
            ("descent", joint_solutions[0], joint_solutions[1], lift_allowed_contact),
            ("lift", joint_solutions[1], joint_solutions[2], lift_allowed_contact),
        )
        checked_samples = 0
        failure_reason = None
        failed_phase = None
        for phase, start, goal, allow_contact in segments:
            collision, checked = _interpolate_path_segment(
                model, path_data, arm_addresses, finger_addresses,
                start, goal, candidate.requested_opening_width_m,
                sample_count, allow_contact,
            )
            checked_samples += checked
            if collision is not None:
                failure_reason, failed_phase = collision, phase
                break

        if failure_reason is None:
            feasible.append(candidate)
            assessments.append(CandidateAssessment(
                candidate.candidate_id, True, "ik_and_path_clear", None,
                tuple(ik_results), checked_samples,
            ))
        else:
            assessments.append(CandidateAssessment(
                candidate.candidate_id, False, failure_reason, failed_phase,
                tuple(ik_results), checked_samples,
            ))

    if not feasible:
        return CandidateFilterResult(False, "no_candidate_passed_ik_and_path", (),
                                     tuple(assessments))
    return CandidateFilterResult(True, "feasible_candidates_found", tuple(feasible),
                                 tuple(assessments))
