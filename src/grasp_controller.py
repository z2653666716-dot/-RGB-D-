"""M3 抓取动作控制：把视觉估计的方块中心转换为 FR3 的抓取动作序列。

本文件负责预抓取、下降、夹爪闭合、抬升与保持阶段的 IK 目标和关节空间
轨迹执行。位置输入必须来自 vision.py 或后续状态估计器；此控制器不读取
仿真物体真值，也不负责最终成功判定（那由独立评估模块完成）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from inverse_kinematics import IkResult, solve_fr3_ik
from grasp_planner import GraspCandidate
from grasp_planner import GraspCandidate


@dataclass(frozen=True)
class GraspExecutionResult:
    """一次抓取动作流程摘要；成功表示动作已执行完，不代表物理抓取已成功。"""

    completed: bool
    reason: str
    phases_completed: tuple[str, ...]
    ik_results: tuple[IkResult, ...]
    estimated_object_center_world_m: tuple[float, float, float]


def _joint_ids_and_addresses(model: mujoco.MjModel) -> tuple[list[int], np.ndarray, np.ndarray]:
    """按七个 FR3 关节名称取编号、qpos 地址和执行器编号。"""
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    actuator_ids: list[int] = []
    for number in range(1, 8):
        name = f"fr3_joint{number}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"模型缺少 FR3 关节或同名位置执行器：{name}")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        actuator_ids.append(actuator_id)
    return joint_ids, np.asarray(qpos_addresses), np.asarray(actuator_ids)


def _make_pose_targets(
    object_center_world_m: np.ndarray,
    object_side_m: float,
    tool_orientation_wxyz: np.ndarray,
    table_top_height_m: float,
    tool_to_fingertip_m: float,
    pregrasp_clearance_m: float,
    lift_distance_m: float,
) -> list[tuple[str, np.ndarray]]:
    """从方块中心生成预抓取、抓取和抬升三个末端 site 目标位置。"""
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, tool_orientation_wxyz)
    rotation = rotation.reshape(3, 3)
    # 末端 site 的局部 Z 轴指向夹爪指尖；取反后得到从指尖回到 site 的向量。
    site_above_fingertip = -rotation[:, 2] * tool_to_fingertip_m
    fingertip_height = table_top_height_m + object_side_m / 2
    grasp_position = object_center_world_m.copy()
    grasp_position[2] = fingertip_height
    grasp_site = grasp_position + site_above_fingertip
    pregrasp_site = grasp_site - rotation[:, 2] * pregrasp_clearance_m
    lift_site = grasp_site - rotation[:, 2] * lift_distance_m
    return [
        ("pregrasp", pregrasp_site),
        ("descend", grasp_site),
        ("lift", lift_site),
    ]


def _trajectory_reference(start: np.ndarray, goal: np.ndarray, fraction: float) -> np.ndarray:
    """用五次多项式平滑插值；起点与终点的速度、加速度均为零。"""
    t = float(np.clip(fraction, 0.0, 1.0))
    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
    return start + blend * (goal - start)


def _move_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    actuator_ids: np.ndarray,
    goal_angles_rad: np.ndarray,
    steps: int,
) -> None:
    """控制 FR3 位置执行器跟随五次曲线关节参考，并按固定步数推进仿真。"""
    start_angles = data.qpos[qpos_addresses].copy()
    for step_number in range(1, steps + 1):
        reference = _trajectory_reference(start_angles, goal_angles_rad,
                                          step_number / steps)
        data.ctrl[actuator_ids] = reference
        mujoco.mj_step(model, data)


def _set_gripper(model: mujoco.MjModel, data: mujoco.MjData, control: float,
                 hold_steps: int) -> None:
    """输入夹爪执行器控制值并保持指定仿真步数，让物理接触自行形成。"""
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    "gripper_actuator8")
    if actuator_id < 0:
        raise ValueError("模型中不存在 gripper_actuator8 平行夹爪执行器")
    low, high = model.actuator_ctrlrange[actuator_id]
    if not low <= control <= high:
        raise ValueError(f"夹爪控制值 {control} 超出执行器范围 [{low}, {high}]")
    data.ctrl[actuator_id] = control
    for _ in range(hold_steps):
        mujoco.mj_step(model, data)


def execute_static_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_center_world_m: np.ndarray,
    object_side_m: float,
    table_top_height_m: float,
    tool_orientation_wxyz: np.ndarray,
    *,
    tool_to_fingertip_m: float = 0.11,
    pregrasp_clearance_m: float = 0.10,
    lift_distance_m: float = 0.12,
    trajectory_steps: int = 180,
    gripper_open_control: float = 255.0,
    gripper_close_control: float = 0.0,
    gripper_hold_steps: int = 120,
    end_effector_site: str = "attachment_site",
    hold_sample_callback: Callable[[], None] | None = None,
) -> GraspExecutionResult:
    """依据视觉估计执行静态方块抓取动作，返回阶段记录和每段 IK 结果。

    输入位置是方块估计中心的世界坐标米数，不可传入仿真真值。输入已知方块
    边长、桌面高度、固定工具姿态；输出仅表示流程是否完成与 IK 是否可解，
    不代表方块已被夹起。不可达目标会在动作执行前返回失败。
    hold_sample_callback 是可选的无参数回调，只在抬升后的保持期每步调用；
    调用方可在回调闭包中采集评估快照，控制器本身不读取快照或真实物体位置。
    """
    center = np.asarray(object_center_world_m, dtype=np.float64)
    orientation = np.asarray(tool_orientation_wxyz, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("方块估计中心须为有限的世界坐标 [x,y,z] 米数")
    if orientation.shape != (4,) or not np.isfinite(orientation).all():
        raise ValueError("工具姿态须是有限的 [w,x,y,z] 四元数")
    orientation_norm = float(np.linalg.norm(orientation))
    if orientation_norm < 1e-12:
        raise ValueError("工具姿态四元数长度不能为零")
    orientation /= orientation_norm
    if object_side_m <= 0 or trajectory_steps <= 0 or gripper_hold_steps < 0:
        raise ValueError("方块边长和轨迹步数须为正，夹爪保持步数不能为负")
    if min(tool_to_fingertip_m, pregrasp_clearance_m, lift_distance_m) < 0:
        raise ValueError("工具偏移、预抓取间距与抬升距离不能为负")

    joint_ids, qpos_addresses, actuator_ids = _joint_ids_and_addresses(model)
    targets = _make_pose_targets(
        center, object_side_m, orientation, table_top_height_m,
        tool_to_fingertip_m, pregrasp_clearance_m, lift_distance_m,
    )

    # 先在临时状态里逐段求解全部目标；任何一段不可达时，不开始真实动作。
    planning_data = mujoco.MjData(model)
    planning_data.qpos[:] = data.qpos
    planning_data.qvel[:] = data.qvel
    planned_angles: list[np.ndarray] = []
    ik_results: list[IkResult] = []
    for phase_name, target_position in targets:
        result = solve_fr3_ik(model, planning_data, target_position, orientation,
                              end_effector_site=end_effector_site)
        ik_results.append(result)
        if not result.success or result.joint_positions_rad is None:
            return GraspExecutionResult(
                False, f"ik_failed_at_{phase_name}:{result.reason}", (),
                tuple(ik_results), tuple(float(value) for value in center),
            )
        angles = np.asarray(result.joint_positions_rad, dtype=np.float64)
        planned_angles.append(angles)
        # 下一段 IK 以当前规划解为起点，避免各段落到不连续的冗余关节分支。
        planning_data.qpos[qpos_addresses] = angles
        mujoco.mj_forward(model, planning_data)

    phases: list[str] = []
    _set_gripper(model, data, gripper_open_control, 1)
    phases.append("gripper_open")
    # 预抓取位姿先走较长的安全路径，随后下降到方块两侧。
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[0], trajectory_steps)
    phases.append("pregrasp")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[1], trajectory_steps)
    phases.append("descend")
    _set_gripper(model, data, gripper_close_control, gripper_hold_steps)
    phases.append("gripper_close")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[2], trajectory_steps)
    phases.append("lift")
    # 抬升后继续保持夹爪控制值；物体是否留在夹爪上交给评估模块判断。
    for _ in range(gripper_hold_steps):
        mujoco.mj_step(model, data)
        # 只通知外部“当前已推进一步”，评估器通过闭包自行读取状态并保存快照；
        # 采样值不会进入后续控制决策，因此真值评估不会泄露给控制器。
        if hold_sample_callback is not None:
            hold_sample_callback()
    phases.append("hold")
    return GraspExecutionResult(
        True, "motion_sequence_completed", tuple(phases), tuple(ik_results),
        tuple(float(value) for value in center),
    )


def execute_grasp_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: GraspCandidate,
    *,
    lift_distance_m: float = 0.12,
    gripper_max_opening_m: float = 0.08,
    trajectory_steps: int = 180,
    gripper_settle_steps: int = 60,
    gripper_close_steps: int = 120,
    post_lift_hold_steps: int = 360,
    end_effector_site: str = "attachment_site",
    hold_sample_callback: Callable[[], None] | None = None,
) -> GraspExecutionResult:
    """按规划器输出的单个候选执行物理抓取，输入候选、模型和当前仿真状态。

    先在只含机器人关节的 IK 副本中求预抓取、抓取、抬升三个目标；全部可解后才
    打开夹爪、走关节轨迹、闭合和抬升。夹爪开口从候选的米制总宽度换算为模型
    执行器控制值。输出动作阶段和 IK 误差；是否真正抓住仍由独立评估器判定。
    """
    if lift_distance_m < 0 or trajectory_steps <= 0:
        raise ValueError("抬升距离不能为负，轨迹步数必须为正")
    if min(gripper_settle_steps, gripper_close_steps, post_lift_hold_steps) < 0:
        raise ValueError("夹爪等待和保持步数不能为负")
    if not 0 < candidate.requested_opening_width_m <= gripper_max_opening_m:
        raise ValueError("候选夹爪开口宽度超出配置能力")

    joint_ids, qpos_addresses, actuator_ids = _joint_ids_and_addresses(model)
    # IK 临时状态只复制七个 FR3 关节，不把方块自由关节真值交给求解器。
    planning_data = mujoco.MjData(model)
    planning_data.qpos[qpos_addresses] = data.qpos[qpos_addresses]
    planning_data.qvel[:] = 0.0
    orientation = np.asarray(candidate.tool_orientation_wxyz, dtype=np.float64)
    lift_position = np.asarray(candidate.tool_site_position_world_m, dtype=np.float64)
    lift_position[2] += lift_distance_m
    targets = (
        ("pregrasp", np.asarray(candidate.pregrasp_site_position_world_m)),
        ("descend", np.asarray(candidate.tool_site_position_world_m)),
        ("lift", lift_position),
    )
    ik_results: list[IkResult] = []
    planned_angles: list[np.ndarray] = []
    for phase, target_position in targets:
        ik = solve_fr3_ik(model, planning_data, target_position, orientation,
                          end_effector_site=end_effector_site)
        ik_results.append(ik)
        if not ik.success or ik.joint_positions_rad is None:
            return GraspExecutionResult(
                False, f"ik_failed_at_{phase}:{ik.reason}", (), tuple(ik_results),
                candidate.grasp_center_world_m,
            )
        angles = np.asarray(ik.joint_positions_rad, dtype=np.float64)
        planned_angles.append(angles)
        planning_data.qpos[qpos_addresses] = angles
        mujoco.mj_forward(model, planning_data)

    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   "gripper_actuator8")
    if gripper_id < 0:
        raise ValueError("模型中不存在 gripper_actuator8")
    control_low, control_high = model.actuator_ctrlrange[gripper_id]
    # Menagerie 夹爪最大总开口与执行器高端对应；按宽度线性缩放到 0..255 范围。
    open_fraction = candidate.requested_opening_width_m / gripper_max_opening_m
    open_control = float(control_low + open_fraction * (control_high - control_low))
    close_control = float(control_low)

    phases: list[str] = []
    _set_gripper(model, data, open_control, gripper_settle_steps)
    phases.append("gripper_open_to_candidate_width")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[0], trajectory_steps)
    phases.append("pregrasp")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[1], trajectory_steps)
    phases.append("descend")
    _set_gripper(model, data, close_control, gripper_close_steps)
    phases.append("gripper_close")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[2], trajectory_steps)
    phases.append("lift")
    for _ in range(post_lift_hold_steps):
        mujoco.mj_step(model, data)
        if hold_sample_callback is not None:
            hold_sample_callback()
    phases.append("hold")
    return GraspExecutionResult(
        True, "candidate_motion_sequence_completed", tuple(phases), tuple(ik_results),
        candidate.grasp_center_world_m,
    )


def execute_grasp_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: GraspCandidate,
    *,
    lift_distance_m: float = 0.12,
    gripper_max_opening_m: float = 0.08,
    trajectory_steps: int = 180,
    gripper_settle_steps: int = 60,
    gripper_close_steps: int = 120,
    post_lift_hold_steps: int = 360,
    end_effector_site: str = "attachment_site",
    hold_sample_callback: Callable[[], None] | None = None,
) -> GraspExecutionResult:
    """按规划器输出的单个候选执行物理抓取，输入候选、模型和当前仿真状态。

    先在只含机器人关节的 IK 副本中求预抓取、抓取、抬升三个目标；全部可解后才
    打开夹爪、走关节轨迹、闭合和抬升。夹爪开口从候选的米制总宽度换算为模型
    执行器控制值。输出动作阶段和 IK 误差；是否真正抓住仍由独立评估器判定。
    """
    if lift_distance_m < 0 or trajectory_steps <= 0:
        raise ValueError("抬升距离不能为负，轨迹步数必须为正")
    if min(gripper_settle_steps, gripper_close_steps, post_lift_hold_steps) < 0:
        raise ValueError("夹爪等待和保持步数不能为负")
    if not 0 < candidate.requested_opening_width_m <= gripper_max_opening_m:
        raise ValueError("候选夹爪开口宽度超出配置能力")

    joint_ids, qpos_addresses, actuator_ids = _joint_ids_and_addresses(model)
    # IK 临时状态只复制七个 FR3 关节，不把方块自由关节真值交给求解器。
    planning_data = mujoco.MjData(model)
    planning_data.qpos[qpos_addresses] = data.qpos[qpos_addresses]
    planning_data.qvel[:] = 0.0
    orientation = np.asarray(candidate.tool_orientation_wxyz, dtype=np.float64)
    lift_position = np.asarray(candidate.tool_site_position_world_m, dtype=np.float64)
    lift_position[2] += lift_distance_m
    targets = (
        ("pregrasp", np.asarray(candidate.pregrasp_site_position_world_m)),
        ("descend", np.asarray(candidate.tool_site_position_world_m)),
        ("lift", lift_position),
    )
    ik_results: list[IkResult] = []
    planned_angles: list[np.ndarray] = []
    for phase, target_position in targets:
        ik = solve_fr3_ik(model, planning_data, target_position, orientation,
                          end_effector_site=end_effector_site)
        ik_results.append(ik)
        if not ik.success or ik.joint_positions_rad is None:
            return GraspExecutionResult(
                False, f"ik_failed_at_{phase}:{ik.reason}", (), tuple(ik_results),
                candidate.grasp_center_world_m,
            )
        angles = np.asarray(ik.joint_positions_rad, dtype=np.float64)
        planned_angles.append(angles)
        planning_data.qpos[qpos_addresses] = angles
        mujoco.mj_forward(model, planning_data)

    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   "gripper_actuator8")
    if gripper_id < 0:
        raise ValueError("模型中不存在 gripper_actuator8")
    control_low, control_high = model.actuator_ctrlrange[gripper_id]
    # Menagerie 夹爪最大总开口与执行器高端对应；按宽度线性缩放到 0..255 范围。
    open_fraction = candidate.requested_opening_width_m / gripper_max_opening_m
    open_control = float(control_low + open_fraction * (control_high - control_low))
    close_control = float(control_low)

    phases: list[str] = []
    _set_gripper(model, data, open_control, gripper_settle_steps)
    phases.append("gripper_open_to_candidate_width")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[0], trajectory_steps)
    phases.append("pregrasp")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[1], trajectory_steps)
    phases.append("descend")
    _set_gripper(model, data, close_control, gripper_close_steps)
    phases.append("gripper_close")
    _move_joints(model, data, qpos_addresses, actuator_ids,
                 planned_angles[2], trajectory_steps)
    phases.append("lift")
    for _ in range(post_lift_hold_steps):
        mujoco.mj_step(model, data)
        if hold_sample_callback is not None:
            hold_sample_callback()
    phases.append("hold")
    return GraspExecutionResult(
        True, "candidate_motion_sequence_completed", tuple(phases), tuple(ik_results),
        candidate.grasp_center_world_m,
    )
