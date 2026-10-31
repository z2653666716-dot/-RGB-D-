"""抓取结果评估：用仿真真值独立判定是否真实抬起并稳定保持方块。

本模块只服务离线评估，不可把读到的物体位置传回视觉、状态估计、IK 或控制器。
评估器按固定配置检查抬升高度、保持时间、掉落、夹爪相对位移和禁止碰撞，
返回成功/失败分类及指标，供之后批量实验记录。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from grasp_controller import (
    GraspExecutionResult,
    execute_grasp_candidate,
    execute_static_grasp,
)
from grasp_planner import GraspCandidate

@dataclass(frozen=True)
class GraspSample:
    """保持阶段的一次评估快照；真值字段只应由评估器采集和消费。"""

    time_seconds: float
    cube_position_world_m: tuple[float, float, float]
    gripper_position_world_m: tuple[float, float, float]
    cube_touches_fingers: bool
    forbidden_collision: bool


@dataclass(frozen=True)
class GraspEvaluation:
    """抓取成功判定和可追溯指标；失败时 reason 指出首要失败条件。"""

    success: bool
    reason: str
    hold_duration_seconds: float
    maximum_lift_m: float
    final_lift_m: float
    drop_from_peak_m: float
    maximum_relative_drift_m: float
    finger_contact_observed: bool
    forbidden_collision_observed: bool


@dataclass(frozen=True)
class GraspTrialResult:
    """一次完整抓取试验的动作结果、独立评估和保持期快照数量。"""

    execution: GraspExecutionResult
    evaluation: GraspEvaluation | None
    hold_sample_count: int


def load_evaluation_config(path: str | Path) -> dict:
    """读取固定的抓取验收阈值配置；输入 JSON 路径，输出检查后的字典。"""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = (
        "minimum_lift_m", "minimum_hold_seconds", "maximum_drop_from_peak_m",
        "maximum_cube_to_gripper_drift_m",
    )
    if any(key not in config for key in required):
        raise ValueError("抓取评估配置缺少必需阈值")
    if any(not np.isfinite(config[key]) or config[key] < 0 for key in required):
        raise ValueError("抓取评估阈值必须是有限的非负数")
    if config["minimum_hold_seconds"] <= 0 or config["minimum_lift_m"] <= 0:
        raise ValueError("最短保持时间和最小抬升高度必须大于零")
    return config


def capture_grasp_sample(model: mujoco.MjModel, data: mujoco.MjData) -> GraspSample:
    """读取一帧仅供评估的方块/夹爪真值、指尖接触与禁止碰撞状态。

    输入当前 MuJoCo 模型和状态，输出 hold 阶段快照。调用方应只在抓取动作完成
    后、评估保持时间内采样；快照不允许回传到控制器作为目标位置。
    """
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     "target_cube_collision")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  "gripper_left_finger")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   "gripper_right_finger")
    if min(cube_body_id, cube_geom_id, site_id, left_body, right_body) < 0:
        raise ValueError("评估所需的方块、夹爪手指或末端 site 不存在")

    finger_body_ids = {left_body, right_body}
    table_geom_ids = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        .startswith("table_")
    }
    cube_touches_fingers = False
    forbidden_collision = False
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first_geom, second_geom = int(contact.geom1), int(contact.geom2)
        first_body = int(model.geom_bodyid[first_geom])
        second_body = int(model.geom_bodyid[second_geom])
        if cube_geom_id in (first_geom, second_geom):
            other_body = second_body if first_geom == cube_geom_id else first_body
            if other_body in finger_body_ids:
                cube_touches_fingers = True
            elif other_body != 0 and other_body not in table_geom_ids:
                # 方块与 FR3 手臂或夹爪手掌碰撞，不算正常指尖夹持。
                other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                               other_body) or ""
                if other_name.startswith(("fr3_", "gripper_hand")):
                    forbidden_collision = True
        # 机器人刚体碰桌属于禁止碰撞；物体落桌在保持期则由“未抬起”条件判失败。
        for geom_id, body_id in ((first_geom, first_body), (second_geom, second_body)):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if geom_id in table_geom_ids and body_name.startswith(("fr3_", "gripper_")):
                forbidden_collision = True

    cube_position = data.xpos[cube_body_id].copy()
    gripper_position = data.site_xpos[site_id].copy()
    return GraspSample(
        time_seconds=float(data.time),
        cube_position_world_m=tuple(float(value) for value in cube_position),
        gripper_position_world_m=tuple(float(value) for value in gripper_position),
        cube_touches_fingers=cube_touches_fingers,
        forbidden_collision=forbidden_collision,
    )


def evaluate_grasp(
    initial_cube_position_world_m: np.ndarray,
    hold_samples: list[GraspSample] | tuple[GraspSample, ...],
    config: dict,
) -> GraspEvaluation:
    """评估一段抓取后保持记录，返回固定阈值下的结果和失败原因。

    输入初始方块真值（仅评估用途）、保持期快照序列和验收配置。成功要求方块
    抬升到阈值、末帧仍保持高度、未从峰值显著掉落、夹爪相对位置稳定、保持期
    至少检测到一次指尖接触，且全程没有禁止碰撞。
    """
    initial = np.asarray(initial_cube_position_world_m, dtype=np.float64)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("初始方块位置必须是有限的世界坐标 [x,y,z] 米数")
    if not hold_samples:
        return GraspEvaluation(False, "no_hold_samples", 0.0, 0.0, 0.0, 0.0,
                               0.0, False, False)
    times = np.asarray([sample.time_seconds for sample in hold_samples])
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("保持期采样时间必须有限且按先后顺序排列")
    cube_positions = np.asarray([sample.cube_position_world_m for sample in hold_samples])
    gripper_positions = np.asarray([sample.gripper_position_world_m for sample in hold_samples])
    if (cube_positions.shape != (len(hold_samples), 3)
            or gripper_positions.shape != (len(hold_samples), 3)
            or not np.isfinite(cube_positions).all()
            or not np.isfinite(gripper_positions).all()):
        raise ValueError("保持期方块和夹爪位置必须为有限的三维坐标")

    lift_values = cube_positions[:, 2] - initial[2]
    maximum_lift = float(np.max(lift_values))
    final_lift = float(lift_values[-1])
    drop_from_peak = maximum_lift - final_lift
    # 以首次指尖接触帧为参考，测量方块相对末端 site 的位移变化。
    contact_indices = [index for index, sample in enumerate(hold_samples)
                       if sample.cube_touches_fingers]
    contact_observed = bool(contact_indices)
    if contact_observed:
        reference_index = contact_indices[0]
        relative_positions = cube_positions - gripper_positions
        relative_drift = np.linalg.norm(
            relative_positions - relative_positions[reference_index], axis=1)
        maximum_drift = float(np.max(relative_drift))
    else:
        maximum_drift = float("inf")
    forbidden_observed = any(sample.forbidden_collision for sample in hold_samples)
    hold_duration = float(times[-1] - times[0])

    # 顺序固定，便于批量实验将同类失败计入同一个分类。
    if forbidden_observed:
        reason = "forbidden_collision"
    elif hold_duration < config["minimum_hold_seconds"]:
        reason = "hold_too_short"
    elif maximum_lift < config["minimum_lift_m"]:
        reason = "not_lifted_enough"
    elif final_lift < config["minimum_lift_m"]:
        reason = "object_dropped_to_low_height"
    elif drop_from_peak > config["maximum_drop_from_peak_m"]:
        reason = "object_dropped_after_lift"
    elif not contact_observed:
        reason = "no_finger_contact"
    elif maximum_drift > config["maximum_cube_to_gripper_drift_m"]:
        reason = "object_slipped_in_gripper"
    else:
        reason = "grasp_success"

    return GraspEvaluation(
        success=reason == "grasp_success", reason=reason,
        hold_duration_seconds=hold_duration, maximum_lift_m=maximum_lift,
        final_lift_m=final_lift, drop_from_peak_m=float(drop_from_peak),
        maximum_relative_drift_m=maximum_drift,
        finger_contact_observed=contact_observed,
        forbidden_collision_observed=forbidden_observed,
    )


def run_and_evaluate_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    estimated_object_center_world_m: np.ndarray,
    object_side_m: float,
    table_top_height_m: float,
    tool_orientation_wxyz: np.ndarray,
    evaluation_config: dict,
    **controller_settings,
) -> GraspTrialResult:
    """运行一次抓取并独立评估；目标输入来自视觉，仿真真值仅留在评估侧。

    在动作开始前，本函数通过评估快照记录初始真实位置；保持阶段的采样回调也
    只由评估器持有。控制器只收到 estimated_object_center_world_m、物体尺寸、
    桌面高度和固定工具姿态。若动作未完成，则 evaluation=None，不伪造成功率。
    """
    initial_sample = capture_grasp_sample(model, data)
    initial_position = np.asarray(initial_sample.cube_position_world_m, dtype=np.float64)
    hold_samples: list[GraspSample] = []

    def collect_evaluation_sample() -> None:
        """控制器每步回调此闭包；仅采集真值快照，不向控制器返回数据。"""
        hold_samples.append(capture_grasp_sample(model, data))

    execution = execute_static_grasp(
        model, data, estimated_object_center_world_m, object_side_m,
        table_top_height_m, tool_orientation_wxyz,
        hold_sample_callback=collect_evaluation_sample,
        **controller_settings,
    )
    if not execution.completed:
        return GraspTrialResult(execution, None, len(hold_samples))
    evaluation = evaluate_grasp(initial_position, hold_samples, evaluation_config)
    return GraspTrialResult(execution, evaluation, len(hold_samples))


def run_and_evaluate_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: GraspCandidate,
    evaluation_config: dict,
    **controller_settings,
) -> GraspTrialResult:
    """执行一个已规划候选并独立评估结果；只有评估器接触仿真真值。"""
    initial_sample = capture_grasp_sample(model, data)
    initial_position = np.asarray(initial_sample.cube_position_world_m, dtype=np.float64)
    hold_samples: list[GraspSample] = []

    def collect_evaluation_sample() -> None:
        """保持期回调只把真值样本存入评估侧，不向控制器返回任何位置。"""
        hold_samples.append(capture_grasp_sample(model, data))

    execution = execute_grasp_candidate(
        model, data, candidate, hold_sample_callback=collect_evaluation_sample,
        **controller_settings,
    )
    if not execution.completed:
        return GraspTrialResult(execution, None, len(hold_samples))
    evaluation = evaluate_grasp(initial_position, hold_samples, evaluation_config)
    return GraspTrialResult(execution, evaluation, len(hold_samples))
