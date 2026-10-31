"""由已估计的平面几何生成多个俯视平行夹爪抓取候选。

本模块只根据视觉几何、已声明尺寸和夹爪开口限制生成候选，不读取场景真值。
输出每个候选的物体抓取中心、末端 site 目标、预抓取点、工具四元数、开口宽度
和评分；IK 可达性与碰撞风险留给后续筛选步骤。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from geometry import PlanarObjectGeometry


@dataclass(frozen=True)
class GraspCandidate:
    """一个可解释的俯视抓取方案；四元数顺序为 MuJoCo [w,x,y,z]。"""

    candidate_id: str
    grasp_center_world_m: tuple[float, float, float]
    tool_site_position_world_m: tuple[float, float, float]
    pregrasp_site_position_world_m: tuple[float, float, float]
    tool_orientation_wxyz: tuple[float, float, float, float]
    closing_axis_world_xy: tuple[float, float]
    requested_opening_width_m: float
    orientation_source: str
    score: float


@dataclass(frozen=True)
class CandidateGenerationResult:
    """候选生成结果；拒绝项按原因保留，方便后续报告失败分类。"""

    success: bool
    reason: str
    candidates: tuple[GraspCandidate, ...]
    rejected_reasons: tuple[str, ...]


def _tool_quaternion_for_closing_axis(closing_axis_xy: np.ndarray) -> np.ndarray:
    """构造 Z 轴朝下、局部 Y 轴沿夹爪闭合方向的工具姿态四元数。"""
    beta = float(np.arctan2(closing_axis_xy[1], closing_axis_xy[0]))
    # 先确定工具局部 X 方向，再绕它的局部 X 轴翻转 180°，令工具 Z 朝向桌面。
    tool_x_yaw = beta + np.pi / 2.0
    cosine, sine = np.cos(tool_x_yaw), np.sin(tool_x_yaw)
    rotation = np.array([
        [cosine, sine, 0.0],
        [sine, -cosine, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion


def generate_grasp_candidates(
    geometry: PlanarObjectGeometry,
    planner_config: dict,
) -> CandidateGenerationResult:
    """从平面中心、长短边和方向估计生成候选，并按夹爪可开宽度筛选。

    输入必须是 geometry.py 的视觉几何结果和 grasp_planner.json 参数。正方形等
    近似对称物体方向不可信时，退化为世界 X/Y 两个等价方向；明显非对称但方向
    不可信时拒绝规划，避免把不稳定主轴角假装成可靠方向。
    """
    if (not geometry.valid or geometry.object_center_world_m is None
            or geometry.footprint_dimensions_long_short_m is None):
        return CandidateGenerationResult(False, "geometry_invalid", (), (geometry.reason,))
    long_size, short_size = geometry.footprint_dimensions_long_short_m
    if min(long_size, short_size) < planner_config["minimum_object_width_m"]:
        return CandidateGenerationResult(False, "object_too_narrow", (),
                                         ("below_minimum_object_width",))

    dimensions_ratio = long_size / max(short_size, 1e-12)
    if geometry.orientation_confident and geometry.major_axis_world_xy is not None:
        major_axis = np.asarray(geometry.major_axis_world_xy, dtype=np.float64)
        major_axis /= max(float(np.linalg.norm(major_axis)), 1e-12)
        minor_axis = np.array([-major_axis[1], major_axis[0]])
        orientation_source = "estimated_geometry"
        closing_directions = [
            ("short_axis", minor_axis, short_size, major_axis),
            ("long_axis", major_axis, long_size, minor_axis),
        ]
    elif dimensions_ratio <= planner_config["symmetric_ratio_tolerance"]:
        # 近似方形绕中心旋转不改变可见轮廓，选用全局 X/Y 作明确的对称退化策略。
        orientation_source = "symmetric_world_axis_fallback"
        closing_directions = [
            ("world_x", np.array([1.0, 0.0]), short_size, np.array([0.0, 1.0])),
            ("world_y", np.array([0.0, 1.0]), long_size, np.array([1.0, 0.0])),
        ]
    else:
        return CandidateGenerationResult(False, "orientation_untrusted", (),
                                         ("asymmetric_object_direction_uncertain",))

    center = np.asarray(geometry.object_center_world_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("目标几何中心必须是有限的三维世界坐标")
    total_margin_m = (planner_config["finger_clearance_m"]
                      + planner_config["position_uncertainty_margin_m"])
    maximum_opening = planner_config["gripper_max_opening_m"]
    if min(total_margin_m, maximum_opening) < 0 or maximum_opening <= 0:
        raise ValueError("夹爪开口和几何裕量配置无效")

    candidates: list[GraspCandidate] = []
    rejected: list[str] = []
    for direction_name, closing_axis, object_width, offset_axis in closing_directions:
        requested_width = object_width + 2.0 * total_margin_m
        if requested_width > maximum_opening:
            rejected.append(f"{direction_name}:opening_exceeds_gripper_limit")
            continue
        tool_orientation = _tool_quaternion_for_closing_axis(closing_axis)
        for offset_fraction in planner_config["center_offset_fractions"]:
            candidate_center = center.copy()
            candidate_center[:2] += offset_axis * (offset_fraction * long_size)
            tool_site = candidate_center.copy()
            tool_site[2] += planner_config["tool_site_offset_above_center_m"]
            pregrasp_site = tool_site.copy()
            pregrasp_site[2] += planner_config["pregrasp_clearance_m"]
            # 中心抓取分数最低；偏离中心越多、夹爪开口越大，排序分数越高。
            score = (abs(float(offset_fraction))
                     + 0.1 * requested_width / maximum_opening)
            candidate_number = len(candidates) + 1
            candidates.append(GraspCandidate(
                candidate_id=f"{direction_name}_{candidate_number}",
                grasp_center_world_m=tuple(float(value) for value in candidate_center),
                tool_site_position_world_m=tuple(float(value) for value in tool_site),
                pregrasp_site_position_world_m=tuple(float(value) for value in pregrasp_site),
                tool_orientation_wxyz=tuple(float(value) for value in tool_orientation),
                closing_axis_world_xy=(float(closing_axis[0]), float(closing_axis[1])),
                requested_opening_width_m=float(requested_width),
                orientation_source=orientation_source,
                score=float(score),
            ))

    candidates.sort(key=lambda item: item.score)
    if not candidates:
        return CandidateGenerationResult(False, "no_candidate_fits_gripper", (), tuple(rejected))
    return CandidateGenerationResult(True, "candidates_generated", tuple(candidates),
                                     tuple(rejected))
