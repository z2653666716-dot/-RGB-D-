"""从视觉目标区域的 RGB-D 点云估计平面物体中心、尺寸和主方向。

输入是图像算法的颜色区域、已组织点云和视觉检测框；按已知桌面高度剔除台面，
从目标最高可见平面提取点，再对 XY 点做二维 PCA 与稳健分位数尺寸估计。
该模块不读取仿真物体 ID；对近似正方形的对称物体会明确标记方向不可信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_cloud import OrganizedPointCloud
from vision import PositionObservation, colored_target_mask


@dataclass(frozen=True)
class PlanarObjectGeometry:
    """一个目标平面几何估计；尺寸顺序为长边、短边，yaw 以世界 +X 轴为零。"""

    valid: bool
    reason: str
    target_point_count: int
    top_surface_center_world_m: tuple[float, float, float] | None
    object_center_world_m: tuple[float, float, float] | None
    footprint_dimensions_long_short_m: tuple[float, float] | None
    major_axis_world_xy: tuple[float, float] | None
    yaw_world_rad: float | None
    orientation_confident: bool
    estimated_top_height_m: float | None
    top_plane_rmse_m: float | None


def _invalid(reason: str, point_count: int = 0) -> PlanarObjectGeometry:
    """统一构造拒绝结果；几何无效时不借用真值或上次位置补齐。"""
    return PlanarObjectGeometry(False, reason, point_count, None, None, None,
                                None, None, False, None, None)


def _select_target_top_points(
    rgb: np.ndarray,
    cloud: OrganizedPointCloud,
    observation: PositionObservation,
    vision_config: dict,
    table_top_height_m: float,
    geometry_config: dict,
) -> np.ndarray:
    """在检测框内用 HSV 颜色与有效深度掩码取目标点，并剔除桌面和侧面点。"""
    if not observation.valid or observation.bbox_xywh_px is None:
        return np.empty((0, 3), dtype=np.float64)
    height, width = cloud.valid_depth_mask.shape
    if rgb.shape != (height, width, 3):
        raise ValueError("RGB 图像和有序点云的高宽必须一致")
    # 只用普通 RGB 颜色阈值筛实际图像，不读取 MuJoCo segmentation 或物体 ID。
    color_mask = colored_target_mask(rgb, vision_config) > 0
    x, y, box_width, box_height = observation.bbox_xywh_px
    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[max(0, y):min(height, y + box_height),
             max(0, x):min(width, x + box_width)] = True
    selected = color_mask & roi_mask & cloud.valid_depth_mask
    points = cloud.points_world_m[selected]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    # 已知桌面是水平面 z=table_top；先剔除桌面及其深度混合边缘，再取最上层。
    above_table = points[:, 2] > table_top_height_m + geometry_config[
        "table_surface_exclusion_m"]
    points = points[above_table]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    highest_surface = float(np.max(points[:, 2]))
    top_band = geometry_config["top_surface_band_m"]
    return points[points[:, 2] >= highest_surface - top_band]


def _estimate_from_top_points(
    top_points_world_m: np.ndarray,
    table_top_height_m: float,
    known_object_height_m: float,
    minimum_orientation_aspect_ratio: float,
    trim_percentile: float,
    minimum_target_points: int,
) -> PlanarObjectGeometry:
    """用 XY 协方差求主轴，再用沿主轴投影的分位数估计物体平面尺寸。"""
    if len(top_points_world_m) < minimum_target_points:
        return _invalid("too_few_top_surface_points", len(top_points_world_m))
    xy = top_points_world_m[:, :2]
    mean_xy = np.mean(xy, axis=0)
    centered = xy - mean_xy
    covariance_xx = float(np.mean(centered[:, 0] ** 2))
    covariance_yy = float(np.mean(centered[:, 1] ** 2))
    covariance_xy = float(np.mean(centered[:, 0] * centered[:, 1]))

    # 对称的 2x2 协方差主轴角可由解析式求出，避免引入额外矩阵分解依赖。
    yaw = 0.5 * float(np.arctan2(2.0 * covariance_xy,
                                 covariance_xx - covariance_yy))
    major_axis = np.array([np.cos(yaw), np.sin(yaw)])
    minor_axis = np.array([-major_axis[1], major_axis[0]])
    major_projection = centered[:, 0] * major_axis[0] + centered[:, 1] * major_axis[1]
    minor_projection = centered[:, 0] * minor_axis[0] + centered[:, 1] * minor_axis[1]
    lower_q = trim_percentile
    upper_q = 100.0 - trim_percentile
    major_limits = np.percentile(major_projection, [lower_q, upper_q])
    minor_limits = np.percentile(minor_projection, [lower_q, upper_q])
    major_size = float(major_limits[1] - major_limits[0])
    minor_size = float(minor_limits[1] - minor_limits[0])
    if minor_size > major_size:
        major_size, minor_size = minor_size, major_size
        major_axis, minor_axis = minor_axis, major_axis
        major_limits, minor_limits = minor_limits, major_limits
    aspect_ratio = major_size / max(minor_size, 1e-12)
    orientation_confident = aspect_ratio >= minimum_orientation_aspect_ratio

    yaw = (float(np.arctan2(major_axis[1], major_axis[0])) + np.pi / 2) % np.pi - np.pi / 2
    # 分位数区间中点还原到世界平面，降低少数边缘深度离群点对中心的影响。
    major_midpoint = float(np.mean(major_limits))
    minor_midpoint = float(np.mean(minor_limits))
    center_xy = (mean_xy + major_axis * major_midpoint
                 + minor_axis * minor_midpoint)
    top_height = float(np.median(top_points_world_m[:, 2]))
    plane_rmse = float(np.sqrt(np.mean((top_points_world_m[:, 2] - top_height) ** 2)))
    object_center = np.array([
        center_xy[0], center_xy[1], table_top_height_m + known_object_height_m / 2.0,
    ])
    reason = "ok" if orientation_confident else "orientation_ambiguous_due_to_symmetry"
    return PlanarObjectGeometry(
        valid=True, reason=reason, target_point_count=len(top_points_world_m),
        top_surface_center_world_m=(float(center_xy[0]), float(center_xy[1]), top_height),
        object_center_world_m=tuple(float(value) for value in object_center),
        footprint_dimensions_long_short_m=(major_size, minor_size),
        major_axis_world_xy=(float(major_axis[0]), float(major_axis[1])),
        yaw_world_rad=float(yaw), orientation_confident=orientation_confident,
        estimated_top_height_m=top_height, top_plane_rmse_m=plane_rmse,
    )


def estimate_planar_object_geometry(
    rgb: np.ndarray,
    cloud: OrganizedPointCloud,
    observation: PositionObservation,
    vision_config: dict,
    table_top_height_m: float,
    geometry_config: dict,
) -> PlanarObjectGeometry:
    """由视觉检测区域和有序点云输出平面中心、长短边、主方向及平面误差。"""
    if not np.isfinite(table_top_height_m):
        raise ValueError("桌面高度必须是有限的世界坐标米数")
    known_height = geometry_config["known_object_height_m"]
    trim = geometry_config["trim_percentile"]
    if known_height <= 0 or not 0 <= trim < 25:
        raise ValueError("已知物体高度须为正，尺寸裁剪分位数须介于 [0,25)")
    top_points = _select_target_top_points(
        rgb, cloud, observation, vision_config, table_top_height_m, geometry_config,
    )
    return _estimate_from_top_points(
        top_points, table_top_height_m, known_height,
        geometry_config["minimum_orientation_aspect_ratio"], trim,
        geometry_config["minimum_target_points"],
    )
