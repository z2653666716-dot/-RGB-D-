"""M2 目标视觉观测：颜色分割、方块轮廓筛选和 RGB-D 三维位置计算。

本模块仅接收相机图像、米制深度、相机/机器人基座位姿和公开实验配置。
它没有 MuJoCo model/data、仿真物体 ID 或方块真值接口，因此无法从仿真器读取
目标真实位置。失败观测返回原因与空位置，不能用真值或旧位置补齐。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera import (
    CameraCalibration,
    CameraPose,
    RobotBasePose,
    camera_to_world,
    pixels_to_camera,
    world_to_robot_base,
)


@dataclass(frozen=True)
class PositionObservation:
    """一个目标检测结果及其定位有效性；位置为空即代表没有有效视觉观测。"""

    valid: bool
    reason: str
    bbox_xywh_px: tuple[int, int, int, int] | None
    center_pixel_uv_px: tuple[float, float] | None
    contour_area_px: float
    valid_depth_fraction: float
    surface_point_camera_m: tuple[float, float, float] | None
    surface_point_world_m: tuple[float, float, float] | None
    object_center_world_m: tuple[float, float, float] | None
    object_center_robot_base_m: tuple[float, float, float] | None


def _failure(
    reason: str,
    bbox: tuple[int, int, int, int] | None = None,
    center: tuple[float, float] | None = None,
    area_px: float = 0.0,
    depth_fraction: float = 0.0,
) -> PositionObservation:
    """统一构造失败结果，保留诊断信息但绝不伪造位置。"""
    return PositionObservation(
        valid=False, reason=reason, bbox_xywh_px=bbox, center_pixel_uv_px=center,
        contour_area_px=float(area_px), valid_depth_fraction=float(depth_fraction),
        surface_point_camera_m=None, surface_point_world_m=None,
        object_center_world_m=None, object_center_robot_base_m=None,
    )


def _red_color_mask(rgb: np.ndarray, settings: dict) -> np.ndarray:
    """将 RGB 图像转成 HSV，合并红色跨 0/179 边界的两个色相区间。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    saturation = settings["minimum_saturation_0_to_255"]
    brightness = settings["minimum_brightness_0_to_255"]
    for hue_low, hue_high in settings["red_hue_ranges_0_to_179"]:
        lower = np.array([hue_low, saturation, brightness], dtype=np.uint8)
        upper = np.array([hue_high, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    # 开运算去除零散红色噪点，闭运算补上方块颜色区域内部的小空洞。
    kernel_size = settings["morphology_kernel_px"]
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("形态处理核尺寸须为正奇数，例如 3")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def colored_target_mask(rgb: np.ndarray, settings: dict) -> np.ndarray:
    """输出由真实 RGB 颜色阈值得到的 0/255 图像掩码，供几何模块筛目标点。

    这是图像算法对颜色的分割结果，不是 MuJoCo 的物体 ID 或理想分割通道。
    输入 RGB uint8 图像与 vision 配置，输出与图像同高宽的 uint8 掩码。
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("颜色分割输入必须是 RGB uint8 三通道图像")
    return _red_color_mask(rgb, settings)


def detect_colored_cube(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    valid_depth: np.ndarray,
    calibration: CameraCalibration,
    camera_pose: CameraPose,
    robot_base_pose: RobotBasePose,
    config: dict,
) -> PositionObservation:
    """从一帧 RGB-D 中估计已知尺寸红色方块的位置，输出有效观测或明确失败原因。

    输入图像由相机模块产生；config 只含已知方块尺寸、桌面和 HSV 阈值。
    算法依次做颜色阈值 → 形态处理 → 轮廓筛选 → 中心区有效深度中位数 →
    相机反投影 → 世界/机器人基座变换。物体中心 Z 由已知桌面高度和方块边长
    推出，假设是相机俯视、方块完整落在水平桌面上；顶面观测点不等于物体中心。
    """
    height, width = rgb.shape[:2] if rgb.ndim == 3 else (0, 0)
    if rgb.dtype != np.uint8 or rgb.shape != (calibration.height_px,
                                               calibration.width_px, 3):
        return _failure("invalid_rgb_shape_or_type")
    if (depth_m.shape != (height, width) or valid_depth.shape != (height, width)
            or not np.issubdtype(depth_m.dtype, np.floating)
            or valid_depth.dtype != np.bool_):
        return _failure("invalid_depth_shape_or_type")

    settings = config["vision"]
    mask = colored_target_mask(rgb, settings)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = height * width
    minimum_area = settings["minimum_contour_area_px"]
    maximum_area = settings["maximum_contour_area_fraction"] * image_area
    candidates = []
    saw_red_region = False
    aspect_low, aspect_high = settings["accepted_square_aspect_range"]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        saw_red_region = True
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width == 0 or box_height == 0 or area > maximum_area:
            continue
        aspect = box_width / box_height
        if not aspect_low <= aspect <= aspect_high:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        candidates.append((area, contour, (x, y, box_width, box_height), center))

    if not candidates:
        return _failure("target_shape_not_found" if saw_red_region else "no_target_detected")
    if len(candidates) > 1:
        # 场景假设只有一个目标；歧义时拒绝乱选，之后可由多目标模块单独扩展。
        return _failure("multiple_target_candidates")

    area, contour, bbox, center = candidates[0]
    x, y, box_width, box_height = bbox
    center_u, center_v = center
    center_fraction = settings["center_depth_region_fraction"]
    if not 0 < center_fraction <= 1:
        raise ValueError("中心深度窗口比例必须大于 0 且不超过 1")
    # 只看轮廓中央区域，避开斜边/像素混合处深度；同时要求窗口位于分割区域内。
    window_width = max(1, int(round(box_width * center_fraction)))
    window_height = max(1, int(round(box_height * center_fraction)))
    left = max(0, int(round(center_u - window_width / 2)))
    top = max(0, int(round(center_v - window_height / 2)))
    right = min(width, left + window_width)
    bottom = min(height, top + window_height)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(object_mask, [contour], -1, color=255, thickness=cv2.FILLED)
    center_region = np.zeros((height, width), dtype=bool)
    center_region[top:bottom, left:right] = object_mask[top:bottom, left:right] > 0
    expected_pixels = int(np.count_nonzero(center_region))
    if expected_pixels == 0:
        return _failure("empty_center_depth_region", bbox, center, area)
    depth_region = center_region & valid_depth & np.isfinite(depth_m)
    valid_count = int(np.count_nonzero(depth_region))
    valid_fraction = valid_count / expected_pixels
    if valid_fraction < settings["minimum_valid_depth_fraction"]:
        return _failure("insufficient_valid_target_depth", bbox, center, area, valid_fraction)

    # 中位数降低深度孔洞或少数边缘像素的影响；不使用区域外的桌面深度。
    median_depth = float(np.median(depth_m[depth_region]))
    if not calibration.near_m < median_depth < calibration.far_m:
        return _failure("target_depth_outside_camera_range", bbox, center, area, valid_fraction)
    pixel_uv = np.asarray([[center_u, center_v]], dtype=np.float64)
    point_camera = pixels_to_camera(pixel_uv, np.asarray([median_depth]), calibration)[0]
    surface_world = camera_to_world(point_camera, camera_pose)

    # 可见顶面中心与方块几何中心不同：已知水平桌面与边长给出中心高度。
    cube = config["cube"]
    table = config["table"]
    # 方块底面在桌面上，顶面比桌面高一个完整边长；中心只高出半个边长。
    target_height = cube.get("dimensions_xyz_m", [cube["side_m"]] * 3)[2]
    expected_surface_z = table["top_height_world_m"] + target_height
    if abs(surface_world[2] - expected_surface_z) > settings["maximum_top_height_error_m"]:
        return _failure("surface_height_inconsistent", bbox, center, area, valid_fraction)
    center_world = surface_world.copy()
    center_world[2] = expected_surface_z - target_height / 2

    bounds = settings["workspace_bounds_xy_world_m"]
    if not (bounds["x"][0] <= center_world[0] <= bounds["x"][1]
            and bounds["y"][0] <= center_world[1] <= bounds["y"][1]):
        return _failure("position_outside_workspace", bbox, center, area, valid_fraction)
    center_base = world_to_robot_base(center_world, robot_base_pose)
    return PositionObservation(
        valid=True, reason="ok", bbox_xywh_px=bbox,
        center_pixel_uv_px=(float(center_u), float(center_v)),
        contour_area_px=area, valid_depth_fraction=float(valid_fraction),
        surface_point_camera_m=tuple(float(value) for value in point_camera),
        surface_point_world_m=tuple(float(value) for value in surface_world),
        object_center_world_m=tuple(float(value) for value in center_world),
        object_center_robot_base_m=tuple(float(value) for value in center_base),
    )
