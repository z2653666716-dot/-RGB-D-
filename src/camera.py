"""固定 MuJoCo 相机的 RGB-D 成像与坐标换算。

本模块由后续的 vision.py 接收彩色图像、米制深度和有效深度掩码。
它不知道方块 ID 或真实位置，也不输出分割掩码；只有离线相机标定检查可以
使用人工构造的已知点。相机坐标统一使用常见视觉约定：X 向右、Y 向下、Z 向前。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """相机针孔内参与有效深度范围；长度用米、像素坐标用像素。

    fx/fy 是水平/垂直焦距（像素），cx/cy 是主点，near_m/far_m 是 MuJoCo
    视锥裁剪面相对距离（米）。
    """

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    near_m: float
    far_m: float


@dataclass(frozen=True)
class CameraPose:
    """固定相机在世界坐标系中的位置与朝向，输入单位均为米。"""

    position_world_m: np.ndarray
    rotation_camera_cv_to_world: np.ndarray


@dataclass(frozen=True)
class RobotBasePose:
    """机器人基座在世界坐标系中的位置与朝向。"""

    position_world_m: np.ndarray
    rotation_base_to_world: np.ndarray


@dataclass(frozen=True)
class RgbdFrame:
    """一次同步 RGB-D 曝光；depth_m 是沿相机光轴测量的米制 Z 深度。"""

    rgb: np.ndarray
    depth_m: np.ndarray
    valid_depth: np.ndarray


def camera_calibration(
    model: mujoco.MjModel,
    camera_name: str,
    width_px: int,
    height_px: int,
) -> CameraCalibration:
    """由模型相机的垂直视场角和图像尺寸生成内参及米制裁剪范围。

    假设像素为正方形，fx=fy；MuJoCo 的 cam_fovy 单位为角度。
    输入相机名必须在 MJCF 中存在，否则显式报错，不猜测相机编号。
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("相机图像宽和高必须是正整数")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"MJCF 中不存在名为 {camera_name!r} 的相机")
    fovy_radians = np.deg2rad(float(model.cam_fovy[camera_id]))
    if not 0 < fovy_radians < np.pi:
        raise ValueError("相机垂直视场角必须介于 0 与 180 度之间")

    # 垂直 FOV 对应整张图像高度：fy = (图像高度/2) / tan(FOV/2)。
    focal_length_px = height_px / (2.0 * np.tan(fovy_radians / 2.0))
    # MuJoCo 以场景尺度乘裁剪参数设置近远面；这里读取模型实际的米制界限。
    scene_extent_m = float(model.stat.extent)
    near_m = float(model.vis.map.znear) * scene_extent_m
    far_m = float(model.vis.map.zfar) * scene_extent_m
    if not 0 < near_m < far_m:
        raise ValueError("模型相机裁剪范围无效，必须满足 0 < near < far")
    return CameraCalibration(
        width_px=width_px, height_px=height_px,
        fx_px=float(focal_length_px), fy_px=float(focal_length_px),
        cx_px=width_px / 2.0, cy_px=height_px / 2.0,
        near_m=near_m, far_m=far_m,
    )


def camera_pose_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
) -> CameraPose:
    """读取 MuJoCo 相机位姿并转换成视觉相机约定，输出旋转矩阵和世界位置。

    MuJoCo 相机局部坐标为 X 向右、Y 向上、看向局部 -Z；视觉相机坐标为
    X 向右、Y 向下、看向 +Z。绕 X 轴翻转 Y/Z 后，两者可正确对应。
    data 必须对应已经运行过 mj_forward/mj_step 的 model。
    """
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"MJCF 中不存在名为 {camera_name!r} 的相机")
    rotation_mujoco_to_world = data.cam_xmat[camera_id].reshape(3, 3).copy()
    # 视觉相机到 MuJoCo 相机约定的转换是右乘 diag(1,-1,-1)，
    # 等价于翻转旋转矩阵的第 2、3 列。逐列处理可避开当前 Windows 环境
    # 中 NumPy 小矩阵 BLAS 乘法触发的原生进程崩溃，数值意义保持相同。
    rotation_cv_to_world = rotation_mujoco_to_world.copy()
    rotation_cv_to_world[:, 1:] *= -1.0
    return CameraPose(
        position_world_m=data.cam_xpos[camera_id].copy(),
        rotation_camera_cv_to_world=rotation_cv_to_world,
    )


def pixels_to_camera(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray | float,
    calibration: CameraCalibration,
) -> np.ndarray:
    """把像素和对应米制 Z 深度反投影到相机坐标，输入 ``[...,2]``，输出 ``[...,3]``。

    深度是从相机平面向前的距离，不是沿像素射线的斜距。非法/缺失深度会
    明确报错，调用方应先从有效区域选取深度；本函数不使用真值替代坏观测。
    """
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    if pixels.ndim < 1 or pixels.shape[-1] != 2:
        raise ValueError("pixels_xy 最后一维必须是像素坐标 [u, v]")
    depth = np.asarray(depth_m, dtype=np.float64)
    try:
        depth = np.broadcast_to(depth, pixels.shape[:-1])
    except ValueError as error:
        raise ValueError("深度数量必须与像素点数量一致，或可广播到像素数组") from error
    if not np.isfinite(pixels).all():
        raise ValueError("像素位置必须是有限数值")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("反投影所需的深度必须是有限的正米数")

    u, v = pixels[..., 0], pixels[..., 1]
    z = depth
    x = (u - calibration.cx_px) * z / calibration.fx_px
    y = (v - calibration.cy_px) * z / calibration.fy_px
    return np.stack((x, y, z), axis=-1)


def camera_to_world(points_camera: np.ndarray, pose: CameraPose) -> np.ndarray:
    """把 ``[...,3]`` 的视觉相机坐标点变换为 MuJoCo 世界坐标，单位保持为米。"""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("相机坐标点须是有限的 [..., 3] 数组")
    return (np.einsum("ij,...j->...i", pose.rotation_camera_cv_to_world, points)
            + pose.position_world_m)


def world_to_camera(points_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    """把 ``[...,3]`` 世界坐标点变换到视觉相机坐标，供规划与标定检查共用。"""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("世界坐标点须是有限的 [..., 3] 数组")
    relative = points - pose.position_world_m
    return np.einsum("ij,...j->...i", pose.rotation_camera_cv_to_world.T, relative)


def robot_base_pose_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = "base",
) -> RobotBasePose:
    """读取机器人基座世界位姿；输入默认使用本项目 FR3 的 ``base`` 刚体名称。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MJCF 中不存在名为 {body_name!r} 的机器人基座刚体")
    return RobotBasePose(
        position_world_m=data.xpos[body_id].copy(),
        rotation_base_to_world=data.xmat[body_id].reshape(3, 3).copy(),
    )


def world_to_robot_base(points_world: np.ndarray, pose: RobotBasePose) -> np.ndarray:
    """将世界坐标点转成 FR3 机器人基座坐标，支持一个点或 ``[...,3]`` 多个点。"""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("世界坐标点须是有限的 [..., 3] 数组")
    return np.einsum("ij,...j->...i", pose.rotation_base_to_world.T,
                     points - pose.position_world_m)


def robot_base_to_world(points_base: np.ndarray, pose: RobotBasePose) -> np.ndarray:
    """把机器人基座坐标点换回 MuJoCo 世界坐标，方便对照验证变换方向。"""
    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("机器人基座坐标点须是有限的 [..., 3] 数组")
    return (np.einsum("ij,...j->...i", pose.rotation_base_to_world, points)
            + pose.position_world_m)


def camera_to_pixels(
    points_camera: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    """把相机坐标点投影至左上角为原点的图像像素，输出 ``[...,2]`` 的 ``[u,v]``。"""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("相机坐标点须是有限的 [..., 3] 数组")
    z = points[..., 2]
    if np.any(z <= 0):
        raise ValueError("相机后方或相机平面上的点不能投影成有效图像位置")
    u = calibration.fx_px * points[..., 0] / z + calibration.cx_px
    v = calibration.fy_px * points[..., 1] / z + calibration.cy_px
    return np.stack((u, v), axis=-1)


class Camera:
    """MuJoCo 相机的受管 RGB-D 渲染器；输入 model/data 和相机参数，输出同步图像帧。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str,
        width_px: int,
        height_px: int,
    ):
        self.model = model
        self.camera_name = camera_name
        self.calibration = camera_calibration(model, camera_name, width_px, height_px)
        self.renderer = mujoco.Renderer(model, height=height_px, width=width_px)

    def capture(self, data: mujoco.MjData) -> RgbdFrame:
        """输入当前仿真状态，输出无分割通道的 RGB、米制光轴深度和有效掩码。

        官方 Renderer 会在内部将深度缓冲反算成米；不能再做一次 PyBullet
        near/far 转换。内置渲染器为提升深度精度会临时使用 ID 色渲染，但本模块
        只返回正常 RGB 与深度，不读取、不输出、不用于检测任何分割 ID 或 mask。
        """
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            # MuJoCo 3.3.7 在 Renderer.render 内部将 OpenGL 深度值换算为米。
            depth_m = self.renderer.render().copy()
        finally:
            # 保证下一次曝光先回到普通 RGB 渲染，即使深度渲染时报错也要复位。
            self.renderer.disable_depth_rendering()
        if rgb.shape != (self.calibration.height_px, self.calibration.width_px, 3):
            raise RuntimeError(f"RGB 图像尺寸异常：{rgb.shape}")
        if depth_m.shape != (self.calibration.height_px, self.calibration.width_px):
            raise RuntimeError(f"深度图尺寸异常：{depth_m.shape}")

        # 零/非有限值以及远裁剪平面的背景都不是有效目标深度。
        valid = (np.isfinite(depth_m) & (depth_m > self.calibration.near_m)
                 & (depth_m < self.calibration.far_m))
        return RgbdFrame(rgb=rgb, depth_m=depth_m, valid_depth=valid)

    def close(self) -> None:
        """释放相机 OpenGL 资源；离开 with 区域时自动调用。"""
        self.renderer.close()

    def __enter__(self) -> Camera:
        """支持 with Camera(...) 的资源管理写法。"""
        return self

    def __exit__(self, error_type, error, traceback):
        """离开 with 时释放 GPU/桌面 OpenGL 上下文，不吞掉原始异常。"""
        self.close()
