"""从米制深度图生成并预处理相机/世界/FR3 基座坐标系点云。

深度每个像素通过相机内参反投影，再使用已知相机外参变换到世界和机器人基座。
无效深度在原始掩码中保留，坐标数组对应位置填 NaN；预处理只使用空间裁剪、
体素质心下采样和邻域占用过滤，不读取仿真物体 ID 或方块真实位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera import CameraCalibration, CameraPose, RobotBasePose


@dataclass(frozen=True)
class OrganizedPointCloud:
    """保持图像高宽排列的点云；像素 (v,u) 对应同位置的空间点。"""

    points_camera_m: np.ndarray
    points_world_m: np.ndarray
    points_robot_base_m: np.ndarray
    valid_depth_mask: np.ndarray


@dataclass(frozen=True)
class ProcessedPointCloud:
    """裁剪、下采样及邻域清理后的点；voxel_counts 表示每个质心代表的像素数。"""

    points_world_m: np.ndarray
    points_robot_base_m: np.ndarray
    voxel_counts: np.ndarray
    voxel_indices: np.ndarray
    input_point_count: int
    workspace_point_count: int
    output_voxel_count: int


def _transform_points(points: np.ndarray, rotation: np.ndarray,
                      translation: np.ndarray) -> np.ndarray:
    """用逐轴乘加变换 N 个三维点，保持单位米并避免小矩阵 BLAS 调用。"""
    output = np.empty_like(points, dtype=np.float64)
    for axis in range(3):
        output[:, axis] = (
            points[:, 0] * rotation[axis, 0]
            + points[:, 1] * rotation[axis, 1]
            + points[:, 2] * rotation[axis, 2]
            + translation[axis]
        )
    return output


def _transform_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """用逐项乘加旋转一个三维向量，避免小型 BLAS 矩阵乘法。"""
    return np.array([
        sum(rotation[axis, column] * vector[column] for column in range(3))
        for axis in range(3)
    ], dtype=np.float64)


def depth_to_point_cloud(
    depth_m: np.ndarray,
    valid_depth: np.ndarray,
    calibration: CameraCalibration,
    camera_pose: CameraPose,
    robot_base_pose: RobotBasePose,
) -> OrganizedPointCloud:
    """输入一帧米制 Z 深度和相机标定，输出三个坐标系的有序点云及有效掩码。"""
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(valid_depth)
    expected_shape = (calibration.height_px, calibration.width_px)
    if depth.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(f"深度和有效掩码须为图像形状 {expected_shape}")
    if mask.dtype != np.bool_:
        raise ValueError("有效深度掩码必须是布尔数组")

    # 深度相机只提供沿光轴的 Z 米数；视锥外、非数值和零深度不参加反投影。
    valid = (mask & np.isfinite(depth) & (depth > calibration.near_m)
             & (depth < calibration.far_m))
    points_camera = np.full((*expected_shape, 3), np.nan, dtype=np.float64)
    points_world = np.full_like(points_camera, np.nan)
    points_base = np.full_like(points_camera, np.nan)
    rows, columns = np.nonzero(valid)
    if rows.size:
        z = depth[rows, columns]
        x = (columns - calibration.cx_px) * z / calibration.fx_px
        y = (rows - calibration.cy_px) * z / calibration.fy_px
        camera_points = np.column_stack((x, y, z))
        world_points = _transform_points(
            camera_points, camera_pose.rotation_camera_cv_to_world,
            camera_pose.position_world_m,
        )
        base_points = _transform_points(
            world_points, robot_base_pose.rotation_base_to_world.T,
            -_transform_vector(robot_base_pose.rotation_base_to_world.T,
                               robot_base_pose.position_world_m),
        )
        points_camera[rows, columns] = camera_points
        points_world[rows, columns] = world_points
        points_base[rows, columns] = base_points
    return OrganizedPointCloud(points_camera, points_world, points_base, valid.copy())


def _voxel_centroids(points: np.ndarray, voxel_size_m: float):
    """按体素索引分组并求每格的点质心和原始点数。"""
    voxel_indices = np.floor(points / voxel_size_m).astype(np.int64)
    unique_voxels, inverse, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True,
    )
    centroids = np.empty((len(unique_voxels), 3), dtype=np.float64)
    for axis in range(3):
        sums = np.bincount(inverse, weights=points[:, axis], minlength=len(unique_voxels))
        centroids[:, axis] = sums / counts
    return centroids, unique_voxels, counts


def _neighbor_occupancy_mask(voxel_indices: np.ndarray,
                             minimum_neighbor_voxels: int) -> np.ndarray:
    """过滤空间上孤立的体素；邻居数包含该体素自身，最多检查周围 26 格。"""
    occupied = {tuple(int(value) for value in voxel) for voxel in voxel_indices}
    keep = np.zeros(len(voxel_indices), dtype=bool)
    for index, voxel in enumerate(voxel_indices):
        x, y, z = (int(value) for value in voxel)
        neighbors = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (x + dx, y + dy, z + dz) in occupied:
                        neighbors += 1
        keep[index] = neighbors >= minimum_neighbor_voxels
    return keep


def preprocess_point_cloud(
    points_world_m: np.ndarray,
    robot_base_pose: RobotBasePose,
    workspace_bounds_world_m: dict,
    voxel_size_m: float,
    minimum_neighbor_voxels: int = 2,
) -> ProcessedPointCloud:
    """裁剪工作空间并体素下采样，输出世界/基座点、质心权重及处理数量。"""
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("输入点云必须是 [N,3] 世界坐标米制数组")
    if voxel_size_m <= 0 or not np.isfinite(voxel_size_m):
        raise ValueError("体素边长必须是有限的正米数")
    if minimum_neighbor_voxels < 1 or minimum_neighbor_voxels > 27:
        raise ValueError("最小邻居体素数必须介于 1 和 27 之间")
    for axis_name in ("x", "y", "z"):
        bounds = workspace_bounds_world_m.get(axis_name)
        if (bounds is None or len(bounds) != 2 or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1]):
            raise ValueError(f"工作空间须为有效的 {axis_name} 坐标 [最小值, 最大值]")

    finite = np.isfinite(points).all(axis=1)
    within = finite.copy()
    for axis, axis_name in enumerate(("x", "y", "z")):
        lower, upper = workspace_bounds_world_m[axis_name]
        within &= (points[:, axis] >= lower) & (points[:, axis] <= upper)
    cropped = points[within]
    if cropped.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float64)
        return ProcessedPointCloud(empty_points, empty_points.copy(),
                                   np.empty(0, dtype=np.int64),
                                   np.empty((0, 3), dtype=np.int64),
                                   len(points), 0, 0)

    centroids, voxel_indices, counts = _voxel_centroids(cropped, voxel_size_m)
    keep = _neighbor_occupancy_mask(voxel_indices, minimum_neighbor_voxels)
    centroids = centroids[keep]
    voxel_indices = voxel_indices[keep]
    counts = counts[keep]
    rotation_world_to_base = robot_base_pose.rotation_base_to_world.T
    translation_world_to_base = -_transform_vector(
        rotation_world_to_base, robot_base_pose.position_world_m)
    base_points = _transform_points(centroids, rotation_world_to_base,
                                    translation_world_to_base)
    return ProcessedPointCloud(
        points_world_m=centroids,
        points_robot_base_m=base_points,
        voxel_counts=counts,
        voxel_indices=voxel_indices,
        input_point_count=len(points),
        workspace_point_count=len(cropped),
        output_voxel_count=len(centroids),
    )
