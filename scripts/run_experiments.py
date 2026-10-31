"""可复现配对抓取批量实验入口：重置场景、运行方法组合并保存 JSON/CSV。

每个场景与随机种子都会为每种估计器/抓取策略重新创建 MuJoCo 仿真。各方法
使用相同的 RGB-D 帧、相机测点噪声和独立的末端执行目标偏移；真值只用于评估。
输出包含配置快照、逐次结果、失败原因和观测序列摘要，便于复核配对公平性。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

# 直接运行 scripts 下入口时，显式加入项目源码目录，免去手动设置 PYTHONPATH。
PROJECT_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from camera import Camera, CameraPose, camera_pose_world, robot_base_pose_world
from evaluation import (
    capture_grasp_sample,
    load_evaluation_config,
    run_and_evaluate_candidate,
    run_and_evaluate_grasp,
)
from geometry import estimate_planar_object_geometry
from grasp_filter import filter_candidates_by_ik_and_path
from grasp_planner import generate_grasp_candidates
from point_cloud import depth_to_point_cloud
from simulation import PROJECT_ROOT, Simulation, load_config
from state_estimator import (
    PositionEstimate,
    compare_estimators,
    inject_position_noise_sequence,
)
from relative_grasp_estimator import (
    RelativeGraspKalmanFilter,
    target_measurement_from_camera,
)
from vision import detect_colored_cube


def _read_json(path: Path) -> dict:
    """从 UTF-8 JSON 文件读取实验参数字典。"""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """将配置中的相对路径解析到项目根目录，绝对路径原样保留。"""
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def _observation_digest(observations: list[np.ndarray | None]) -> str:
    """对某条件的带噪位置序列求 SHA-256 摘要，用于验证方法间输入一致。"""
    digest = hashlib.sha256()
    for position in observations:
        if position is None:
            digest.update(b"missing;")
        else:
            digest.update(np.asarray(position, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _camera_observation_sequence(
    surface_points_camera_m: list[np.ndarray],
    pose: CameraPose,
    known_center_height_world_m: float,
    sensor_noise_std_m: float,
    sensor_noise_axis_scales,
    noise_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """输入视觉顶面测点及噪声参数，返回带噪相机点、世界中心及相机测量协方差。

    噪声只在相机 XYZ 点上采样一次，再通过相机位姿变换；各估计器共享这批点。
    axis_scales 可表达深度轴与横向不同的误差大小，例如 [1,1,2]。
    这里是可控的三维测点扰动，没有伪称它来自真实传感器标定或像素级图像噪声。
    """
    points = np.asarray(surface_points_camera_m, dtype=np.float64)
    scales = np.asarray(sensor_noise_axis_scales, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("相机测点必须是有限的 [N,3] 数组")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("sensor_noise_axis_scales 必须为三个正数")
    if not np.isfinite(sensor_noise_std_m) or sensor_noise_std_m < 0:
        raise ValueError("传感器位置噪声标准差必须非负且有限")
    axis_std_m = float(sensor_noise_std_m) * scales
    generator = np.random.default_rng(noise_seed)
    noisy_points = points + generator.normal(0.0, axis_std_m, size=points.shape)
    covariance_camera = np.diag(axis_std_m ** 2)
    positions_world = []
    for point in noisy_points:
        # 均值与协方差使用同一个变换；中心高度采用公开桌面和物体尺寸假设。
        measurement = target_measurement_from_camera(
            point, pose.rotation_camera_cv_to_world, pose.position_world_m,
            covariance_camera, known_center_height_world_m=known_center_height_world_m)
        positions_world.append(np.asarray(measurement.target_position_world_m))
    return list(noisy_points), positions_world, covariance_camera


def _estimate_with_relative_kalman(
    noisy_surface_points_camera_m: list[np.ndarray],
    gripper_position_world_m: np.ndarray,
    pose: CameraPose,
    sensor_covariance_camera_m2: np.ndarray,
    known_center_height_world_m: float,
    sample_period_seconds: float,
    filter_config: dict,
) -> tuple[list[PositionEstimate], dict]:
    """输入带噪相机点及其协方差，输出位置估计序列和可复核的逐帧滤波记录。

    第一帧以其测量 R 初始化均值与 P，从第二帧开始融合，不重复使用第一帧。
    采集期间机器人没有运动，因此每帧 Q=0。随后执行产生的独立运动偏移，
    由 _run_one_trial 在执行边界采样并将 Q 加到执行位置误差的预测协方差。
    """
    if not noisy_surface_points_camera_m:
        return [], {"updates": []}
    minimum_std_m = float(filter_config["minimum_sensor_position_std_m"])
    if not np.isfinite(minimum_std_m) or minimum_std_m <= 0:
        raise ValueError("minimum_sensor_position_std_m 必须为有限正数")
    covariance_camera = np.asarray(sensor_covariance_camera_m2, dtype=np.float64).copy()
    # 方差下限只影响滤波器的置信度，不改变实际随机噪声样本。
    for axis in range(3):
        covariance_camera[axis, axis] = max(covariance_camera[axis, axis], minimum_std_m ** 2)
    first = target_measurement_from_camera(
        noisy_surface_points_camera_m[0], pose.rotation_camera_cv_to_world,
        pose.position_world_m, covariance_camera,
        known_center_height_world_m=known_center_height_world_m)
    initial_position = np.asarray(first.target_position_world_m)
    initial_covariance = np.asarray(first.covariance_world_m2)
    initial_relative = initial_position - gripper_position_world_m
    filter_instance = RelativeGraspKalmanFilter(
        initial_relative, initial_covariance,
        initial_gripper_position_world_m=gripper_position_world_m)
    estimates = [PositionEstimate(
        valid=True, reason="initialized_once_from_first_camera_observation",
        timestamp_seconds=0.0, position_world_m=first.target_position_world_m,
        velocity_world_m_s=None,
        position_variance_m2=tuple(float(value) for value in np.diag(initial_covariance)))]
    updates = [{
        "frame_index": 0, "phase": "initialization",
        "relative_position_world_m": initial_relative.tolist(),
        "target_position_world_m": first.target_position_world_m,
        "covariance_world_m2": first.covariance_world_m2,
        "measurement_covariance_world_m2": first.covariance_world_m2,
        "motion_covariance_world_m2": np.zeros((3, 3)).tolist(),
        "predicted_covariance_world_m2": None,
        "kalman_gain": None, "innovation_world_m": None,
    }]
    for frame_index, point in enumerate(noisy_surface_points_camera_m[1:], start=1):
        result = filter_instance.update_from_camera(
            point, pose.rotation_camera_cv_to_world, pose.position_world_m,
            covariance_camera, gripper_position_world_m,
            known_center_height_world_m=known_center_height_world_m)
        estimates.append(PositionEstimate(
            valid=True, reason="relative_position_updated_from_camera",
            timestamp_seconds=frame_index * sample_period_seconds,
            position_world_m=result.target_position_world_m, velocity_world_m_s=None,
            position_variance_m2=tuple(float(value) for value in np.diag(result.covariance_world_m2))))
        # asdict 把结果类变为普通字典，保存完整矩阵，避免只存对角线丢失相关性。
        updates.append({"frame_index": frame_index, "phase": "measurement_update", **asdict(result)})
    return estimates, {
        "assumed_sensor_covariance_camera_m2": covariance_camera.tolist(),
        "sensor_jacobian_world": first.sensor_jacobian_world,
        "pose_jacobian_world": first.pose_jacobian_world,
        "assumed_measurement_covariance_world_m2": first.covariance_world_m2,
        "updates": updates,
    }


def _run_one_trial(
    project_root: Path,
    batch_config: dict,
    scenario: dict,
    scene_seed: int,
    estimator_name: str,
    grasp_strategy: str,
    scenario_index: int,
    position_noise_std_m: float,
    motion_position_offset_std_m: float,
) -> dict:
    """运行一个独立配对试次；输入场景/种子/方法，输出一条可追溯结果记录。"""
    started = time.perf_counter()
    scene_path = _resolve_project_path(project_root, batch_config["scene_config_path"])
    estimator_path = _resolve_project_path(project_root, batch_config["estimator_config_path"])
    geometry_path = _resolve_project_path(project_root, batch_config["geometry_config_path"])
    planner_path = _resolve_project_path(project_root, batch_config["planner_config_path"])
    filter_path = _resolve_project_path(project_root, batch_config["filter_config_path"])
    evaluation_path = _resolve_project_path(project_root, batch_config["evaluation_config_path"])
    relative_filter_config = None
    if estimator_name == "relative_kalman":
        relative_filter_path = _resolve_project_path(
            project_root, batch_config["relative_grasp_filter_config_path"])
        relative_filter_config = _read_json(relative_filter_path)

    scene_config = load_config(scene_path)
    scene_config["seed"] = int(scene_seed)
    scene_config["cube"]["center_xy_world_m"] = scenario["center_xy_world_m"]
    scene_config["cube"]["dimensions_xyz_m"] = scenario["dimensions_xyz_m"]
    scene_config["cube"]["side_m"] = float(scenario["dimensions_xyz_m"][2])
    scene_config["cube"]["yaw_world_rad"] = float(np.deg2rad(scenario["yaw_world_degrees"]))
    scene_config["cube"]["xy_jitter_m"] = float(batch_config["cube_xy_jitter_m"])
    scene_config["visual_distractors"] = scenario.get("visual_distractors", [])
    # 批量场景可能是长方体，使用任务声明的尺寸范围扩展方形外观筛选。
    scene_config["vision"]["accepted_square_aspect_range"] = [0.35, 2.5]

    estimator_config = _read_json(estimator_path)
    geometry_config = _read_json(geometry_path)
    geometry_config["known_object_height_m"] = float(scenario["dimensions_xyz_m"][2])
    planner_config = _read_json(planner_path)
    filter_config = _read_json(filter_path)
    evaluation_config = load_evaluation_config(evaluation_path)

    noise_label = f"{position_noise_std_m:.4f}".replace(".", "p")
    motion_label = f"{motion_position_offset_std_m:.4f}".replace(".", "p")
    pair_id = (f"{scenario['name']}__seed_{scene_seed}__sensor_{noise_label}"
               f"__motion_{motion_label}")
    run_id = f"{pair_id}__{estimator_name}__{grasp_strategy}"
    sequence_digest = None
    detected_frame_count = 0
    geometry = None
    point_cloud = None
    last_valid_observation = None
    last_valid_rgb = None
    ground_truth_for_evaluation = None
    execution = None
    evaluation = None
    generated_candidate_count = 0
    feasible_candidate_count = 0
    selected_candidate_id = None
    failure_reason = ""
    estimate_position = None
    position_error = None
    sensor_noise_frame = batch_config.get("sensor_noise_coordinate_frame", "world")
    sensor_covariance_camera = None
    measurement_covariance_world = None
    relative_filter_diagnostics = None
    camera_sequence_digest = None
    execution_error_covariance = None
    execution_target_error = None
    grasp_target_position = None
    motion_covariance_world = np.eye(3) * float(motion_position_offset_std_m) ** 2
    motion_offset_world_m = np.zeros(3, dtype=np.float64)
    position_rmse = None
    mean_position_bias = None
    static_jitter = None
    valid_estimates = []

    try:
        with Simulation(scene_config) as simulation:
            simulation.step(scene_config["physics"]["settle_steps"])
            # 评估真值在独立变量中保管；控制分支只使用下面的 RGB-D 观测结果。
            ground_truth_for_evaluation = np.asarray(
                capture_grasp_sample(simulation.model, simulation.data)
                .cube_position_world_m, dtype=np.float64,
            )
            simulation.close()
            vision_config = scene_config["vision"]
            camera_config = scene_config["camera"]
            positions = []
            surface_points_camera = []
            with Camera(simulation.model, camera_config["name"],
                        camera_config["width_px"], camera_config["height_px"]) as camera:
                pose = camera_pose_world(simulation.model, simulation.data, camera_config["name"])
                base_pose = robot_base_pose_world(simulation.model, simulation.data)
                for _ in range(int(batch_config["observation_frame_count"])):
                    frame = camera.capture(simulation.data)
                    observation = detect_colored_cube(
                        frame.rgb, frame.depth_m, frame.valid_depth, camera.calibration,
                        pose, base_pose, scene_config,
                    )
                    if observation.valid and observation.object_center_world_m:
                        positions.append(observation.object_center_world_m)
                        surface_points_camera.append(np.asarray(observation.surface_point_camera_m))
                        detected_frame_count += 1
                        last_valid_observation = observation
                        # 保留与最后一条有效检测严格对应的图像和点云。
                        last_valid_rgb = frame.rgb.copy()
                        point_cloud = depth_to_point_cloud(
                            frame.depth_m, frame.valid_depth, camera.calibration,
                            pose, base_pose,
                        )
            if not positions:
                failure_reason = "target_not_detected"
                raise RuntimeError(failure_reason)

            noise_seed = int(scene_seed + batch_config["noise_seed_offset"] + scenario_index)
            known_center_height = (float(scene_config["table"]["top_height_world_m"])
                                   + float(scenario["dimensions_xyz_m"][2]) / 2.0)
            if sensor_noise_frame == "camera":
                noisy_camera_points, noisy_positions, sensor_covariance_camera = _camera_observation_sequence(
                    surface_points_camera, pose, known_center_height, position_noise_std_m,
                    batch_config.get("sensor_noise_axis_scales", [1.0, 1.0, 1.0]), noise_seed)
                camera_sequence_digest = _observation_digest(noisy_camera_points)
                first_measurement = target_measurement_from_camera(
                    noisy_camera_points[0], pose.rotation_camera_cv_to_world,
                    pose.position_world_m, sensor_covariance_camera,
                    known_center_height_world_m=known_center_height)
                measurement_covariance_world = np.asarray(first_measurement.covariance_world_m2)
            else:
                # 旧实验配置仍保持原有的世界 XYZ 点扰动，便于复现已有结果。
                noisy_positions = inject_position_noise_sequence(positions, position_noise_std_m, noise_seed)
                measurement_covariance_world = np.eye(3) * float(position_noise_std_m) ** 2
            sequence_digest = _observation_digest(noisy_positions)
            timestamps = (np.arange(len(noisy_positions), dtype=np.float64)
                          * estimator_config["sample_period_seconds"])
            estimates_by_method = compare_estimators(noisy_positions, timestamps, estimator_config)
            if estimator_name == "relative_kalman":
                site_id = simulation.model.site("attachment_site").id
                gripper_position = simulation.data.site_xpos[site_id].copy()
                relative_estimates, relative_filter_diagnostics = _estimate_with_relative_kalman(
                    noisy_camera_points, gripper_position, pose, sensor_covariance_camera,
                    known_center_height, float(estimator_config["sample_period_seconds"]),
                    relative_filter_config)
                estimates_by_method[estimator_name] = relative_estimates
            selected_estimate = estimates_by_method[estimator_name][-1]
            if not selected_estimate.valid or selected_estimate.position_world_m is None:
                failure_reason = f"estimator_invalid:{selected_estimate.reason}"
                raise RuntimeError(failure_reason)
            estimate_position = np.asarray(selected_estimate.position_world_m, dtype=np.float64)
            # 运动噪声与传感器噪声使用不同随机种子；一组配对方法共享同一 XYZ 样本。
            motion_seed = int(scene_seed + batch_config.get("motion_seed_offset", 26000) + scenario_index)
            motion_generator = np.random.default_rng(motion_seed)
            motion_offset_world_m = motion_generator.normal(0.0, motion_position_offset_std_m, size=3)
            # 受控的末端执行目标偏移：近似一次动作的终点位置误差，不冒充全程力/关节扰动。
            # 未观测到这个随机样本，估计器不使用它补偿目标位置。
            grasp_target_position = estimate_position + motion_offset_world_m
            if relative_filter_diagnostics is not None:
                last_covariance = np.asarray(relative_filter_diagnostics["updates"][-1]["covariance_world_m2"])
            elif selected_estimate.position_variance_m2 is not None:
                last_covariance = np.diag(selected_estimate.position_variance_m2)
            else:
                last_covariance = measurement_covariance_world
            # 此处是观测结束、执行开始的边界：独立运动误差 Q 仅计一次。
            # 后面没有视觉反馈，因此没有再做一次 Kalman 修正来缩小这个 Q。
            execution_error_covariance = last_covariance + motion_covariance_world
            execution_target_error = float(np.linalg.norm(grasp_target_position - ground_truth_for_evaluation))
            position_error = float(np.linalg.norm(estimate_position - ground_truth_for_evaluation))
            valid_estimates = [item for item in estimates_by_method[estimator_name]
                               if item.valid and item.position_world_m is not None]
            estimate_positions = np.asarray(
                [item.position_world_m for item in valid_estimates], dtype=np.float64)
            position_errors = estimate_positions - ground_truth_for_evaluation
            position_rmse = float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1))))
            mean_position_bias = np.mean(position_errors, axis=0)
            temporal_center = np.mean(estimate_positions, axis=0)
            static_jitter = float(np.sqrt(np.mean(np.sum(
                (estimate_positions - temporal_center) ** 2, axis=1))))

            if grasp_strategy == "fixed_pose":
                site_id = simulation.model.site("attachment_site").id
                tool_orientation = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(tool_orientation, simulation.data.site_xmat[site_id])
                trial = run_and_evaluate_grasp(
                    simulation.model, simulation.data, grasp_target_position,
                    float(scenario["dimensions_xyz_m"][2]),
                    scene_config["table"]["top_height_world_m"], tool_orientation,
                    evaluation_config,
                    trajectory_steps=int(batch_config["controller"]["trajectory_steps"]),
                    gripper_hold_steps=int(batch_config["controller"]["post_lift_hold_steps"]),
                )
            elif grasp_strategy == "geometry_candidate":
                geometry = estimate_planar_object_geometry(
                    last_valid_rgb, point_cloud, last_valid_observation, vision_config,
                    scene_config["table"]["top_height_world_m"], geometry_config,
                )
                if not geometry.valid or geometry.object_center_world_m is None:
                    failure_reason = f"geometry_invalid:{geometry.reason}"
                    raise RuntimeError(failure_reason)
                # 位置取当前滤波器结果；几何模块提供尺寸与方向，不回填仿真真值。
                geometry = replace(geometry,
                                   object_center_world_m=tuple(
                                       float(v) for v in grasp_target_position))
                generated = generate_grasp_candidates(geometry, planner_config)
                generated_candidate_count = len(generated.candidates)
                if not generated.success:
                    failure_reason = f"no_candidate:{generated.reason}"
                    raise RuntimeError(failure_reason)
                filtered = filter_candidates_by_ik_and_path(
                    simulation.model, simulation.data, geometry, generated, filter_config,
                )
                feasible_candidate_count = len(filtered.feasible_candidates)
                if not filtered.success:
                    failure_reason = f"no_ik_path_candidate:{filtered.reason}"
                    raise RuntimeError(failure_reason)
                chosen = filtered.feasible_candidates[0]
                selected_candidate_id = chosen.candidate_id
                trial = run_and_evaluate_candidate(
                    simulation.model, simulation.data, chosen, evaluation_config,
                    lift_distance_m=float(filter_config["lift_distance_m"]),
                    gripper_max_opening_m=float(planner_config["gripper_max_opening_m"]),
                    trajectory_steps=int(batch_config["controller"]["trajectory_steps"]),
                    gripper_settle_steps=int(batch_config["controller"]["gripper_settle_steps"]),
                    gripper_close_steps=int(batch_config["controller"]["gripper_close_steps"]),
                    post_lift_hold_steps=int(batch_config["controller"]["post_lift_hold_steps"]),
                )
            else:
                raise ValueError(f"未知抓取策略: {grasp_strategy}")
            execution, evaluation = trial.execution, trial.evaluation
            if not execution.completed:
                failure_reason = f"execution_failed:{execution.reason}"
            elif evaluation is None:
                failure_reason = "evaluation_missing"
            elif not evaluation.success:
                failure_reason = evaluation.reason
            else:
                failure_reason = ""
    except Exception as error:
        if not failure_reason:
            failure_reason = f"runtime_error:{type(error).__name__}:{error}"

    return {
        "run_id": run_id,
        "pair_id": pair_id,
        "scenario": scenario["name"],
        "seed": int(scene_seed),
        "estimator": estimator_name,
        "grasp_strategy": grasp_strategy,
        "observation_frame_count": int(batch_config["observation_frame_count"]),
        "valid_visual_frame_count": detected_frame_count,
        "position_noise_std_m": float(position_noise_std_m),
        "position_noise_level_m": float(position_noise_std_m),
        "motion_position_offset_std_m": float(motion_position_offset_std_m),
        "motion_noise_model": "independent_world_xyz_execution_target_offset",
        "motion_covariance_world_m2": motion_covariance_world.tolist(),
        "sensor_noise_coordinate_frame": sensor_noise_frame,
        "sensor_covariance_camera_m2": (None if sensor_covariance_camera is None else sensor_covariance_camera.tolist()),
        "measurement_covariance_world_m2": (None if measurement_covariance_world is None else measurement_covariance_world.tolist()),
        "camera_observation_sequence_sha256": camera_sequence_digest,
        "relative_filter_diagnostics": relative_filter_diagnostics,
        "execution_error_covariance_world_m2": (None if execution_error_covariance is None else execution_error_covariance.tolist()),
        "execution_target_position_world_m": (None if grasp_target_position is None else grasp_target_position.tolist()),
        "offline_execution_target_error_m": execution_target_error,
        "sampled_motion_position_offset_world_m": motion_offset_world_m.tolist(),
        "observation_sequence_sha256": sequence_digest,
        "estimated_position_world_m": None if estimate_position is None else estimate_position.tolist(),
        "offline_position_error_m": position_error,
        "position_rmse_3d_m": position_rmse,
        "mean_position_bias_xyz_m": (None if mean_position_bias is None
                                     else [float(value) for value in mean_position_bias]),
        "static_jitter_3d_m": static_jitter,
        "valid_estimator_frame_count": len(valid_estimates) or None,
        "geometry_reason": None if geometry is None else geometry.reason,
        "geometry_orientation_confident": None if geometry is None else geometry.orientation_confident,
        "generated_candidate_count": generated_candidate_count,
        "feasible_candidate_count": feasible_candidate_count,
        "selected_candidate_id": selected_candidate_id,
        "execution_completed": None if execution is None else execution.completed,
        "execution_reason": None if execution is None else execution.reason,
        "evaluation_success": None if evaluation is None else evaluation.success,
        "evaluation_reason": None if evaluation is None else evaluation.reason,
        "hold_duration_seconds": None if evaluation is None else evaluation.hold_duration_seconds,
        "maximum_lift_m": None if evaluation is None else evaluation.maximum_lift_m,
        "failure_reason": failure_reason,
        "success": bool(evaluation is not None and evaluation.success and execution is not None
                        and execution.completed),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }


def run_batch(config_path: Path, output_override: Path | None = None) -> dict:
    """运行所有场景/种子/估计器/策略组合，保存可追溯 JSON、CSV 和汇总。"""
    project_root = PROJECT_ROOT
    batch_config = _read_json(config_path)
    noise_frame = batch_config.get("sensor_noise_coordinate_frame", "world")
    if noise_frame not in ("world", "camera"):
        raise ValueError("sensor_noise_coordinate_frame 只能为 world 或 camera")
    if "relative_kalman" in batch_config["estimators"] and noise_frame != "camera":
        raise ValueError("新相对滤波实验需设置 sensor_noise_coordinate_frame=camera，直接使用相机测点")
    if any(float(value) != 0.0 for value in batch_config.get("motion_joint_error_levels_degrees", [])):
        raise ValueError("请用 motion_position_offset_levels_m 指定末端米制偏移，不能把旧关节角度值当作米")
    motion_levels = batch_config.get("motion_position_offset_levels_m", [0.0])
    noise_levels = batch_config.get("position_noise_levels_m", [batch_config.get("position_noise_std_m", 0.0)])
    if not motion_levels or not noise_levels or not all(
            np.isfinite(value) and value >= 0 for value in [*motion_levels, *noise_levels]):
        raise ValueError("传感器噪声和运动偏移级别必须是非空的有限非负数列表")
    output_directory = output_override or _resolve_project_path(
        project_root, batch_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "trial_checkpoints.jsonl"
    # 每条试次完成后立即追加到检查点，长实验中断时仍保留已完成记录。
    checkpoint_path.write_text("", encoding="utf-8")
    rows = []
    total_trials = (len(batch_config["scenarios"]) * len(batch_config["test_seeds"])
                    * len(batch_config.get("position_noise_levels_m", [
                        batch_config.get("position_noise_std_m", 0.0)]))
                    * len(motion_levels)
                    * len(batch_config["estimators"])
                    * len(batch_config["grasp_strategies"]))
    completed_trials = 0
    noise_levels = batch_config.get(
        "position_noise_levels_m", [batch_config.get("position_noise_std_m", 0.0)])
    for scenario_index, scenario in enumerate(batch_config["scenarios"]):
        for seed in batch_config["test_seeds"]:
            for noise_level in noise_levels:
                for motion_level in motion_levels:
                    for estimator_name in batch_config["estimators"]:
                        for strategy in batch_config["grasp_strategies"]:
                            row = _run_one_trial(
                                project_root, batch_config, scenario, seed,
                                estimator_name, strategy, scenario_index,
                                float(noise_level), float(motion_level),
                            )
                            rows.append(row)
                            with checkpoint_path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            completed_trials += 1
                            print(f"Trial {completed_trials}/{total_trials}: {row['run_id']} "
                                  f"success={row['success']} ({row['elapsed_wall_seconds']:.1f}s)",
                                  flush=True)

    # 同一场景/种子下所有方法的观测摘要应相同，否则这组配对实验不能直接比较。
    paired: dict[str, set[str]] = {}
    paired_row_count: dict[str, int] = {}
    paired_motion_offsets: dict[str, set[tuple[float, ...]]] = {}
    paired_camera_digests: dict[str, set[str]] = {}
    for row in rows:
        paired_row_count[row["pair_id"]] = paired_row_count.get(row["pair_id"], 0) + 1
        paired_motion_offsets.setdefault(row["pair_id"], set()).add(
            tuple(float(value) for value in row["sampled_motion_position_offset_world_m"]))
        if row["camera_observation_sequence_sha256"] is not None:
            paired_camera_digests.setdefault(row["pair_id"], set()).add(row["camera_observation_sequence_sha256"])
        if row["observation_sequence_sha256"] is not None:
            paired.setdefault(row["pair_id"], set()).add(row["observation_sequence_sha256"])
    expected_methods_per_pair = (len(batch_config["estimators"])
                                 * len(batch_config["grasp_strategies"]))
    pair_checks = {
        pair_id: (len(paired.get(pair_id, set())) == 1
                  and paired_row_count[pair_id] == expected_methods_per_pair
                  and len(paired_motion_offsets.get(pair_id, set())) == 1
                  and (noise_frame != "camera" or len(paired_camera_digests.get(pair_id, set())) == 1))
        for pair_id in paired_row_count
    }

    by_method = {}
    for row in rows:
        method = (f"sensor={row['position_noise_level_m']:.3f}|"
                  f"motion={row['motion_position_offset_std_m']:.4f}m|"
                  f"{row['estimator']}|{row['grasp_strategy']}")
        stats = by_method.setdefault(method, {"attempts": 0, "successes": 0, "failures": {}})
        stats["attempts"] += 1
        stats["successes"] += int(row["success"])
        if not row["success"]:
            reason = row["failure_reason"] or "unknown_failure"
            stats["failures"][reason] = stats["failures"].get(reason, 0) + 1

    # 末帧定位 RMSE 按不同随机种子的末帧误差计算，区别于一条序列内部的时间 RMSE。
    for method, stats in by_method.items():
        method_rows = [row for row in rows if (
            f"sensor={row['position_noise_level_m']:.3f}|"
            f"motion={row['motion_position_offset_std_m']:.4f}m|"
            f"{row['estimator']}|{row['grasp_strategy']}") == method]
        errors = [row["offline_position_error_m"] for row in method_rows if row["offline_position_error_m"] is not None]
        execution_errors = [row["offline_execution_target_error_m"] for row in method_rows
                            if row["offline_execution_target_error_m"] is not None]
        stats["success_rate"] = stats["successes"] / stats["attempts"]
        stats["final_position_rmse_m"] = (None if not errors else float(np.sqrt(np.mean(np.square(errors)))))
        stats["execution_target_rmse_m"] = (None if not execution_errors else float(np.sqrt(np.mean(np.square(execution_errors)))))
    runtime_error_count = sum(row["failure_reason"].startswith("runtime_error:") for row in rows)

    snapshot = {
        "batch_experiment": batch_config,
        "scene": _read_json(_resolve_project_path(project_root, batch_config["scene_config_path"])),
        "estimator": _read_json(_resolve_project_path(project_root, batch_config["estimator_config_path"])),
        "geometry": _read_json(_resolve_project_path(project_root, batch_config["geometry_config_path"])),
        "planner": _read_json(_resolve_project_path(project_root, batch_config["planner_config_path"])),
        "filter": _read_json(_resolve_project_path(project_root, batch_config["filter_config_path"])),
        "evaluation": _read_json(_resolve_project_path(project_root, batch_config["evaluation_config_path"])),
    }
    if "relative_kalman" in batch_config["estimators"]:
        snapshot["relative_grasp_filter"] = _read_json(
            _resolve_project_path(project_root,
                                  batch_config["relative_grasp_filter_config_path"]))
    report = {
        "experiment_name": batch_config["experiment_name"],
        "status": ("formal_frozen_protocol" if batch_config.get("formal_statistics", False)
                   else "smoke_check_not_formal_statistics"),
        "configuration_snapshot": snapshot,
        "paired_observation_sequences_identical": pair_checks,
        "summary_by_method": by_method,
        "trial_count": len(rows),
        "runtime_error_count": runtime_error_count,
        "noise_model_note": "Camera XYZ measurement perturbations and independent world XYZ execution-target offsets; fixed extrinsics and known center height. Motion offsets are an endpoint proxy, without in-motion visual correction.",
        "source_sha256": {
            "experiment_runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "relative_filter": hashlib.sha256((project_root / "src/relative_grasp_estimator.py").read_bytes()).hexdigest(),
        },
        "trials": rows,
    }
    (output_directory / "trials.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_fields = list(rows[0]) if rows else []
    with (output_directory / "trials.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        # 矩阵与逐帧记录在 CSV 单元格中用 JSON 保存，重新读取时可直接 json.loads。
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
                          for key, value in row.items()} for row in rows)
    (output_directory / "summary.json").write_text(
        json.dumps({"summary_by_method": by_method,
                    "paired_observation_sequences_identical": pair_checks,
                    "trial_count": len(rows), "runtime_error_count": runtime_error_count}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Batch trials: {len(rows)}")
    for method, summary in by_method.items():
        print(f"  {method}: {summary['successes']}/{summary['attempts']} success")
    print(f"Paired inputs identical: {all(pair_checks.values()) if pair_checks else False}")
    print(f"Reports: {output_directory.resolve()}")
    return report


def main() -> int:
    """解析命令行路径并运行批量实验，正常完成返回 0。"""
    parser = argparse.ArgumentParser(description="运行配对 MuJoCo 抓取实验")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "batch_experiment.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="可选覆盖结果目录；默认读取批量配置")
    args = parser.parse_args()
    report = run_batch(args.config, args.output)
    pair_checks = report["paired_observation_sequences_identical"]
    return 0 if (report["runtime_error_count"] == 0 and pair_checks and all(pair_checks.values())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
