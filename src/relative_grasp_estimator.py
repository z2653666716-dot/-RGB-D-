"""把相机测点变成目标中心，并估计目标相对夹爪的位置。

本文件属于相对位置卡尔曼滤波功能：vision.py 提供相机坐标下的可见顶面点，
camera.py 提供相机位姿。本模块用同一个坐标变换计算目标中心和误差协方差，
再依次进行运动预测、视觉修正，输出供抓取规划使用的目标世界坐标。
位置单位为米，位置协方差为平方米；这里的运动噪声是末端 XYZ 位置偏移，
不把关节角的度数直接当作位置标准差，也不读取仿真目标真值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_vector(value, name: str, length: int = 3) -> np.ndarray:
    """输入任意数组形式的向量，返回长度正确、全部有限的浮点数组。"""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} 必须是长度为 {length} 的有限数值向量")
    return vector.copy()


def _covariance_matrix(value, name: str, size: int) -> np.ndarray:
    """检查输入协方差，返回对称矩阵；允许已知高度等对应的零方差。

    先按矩阵本身的大小缩放，再用小矩阵 Cholesky 检查半正定性。
    这样不会把很小的平方米数值误判成零，也避开当前环境的 BLAS DLL 冲突。
    """
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须是有限的 {size}x{size} 矩阵")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return matrix.copy()
    if not np.allclose(matrix, matrix.T, atol=scale * 1e-12, rtol=0):
        raise ValueError(f"{name} 必须对称")
    matrix = 0.5 * (matrix + matrix.T)
    scaled = matrix / scale
    lower = np.zeros_like(scaled)
    tolerance = 1e-12
    for row in range(size):
        diagonal = scaled[row, row] - sum(lower[row, k] ** 2 for k in range(row))
        if diagonal < -tolerance:
            raise ValueError(f"{name} 必须是半正定矩阵")
        if diagonal <= tolerance:
            # 半正定矩阵可以有零主元，但该列剩余的残差也必须为零。
            for next_row in range(row + 1, size):
                residual = scaled[next_row, row] - sum(
                    lower[next_row, k] * lower[row, k] for k in range(row))
                if abs(residual) > tolerance:
                    raise ValueError(f"{name} 必须是半正定矩阵")
        else:
            lower[row, row] = np.sqrt(diagonal)
            for next_row in range(row + 1, size):
                residual = scaled[next_row, row] - sum(
                    lower[next_row, k] * lower[row, k] for k in range(row))
                lower[next_row, row] = residual / lower[row, row]
    return matrix


def _matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """输入两个兼容的小矩阵，逐项乘加得到乘积，避免调用外部 BLAS。"""
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                left[row, inner] * right[inner, column]
                for inner in range(left.shape[1]))
    return result


def _solve_linear_system(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """解协方差线性方程，支持零方差产生的秩亏；输入方阵及一列或多列右侧。

    高斯消元逐列选择主元。没有主元的列代表已完全确定的方向，取自由变量为零；
    如果该方向的观测却发生变化，方程无解，明确报错，避免用人为噪声掩盖矛盾。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    right = np.asarray(right_hand_side, dtype=np.float64)
    vector_result = right.ndim == 1
    if vector_result:
        right = right[:, None]
    size = matrix.shape[0]
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        if np.any(right != 0.0):
            raise ArithmeticError("零方差方向的观测与预测不一致")
        return np.zeros_like(right_hand_side, dtype=np.float64)
    augmented = np.concatenate((matrix / scale, right / scale), axis=1)
    right_scale = max(1.0, float(np.max(np.abs(right / scale))))
    pivot_columns = []
    pivot_row = 0
    for column in range(size):
        if pivot_row == size:
            break
        pivot = pivot_row + int(np.argmax(np.abs(augmented[pivot_row:, column])))
        if abs(augmented[pivot, column]) <= 1e-12:
            continue
        if pivot != pivot_row:
            augmented[[pivot_row, pivot]] = augmented[[pivot, pivot_row]]
        augmented[pivot_row] /= augmented[pivot_row, column]
        for row in range(size):
            if row != pivot_row:
                augmented[row] -= augmented[row, column] * augmented[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    if pivot_row < size and np.any(
            np.abs(augmented[pivot_row:, size:]) > 1e-10 * right_scale):
        raise ArithmeticError("零方差方向的观测与预测不一致")
    solution = np.zeros((size, right.shape[1]), dtype=np.float64)
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row, size:]
    return solution[:, 0] if vector_result else solution


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """将内部数组变为不可变的普通数字元组，方便结果保存为 JSON。"""
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _skew_matrix(vector: np.ndarray) -> np.ndarray:
    """输入三维向量，输出叉乘矩阵；用于计算小角度旋转对位置的影响。"""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_matrix(value) -> np.ndarray:
    """验证相机到世界的旋转矩阵，防止把缩放矩阵或镜像变换当成旋转。"""
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_camera_to_world 必须是有限的 3x3 矩阵")
    gram = _matrix_product(rotation.T, rotation)
    determinant = float(np.sum(np.cross(rotation[:, 0], rotation[:, 1]) * rotation[:, 2]))
    if not np.allclose(gram, np.eye(3), atol=1e-8, rtol=0) or abs(determinant - 1.0) > 1e-8:
        raise ValueError("rotation_camera_to_world 必须是正交且行列式为 +1 的旋转矩阵")
    return rotation.copy()


@dataclass(frozen=True)
class TargetMeasurement:
    """相机测点转换后的目标中心、总协方差及其来源，供滤波和诊断使用。

    sensor_jacobian_world 是 3x3，pose_jacobian_world 是 3x6。
    位姿误差的排列为 [世界系平移 dx,dy,dz, 相机系小旋转 dθx,dθy,dθz]。
    """

    target_position_world_m: tuple[float, float, float]
    covariance_world_m2: tuple[tuple[float, ...], ...]
    sensor_covariance_world_m2: tuple[tuple[float, ...], ...]
    pose_covariance_world_m2: tuple[tuple[float, ...], ...]
    height_covariance_world_m2: tuple[tuple[float, ...], ...]
    sensor_jacobian_world: tuple[tuple[float, ...], ...]
    pose_jacobian_world: tuple[tuple[float, ...], ...]


def target_measurement_from_camera(
    surface_point_camera_m,
    rotation_camera_to_world,
    camera_position_world_m,
    sensor_covariance_camera_m2,
    *,
    camera_pose_covariance=None,
    surface_to_center_offset_world_m=(0.0, 0.0, 0.0),
    known_center_height_world_m=None,
    known_center_height_variance_m2=0.0,
) -> TargetMeasurement:
    """输入相机三维测点和位姿，输出目标中心及传播后的测量协方差。

    位置变换为 c_world = R * p_camera + t + offset。offset 用于将可见顶面点
    转成物体中心，例如水平物体可取 [0,0,-height/2]；该偏移在此视为已知。
    若另给 known_center_height_world_m，则以已知中心高度覆盖 z，这与现有
    vision.py 的桌面约束一致。高度本身的不确定性由独立的高度方差提供。

    论文的高度投影在这里扩展为 XYZ：Js = A*R，Jpose = A*[I, -R*[p]x]。
    A 默认为 I；固定中心高度时 A=diag(1,1,0)。总协方差为
    Js*Σsensor*Jsᵀ + Jpose*Σpose*Jposeᵀ + Σheight。
    相机点测量误差与位姿误差假定独立；6x6 位姿协方差内部可含相关项。
    位姿项应表示本帧独立误差：固定标定偏差不能被当作逐帧独立噪声反复融合。
    当前项目假定固定相机外参准确，默认该项为零，不声称实现了外参标定。
    """
    point = _finite_vector(surface_point_camera_m, "surface_point_camera_m")
    rotation = _rotation_matrix(rotation_camera_to_world)
    translation = _finite_vector(camera_position_world_m, "camera_position_world_m")
    offset = _finite_vector(surface_to_center_offset_world_m, "surface_to_center_offset_world_m")
    sensor_covariance = _covariance_matrix(
        sensor_covariance_camera_m2, "sensor_covariance_camera_m2", 3)
    pose_covariance = (np.zeros((6, 6)) if camera_pose_covariance is None else
                       _covariance_matrix(camera_pose_covariance, "camera_pose_covariance", 6))
    height_variance = float(known_center_height_variance_m2)
    if not np.isfinite(height_variance) or height_variance < 0:
        raise ValueError("known_center_height_variance_m2 必须为有限的非负数")
    center = _matrix_product(rotation, point[:, None])[:, 0] + translation + offset
    center_jacobian = np.eye(3)
    height_covariance = np.zeros((3, 3))
    if known_center_height_world_m is not None:
        height = float(known_center_height_world_m)
        if not np.isfinite(height):
            raise ValueError("known_center_height_world_m 必须为有限数值")
        center[2] = height
        # 中心高度来自桌面/尺寸约束后，相机测点的 z 误差不再直接成为中心 z 误差。
        center_jacobian[2, 2] = 0.0
        height_covariance[2, 2] = height_variance
    elif height_variance != 0.0:
        raise ValueError("指定高度方差时也必须指定已知中心高度")
    sensor_jacobian = _matrix_product(center_jacobian, rotation)
    # 采用相机局部小旋转：R*exp([dθ]x)*p 对 dθ 的导数为 -R*[p]x。
    rotation_jacobian = -_matrix_product(rotation, _skew_matrix(point))
    pose_jacobian = _matrix_product(
        center_jacobian, np.column_stack((np.eye(3), rotation_jacobian)))
    sensor_world = _matrix_product(
        _matrix_product(sensor_jacobian, sensor_covariance), sensor_jacobian.T)
    pose_world = _matrix_product(
        _matrix_product(pose_jacobian, pose_covariance), pose_jacobian.T)
    total = sensor_world + pose_world + height_covariance
    return TargetMeasurement(
        target_position_world_m=tuple(float(value) for value in center),
        covariance_world_m2=_matrix_tuple(0.5 * (total + total.T)),
        sensor_covariance_world_m2=_matrix_tuple(sensor_world),
        pose_covariance_world_m2=_matrix_tuple(pose_world),
        height_covariance_world_m2=_matrix_tuple(height_covariance),
        sensor_jacobian_world=_matrix_tuple(sensor_jacobian),
        pose_jacobian_world=_matrix_tuple(pose_jacobian),
    )


@dataclass(frozen=True)
class RelativeGraspEstimate:
    """一次更新的结果：位置、后验 P、测量 R、运动 Q、增益 K 和诊断量。

    covariance_world_m2 是相对位置的后验协方差。重建目标世界坐标时，夹爪位置
    在本接口中视为准确的外部反馈；若反馈也有误差，需要进一步建模其相关性。
    """

    relative_position_world_m: tuple[float, float, float]
    target_position_world_m: tuple[float, float, float]
    covariance_world_m2: tuple[tuple[float, ...], ...]
    measurement_covariance_world_m2: tuple[tuple[float, ...], ...]
    motion_covariance_world_m2: tuple[tuple[float, ...], ...]
    kalman_gain: tuple[tuple[float, ...], ...]
    predicted_covariance_world_m2: tuple[tuple[float, ...], ...]
    innovation_world_m: tuple[float, float, float]


class RelativeGraspKalmanFilter:
    """估计世界坐标轴下的目标减夹爪向量，供实验入口和抓取规划调用。

    一段运动先调用 predict(计划位移, 运动偏移协方差)，新图像到来后调用
    update_from_camera；update 则保留已经有世界坐标观测的调用方式。
    视觉相对观测需要当前夹爪位置的独立反馈，不能用未知偏移后的名义位置冒充。
    固定相机只观察静止目标时，无法消除机器人之后才发生、又未观测到的运动偏移。
    """

    def __init__(
        self, initial_relative_position_world_m, initial_covariance_world_m2,
        *, initial_gripper_position_world_m=None,
    ):
        """输入相对位置先验和先验 P；可选输入该先验对应的夹爪位置。

        先验可以来自独立粗定位；如果用第一帧初始化，就从第二帧开始 update，
        避免把同一帧重复当作两份独立证据。
        """
        self._relative_position = _finite_vector(
            initial_relative_position_world_m, "initial_relative_position_world_m")
        self.initial_covariance = _covariance_matrix(
            initial_covariance_world_m2, "initial_covariance_world_m2", 3)
        self._covariance = self.initial_covariance.copy()
        self._previous_gripper_position = (
            None if initial_gripper_position_world_m is None else _finite_vector(
                initial_gripper_position_world_m, "initial_gripper_position_world_m"))
        self._prediction_pending = False
        self._pending_motion_covariance = np.zeros((3, 3))

    def predict(self, gripper_displacement_world_m, motion_covariance_world_m2) -> None:
        """输入这一段的名义夹爪位移和米制位置偏移 Q，更新内部先验，无返回值。

        静止目标满足 r_next = r_current - gripper_displacement + w，F=I，
        因此均值减去夹爪位移、P 加上 Q。每段独立运动调用一次；没有图像也可预测。
        多段运动的 Q 可累加，但这些运动误差必须相互独立；相同的固定偏差不能重复加。
        """
        displacement = _finite_vector(gripper_displacement_world_m, "gripper_displacement_world_m")
        motion_covariance = _covariance_matrix(
            motion_covariance_world_m2, "motion_covariance_world_m2", 3)
        self._relative_position = self._relative_position - displacement
        self._covariance = self._covariance + motion_covariance
        self._pending_motion_covariance = self._pending_motion_covariance + motion_covariance
        self._prediction_pending = True

    def update_from_camera(
        self, surface_point_camera_m, rotation_camera_to_world, camera_position_world_m,
        sensor_covariance_camera_m2, gripper_position_world_m, *,
        camera_pose_covariance=None,
        surface_to_center_offset_world_m=(0.0, 0.0, 0.0),
        known_center_height_world_m=None,
        known_center_height_variance_m2=0.0,
    ) -> RelativeGraspEstimate:
        """输入本帧相机测点、位姿和当前夹爪反馈，输出一次完整的滤波结果。

        先调用 target_measurement_from_camera 计算世界坐标中心和 R，
        再交给 update 做视觉修正；运动 Q 由前面的 predict 单独加入。
        """
        measurement = target_measurement_from_camera(
            surface_point_camera_m, rotation_camera_to_world, camera_position_world_m,
            sensor_covariance_camera_m2, camera_pose_covariance=camera_pose_covariance,
            surface_to_center_offset_world_m=surface_to_center_offset_world_m,
            known_center_height_world_m=known_center_height_world_m,
            known_center_height_variance_m2=known_center_height_variance_m2,
        )
        return self.update(
            measurement.target_position_world_m, gripper_position_world_m,
            measurement.covariance_world_m2)

    def update(
        self, target_sensor_position_world_m, gripper_position_world_m,
        sensor_covariance_world_m2, gripper_motion_covariance_world_m2=None,
    ) -> RelativeGraspEstimate:
        """输入目标世界坐标观测、夹爪反馈和 R，输出后验、K、创新及本次使用的 Q。

        显式 predict 后只做测量修正，避免重复减位移或重复加入 Q。
        没有显式 predict 时，兼容原接口：用两次夹爪位置差作确定的坐标参考点变化。
        此时默认 Q=0；已由准确反馈解释的位移不能再假装成未知的运动误差。
        旧参数 gripper_motion_covariance_world_m2 仅用于尚未显式预测的运动区间。
        """
        target = _finite_vector(target_sensor_position_world_m, "target_sensor_position_world_m")
        gripper = _finite_vector(gripper_position_world_m, "gripper_position_world_m")
        measurement_covariance = _covariance_matrix(
            sensor_covariance_world_m2, "sensor_covariance_world_m2", 3)
        extra_motion = (np.zeros((3, 3)) if gripper_motion_covariance_world_m2 is None else
                        _covariance_matrix(gripper_motion_covariance_world_m2,
                                           "gripper_motion_covariance_world_m2", 3))
        predicted_position = self._relative_position.copy()
        if self._prediction_pending:
            if np.any(extra_motion != 0.0):
                raise ValueError("已通过 predict 加入运动 Q，请勿在 update 中重复加入")
            motion_covariance = self._pending_motion_covariance.copy()
            predicted_covariance = self._covariance.copy()
        else:
            if self._previous_gripper_position is not None:
                predicted_position -= gripper - self._previous_gripper_position
            motion_covariance = extra_motion
            predicted_covariance = self._covariance + motion_covariance
        # H=I：观测与状态都是按世界坐标轴表达的目标相对夹爪 XYZ。
        relative_measurement = target - gripper
        innovation = relative_measurement - predicted_position
        innovation_covariance = predicted_covariance + measurement_covariance
        # 验证精确约束是否一致；零高度方差时不能用一个小噪声掩盖高度矛盾。
        _solve_linear_system(innovation_covariance, innovation)
        kalman_gain = _solve_linear_system(
            innovation_covariance.T, predicted_covariance.T).T
        posterior_position = predicted_position + _matrix_product(
            kalman_gain, innovation[:, None])[:, 0]
        identity_minus_gain = np.eye(3) - kalman_gain
        # Joseph 形式：P+ = (I-K)P-(I-K)ᵀ + K R Kᵀ，减少数值误差造成的负方差。
        posterior = (
            _matrix_product(_matrix_product(identity_minus_gain, predicted_covariance),
                            identity_minus_gain.T)
            + _matrix_product(_matrix_product(kalman_gain, measurement_covariance),
                              kalman_gain.T))
        self._relative_position = posterior_position
        self._covariance = 0.5 * (posterior + posterior.T)
        self._previous_gripper_position = gripper.copy()
        self._prediction_pending = False
        self._pending_motion_covariance = np.zeros((3, 3))
        # 将滤波后的相对向量加回当前夹爪反馈，生成抓取规划所需的目标世界坐标。
        target_estimate = gripper + self._relative_position
        return RelativeGraspEstimate(
            relative_position_world_m=tuple(float(value) for value in self._relative_position),
            target_position_world_m=tuple(float(value) for value in target_estimate),
            covariance_world_m2=_matrix_tuple(self._covariance),
            measurement_covariance_world_m2=_matrix_tuple(measurement_covariance),
            motion_covariance_world_m2=_matrix_tuple(motion_covariance),
            kalman_gain=_matrix_tuple(kalman_gain),
            predicted_covariance_world_m2=_matrix_tuple(predicted_covariance),
            innovation_world_m=tuple(float(value) for value in innovation),
        )
