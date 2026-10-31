"""位置状态估计：对同一条三维观测序列运行原始、EMA 与线性 Kalman 方案。

每个估计器只接收带噪位置观测和时间戳，不接收方块真实位置。噪声注入函数
属于实验输入生成边界；无噪声基线仅在此处被扰动，评估真值另行保管。Kalman 采用
六维常速度状态 [x,y,z,vx,vy,vz]，缺测时只预测并受最大观测年龄限制。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionEstimate:
    """一个时刻的位置估计；无效结果的位置为 None，并附带原因。"""

    valid: bool
    reason: str
    timestamp_seconds: float
    position_world_m: tuple[float, float, float] | None
    velocity_world_m_s: tuple[float, float, float] | None
    position_variance_m2: tuple[float, float, float] | None


def _as_position(
    observation_world_m: np.ndarray | tuple[float, float, float] | None,
) -> np.ndarray | None:
    """将有效观测标准化为有限的三维浮点位置；None 保持为缺测。"""
    if observation_world_m is None:
        return None
    position = np.asarray(observation_world_m, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("位置观测必须是有限的三维世界坐标米数，或 None")
    return position


def _make_estimate(timestamp: float, position: np.ndarray | None,
                   velocity: np.ndarray | None = None,
                   variance: np.ndarray | None = None,
                   reason: str = "ok") -> PositionEstimate:
    """统一把内部数组转成易保存、易查看的不可变估计结果。"""
    return PositionEstimate(
        valid=position is not None, reason=reason, timestamp_seconds=float(timestamp),
        position_world_m=None if position is None else tuple(float(v) for v in position),
        velocity_world_m_s=None if velocity is None else tuple(float(v) for v in velocity),
        position_variance_m2=None if variance is None else tuple(float(v) for v in variance),
    )


class RawPositionEstimator:
    """直接输出当前观测作基线；输入缺失时不回填旧位置。"""

    def update(self, timestamp_seconds: float, observation_world_m) -> PositionEstimate:
        """接收时间戳和一个位置/None，原样输出位置或 missing_measurement。"""
        position = _as_position(observation_world_m)
        return _make_estimate(timestamp_seconds, position,
                              reason="ok" if position is not None else "missing_measurement")


class ExponentialMovingAverage:
    """三轴指数移动平均；alpha 越小越平滑，但跟随变化越慢。"""

    def __init__(self, alpha: float):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("EMA alpha 必须满足 0 < alpha <= 1")
        self.alpha = float(alpha)
        self._position: np.ndarray | None = None
        self._previous_time: float | None = None

    def update(self, timestamp_seconds: float, observation_world_m) -> PositionEstimate:
        """输入带噪三维位置或缺测，输出 EMA 位置和由相邻估计差分得出的速度。"""
        position = _as_position(observation_world_m)
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp):
            raise ValueError("EMA 时间戳必须是有限数值")
        if position is None:
            return _make_estimate(timestamp, None, reason="missing_measurement")
        velocity = None
        if self._position is None:
            # 不用真实初始位置；第一条有效观测就是 EMA 初值。
            self._position = position.copy()
        else:
            if self._previous_time is not None and timestamp <= self._previous_time:
                raise ValueError("EMA 时间戳必须严格递增")
            old_position = self._position.copy()
            self._position = self.alpha * position + (1.0 - self.alpha) * self._position
            dt = timestamp - float(self._previous_time)
            velocity = (self._position - old_position) / dt
        self._previous_time = timestamp
        return _make_estimate(timestamp, self._position, velocity=velocity)


def _solve_linear_system(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """用带主元高斯消元解 3x3/多右端线性系统，不显式求矩阵逆。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    right = np.asarray(right_hand_side, dtype=np.float64)
    if right.ndim == 1:
        right = right[:, None]
        vector_result = True
    else:
        vector_result = False
    augmented = np.concatenate((matrix.copy(), right.copy()), axis=1)
    size = matrix.shape[0]
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) < 1e-14:
            raise ArithmeticError("Kalman 创新协方差数值退化")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    solution = augmented[:, size:]
    return solution[:, 0] if vector_result else solution


def _matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """以简单循环完成小型矩阵乘法，避免依赖 BLAS 并方便初学者看清索引含义。"""
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                left[row, inner] * right[inner, column]
                for inner in range(left.shape[1])
            )
    return result


class ConstantVelocityKalmanFilter:
    """三维常速度 Kalman 滤波器，状态顺序为 [x,y,z,vx,vy,vz]。"""

    def __init__(
        self,
        measurement_noise_std_m: float,
        process_noise_spectral_density_m2_s3: float,
        initial_position_std_m: float,
        initial_velocity_std_m_s: float,
        maximum_measurement_age_seconds: float,
    ):
        values = (measurement_noise_std_m, process_noise_spectral_density_m2_s3,
                  initial_position_std_m, initial_velocity_std_m_s,
                  maximum_measurement_age_seconds)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Kalman 噪声、初始不确定性和缺测超时参数必须为正数")
        self.measurement_variance = float(measurement_noise_std_m) ** 2
        self.process_spectral_density = float(process_noise_spectral_density_m2_s3)
        self.initial_position_variance = float(initial_position_std_m) ** 2
        self.initial_velocity_variance = float(initial_velocity_std_m_s) ** 2
        self.maximum_measurement_age = float(maximum_measurement_age_seconds)
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_time: float | None = None
        self._last_measurement_time: float | None = None

    def _predict(self, dt: float) -> None:
        """按常速度模型推进状态，并用连续白噪声加速度假设构造过程噪声 Q。"""
        transition = np.eye(6, dtype=np.float64)
        for axis in range(3):
            transition[axis, axis + 3] = dt
        process = np.zeros((6, 6), dtype=np.float64)
        q = self.process_spectral_density
        # 每轴的离散 Q 块为 q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]。
        for axis in range(3):
            velocity_axis = axis + 3
            process[axis, axis] = q * dt**3 / 3.0
            process[axis, velocity_axis] = q * dt**2 / 2.0
            process[velocity_axis, axis] = q * dt**2 / 2.0
            process[velocity_axis, velocity_axis] = q * dt
        self._state = np.array([
            sum(transition[row, column] * self._state[column] for column in range(6))
            for row in range(6)
        ])
        predicted = _matrix_product(
            _matrix_product(transition, self._covariance), transition.T
        ) + process
        self._covariance = 0.5 * (predicted + predicted.T)

    def update(self, timestamp_seconds: float, observation_world_m) -> PositionEstimate:
        """输入按时间递增的米制位置观测或缺测，输出更新/预测估计及协方差对角。"""
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp):
            raise ValueError("Kalman 时间戳必须是有限数值")
        observation = _as_position(observation_world_m)
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError("Kalman 时间戳必须严格递增")

        if self._state is None:
            self._last_time = timestamp
            if observation is None:
                return _make_estimate(timestamp, None, reason="not_initialized")
            # 初值来自第一条有效观测，速度初值明确设为零，不读取真实初始位置。
            self._state = np.zeros(6, dtype=np.float64)
            self._state[:3] = observation
            self._covariance = np.diag([
                self.initial_position_variance,
                self.initial_position_variance,
                self.initial_position_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ])
            self._last_measurement_time = timestamp
            return _make_estimate(timestamp, self._state[:3], self._state[3:],
                                  np.diag(self._covariance)[:3], "initialized_from_observation")

        dt = timestamp - float(self._last_time)
        self._predict(dt)
        self._last_time = timestamp
        if observation is not None:
            # H 直接选出状态前三项位置；创新协方差 S = HPH' + R。
            innovation = observation - self._state[:3]
            innovation_covariance = self._covariance[:3, :3].copy()
            for axis in range(3):
                innovation_covariance[axis, axis] += self.measurement_variance
            cross_covariance = self._covariance[:, :3].copy()
            # 解 S X = (P H')'，转置后得到 K；不用显式计算 S 的逆。
            gain = _solve_linear_system(innovation_covariance,
                                        cross_covariance.T).T
            self._state += np.array([
                sum(gain[row, axis] * innovation[axis] for axis in range(3))
                for row in range(6)
            ])
            identity_minus_kh = np.eye(6, dtype=np.float64)
            identity_minus_kh[:, :3] -= gain
            # Joseph 形式保证有限精度下协方差仍接近对称半正定。
            first = _matrix_product(
                _matrix_product(identity_minus_kh, self._covariance),
                identity_minus_kh.T,
            )
            second = self.measurement_variance * _matrix_product(gain, gain.T)
            self._covariance = 0.5 * (first + second + (first + second).T)
            self._last_measurement_time = timestamp
            reason = "updated"
        else:
            age = timestamp - float(self._last_measurement_time)
            if age > self.maximum_measurement_age:
                return _make_estimate(timestamp, None, self._state[3:],
                                      np.diag(self._covariance)[:3],
                                      "stale_observation_timeout")
            reason = "predicted_without_measurement"

        return _make_estimate(timestamp, self._state[:3], self._state[3:],
                              np.diag(self._covariance)[:3], reason)


def add_position_noise(
    baseline_position_world_m: np.ndarray,
    standard_deviation_m: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """在实验输入边界加入三轴独立高斯位置扰动，输出带噪观测（米）。"""
    position = _as_position(baseline_position_world_m)
    if (position is None or not np.isfinite(standard_deviation_m)
            or standard_deviation_m < 0):
        raise ValueError("噪声注入需要有效基准位置和非负标准差")
    return position + generator.normal(0.0, standard_deviation_m, size=3)


def inject_position_noise_sequence(
    baseline_positions_world_m: list[np.ndarray | tuple[float, float, float] | None],
    standard_deviation_m: float,
    seed: int,
) -> list[np.ndarray | None]:
    """用固定随机种子生成整段带噪观测，保留 None 缺测位置以便公平复现。"""
    if not np.isfinite(standard_deviation_m) or standard_deviation_m < 0:
        raise ValueError("位置噪声标准差必须是有限的非负数")
    generator = np.random.default_rng(seed)
    return [None if position is None else
            add_position_noise(position, standard_deviation_m, generator)
            for position in baseline_positions_world_m]


def compare_estimators(
    observations_world_m: list[np.ndarray | tuple[float, float, float] | None],
    timestamps_seconds: list[float] | np.ndarray,
    config: dict,
) -> dict[str, list[PositionEstimate]]:
    """把同一条观测序列依次送入原始、EMA、KF，返回三种结果作公平对照。"""
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if timestamps.shape != (len(observations_world_m),):
        raise ValueError("时间戳数量必须与观测数量一致")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("比较用时间戳必须有限且严格递增")
    raw = RawPositionEstimator()
    ema = ExponentialMovingAverage(config["ema"]["alpha"])
    kalman_settings = config["kalman"]
    kalman = ConstantVelocityKalmanFilter(
        measurement_noise_std_m=kalman_settings["measurement_noise_std_m"],
        process_noise_spectral_density_m2_s3=(
            kalman_settings["process_noise_spectral_density_m2_s3"]),
        initial_position_std_m=kalman_settings["initial_position_std_m"],
        initial_velocity_std_m_s=kalman_settings["initial_velocity_std_m_s"],
        maximum_measurement_age_seconds=(
            kalman_settings["maximum_measurement_age_seconds"]),
    )
    output = {"raw": [], "ema": [], "kalman": []}
    for timestamp, observation in zip(timestamps, observations_world_m):
        # 相同时间戳、相同观测依次输入三个估计器，比较才不会受随机序列差异影响。
        output["raw"].append(raw.update(timestamp, observation))
        output["ema"].append(ema.update(timestamp, observation))
        output["kalman"].append(kalman.update(timestamp, observation))
    return output


def calculate_position_metrics(
    estimates: list[PositionEstimate],
    true_positions_world_m: np.ndarray,
) -> dict:
    """离线比较有效估计与真值，输出三维欧氏 RMSE、逐轴 RMSE、偏差和样本数。"""
    truth = np.asarray(true_positions_world_m, dtype=np.float64)
    if truth.shape != (len(estimates), 3) or not np.isfinite(truth).all():
        raise ValueError("真值数组须是与估计等长的有限 [N,3] 米制数组")
    valid_indices = [index for index, estimate in enumerate(estimates)
                     if estimate.valid and estimate.position_world_m is not None]
    if not valid_indices:
        return {"valid_sample_count": 0, "position_rmse_3d_m": None,
                "axis_rmse_m": None, "mean_error_xyz_m": None}
    predicted = np.asarray([estimates[index].position_world_m for index in valid_indices])
    errors = predicted - truth[valid_indices]
    euclidean = np.sqrt(np.sum(errors**2, axis=1))
    axis_rmse = np.sqrt(np.mean(errors**2, axis=0))
    mean_error = np.mean(errors, axis=0)
    return {
        "valid_sample_count": len(valid_indices),
        "position_rmse_3d_m": float(np.sqrt(np.mean(euclidean**2))),
        "axis_rmse_m": [float(value) for value in axis_rmse],
        "mean_error_xyz_m": [float(value) for value in mean_error],
    }
