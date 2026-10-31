"""FR3 末端逆运动学：把期望工具位姿换算为七个机械臂关节角。

本模块只做运动学数值求解，不推进仿真物理、不驱动执行器，也不读取方块真值。
调用方提供 MuJoCo 模型/当前状态、末端 site 名称和目标位姿；求解器返回关节角
以及位置/姿态误差，供后续抓取规划和控制器使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IkResult:
    """一次 IK 求解结果；失败时 joint_positions_rad 为空并带有原因。"""

    success: bool
    reason: str
    joint_names: tuple[str, ...]
    joint_positions_rad: tuple[float, ...] | None
    position_error_m: float
    orientation_error_rad: float
    iterations: int


def _orientation_error(target_quaternion_wxyz: np.ndarray,
                       current_quaternion_wxyz: np.ndarray) -> np.ndarray:
    """将目标与当前四元数差转换为世界坐标系旋转向量（弧度）。"""
    local_error = np.zeros(3, dtype=np.float64)
    # subQuat 的旋转差在当前末端局部坐标系表达；mj_jacSite 的角速度列
    # 则在世界系表达，因此需先用当前末端旋转矩阵转到世界系再用于雅可比。
    mujoco.mju_subQuat(local_error, target_quaternion_wxyz, current_quaternion_wxyz)
    current_rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(current_rotation, current_quaternion_wxyz)
    return current_rotation.reshape(3, 3) @ local_error


def _solve_small_system(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """用带主元的高斯消元解小型 6x6 系统，避免依赖环境的 BLAS 求解器。"""
    augmented = np.column_stack((matrix.astype(np.float64), vector.astype(np.float64)))
    size = len(vector)
    for column in range(size):
        # 选当前列绝对值最大的行作主元，减少除以很小数字带来的数值误差。
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) < 1e-14:
            raise ArithmeticError("阻尼雅可比系统数值退化")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    return augmented[:, -1]


def solve_fr3_ik(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    target_position_world_m: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
    end_effector_site: str = "attachment_site",
    max_iterations: int = 200,
    position_tolerance_m: float = 0.002,
    orientation_tolerance_rad: float = 0.02,
    damping: float = 0.03,
    max_joint_step_rad: float = 0.15,
) -> IkResult:
    """用阻尼最小二乘法求 FR3 末端目标位姿对应的七轴关节角。

    输入目标是世界坐标米数与 MuJoCo 顺序 [w,x,y,z] 的单位四元数。函数在新的
    MjData 副本上迭代，因此不会改动调用方正在运行的仿真状态。输出只包含七个
    ``fr3_joint1`` 至 ``fr3_joint7`` 的弧度角；夹爪关节不参与 IK。
    """
    target_position = np.asarray(target_position_world_m, dtype=np.float64)
    target_quaternion = np.asarray(target_quaternion_wxyz, dtype=np.float64)
    if target_position.shape != (3,) or not np.isfinite(target_position).all():
        raise ValueError("目标位置必须是有限的世界坐标 [x,y,z] 米数")
    if target_quaternion.shape != (4,) or not np.isfinite(target_quaternion).all():
        raise ValueError("目标姿态必须是有限的 MuJoCo [w,x,y,z] 四元数")
    norm = float(np.linalg.norm(target_quaternion))
    if norm < 1e-12:
        raise ValueError("目标四元数长度不能为零")
    target_quaternion /= norm
    if max_iterations <= 0 or damping <= 0 or max_joint_step_rad <= 0:
        raise ValueError("迭代次数、阻尼和单步关节上限必须为正数")
    if position_tolerance_m <= 0 or orientation_tolerance_rad <= 0:
        raise ValueError("位置和姿态误差容许值必须为正数")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
    if site_id < 0:
        raise ValueError(f"模型中不存在末端 site：{end_effector_site!r}")

    # 按关节名称取 qpos/dof 地址；不要把执行器编号当成关节状态数组下标。
    joint_ids = []
    for number in range(1, 8):
        joint_name = f"fr3_joint{number}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"模型缺少有效的 FR3 转动关节 {joint_name}")
        joint_ids.append(joint_id)

    dof_addresses = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    qpos_addresses = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]
    work_data = mujoco.MjData(model)
    # 只复制七个 FR3 关节；不复制夹爪以外的自由物体状态，避免目标真值进入 IK。
    work_data.qpos[qpos_addresses] = current_data.qpos[qpos_addresses]
    work_data.qvel[:] = 0.0
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        mujoco.mj_forward(model, work_data)
        current_position = work_data.site_xpos[site_id].copy()
        current_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(current_quaternion, work_data.site_xmat[site_id])
        position_error = target_position - current_position
        rotation_error = _orientation_error(target_quaternion, current_quaternion)
        position_error_norm = float(np.linalg.norm(position_error))
        rotation_error_norm = float(np.linalg.norm(rotation_error))
        if (position_error_norm <= position_tolerance_m
                and rotation_error_norm <= orientation_tolerance_rad):
            return IkResult(
                True, "converged", tuple(f"fr3_joint{i}" for i in range(1, 8)),
                tuple(float(work_data.qpos[address]) for address in qpos_addresses),
                position_error_norm, rotation_error_norm, iterations,
            )

        mujoco.mj_jacSite(model, work_data, jacobian_position, jacobian_rotation, site_id)
        # 拼成 6x7 位姿雅可比，只取七个 FR3 自由度，不动夹爪或物体自由度。
        jacobian = np.vstack((jacobian_position[:, dof_addresses],
                              jacobian_rotation[:, dof_addresses]))
        error = np.concatenate((position_error, rotation_error))
        # 解 6x6 阻尼系统，避免直接求逆；阻尼可抑制奇异附近的大幅关节跳变。
        # 明确逐项计算 6x6 雅可比积；七轴矩阵很小，循环开销可忽略，
        # 同时避免依赖当前 Windows 环境中不稳定的 NumPy BLAS 小矩阵乘法。
        system = np.zeros((6, 6), dtype=np.float64)
        for row in range(6):
            for column in range(6):
                system[row, column] = sum(
                    jacobian[row, joint] * jacobian[column, joint]
                    for joint in range(7)
                )
            system[row, row] += damping**2
        try:
            system_solution = _solve_small_system(system, error)
            joint_step = np.array([
                sum(jacobian[row, joint] * system_solution[row] for row in range(6))
                for joint in range(7)
            ], dtype=np.float64)
        except ArithmeticError:
            return IkResult(False, "singular_jacobian", (), None,
                            position_error_norm, rotation_error_norm, iterations)
        largest_step = float(np.max(np.abs(joint_step)))
        if largest_step > max_joint_step_rad:
            joint_step *= max_joint_step_rad / largest_step

        # 关节是 hinge，一自由度可直接叠加弧度增量；随后裁到模型声明的限位内。
        proposed = work_data.qpos[qpos_addresses] + joint_step
        work_data.qpos[qpos_addresses] = np.clip(proposed, lower_limits, upper_limits)

    # 把最终实际误差算出来，失败结果仍提供数值，便于上层记录“不可达/未收敛”。
    mujoco.mj_forward(model, work_data)
    final_position_error = float(np.linalg.norm(target_position - work_data.site_xpos[site_id]))
    final_quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(final_quaternion, work_data.site_xmat[site_id])
    final_rotation_error = float(np.linalg.norm(
        _orientation_error(target_quaternion, final_quaternion)))
    return IkResult(
        False, "iteration_limit", tuple(f"fr3_joint{i}" for i in range(1, 8)), None,
        final_position_error, final_rotation_error, iterations,
    )
