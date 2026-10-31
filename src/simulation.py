"""M1 场景模块：组装 MuJoCo FR3、平行夹爪、桌面、方块与固定俯视相机。

主程序 main.py 创建 Simulation、推进固定时间步并保存诊断与相机画面。
Frankа FR3 机械臂与 Panda 两指夹爪来自分开授权的上游 MJCF；本模块在 FR3 法兰
挂点把两个上游模型组合，不将场景真值传给后续的视觉定位或抓取规划模块。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "third_party" / "mujoco_menagerie"


def load_config(path: str | Path) -> dict:
    """输入 JSON 配置路径，输出经过基本物理范围检查的字典；入口与模拟器共用。"""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    physics, camera, cube = config["physics"], config["camera"], config["cube"]
    if physics["time_step_seconds"] <= 0:
        raise ValueError("物理时间步长必须大于 0 秒")
    if physics["solver_iterations"] <= 0 or physics["settle_steps"] <= 0:
        raise ValueError("求解器迭代次数和稳定步数必须大于 0")
    if len(config["robot"]["arm_joint_angles_rad"]) != 7:
        raise ValueError("FR3 机械臂必须配置 7 个关节角，单位为弧度")
    if len(config["robot"]["gripper_finger_positions_m"]) != 2:
        raise ValueError("平行夹爪必须配置左右两个手指的位置，单位为米")
    if cube["side_m"] <= 0 or cube["mass_kg"] <= 0 or cube["xy_jitter_m"] < 0:
        raise ValueError("方块边长和质量须大于 0，出生位置扰动不能小于 0")
    dimensions = np.asarray(cube.get("dimensions_xyz_m", [cube["side_m"]] * 3), dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("目标长宽高 dimensions_xyz_m 必须是三个正的米制长度")
    if not np.isfinite(cube.get("yaw_world_rad", 0.0)):
        raise ValueError("目标平面朝向 yaw_world_rad 必须是有限弧度")
    dimensions = np.asarray(cube.get("dimensions_xyz_m", [cube["side_m"]] * 3), dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("目标长宽高 dimensions_xyz_m 必须是三个正的米制长度")
    if not np.isfinite(cube.get("yaw_world_rad", 0.0)):
        raise ValueError("目标平面朝向 yaw_world_rad 必须是有限弧度")
    if camera["width_px"] <= 0 or camera["height_px"] <= 0:
        raise ValueError("相机图像宽高必须是正整数")
    if not 0 < camera["vertical_fov_degrees"] < 180:
        raise ValueError("相机垂直视场角必须介于 0 和 180 度之间")
    expected_top = config["table"]["center_world_m"][2] + config["table"]["half_size_m"][2]
    if abs(expected_top - config["table"]["top_height_world_m"]) > 1e-9:
        raise ValueError("桌面中心高度加半厚度必须等于配置中的桌面表面高度")
    return config


class Simulation:
    """控制一个 MuJoCo 场景；输入场景配置，提供步进、渲染与只读关节映射。"""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.data = None
        self.renderer = None
        self.steps = 0
        self.joint_metadata: list[dict] = []

    def __enter__(self) -> Simulation:
        """进入 with 时建立模型、附加夹爪并放置物体；出错时自动关闭渲染器。"""
        self._build_model()
        try:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.config["camera"]["height_px"],
                width=self.config["camera"]["width_px"],
            )
        except Exception as error:
            raise RuntimeError(
                "MuJoCo 模型已加载，但未能建立相机的 OpenGL 渲染环境；"
                "请检查当前 Windows 桌面图形驱动和渲染能力。"
            ) from error
        return self

    def __exit__(self, error_type, error, traceback):
        """离开 with 时释放 OpenGL 渲染器，确保连续运行检查能重新打开场景。"""
        self.close()

    def close(self) -> None:
        """释放本实例持有的 OpenGL 上下文；重复调用安全，无返回值。"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _build_model(self) -> None:
        """读取上游 MJCF、用法兰 site 挂接独立夹爪，并从配置构造场景物体。"""
        config, physics = self.config, self.config["physics"]
        fr3_path = ASSET_ROOT / "franka_fr3" / "scene.xml"
        gripper_path = ASSET_ROOT / "franka_emika_panda" / "hand.xml"
        if not fr3_path.is_file() or not gripper_path.is_file():
            raise FileNotFoundError(
                "缺少锁定版本的 FR3 或两指夹爪 MJCF。请先准备 "
                "assets/third_party/mujoco_menagerie/ 模型资源。"
            )

        # MjSpec 保留两个上游模型文件不变，并把完整的夹爪子树和对应 actuator、
        # tendon、关节约束一并附加到 FR3 末端法兰的 attachment_site。
        spec = mujoco.MjSpec.from_file(str(fr3_path))
        gripper_spec = mujoco.MjSpec.from_file(str(gripper_path))
        spec.site("attachment_site").attach_body(
            gripper_spec.body("hand"), prefix="gripper_",
        )
        spec.option.timestep = physics["time_step_seconds"]
        spec.option.iterations = physics["solver_iterations"]
        spec.option.gravity = np.asarray(physics["gravity_world_m_s2"], dtype=float)
        spec.body("base").pos = np.asarray(config["robot"]["base_world_position_m"], dtype=float)
        self._add_table(spec.worldbody, config["table"])
        self._add_cube(spec.worldbody, config["table"], config["cube"], config["seed"])
        self._add_visual_distractors(spec.worldbody, config.get("visual_distractors", []))
        camera = config["camera"]
        spec.worldbody.add_camera(
            name=camera["name"], pos=camera["position_world_m"],
            fovy=camera["vertical_fov_degrees"],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._set_initial_joint_positions()
        mujoco.mj_forward(self.model, self.data)
        self.initial_contacts = self.contacts()
        self.joint_metadata = self._read_joint_metadata()

    @staticmethod
    def _add_table(world, table: dict) -> None:
        """输入桌面尺寸与中心，输出一个静态台面和四条有物理碰撞的桌腿。"""
        center = table["center_world_m"]
        half_size = table["half_size_m"]
        leg_half = table["leg_width_m"] / 2
        leg_length = table["leg_height_m"]
        x_edge = half_size[0] - leg_half
        y_edge = half_size[1] - leg_half
        z_center = table["top_height_world_m"] - leg_length / 2
        color = table["color_rgba"]
        world.add_geom(
            name="table_surface", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=center, size=half_size, rgba=color,
        )
        for number, x_sign, y_sign in (
            (1, -1, -1), (2, -1, 1), (3, 1, -1), (4, 1, 1),
        ):
            world.add_geom(
                name=f"table_leg_{number}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[center[0] + x_sign * x_edge,
                     center[1] + y_sign * y_edge, z_center],
                size=[leg_half, leg_half, leg_length / 2], rgba=color,
            )

    @staticmethod
    def _add_cube(world, table: dict, cube: dict, seed: int) -> None:
        """用固定种子生成目标长方体；可选尺寸和 yaw 仅用于场景/实验设置。"""
        generator = np.random.default_rng(seed)
        xy = np.asarray(cube["center_xy_world_m"], dtype=float) + generator.uniform(
            -cube["xy_jitter_m"], cube["xy_jitter_m"], size=2,
        )
        side = cube["side_m"]
        dimensions = np.asarray(cube.get("dimensions_xyz_m", [side, side, side]),
                                dtype=np.float64)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("目标尺寸 dimensions_xyz_m 必须是三个正的米制长度")
        half_sizes = dimensions / 2.0
        z = table["top_height_world_m"] + half_sizes[2] + cube["spawn_clearance_m"]
        body = world.add_body(name="target_cube", pos=[xy[0], xy[1], z])
        body.add_freejoint(name="target_cube_free")
        body.add_geom(
            name="target_cube_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_sizes.tolist(), density=cube["mass_kg"] / float(np.prod(dimensions)),
            friction=cube["friction"], rgba=cube["color_rgba"],
        )

    @staticmethod
    def _add_visual_distractors(world, distractors: list[dict]) -> None:
        """添加不参与物理碰撞的视觉干扰物，测试目标颜色分割与有限遮挡。"""
        for index, item in enumerate(distractors, start=1):
            name = item.get("name", f"visual_distractor_{index}")
            half_sizes = np.asarray(item["half_size_m"], dtype=np.float64)
            position = np.asarray(item["position_world_m"], dtype=np.float64)
            if (half_sizes.shape != (3,) or position.shape != (3,)
                    or not np.isfinite(half_sizes).all() or not np.isfinite(position).all()
                    or np.any(half_sizes <= 0)):
                raise ValueError("视觉干扰物须配置有限位置和三个正的半尺寸")
            world.add_geom(
                name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=position.tolist(), size=half_sizes.tolist(),
                rgba=item["color_rgba"], contype=0, conaffinity=0,
            )

    def _set_initial_joint_positions(self) -> None:
        """按 MJCF 中的关节名写入 qpos 并按 actuator 名设置控制量，不猜数组序号。"""
        arm = self.config["robot"]["arm_joint_angles_rad"]
        fingers = self.config["robot"]["gripper_finger_positions_m"]
        for number, angle in enumerate(arm, start=1):
            joint_name = f"fr3_joint{number}"
            self._set_joint_position(joint_name, angle)
            self.data.ctrl[self.model.actuator(f"fr3_joint{number}").id] = angle
        for number, position in enumerate(fingers, start=1):
            self._set_joint_position(f"gripper_finger_joint{number}", position)
        self.data.ctrl[self.model.actuator("gripper_actuator8").id] = (
            self.config["robot"]["gripper_open_control"]
        )
        # 自由目标有 7 个 qpos（位置 + [w,x,y,z] 四元数）；可按配置绕桌面法线旋转。
        address = self.model.joint("target_cube_free").qposadr[0]
        yaw = float(self.config["cube"].get("yaw_world_rad", 0.0))
        if not np.isfinite(yaw):
            raise ValueError("目标平面朝向 yaw_world_rad 必须是有限弧度")
        self.data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0),
        ]

    def _set_joint_position(self, name: str, value: float) -> None:
        """输入 MJCF 关节名称与位置，按该关节的 qpos 地址更新状态。"""
        joint = self.model.joint(name)
        lower, upper = self.model.jnt_range[joint.id]
        if self.model.jnt_limited[joint.id] and not lower <= value <= upper:
            raise ValueError(f"关节 {name} 的初始位置 {value} 超出限位 [{lower}, {upper}]")
        self.data.qpos[joint.qposadr[0]] = value

    def _read_joint_metadata(self) -> list[dict]:
        """输出具名关节/执行器映射及限位，为诊断和之后的 IK 提供可靠索引。"""
        items = []
        actuator_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act): act
            for act in range(self.model.nu)
        }
        for index in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            actuator_id = actuator_names.get(name)
            limited = bool(self.model.jnt_limited[index])
            items.append({
                "index": index, "name": name,
                "joint_type": int(self.model.jnt_type[index]),
                "qpos_address": int(self.model.jnt_qposadr[index]),
                "dof_address": int(self.model.jnt_dofadr[index]),
                "limited": limited,
                "range": self.model.jnt_range[index].tolist() if limited else None,
                "actuator_id": int(actuator_id) if actuator_id is not None else None,
                "actuator_name": name if actuator_id is not None else None,
            })
        return items

    def step(self, count: int = 1) -> None:
        """输入非负物理步数；每一步推进 time_step_seconds，不按电脑速度改变物理时间。"""
        if not isinstance(count, int) or count < 0:
            raise ValueError("物理步数必须是非负整数")
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
        self.steps += count

    def render_camera(self) -> np.ndarray:
        """使用配置的固定相机渲染 RGB uint8 图像；本函数不读取物体 ID 或真值位置。"""
        if self.renderer is None:
            raise RuntimeError("渲染器尚未建立；请使用 with Simulation(config) as scene")
        self.renderer.update_scene(self.data, camera=self.config["camera"]["name"])
        return self.renderer.render()

    def cube_truth_for_diagnostics(self) -> dict:
        """仅供 M1/M3 离线场景检查，输出方块世界位置与速度，感知模块不得调用。"""
        body = self.model.body("target_cube")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0,
        )
        return {"position_world_m": self.data.xpos[body.id].tolist(),
                "linear_velocity_world_m_s": velocity[3:6].tolist()}

    def contacts(self) -> list[dict]:
        """返回当前碰撞对的具名几何体和间距，用于场景诊断；不用于物体定位。"""
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            contacts.append({
                "geom_a": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                "geom_b": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                "distance_m": float(contact.dist),
            })
        return contacts
