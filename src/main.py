"""M1 启动入口：读取 MuJoCo 场景参数，运行物理场景并保存检查结果。"""

import argparse
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np

from simulation import PROJECT_ROOT, Simulation, load_config


def inspect_scene(scene: Simulation) -> dict:
    """读取仿真状态作离线验收；输入已运行场景，输出含配置快照的可追溯检查报告。

    本函数通过刚体真值判断物理是否稳定，只供场景验收/以后独立评估器使用；
    视觉模块与规划器不得把本函数的方块位置当成观测或目标位置。
    """
    cube_truth = scene.cube_truth_for_diagnostics()
    contacts = scene.contacts()
    table_contacts = [pair for pair in contacts if {
        pair["geom_a"], pair["geom_b"],
    } == {"table_surface", "target_cube_collision"}]
    cube_geom = scene.model.geom("target_cube_collision").id
    table_geom = scene.model.geom("table_surface").id
    cube_on_table = any(
        {scene.data.contact[index].geom1, scene.data.contact[index].geom2}
        == {cube_geom, table_geom}
        for index in range(scene.data.ncon)
    )
    camera_frame = scene.render_camera()
    cube = scene.config["cube"]
    cube_height = cube.get("dimensions_xyz_m", [cube["side_m"]] * 3)[2]
    expected_z = scene.config["table"]["top_height_world_m"] + cube_height / 2
    drift = float(np.linalg.norm(np.asarray(cube_truth["position_world_m"]) - scene.settled_cube_position))
    cube_geom_id = scene.model.geom("target_cube_collision").id
    robot_geom_ids = {
        index for index in range(scene.model.ngeom)
        if (name := mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, index))
        and (name.startswith("fr3_") or name.startswith("gripper_"))
    }
    cube_touches_robot = any(
        cube_geom_id in (scene.data.contact[index].geom1, scene.data.contact[index].geom2)
        and bool({scene.data.contact[index].geom1, scene.data.contact[index].geom2} & robot_geom_ids)
        for index in range(scene.data.ncon)
    )
    checks = {
        "fr3_has_seven_arm_joints": sum(item["name"].startswith("fr3_joint") for item in scene.joint_metadata) == 7,
        "parallel_gripper_has_two_fingers": sum(item["name"].startswith("gripper_finger_joint") for item in scene.joint_metadata) == 2,
        "all_eight_actuators_loaded": scene.model.nu == 8,
        "no_deep_initial_penetration": not any(
            contact["distance_m"] < -scene.config["physics"]["penetration_tolerance_m"]
            for contact in scene.initial_contacts
        ),
        # 小于微米的越限只可能是浮点舍入，允许 1e-6 m/rad 容差。
        "all_joint_limits_valid": all(item["range"] is None or
                                     item["range"][0] - 1e-6 <= scene.data.qpos[item["qpos_address"]]
                                     <= item["range"][1] + 1e-6
                                     for item in scene.joint_metadata),
        "cube_contacts_table": bool(cube_on_table and table_contacts),
        "cube_does_not_touch_robot": not cube_touches_robot,
        "cube_height_matches_table": abs(cube_truth["position_world_m"][2] - expected_z) < 0.002,
        "cube_speed_settled": bool(np.linalg.norm(cube_truth["linear_velocity_world_m_s"]) < 0.005),
        "cube_stays_stable": drift < 0.001,
        "camera_rgb_shape_valid": camera_frame.shape == (scene.config["camera"]["height_px"],
                                                            scene.config["camera"]["width_px"], 3),
        "camera_rgb_has_scene_content": bool(np.any(camera_frame != 0)),
    }
    # 按刚体名称取编号，不依赖关节或网格在 XML 中排列的位置。
    body_id = scene.model.body("target_cube").id
    return {
        "stage": "M1_mujoco_fr3_scene_validation",
        "all_checks_passed": all(checks.values()), "checks": checks,
        "config_snapshot": scene.config,
        "runtime": {"mujoco": mujoco.__version__, "mujoco_api_version": mujoco.mj_versionString()},
        "physics": {"steps": scene.steps, "time_step_seconds": scene.model.opt.timestep,
                    "simulated_seconds": scene.steps * scene.model.opt.timestep,
                    "cube_drift_during_stability_check_m": drift},
        "frames": {"world": "MuJoCo world frame (x/right, y/forward, z/up), metres and radians",
                   "robot_base_position_world_m": scene.config["robot"]["base_world_position_m"],
                   "camera_position_world_m": scene.config["camera"]["position_world_m"],
                   "end_effector_reference_site": "attachment_site"},
        "model_counts": {"nq": scene.model.nq, "nv": scene.model.nv,
                         "nu": scene.model.nu, "nbody": scene.model.nbody,
                         "ngeom": scene.model.ngeom, "ncam": scene.model.ncam},
        "joint_metadata": scene.joint_metadata,
        "joint_positions": {
            item["name"]: float(scene.data.qpos[item["qpos_address"]])
            for item in scene.joint_metadata
        },
        "end_effector": {"site_name": "attachment_site",
                          "world_position_m": scene.data.site("attachment_site").xpos.tolist(),
                          "body_name": mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, body_id)},
        "cube_truth_for_diagnostics_only": cube_truth,
        "contacts_for_diagnostics_only": contacts,
        "initial_contacts_for_diagnostics_only": scene.initial_contacts,
        "render": {"rgb_shape": list(camera_frame.shape),
                   "rgb_dtype": str(camera_frame.dtype),
                   "rgb_min_max": [int(camera_frame.min()), int(camera_frame.max())]},
    }


def run_headless(config: dict, output: Path) -> dict:
    """输入 M1 配置与输出目录，完成无可见窗口的仿真、相机渲染和复现检查。"""
    output.mkdir(parents=True, exist_ok=True)
    with Simulation(config) as scene:
        scene.step(config["physics"]["settle_steps"])
        scene.settled_cube_position = scene.data.xpos[scene.model.body("target_cube").id].copy()
        scene.step(config["physics"]["stability_check_steps"])
        report = inspect_scene(scene)
        image = scene.render_camera()
        # MuJoCo Renderer 输出 RGB，OpenCV 文件编码要求 BGR，保存时因此转换通道顺序。
        image_path = output / "scene_rgb.png"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"相机图像无法写入：{image_path}")
    # 同一份配置和种子重新初始化，比较关节与方块状态以检查可复现性和正常退出。
    with Simulation(config) as repeated_scene:
        repeated_scene.step(config["physics"]["settle_steps"] + config["physics"]["stability_check_steps"])
        repeated_image = repeated_scene.render_camera()
        same_cube = np.allclose(repeated_scene.cube_truth_for_diagnostics()["position_world_m"],
                                report["cube_truth_for_diagnostics_only"]["position_world_m"], atol=1e-10)
        repeated_joint_positions = [
            float(repeated_scene.data.qpos[item["qpos_address"]])
            for item in repeated_scene.joint_metadata
        ]
        first_joint_positions = list(report["joint_positions"].values())
        same_joint_state = np.allclose(repeated_joint_positions, first_joint_positions,
                                       rtol=0, atol=1e-6)
    rgb_difference = np.abs(image.astype(np.int16) - repeated_image.astype(np.int16))
    red_cube_pixels = int(np.sum((image[:, :, 0] > 120) &
                                 (image[:, :, 1] < 90) & (image[:, :, 2] < 90)))
    report["checks"]["same_seed_repeats_cube_state"] = bool(same_cube)
    report["checks"]["same_seed_repeats_joint_state"] = bool(same_joint_state)
    report["checks"]["same_seed_repeats_camera_shape"] = repeated_image.shape == image.shape
    report["checks"]["red_cube_visible_in_camera"] = red_cube_pixels >= 20
    report["checks"]["both_renderers_closed_cleanly"] = scene.renderer is None and repeated_scene.renderer is None
    report["all_checks_passed"] = all(report["checks"].values())
    report["saved_rgb_image"] = str(image_path.resolve())
    report["render"]["visible_red_cube_pixel_count"] = red_cube_pixels
    report["render"]["repeat_render_mean_absolute_difference"] = float(rgb_difference.mean())
    report["render"]["repeat_render_max_absolute_difference"] = int(rgb_difference.max())
    report_path = output / "m1_scene_diagnostics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_gui(config: dict) -> None:
    """打开用户明确请求的可见窗口；退出方式是在窗口关闭后结束演示。"""
    import mujoco.viewer

    with Simulation(config) as scene:
        with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                scene.step(1)
                viewer.sync()
                time.sleep(max(0, scene.model.opt.timestep - (time.perf_counter() - start)))


def main() -> int:
    """解析命令行选项；返回 0 表示检查通过，返回 1 表示至少一项验收失败。"""
    parser = argparse.ArgumentParser(description="运行 MuJoCo FR3 + 平行夹爪 M1 场景")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "scene.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs" / "scene")
    parser.add_argument("--gui", action="store_true", help="显式打开可见的 MuJoCo 窗口")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.gui:
        run_gui(config)
        return 0
    report = run_headless(config, args.output)
    print(f"M1 checks: {'PASS' if report['all_checks_passed'] else 'FAIL'}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"Reports: {args.output.resolve()}")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    # 仅用户直接运行此文件时启动；导入验收函数不会自行打开相机或仿真窗口。
    raise SystemExit(main())
