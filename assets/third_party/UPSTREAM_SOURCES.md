# 机器人模型来源

模型来自 [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
固定提交：`a662de603224c47123b0783821c58ad4402ad3e3`。

本项目使用 `franka_fr3/` 的七轴机械臂，以及 `franka_emika_panda/hand.xml` 的两指夹爪。
两者通过 FR3 的 `attachment_site` 在仿真中装配。该装配是项目使用的模拟组合，
不表示经过实机标定的 FR3 Hand。

两个上游模型目录采用 Apache-2.0 许可。下载时保留目录内的 LICENSE、README、
MJCF 和网格文件。模型目录由 `.gitignore` 排除，获取命令见仓库根目录 README。
