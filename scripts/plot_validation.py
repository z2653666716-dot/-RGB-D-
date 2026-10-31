"""从已完成的抓取验证结果生成 GitHub 展示图，不运行仿真。

输入是 validation_summary.json。脚本分别绘制总体表现、分场景成功率和主验证的
分噪声成功率，导出 PNG、SVG 及数据指纹，便于 README 引用和以后重新生成。
图中 raw 指直接采用末帧视觉位置；relative_kalman 指融合十帧的相对位置滤波。
这批数据没有比较旧版常速度 KF，因此不能把图例写成“传统 KF”和“改进 KF”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# 将当前 Python 环境的 DLL 目录放在前面，减少 Windows 桌面软件的数值库干扰。
if sys.platform == "win32":
    dll_directories = [Path(sys.prefix) / "Library/bin", Path(sys.prefix) / "DLLs"]
    os.environ["PATH"] = os.pathsep.join(
        [str(path) for path in dll_directories if path.exists()]
        + [os.environ.get("PATH", "")])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("raw", "relative_kalman")
METHOD_LABELS = ("Raw (last frame)", "Relative Kalman fusion (10 frames)")
METHOD_COLORS = ("#87919C", "#217B9E")
PRIMARY_EXPERIMENT = "relative_kalman_large_primary"
SCENE_LABELS = {
    "relative_kalman_large_primary": "Square\nFixed pose",
    "relative_kalman_large_square": "Square\nGeometry candidates",
    "relative_kalman_large_rectangle": "Rotated rectangle\nGeometry candidates",
    "relative_kalman_large_distractor": "Target + green distractor\nGeometry candidates",
}


def load_summary(path: Path) -> dict:
    """读取汇总 JSON，检查成功率与成功次数一致，返回可供绘图的字典。

    本函数由 build_figures 调用。校验的目的是让图直接对应实验记录，避免把一个
    百分比复制错，或者把滤波和原始观测的分母写成不同的次数。
    """
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["audit"]["runtime_errors"] != 0:
        raise ValueError("输入结果包含运行错误，请先检查原始记录")
    if not summary["audit"]["all_pair_inputs_identical"]:
        raise ValueError("两种方法的配对输入不一致，不能直接生成比较图")
    groups = [summary["overall"], *summary["experiments"].values(), *summary["conditions"]]
    for group in groups:
        for method in METHODS:
            stats = group["methods"][method]
            attempts, successes = stats["attempts"], stats["successes"]
            if attempts != group["pair_count"] or not 0 <= successes <= attempts:
                raise ValueError(f"成功次数或分母不一致：{method}")
            if not math.isclose(stats["success_rate"], successes / attempts, abs_tol=1e-12):
                raise ValueError(f"成功率和成功次数不一致：{method}")
            if not math.isfinite(stats["final_position_rmse_m"]) or stats["final_position_rmse_m"] < 0:
                raise ValueError("定位 RMSE 必须是有限的非负数")
    return summary


def _style_axis(axis) -> None:
    """统一坐标轴外观，输入 Matplotlib 坐标轴，直接修改其样式，无返回值。"""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#B9C0C6")
    axis.tick_params(colors="#3A434B", length=3)
    axis.grid(axis="y", color="#E4E8EB", linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure, output_directory: Path, name: str) -> list[str]:
    """把一张图保存为 PNG 和 SVG，关闭绘图对象，返回两个文件名。

    PNG 用于 GitHub 首页预览；SVG 保留矢量线条，后续放入报告时放大仍然清晰。
    每张图完成后关闭 figure，避免连续生成多张图时一直占用内存。
    """
    filenames = []
    for extension in ("png", "svg"):
        path = output_directory / f"{name}.{extension}"
        figure.savefig(path, dpi=190, facecolor="white", bbox_inches="tight")
        filenames.append(path.name)
    plt.close(figure)
    return filenames


def plot_overview(summary: dict, output_directory: Path) -> list[str]:
    """输入完整汇总，输出总体成功率与最终定位 RMSE 的双面板图片文件名。

    成功率的分母是每种方法 540 次抓取；定位指标则使用去重后的 270 个定位序列。
    两个指标的样本口径不同，分别标注，不把重复运动条件当成新增定位样本。
    """
    overall = summary["overall"]
    stats = [overall["methods"][method] for method in METHODS]
    rates = [item["success_rate"] * 100 for item in stats]
    errors_mm = [item["final_position_rmse_m"] * 1000 for item in stats]
    gain = overall["success_rate_difference"] * 100
    interval = [value * 100 for value in overall["success_rate_difference_ci95"]]
    reduction = (1 - errors_mm[1] / errors_mm[0]) * 100
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.3))
    figure.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.28, wspace=0.30)
    figure.suptitle("Grasping with relative-position Kalman fusion", fontsize=16, x=0.075, ha="left", y=0.98)
    audit = summary["audit"]
    figure.text(0.075, 0.885,
                f"{audit['actual_trials']:,} simulated grasps | {audit['paired_comparisons']} matched pairs | "
                f"{audit['independent_seed_count']} seeds", fontsize=10, color="#525D66")
    for axis, values, title, ylabel in zip(
            axes, (rates, errors_mm), ("Grasp success", "Final position error"),
            ("Success rate (%)", "Position RMSE (mm)")):
        _style_axis(axis)
        bars = axis.bar([0, 1], values, width=0.53, color=METHOD_COLORS)
        axis.set_xticks([0, 1], ["Raw\nlast frame", "Relative KF\n10 frames"])
        axis.set_xlabel("Position estimator")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=12, pad=13)
        axis.set_ylim(0, 115 if axis is axes[0] else max(errors_mm) * 1.32)
        for bar, value, item in zip(bars, values, stats):
            label = (f"{value:.1f}%\n{item['successes']}/{item['attempts']}" if axis is axes[0]
                     else f"{value:.2f} mm")
            axis.text(bar.get_x() + bar.get_width() / 2, value + (2 if axis is axes[0] else 0.35),
                      label, ha="center", va="bottom", fontsize=10, color="#252F37")
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].text(0.98, 0.96, f"+{gain:.1f} percentage points", transform=axes[0].transAxes,
                 ha="right", va="top", color=METHOD_COLORS[1], fontsize=10)
    axes[1].text(0.98, 0.96, f"{reduction:.1f}% lower RMSE", transform=axes[1].transAxes,
                 ha="right", va="top", color=METHOD_COLORS[1], fontsize=10)
    figure.text(0.075, 0.155,
                f"Paired success gain: +{gain:.1f} pp; 95% seed-cluster bootstrap CI: "
                f"+{interval[0]:.1f} to +{interval[1]:.1f} pp.", fontsize=9)
    figure.text(0.075, 0.103,
                f"RMSE: {stats[0]['unique_localization_cases']} unique localization cases per method; "
                "repeated motion-noise conditions are deduplicated.", fontsize=8.5, color="#525D66")
    figure.text(0.075, 0.047, "Source: validation_summary.json | 2026-10-31 | All scenes and failures included.",
                fontsize=8, color="#66727C")
    return _save_figure(figure, output_directory, "filter_comparison_overview")


def plot_scene_success(summary: dict, output_directory: Path) -> list[str]:
    """输入各实验汇总，输出四组场景的成功率图；保留零成功的组合场景。"""
    names = [name for name in SCENE_LABELS if name in summary["experiments"]]
    if not names:
        raise ValueError("没有可绘制的场景结果")
    figure, axis = plt.subplots(figsize=(11.6, 5.8))
    figure.subplots_adjust(left=0.075, right=0.98, top=0.74, bottom=0.26)
    figure.suptitle("Grasp success by scene and grasp strategy", fontsize=16, x=0.075, ha="left", y=0.98)
    _style_axis(axis)
    positions = np.arange(len(names), dtype=float)
    width = 0.29
    for method_index, method in enumerate(METHODS):
        stats = [summary["experiments"][name]["methods"][method] for name in names]
        values = [item["success_rate"] * 100 for item in stats]
        bars = axis.bar(positions + (method_index - 0.5) * width, values, width=width,
                        color=METHOD_COLORS[method_index], label=METHOD_LABELS[method_index])
        for bar, value, item in zip(bars, values, stats):
            # 零高度柱也明确显示 0/80，防止读者把失败场景误以为没有数据。
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2,
                      f"{value:.1f}%\n{item['successes']}/{item['attempts']}",
                      ha="center", va="bottom", fontsize=9)
    axis.set_xticks(positions, [SCENE_LABELS[name] for name in names], fontsize=9)
    axis.set_ylabel("Success rate (%)")
    axis.set_xlabel("Scene and grasp strategy")
    axis.set_ylim(0, 115)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.068, 0.90),
                  ncol=2, frameon=False, fontsize=10)
    figure.text(0.075, 0.125,
                "Fixed-pose and geometry-candidate protocols are shown separately; every failure stays in the denominator.",
                fontsize=8.5, color="#525D66")
    figure.text(0.075, 0.073,
                "The target + green-distractor case failed the lift criterion for both methods; its cause remains unresolved.",
                fontsize=8.5, color="#525D66")
    figure.text(0.075, 0.025, "Source: validation_summary.json | 2026-10-31", fontsize=8, color="#66727C")
    return _save_figure(figure, output_directory, "filter_success_by_scene")


def plot_noise_success(summary: dict, output_directory: Path) -> list[str]:
    """读取主验证的各噪声条件，按运动偏移分面，输出成功率与 Wilson 95% 区间。

    每根柱内的 50 个种子是独立样本，可以使用该条件的二项区间；不同柱之间复用
    相同种子，不能将它们当作独立样本来计算总体区间。总体区间已在上一张图注明。
    """
    rows = [row for row in summary["conditions"] if row["experiment"] == PRIMARY_EXPERIMENT]
    if not rows:
        raise ValueError("缺少主验证的分噪声结果")
    sensor_levels = sorted({row["sensor_xy_std_m"] for row in rows})
    motion_levels = sorted({row["motion_xyz_std_m"] for row in rows})
    lookup = {(row["sensor_xy_std_m"], row["motion_xyz_std_m"]): row for row in rows}
    figure, axes = plt.subplots(1, len(motion_levels), figsize=(6.0 * len(motion_levels), 5.8), squeeze=False)
    figure.subplots_adjust(left=0.075, right=0.98, top=0.73, bottom=0.26, wspace=0.27)
    figure.suptitle("Primary validation: success under measurement and execution noise",
                     fontsize=15, x=0.075, ha="left", y=0.98)
    positions = np.arange(len(sensor_levels), dtype=float)
    width = 0.30
    for axis, motion_std in zip(axes[0], motion_levels):
        _style_axis(axis)
        for method_index, method in enumerate(METHODS):
            stats = [lookup[(sensor_std, motion_std)]["methods"][method] for sensor_std in sensor_levels]
            rates = np.array([item["success_rate"] for item in stats]) * 100
            lower = np.array([item["success_rate_wilson_ci95"][0] for item in stats]) * 100
            upper = np.array([item["success_rate_wilson_ci95"][1] for item in stats]) * 100
            x = positions + (method_index - 0.5) * width
            bars = axis.bar(x, rates, width=width, color=METHOD_COLORS[method_index], label=METHOD_LABELS[method_index])
            axis.errorbar(x, rates, yerr=np.maximum(0.0, np.vstack((rates - lower, upper - rates))),
                           fmt="none", color="#303A43", linewidth=0.9, capsize=3)
            for bar, value, ceiling in zip(bars, rates, upper):
                axis.text(bar.get_x() + bar.get_width() / 2, ceiling + 2,
                          f"{value:.0f}%", ha="center", va="bottom", fontsize=9)
        axis.set_title(f"Execution offset: {motion_std * 1000:g} mm per-axis SD", loc="left", fontsize=11, pad=12)
        axis.set_xticks(positions, [f"{value * 1000:g}" for value in sensor_levels])
        axis.set_xlabel("Sensor position noise, xy SD (mm)")
        axis.set_ylabel("Success rate (%)")
        axis.set_ylim(0, 113)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.068, 0.91),
                  ncol=2, frameon=False, fontsize=10)
    attempts = sorted({row["pair_count"] for row in rows})
    figure.text(0.075, 0.155,
                f"Trials per method and condition: {', '.join(map(str, attempts))}. Error bars: Wilson 95% intervals within each condition.",
                fontsize=8.5)
    figure.text(0.075, 0.105,
                "Camera XYZ noise SD ratio 1:1:2; independent world XYZ endpoint offsets; known target-center height.",
                fontsize=8.5, color="#525D66")
    figure.text(0.075, 0.055,
                "Source: validation_summary.json | 2026-10-31 | Primary square scene, fixed-pose grasps.",
                fontsize=8, color="#66727C")
    return _save_figure(figure, output_directory, "filter_success_by_noise")


def build_figures(summary_path: Path, output_directory: Path) -> dict:
    """输入汇总路径和图片目录，依次生成三组图，返回并保存可追溯的清单。

    main 调用本函数。清单记录输入内容的 SHA-256 指纹，说明图片对应哪份数据；
    它不保存本机绝对路径，便于后续随 GitHub 仓库一起公开。
    """
    summary = load_summary(summary_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 12, "svg.fonttype": "path"})
    filenames = []
    for plot in (plot_overview, plot_scene_success, plot_noise_success):
        filenames.extend(plot(summary, output_directory))
    manifest = {
        "source_file": summary_path.name,
        "source_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "baseline": METHOD_LABELS[0], "method": METHOD_LABELS[1],
        "note": "No old-Kalman baseline was included in this validation. All figures retain failed scenes.",
        "figures": filenames,
    }
    (output_directory / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """读取命令行的输入汇总和输出目录，生成图片并打印文件位置，正常返回 0。"""
    parser = argparse.ArgumentParser(description="从配对抓取验证汇总生成 README 对比图")
    parser.add_argument("--summary", type=Path, required=True, help="validation_summary.json 的位置")
    parser.add_argument("--output", type=Path, required=True, help="PNG、SVG 和清单的保存目录")
    args = parser.parse_args()
    manifest = build_figures(args.summary, args.output)
    for filename in manifest["figures"]:
        print(args.output / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
