"""Part 1 第一课：未配准的灰度绝对差分（教学对照，不是完整检测器）。"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 保存图片，无需桌面窗口。
import matplotlib.pyplot as plt
from intro_to_cv.common.images import read_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 1. 图片成为 H×W×3 的 uint8 数组，取值范围 0–255。
    reference = read_rgb(args.reference)
    test = read_rgb(args.test)
    if reference.shape != test.shape:
        raise ValueError(
            f"图片尺寸不同：{reference.shape} 与 {test.shape}。"
            "本课不自动缩放，请选择同尺寸图片。"
        )

    # 2. 转为灰度，再转浮点并归一化。避免 uint8 相减发生回绕。
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    reference_gray = reference_gray.astype(np.float32) / 255.0
    test_gray = test_gray.astype(np.float32) / 255.0

    # 3. 同一位置的灰度差越大，差分热图越亮。
    difference = np.abs(test_gray - reference_gray)

    # 4. 数值与展示分开保存；两次运行使用相同的热图色标。
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "difference.npy", difference)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reference)
    axes[0].set_title("Normal reference")
    axes[1].imshow(test)
    axes[1].set_title("Test image")
    heatmap = axes[2].imshow(difference, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("Absolute grayscale difference")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)

    print(f"参考图：{args.reference}")
    print(f"测试图：{args.test}")
    print(f"RGB数组：shape={test.shape}, dtype={test.dtype}")
    print(f"差分数组：shape={difference.shape}, dtype={difference.dtype}")
    print(f"差值 min={difference.min():.4f}, max={difference.max():.4f}")
    print(f"平均差值={difference.mean():.4f}（描述统计，不是已校准的判决）")
    print(f"对比图：{args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
