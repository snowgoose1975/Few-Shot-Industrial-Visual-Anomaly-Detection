"""Part 1 第三课：配准后比较绝对差、平滑差分和SSIM差异。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from intro_to_cv.common.images import read_rgb
from intro_to_cv.common.images import gray_float
from intro_to_cv.baselines.template.alignment import register_ecc
from intro_to_cv.baselines.template.foreground import foreground_region
import matplotlib.pyplot as plt


def difference_maps(reference, aligned, valid, blur_kernel=11, ssim_window=11):
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur-kernel必须是正奇数。")
    if ssim_window < 3 or ssim_window % 2 == 0:
        raise ValueError("ssim-window必须是至少3的奇数。")
    ref_gray, test_gray = gray_float(reference), gray_float(aligned)
    absolute = np.abs(test_gray - ref_gray)
    # 先取绝对差再平滑：平均邻域差值，不等于先平滑图像再相减。
    smoothed = cv2.GaussianBlur(absolute, (blur_kernel, blur_kernel), 0)
    _, similarity = structural_similarity(
        ref_gray, test_gray, data_range=1.0, win_size=ssim_window, full=True
    )
    # SSIM可为负；(1-SSIM)/2将理论[-1,1]对应到差异[0,1]。
    ssim_difference = (1.0 - similarity) / 2.0

    # 局部窗口不能碰到warp填充区，也不能超出图像范围。
    # 三种方法使用相同区域，才能公平比较。
    kernel_size = max(blur_kernel, ssim_window)
    safe = cv2.erode(
        valid.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    if not safe.any():
        raise ValueError("局部窗口太大，没有可用区域。")
    maps = {"absolute": absolute, "smoothed": smoothed, "ssim": ssim_difference}
    for anomaly_map in maps.values():
        anomaly_map[~safe] = np.nan
    return maps, safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="仅展示真实缺陷，不参与配准或打分")
    parser.add_argument("--blur-kernel", type=int, default=11)
    parser.add_argument("--ssim-window", type=int, default=11)
    parser.add_argument("--foreground", action="store_true", help="亮腰果/暗背景分割，排除背景")
    args = parser.parse_args()
    reference, test = read_rgb(args.reference), read_rgb(args.test)
    aligned, valid, warp, correlation = register_ecc(reference, test)
    maps, safe = difference_maps(reference, aligned, valid, args.blur_kernel, args.ssim_window)
    foreground = None
    foreground_threshold = None
    if args.foreground:
        foreground, ref_mask, test_mask, foreground_threshold = foreground_region(reference, aligned, valid)
        safe = safe & foreground
        if not safe.any():
            raise ValueError("前景没有有效差分区域。")
        for anomaly_map in maps.values():
            anomaly_map[~safe] = np.nan

    # 标注在原测试坐标系；使用同一变换和最近邻插值对齐到参考坐标。
    ground_truth = np.zeros(valid.shape, dtype=np.uint8)
    if args.mask is not None:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != valid.shape:
            raise ValueError("无法读取mask或mask尺寸不匹配。")
        ground_truth = cv2.warpAffine(
            (mask > 0).astype(np.uint8), warp, (valid.shape[1], valid.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "valid_mask.npy", safe)
    if foreground is not None:
        np.save(args.output / "foreground_roi.npy", foreground)
        np.save(args.output / "foreground_reference.npy", ref_mask)
        np.save(args.output / "foreground_test.npy", test_mask)
        fig_roi, axes_roi = plt.subplots(1, 3, figsize=(15, 5))
        for axis, image, region, title in zip(axes_roi, [reference, aligned, aligned], [ref_mask, test_mask, safe],
                                      ["Reference foreground", "Test foreground", "Detection ROI (union + margin)"]):
            axis.imshow(image)
            axis.contour(region, levels=[0.5], colors="lime", linewidths=1)
            axis.set_title(title)
            axis.axis("off")
        fig_roi.tight_layout()
        fig_roi.savefig(args.output / "foreground.png", dpi=150)
        plt.close(fig_roi)
    for name, anomaly_map in maps.items():
        np.save(args.output / f"{name}.npy", anomaly_map)
    if args.mask is not None:
        np.save(args.output / "ground_truth_aligned.npy", ground_truth)
    summary = {
        "reference": str(args.reference), "test": str(args.test),
        "mask": str(args.mask) if args.mask else None,
        "blur_kernel": args.blur_kernel, "ssim_window": args.ssim_window,
        "ssim_difference_formula": "(1 - SSIM) / 2", "ecc": correlation,
        "coordinates": "reference", "warp_reference_to_test": warp.tolist(),
        "safe_fraction": float(safe.mean()),
        "foreground": args.foreground,
        "foreground_brightness_threshold_from_reference": foreground_threshold,
        "mean_difference": {name: float(np.nanmean(m)) for name, m in maps.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(reference)
    axes[0, 0].set_title("Normal reference")
    axes[0, 1].imshow(aligned)
    axes[0, 1].set_title("Aligned test")
    if args.mask:
        axes[0, 2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title("Aligned ground truth (display only)")
    else:
        axes[0, 2].imshow(safe, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title("Valid window centers")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("gray")
    titles = ["Absolute difference", f"Smoothed difference ({args.blur_kernel}x{args.blur_kernel})",
              f"SSIM difference ({args.ssim_window}x{args.ssim_window})"]
    for axis, anomaly_map, title in zip(axes[1], maps.values(), titles):
        heatmap = axis.imshow(anomaly_map, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        fig.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"对比图：{args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
