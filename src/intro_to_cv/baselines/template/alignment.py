"""Part 1 第二课：ECC 配准，以及同一有效区域内的差分对照。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from intro_to_cv.common.images import read_rgb, gray_float
import matplotlib.pyplot as plt


def register_ecc(reference: np.ndarray, test: np.ndarray):
    """返回对齐图、有效像素、参考坐标→测试坐标的变换及ECC值。"""
    if reference.shape != test.shape:
        raise ValueError("本课要求参考图与测试图尺寸相同。")
    height, width = reference.shape[:2]

    # 小图估计变换以降低CPU耗时；平滑减弱背景细纹理的干扰。
    scale = min(1.0, 512 / max(height, width))
    size = (round(width * scale), round(height * scale))
    ref_small = cv2.resize(gray_float(reference), size, interpolation=cv2.INTER_AREA)
    test_small = cv2.resize(gray_float(test), size, interpolation=cv2.INTER_AREA)
    ref_small = cv2.GaussianBlur(ref_small, (11, 11), 0)
    test_small = cv2.GaussianBlur(test_small, (11, 11), 0)

    # 单位矩阵表示初始假设：没有平移、没有旋转。
    warp_small = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        correlation, warp_small = cv2.findTransformECC(
            ref_small, test_small, warp_small, cv2.MOTION_EUCLIDEAN, criteria
        )
    except cv2.error as error:
        raise RuntimeError("ECC未收敛；请保留为配准失败案例，不将其当成功结果。") from error

    # 从小图坐标恢复原图坐标。使用实际缩放比例，处理尺寸取整。
    resize_matrix = np.diag([size[0] / width, size[1] / height, 1.0])
    homogeneous = np.vstack([warp_small, [0, 0, 1]])
    warp = (np.linalg.inv(resize_matrix) @ homogeneous @ resize_matrix)[:2]
    warp = warp.astype(np.float32)

    # ECC返回参考→测试的坐标映射；逆向采样将测试图对齐到参考图。
    aligned = cv2.warpAffine(
        test, warp, (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # 全1图随同变换；仅保留插值采样完全位于原图内的像素。
    coverage = cv2.warpAffine(
        np.ones((height, width), dtype=np.float32), warp, (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = coverage > 0.999
    if not valid.any():
        raise RuntimeError("配准后无有效重叠区域。")
    return aligned, valid, warp, float(correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, test = read_rgb(args.reference), read_rgb(args.test)
    aligned, valid, warp, correlation = register_ecc(reference, test)
    before = np.abs(gray_float(test) - gray_float(reference))
    after = np.abs(gray_float(aligned) - gray_float(reference))
    # 无效区域存为NaN，不能作为零异常参与平均或未来评价。
    before[~valid] = np.nan
    after[~valid] = np.nan
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "difference_before.npy", before)
    np.save(args.output / "difference_after.npy", after)
    np.save(args.output / "valid_mask.npy", valid)
    cv2.imwrite(str(args.output / "aligned.png"), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    metrics = {
        "reference": str(args.reference), "test": str(args.test),
        "motion": "euclidean", "ecc": correlation,
        "warp_reference_to_test": warp.tolist(),
        "valid_fraction": float(valid.mean()),
        "mean_before": float(np.nanmean(before)),
        "mean_after": float(np.nanmean(after)),
        "heatmap_coordinates": "reference", "estimation_max_edge": 512,
        "estimation_blur_kernel": 11,
    }
    (args.output / "registration.json").write_text(json.dumps(metrics, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, image, title in zip(axes[0], [reference, test, aligned],
                                  ["Reference", "Test before alignment", "Test after ECC"]):
        axis.imshow(image)
        axis.set_title(title)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("gray")
    for axis, difference, title in zip(axes[1, :2], [before, after],
                                      ["Difference before", "Difference after"]):
        heatmap = axis.imshow(difference, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        fig.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    axes[1, 2].imshow(valid, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Valid overlap (white)")
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"对比图：{args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
