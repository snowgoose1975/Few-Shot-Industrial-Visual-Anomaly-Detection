"""Part 1：正常图校准图像阈值，再用固定阈值预测。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from intro_to_cv.common.images import read_rgb
from intro_to_cv.common.scoring import normal_threshold, clean_regions
from intro_to_cv.common.visa import read_split
from intro_to_cv.baselines.template.pipeline import score_details
import matplotlib.pyplot as plt


def calibrate(args):
    root = args.data_root.resolve()
    rows = read_split(root)
    paths = sorted(r["image"] for r in rows if r["object"] == "cashew"
                   and r["split"] == "train" and r["label"] == "normal"
                   and Path(r["image"]).name not in {"000.JPG", "001.JPG"})
    if args.num_calibration < 10 or args.num_calibration + 1 > len(paths):
        raise ValueError("正常校准图数量应至少10，且为参考图留出一张。")
    if not 0 < args.target_fpr < 1 or not 0 < args.pixel_fpr < 1:
        raise ValueError("target-fpr与pixel-fpr应位于0和1之间。")
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(paths)[:args.num_calibration + 1].tolist()
    reference_path = root / chosen[0]
    reference = read_rgb(reference_path)
    records, failures, normal_pixels = [], [], []
    for index, relative in enumerate(chosen[1:], start=1):
        try:
            score, values, *_ = score_details(reference, read_rgb(root / relative))
        except (ValueError, RuntimeError, cv2.error) as error:
            # 失败不当成正常，也不悄悄丢弃：后续单图预测返回UNCERTAIN。
            failures.append({"image": relative, "error": str(error)})
            print(f"[{index}/{args.num_calibration}] 配准/预处理失败：{relative}", flush=True)
            continue
        records.append({"image": relative, "score": score})
        normal_pixels.append(values)
        print(f"[{index}/{args.num_calibration}] {relative}: {score:.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"reference": chosen[0], "normal_calibration": records, "failures": failures}
    (args.output / "calibration_records.json").write_text(json.dumps(audit, indent=2))
    if len(records) < 10:
        raise RuntimeError("成功校准不足10张；已保存失败记录，不生成阈值。")
    scores = np.array([record["score"] for record in records])
    # 保守向上取分位数；测试时使用score > threshold，平分视为OK。
    threshold = normal_threshold(scores, args.target_fpr)
    pixels = np.concatenate(normal_pixels)
    pixel_threshold = normal_threshold(pixels, args.pixel_fpr)
    config = {
        "data_root": str(root), "reference": chosen[0], "category": "cashew",
        "seed": args.seed, "num_requested": args.num_calibration,
        "num_successful": len(records), "num_failed": len(failures),
        "score_method": "mean of top 1% smoothed absolute differences in foreground ROI",
        "blur_kernel": 11, "ssim_window": 11, "foreground_margin_ratio": 0.1,
        "target_fpr": args.target_fpr, "image_threshold": threshold,
        "pixel_target_fpr_within_roi": args.pixel_fpr, "pixel_threshold": pixel_threshold,
        "pixel_calibration": "pooled normal ROI pixels; before morphology",
        "pixel_calibration_fpr": float(np.mean(pixels > pixel_threshold)),
        "morphology": "3x3 opening then closing; intersect ROI after each step",
        "decision_rule": "NG if score > threshold; preprocessing failure => UNCERTAIN",
        "calibration_fpr_successful_only": float(np.mean(scores > threshold)),
        "protocol": "teaching calibration; only official train/normal; not final benchmark",
    }
    (args.output / "threshold.json").write_text(json.dumps(config, indent=2))
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(scores, bins=10, edgecolor="black")
    axis.axvline(threshold, color="red", label=f"Threshold = {threshold:.4f}")
    axis.set_xlabel("Normal image score (top 1% mean)")
    axis.set_ylabel("Number of normal images")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "normal_scores.png", dpi=150)
    plt.close(fig)
    print(json.dumps(config, indent=2))


def predict(args):
    config = json.loads(args.calibration.read_text())
    reference_path = Path(config["data_root"]) / config["reference"]
    result = {"test": str(args.test), "reference": str(reference_path),
              "threshold": config["image_threshold"]}
    try:
        if "pixel_threshold" not in config:
            raise ValueError("旧校准文件没有像素阈值，请重新运行calibrate。")
        reference, test = read_rgb(reference_path), read_rgb(args.test)
        score, _, aligned, anomaly_map, roi, warp = score_details(reference, test)
        result.update(score=score, decision="NG" if score > result["threshold"] else "OK")
        # 像素阈值独立于图像阈值；即使整图OK，也保留局部预测供诊断。
        raw_prediction = ((anomaly_map > config["pixel_threshold"]) & roi).astype(np.uint8)
        prediction = clean_regions(raw_prediction, roi)
        args.output.mkdir(parents=True, exist_ok=True)
        # 当前热图在参考坐标；映回原测试图坐标后才可与官方mask比较。
        size = (test.shape[1], test.shape[0])
        original_prediction = cv2.warpAffine(prediction, warp, size, flags=cv2.INTER_NEAREST)
        cv2.imwrite(str(args.output / "prediction.png"), original_prediction * 255)
        cv2.imwrite(str(args.output / "prediction_reference.png"), prediction * 255)
        cv2.imwrite(str(args.output / "prediction_raw_reference.png"), raw_prediction * 255)
        np.save(args.output / "anomaly_map_reference.npy", np.where(roi, anomaly_map, np.nan))
        np.save(args.output / "valid_roi_reference.npy", roi)
        mapped_map = cv2.warpAffine(np.where(roi, anomaly_map, 0), warp, size, flags=cv2.INTER_LINEAR)
        mapped_roi = cv2.warpAffine(roi.astype(np.uint8), warp, size, flags=cv2.INTER_NEAREST)
        np.save(args.output / "anomaly_map_test.npy", mapped_map)
        np.save(args.output / "valid_roi_test.npy", mapped_roi.astype(bool))
        result.update(pixel_threshold=config["pixel_threshold"],
                      predicted_pixels=int(original_prediction.sum()),
                      warp_reference_to_test=warp.tolist(),
                      localization_note="full-image evaluation must include background with zero score; report ROI coverage")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(test)
        axes[0].set_title(f"Test: {result['decision']} (score={score:.4f})")
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("gray")
        heatmap = axes[1].imshow(np.where(roi, anomaly_map, np.nan), cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title("Heatmap in reference coordinates")
        fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(test)
        overlay = np.zeros((*original_prediction.shape, 4))
        overlay[original_prediction > 0] = [1, 0, 0, 0.7]
        axes[2].imshow(overlay)
        axes[2].set_title("Predicted regions (red), test coordinates")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(args.output / "prediction_comparison.png", dpi=150)
        plt.close(fig)
    except (ValueError, RuntimeError, cv2.error) as error:
        result.update(score=None, decision="UNCERTAIN", error=str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "decision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="只用正常训练图校准")
    calibration.add_argument("--data-root", type=Path, default=Path("data/raw/visa"))
    calibration.add_argument("--num-calibration", type=int, default=40)
    calibration.add_argument("--target-fpr", type=float, default=0.05)
    calibration.add_argument("--pixel-fpr", type=float, default=0.005,
                             help="正常ROI像素的目标超阈比例，形态学处理前")
    calibration.add_argument("--seed", type=int, default=0)
    calibration.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict", help="使用固定阈值，不读取测试标签")
    prediction.add_argument("--calibration", type=Path, required=True)
    prediction.add_argument("--test", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
