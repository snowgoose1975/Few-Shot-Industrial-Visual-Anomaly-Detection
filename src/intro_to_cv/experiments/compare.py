"""腰果教学基线：固定128分辨率、完整测试清单、正常校准与同一评价代码。"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from intro_to_cv.baselines.autoencoder.pipeline import load_model
from intro_to_cv.common.images import read_rgb, resize_rgb
from intro_to_cv.baselines.template.foreground import foreground_region
from intro_to_cv.baselines.template.maps import difference_maps
from intro_to_cv.baselines.template.pipeline import score_details
from intro_to_cv.common.metrics import localization_metrics, image_metrics
from intro_to_cv.common.scoring import top_fraction_mean, normal_threshold, clean_regions
from intro_to_cv.common.visa import read_split, ground_truth
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normal-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, size = load_model(args.checkpoint)
    if size != 128:
        raise ValueError("当前对照固定128×128，请使用教学模型。")
    config = json.loads(args.normal_calibration.read_text())
    root = Path(config["data_root"])
    audit = json.loads((args.normal_calibration.parent / "calibration_records.json").read_text())
    calibration_paths = [r["image"] for r in audit["normal_calibration"]]
    training = json.loads((args.checkpoint.parent / "training.json").read_text())
    if set(calibration_paths) & set(training["train_images"]):
        raise ValueError("AE训练与正常校准重叠。")
    all_rows = read_split(root)
    index = {r["image"]: r for r in all_rows}
    if not all(index[p]["split"] == "train" and index[p]["label"] == "normal" for p in calibration_paths):
        raise ValueError("正常校准集不符合划分。")
    test_rows = [r for r in all_rows if r["object"] == "cashew" and r["split"] == "test"]

    def read_small(path):
        return resize_rgb(read_rgb(root / path), size)

    reference = read_small(config["reference"])
    fallback_events = []

    def maps_for(image, tag):
        # 两种方法输入同一个resize后的RGB数组；方法内部的配准/前景处理分别披露。
        start = time.perf_counter()
        try:
            _, _, _, difference, roi, warp = score_details(reference, image)
        except RuntimeError as error:
            if "ECC未收敛" not in str(error):
                raise
            # 固定回退规则；不删除失败图、不试到配准成功为止。
            valid = np.ones((size, size), bool)
            unaligned, safe = difference_maps(reference, image, valid)
            foreground, _, _, _ = foreground_region(reference, image, valid)
            difference, roi = unaligned["smoothed"], safe & foreground
            warp = np.eye(2, 3, dtype=np.float32)
            fallback_events.append({"image": tag, "action": "unaligned difference", "reason": str(error)})
        diff = cv2.warpAffine(np.where(roi, difference, 0), warp, (size, size), flags=cv2.INTER_LINEAR)
        diff_seconds = time.perf_counter() - start
        start = time.perf_counter()
        tensor = torch.from_numpy((image.astype(np.float32) / 255).transpose(2, 0, 1).copy()).unsqueeze(0)
        with torch.no_grad():
            reconstructed = model(tensor)
        ae = (tensor - reconstructed).abs().mean(dim=1)[0].numpy()
        ae_seconds = time.perf_counter() - start
        return {"difference": diff, "autoencoder": ae}, {"difference": diff_seconds, "autoencoder": ae_seconds}

    args.output.mkdir(parents=True, exist_ok=True)
    normals = {"difference": [], "autoencoder": []}
    for path in calibration_paths:
        maps, _ = maps_for(read_small(path), "calibration:" + path)
        for name in normals:
            normals[name].append(maps[name])
    thresholds = {}
    for name, maps in normals.items():
        thresholds[name] = {
            "image": normal_threshold([top_fraction_mean(m) for m in maps], .05),
            "pixel": normal_threshold(np.concatenate([m.ravel() for m in maps]), .005),
        }
    (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    collected = {name: [] for name in normals}
    timings = {name: [] for name in normals}
    records, masks, labels, lighting = [], [], [], []
    for row in test_rows:
        image = read_small(row["image"])
        maps, times = maps_for(image, "test:" + row["image"])
        anomalous = row["label"] == "anomaly"
        mask = ground_truth(root, row, size)
        masks.append(mask)
        labels.append(int(anomalous))
        for name, anomaly_map in maps.items():
            collected[name].append(anomaly_map)
            timings[name].append(times[name])
            score = top_fraction_mean(anomaly_map)
            ng = score > thresholds[name]["image"]
            outcome = ("TP" if ng else "FN") if anomalous else ("FP" if ng else "TN")
            records.append({"method": name, "image": row["image"], "label": row["label"],
                            "score": score, "decision": "NG" if ng else "OK", "outcome": outcome})
        # 固定0.8倍亮度：合成正常扰动，只测原阈值，不参与校准或调参。
        if not anomalous:
            perturbed = (image.astype(np.float32) * .8).astype(np.uint8)
            altered, _ = maps_for(perturbed, "brightness0.8:" + row["image"])
            for name, anomaly_map in altered.items():
                score = top_fraction_mean(anomaly_map)
                lighting.append({"method": name, "image": row["image"], "brightness_factor": .8,
                                 "score": score, "false_alarm": score > thresholds[name]["image"]})
        if anomalous and Path(row["image"]).name in {"000.JPG", "001.JPG"}:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for axis_row, name in zip(axes, normals):
                pred = (maps[name] > thresholds[name]["pixel"]).astype(np.uint8)
                # 保留差分基线已采用的3×3开闭运算；AE不做形态学。
                if name == "difference":
                    pred = clean_regions(pred)
                for j, (axis, array, title) in enumerate(zip(axis_row, [image, mask, maps[name], pred],
                                                            [name + ': input', 'GT', 'Anomaly map', 'Prediction'])):
                    axis.imshow(array, cmap="inferno" if j == 2 else "gray", vmin=0, vmax=1 if j else None)
                    axis.set_title(title)
                    axis.axis("off")
            fig.tight_layout()
            fig.savefig(args.output / f"comparison_{Path(row['image']).stem}.png", dpi=150)
            plt.close(fig)
    summaries = {}
    for name, maps in collected.items():
        method_records = [r for r in records if r["method"] == name]
        scores = np.array([r["score"] for r in method_records])
        metrics = image_metrics(labels, scores, thresholds[name]["image"])
        metrics.update(
            normal_brightness_0_8_FPR=float(np.mean([r["false_alarm"] for r in lighting if r["method"] == name])),
            mean_scoring_seconds_without_file_read=float(np.mean(timings[name])),
        )
        metrics.update(localization_metrics(masks, maps))
        summaries[name] = metrics
    protocol = {"size": size, "test_images": len(test_rows), "normal_calibration": calibration_paths,
                "reference": config["reference"], "AE_normal_training_count": len(training["train_images"]),
                "difference_normal_reference_count": 1, "score": "top 1% mean over full 128x128 map",
                "pixel_protocol": "full image; GT nearest-neighbor resize; difference outside ROI/warp coverage=0",
                "method_preprocessing": "difference: ECC + foreground + gray smooth; AE: RGB reconstruction",
                "limitations": "unequal training data budgets; explored000/001 included; only cashew; low-resolution defects may vanish",
                "threshold_source": "normal only; recalibrated for shared scoring rule",
                "synthetic_robustness": "normal test images multiplied by 0.8; not a real domain-shift dataset",
                "failures": "ECC nonconvergence uses logged unaligned difference; other failures abort",
                "ECC_fallback_events": fallback_events}
    (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2))
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for filename, data in [("predictions.csv", records), ("lighting_probe.csv", lighting)]:
        with (args.output / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = ["# 腰果基线对照（教学实验）", "", "两种方法使用完整150张官方测试图、128×128输入、同40张正常校准图、相同整图聚合与像素评价。数据预算不同：差分1张模板，自编码器407张正常训练图；不能称相同few-shot预算。", "", "|方法|图像AUROC|图像AP|FPR|FNR|像素AUROC|像素AP|AUPRO|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summaries.items():
        report.append("|" + name + "|" + "|".join(f"{m[k]:.4f}" for k in ["image_AUROC", "image_AP", "FPR", "FNR", "pixel_AUROC", "pixel_AP", "AUPRO"]) + "|")
    report += ["", "AUPRO：8连通真实区域等权，FPR积分上限0.3，包含正常图。分数评价用连续热图；二值预测形态学不改变AUROC/AP/AUPRO。", "", "## 已观察到的局限", "", "- 模板差分受个体轮廓、姿态、光照和背景纹理影响；固定阈值可漏掉真实缺陷。", "- 自编码器重建出大体形状，但正常细纹理与轮廓也产生误差；图像检测较好不代表定位准确。", "- 缩小到128会改变缺陷可见性；异常000/001参与过方法探索，完整测试成绩需披露这一事实。", "- 当前只完成腰果类别，六类、失败分类和最终统一复现协议仍需完成。", "", "## 光照探针", "", "把50张正常测试图亮度固定乘0.8，用原阈值评价误报；这是合成扰动而非真实跨域测试。"]
    report.append(f"ECC失败回退共{len(fallback_events)}次（含校准/测试/光照探针），完整清单见protocol.json。")
    for name, m in summaries.items():
        report.append(f"- {name}：原正常FPR {m['FPR']:.1%}，0.8倍亮度FPR {m['normal_brightness_0_8_FPR']:.1%}。")
    (args.output / "report_notes.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"报告笔记：{args.output / 'report_notes.md'}")


if __name__ == "__main__":
    main()
