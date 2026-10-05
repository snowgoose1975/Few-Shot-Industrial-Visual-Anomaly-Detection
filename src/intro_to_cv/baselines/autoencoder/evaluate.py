"""低分辨率AE的正常校准与批量评价；所有参数固定，不优化测试结果。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from intro_to_cv.common.images import read_rgb
from intro_to_cv.common.metrics import localization_metrics, image_metrics
from intro_to_cv.common.scoring import normal_threshold
from intro_to_cv.common.visa import read_split, ground_truth
from intro_to_cv.baselines.autoencoder.pipeline import load_model, infer
import matplotlib.pyplot as plt


def calibrate(args):
    if not 0 < args.target_fpr < 1 or not 0 < args.pixel_fpr < 1:
        raise ValueError("误报率参数必须位于0和1之间。")
    training = json.loads((args.checkpoint.parent / "training.json").read_text())
    root = Path(training["data_root"])
    original_calibration = Path(training["normal_calibration_source"])
    audit = json.loads((original_calibration.parent / "calibration_records.json").read_text())
    paths = [r["image"] for r in audit["normal_calibration"]]
    if set(paths) & set(training["train_images"]):
        raise ValueError("校准图与训练图重叠。")
    rows = {r["image"]: r for r in read_split(root)}
    if not paths or not all(rows[p]["split"] == "train" and rows[p]["label"] == "normal" for p in paths):
        raise ValueError("校准必须使用官方正常训练图。")
    model, size = load_model(args.checkpoint)
    scores, maps, records = [], [], []
    for path in paths:
        score, error, _ = infer(model, size, root / path)
        scores.append(score)
        maps.append(error)
        records.append({"image": path, "score": score})
    pixels = np.concatenate([m.ravel() for m in maps])
    image_threshold = normal_threshold(scores, args.target_fpr)
    pixel_threshold = normal_threshold(pixels, args.pixel_fpr)
    config = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "data_root": str(root), "size": size, "num_calibration": len(paths),
              "image_threshold": image_threshold, "pixel_threshold": pixel_threshold,
              "target_fpr": args.target_fpr, "pixel_target_fpr": args.pixel_fpr,
              "image_score": "mean top 1% RGB absolute reconstruction error, full resized image",
              "calibration_image_fpr": float(np.mean(np.array(scores) > image_threshold)),
              "calibration_pixel_fpr": float(np.mean(pixels > pixel_threshold)),
              "decision_rule": "strictly greater than threshold",
              "morphology": "none; preserve low-resolution small regions"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "threshold.json").write_text(json.dumps(config, indent=2))
    (args.output / "calibration_records.json").write_text(json.dumps(records, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=10)
    ax.axvline(image_threshold, color="red", label="Fixed image threshold")
    ax.set_xlabel("Normal AE image score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "normal_scores.png", dpi=150)
    plt.close(fig)
    print(json.dumps(config, indent=2))


def evaluate(args):
    config = json.loads(args.calibration.read_text())
    checkpoint = Path(config["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("权重已改变，需要重新校准。")
    model, size = load_model(checkpoint)
    root = Path(config["data_root"])
    rows = [r for r in read_split(root) if r["object"] == "cashew" and r["split"] == "test"]
    args.output.mkdir(parents=True, exist_ok=True)
    records, maps, masks = [], [], []
    saved = {"normal": 0, "anomaly": 0}
    for row in rows:
        score, error, reconstructed = infer(model, size, root / row["image"])
        gt = ground_truth(root, row, size)
        ng, anomalous = score > config["image_threshold"], row["label"] == "anomaly"
        outcome = ("TP" if ng else "FN") if anomalous else ("FP" if ng else "TN")
        record = {"image": row["image"], "label": row["label"], "score": score,
                  "decision": "NG" if ng else "OK", "outcome": outcome}
        records.append(record)
        maps.append(error)
        masks.append(gt)
        if saved[row["label"]] < 2:
            prediction = error > config["pixel_threshold"]
            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            input_image = cv2.resize(read_rgb(root / row["image"]), (size, size), interpolation=cv2.INTER_AREA)
            images = [input_image, reconstructed, gt, error, prediction]
            titles = [f"Input: {record['decision']}", "Reconstruction", "GT", "Error", "Prediction"]
            for i, (axis, image, title) in enumerate(zip(axes, images, titles)):
                axis.imshow(image, cmap="inferno" if i == 3 else "gray", vmin=0, vmax=1 if i >= 2 else None)
                axis.set_title(title)
                axis.axis("off")
            fig.tight_layout()
            fig.savefig(args.output / f"example_{row['label']}_{Path(row['image']).stem}.png", dpi=150)
            plt.close(fig)
            saved[row["label"]] += 1
    labels = np.array([r["label"] == "anomaly" for r in records], dtype=int)
    scores = np.array([r["score"] for r in records])
    image_results = image_metrics(labels, scores, config["image_threshold"])
    metrics = {"scope": "full official cashew test split, low-resolution teaching AE",
               "size": size, "num_images": len(rows), **image_results,
               "image_threshold": config["image_threshold"], "pixel_threshold": config["pixel_threshold"],
               "explored_samples_included": ["Anomaly/000.JPG", "Anomaly/001.JPG"],
               "pixel_protocol": "128x128 whole image; GT nearest-neighbor resize; no foreground masking",
               "limitation": "small defects can vanish when resizing; not directly comparable to original-resolution difference baseline"}
    metrics.update(localization_metrics(masks, maps, pro_limit=0.3))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fields = ["image", "label", "score", "decision", "outcome"]
    with (args.output / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output / "pixel_outputs.npz", maps=np.stack(maps), masks=np.stack(masks))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    cal = commands.add_parser("calibrate")
    cal.add_argument("--checkpoint", type=Path, required=True)
    cal.add_argument("--target-fpr", type=float, default=0.05)
    cal.add_argument("--pixel-fpr", type=float, default=0.005)
    cal.add_argument("--output", type=Path, required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--calibration", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads必须为正。")
    torch.set_num_threads(args.threads)
    if args.command == "calibrate":
        calibrate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
