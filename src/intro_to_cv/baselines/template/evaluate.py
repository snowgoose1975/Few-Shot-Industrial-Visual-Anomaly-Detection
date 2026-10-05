"""固定阈值的腰果差分基线：图像级批量评价，不调整参数。"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from intro_to_cv.common.images import read_rgb
from intro_to_cv.baselines.template.pipeline import image_score
from intro_to_cv.common.metrics import image_metrics
from intro_to_cv.common.visa import read_split
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, help="仅用于小规模代码验证；省略则跑全部")
    parser.add_argument("--exclude-exploration", action="store_true", help="可选排除开发异常000/001；默认完整测试集")
    args = parser.parse_args()
    config = json.loads(args.calibration.read_text())
    if config["category"] != "cashew":
        raise ValueError("当前教学版仅支持cashew亮前景分割。")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        raise ValueError("limit-per-class必须大于0。")
    root = Path(config["data_root"])
    reference = read_rgb(root / config["reference"])
    rows = [r for r in read_split(root) if r["object"] == "cashew" and r["split"] == "test"]
    # 已用来观察/修改方法的异常样例不再算未见测试。
    excluded = [r["image"] for r in rows if r["label"] == "anomaly"
                and Path(r["image"]).name in {"000.JPG", "001.JPG"}]
    if not args.exclude_exploration:
        excluded = []
    rows = [r for r in rows if r["image"] not in excluded]
    selected = []
    for label in ["normal", "anomaly"]:
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: r["image"])
        selected.extend(group if args.limit_per_class is None else group[:args.limit_per_class])
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    # CSV逐图写入；标签仅在分数产生后用于评价。
    fields = ["image", "label", "score", "decision", "outcome", "seconds", "error"]
    with (args.output / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            start = time.perf_counter()
            record = dict(image=row["image"], label=row["label"], score=None,
                          decision="UNCERTAIN", outcome="preprocessing_failure", error="")
            try:
                score = image_score(reference, read_rgb(root / row["image"]))
                is_ng = score > config["image_threshold"]
                record.update(score=score, decision="NG" if is_ng else "OK",
                              outcome=("TP" if is_ng else "FN") if row["label"] == "anomaly"
                              else ("FP" if is_ng else "TN"))
            except (ValueError, RuntimeError, cv2.error) as error:
                record["error"] = str(error)
            record["seconds"] = time.perf_counter() - start
            records.append(record)
            writer.writerow(record)
            file.flush()
            print(f"[{index}/{len(selected)}] {row['image']}: {record['outcome']}", flush=True)
    success = [r for r in records if r["score"] is not None]
    labels = np.array([r["label"] == "anomaly" for r in success], dtype=int)
    scores = np.array([r["score"] for r in success])
    shared = image_metrics(labels, scores, config["image_threshold"])
    metrics = {
        "category": "cashew", "scope": "image-level teaching evaluation; not full course benchmark",
        "calibration_file": str(args.calibration),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "excluded_exploration_images": excluded, "limit_per_class": args.limit_per_class,
        "num_requested": len(records), "num_successful": len(success),
        "num_preprocessing_failures": len(records) - len(success),
        "coverage": len(success) / len(records),
        "metrics_population": "successful preprocessing only; failures separately reported",
        "image_threshold": config["image_threshold"],
        "AUROC": shared["image_AUROC"], "AP": shared["image_AP"],
        **{k: shared[k] for k in ["TN", "FP", "FN", "TP", "FPR", "FNR"]},
        "recall": 1 - shared["FNR"] if shared["FNR"] is not None else None,
        "mean_seconds_per_requested_image": float(np.mean([r["seconds"] for r in records])),
        "pixel_metrics": "not computed in this lesson",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output / "failure_cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields + ["cause_manual"])
        writer.writeheader()
        for record in records:
            if record["outcome"] in {"FP", "FN", "preprocessing_failure"}:
                writer.writerow({**record, "cause_manual": ""})
    fig, axis = plt.subplots(figsize=(8, 4))
    for label, name in [(0, "Normal test"), (1, "Anomalous test")]:
        axis.hist(scores[labels == label], bins=15, alpha=0.5, label=name)
    axis.axvline(config["image_threshold"], color="red", label="Fixed normal-calibrated threshold")
    axis.set_xlabel("Image anomaly score")
    axis.set_ylabel("Number of images")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "test_score_distribution.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
