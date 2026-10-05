"""紧凑CPU自编码器：正常图训练，以及重建误差展示。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from intro_to_cv.common.images import image_tensor
from intro_to_cv.baselines.autoencoder.model import Autoencoder
from intro_to_cv.common.visa import read_split
import matplotlib.pyplot as plt


def train(args):
    if args.size < 8 or args.size % 8 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("size必须是至少8的8倍数；epochs和batch-size必须为正。")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    config = json.loads(args.normal_calibration.read_text())
    root = Path(config["data_root"])
    records_file = args.normal_calibration.parent / "calibration_records.json"
    audit = json.loads(records_file.read_text())
    excluded = {audit["reference"]} | {r["image"] for r in audit["normal_calibration"]}
    excluded |= {r["image"] for r in audit["failures"]}
    excluded |= {"cashew/Data/Images/Normal/000.JPG", "cashew/Data/Images/Normal/001.JPG"}
    paths = sorted(r["image"] for r in read_split(root)
                   if r["object"] == "cashew" and r["split"] == "train"
                   and r["label"] == "normal" and r["image"] not in excluded)
    if not paths:
        raise ValueError("没有剩余正常训练图片。")
    # 预先缩小并缓存几百张正常图，减少每个epoch的JPEG读取成本。
    images = torch.stack([image_tensor(root / p, args.size) for p in paths])
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True)
    model = Autoencoder().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        start = time.perf_counter()
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)  # 输入正常图，也是重建目标。
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        mean_loss = total / len(images)
        history.append({"epoch": epoch + 1, "loss": mean_loss,
                        "seconds": time.perf_counter() - start})
        print(f"Epoch {epoch + 1}/{args.epochs}: MSE={mean_loss:.6f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "size": args.size}, args.output / "model.pt")
    training = {"train_images": paths, "excluded_normal_images": sorted(excluded),
                "data_root": str(root), "size": args.size, "epochs": args.epochs,
                "seed": args.seed, "threads": args.threads, "batch_size": args.batch_size,
                "normal_calibration_source": str(args.normal_calibration),
                "history": history, "loss": "RGB MSE", "learning_rate": 0.001,
                "device": "cpu", "protocol": "initial low-resolution teaching run; no test-driven tuning"}
    (args.output / "training.json").write_text(json.dumps(training, indent=2))
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([h["epoch"] for h in history], [h["loss"] for h in history], marker="o")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normal training MSE")
    fig.tight_layout()
    fig.savefig(args.output / "loss.png", dpi=150)
    plt.close(fig)
    print(f"训练正常图数：{len(paths)}；权重：{args.output / 'model.pt'}")


def reconstruct(args):
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Autoencoder().cpu()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = image_tensor(args.test, checkpoint["size"]).unsqueeze(0)
    with torch.no_grad():
        reconstruction = model(x)
    # 每个像素对RGB三个通道取平均绝对重建误差。
    anomaly_map = (x - reconstruction).abs().mean(dim=1)[0].numpy()
    original = x[0].permute(1, 2, 0).numpy()
    reconstructed = reconstruction[0].permute(1, 2, 0).numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "reconstruction_error.npy", anomaly_map)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Resized input")
    axes[1].imshow(reconstructed)
    axes[1].set_title("Autoencoder reconstruction")
    heatmap = axes[2].imshow(anomaly_map, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("RGB absolute reconstruction error")
    fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"重建图：{args.output / 'comparison.png'}；当前还没有AE专用阈值或OK/NG。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--normal-calibration", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--size", type=int, default=128)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("reconstruct")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--test", type=Path, required=True)
    inference.add_argument("--threads", type=int, default=4)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads必须为正数。")
    if args.command == "train":
        train(args)
    else:
        reconstruct(args)


if __name__ == "__main__":
    main()
