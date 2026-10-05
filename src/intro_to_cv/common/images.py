"""Shared image loading, resizing, channel conversion and visualization backend."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图片：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Direct square resize, without cropping or letterboxing."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def gray_float(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def image_tensor(path: Path, size: int):
    import torch

    array = resize_rgb(read_rgb(path), size).astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())
