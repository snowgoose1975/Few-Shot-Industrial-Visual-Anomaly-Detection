"""VisA metadata; use official 1-class CSV and binary masks (nonzero = defect)."""

import csv
from pathlib import Path

import cv2
import numpy as np


def read_split(root: Path):
    with (root / "split_csv/1cls.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def ground_truth(root: Path, row: dict, size: int):
    if row["label"] == "normal":
        return np.zeros((size, size), dtype=bool)
    raw = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"无法读取GT：{row['mask']}")
    return cv2.resize((raw > 0).astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
