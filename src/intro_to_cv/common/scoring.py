"""Shared image aggregation, normal-only thresholds and binary-region cleanup."""

import cv2
import numpy as np


def top_fraction_mean(values, fraction=0.01):
    values = np.asarray(values).ravel()
    if values.size == 0 or not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError("分数应有限且非空；fraction必须位于(0,1]。")
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def normal_threshold(values, target_fpr):
    values = np.asarray(values).ravel()
    if values.size == 0 or not np.isfinite(values).all() or not 0 < target_fpr < 1:
        raise ValueError("校准分数应有限且非空；target_fpr应位于(0,1)。")
    return float(np.quantile(values, 1 - target_fpr, method="higher"))


def clean_regions(prediction, roi=None):
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if roi is not None:
        opened[~roi] = 0
    result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    if roi is not None:
        result[~roi] = 0
    return result
