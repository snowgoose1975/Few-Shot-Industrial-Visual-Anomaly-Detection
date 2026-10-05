"""Template feature-free scoring, independent of command-line parsing."""

import numpy as np
from intro_to_cv.baselines.template.alignment import register_ecc
from intro_to_cv.baselines.template.foreground import foreground_region
from intro_to_cv.baselines.template.maps import difference_maps
from intro_to_cv.common.scoring import top_fraction_mean


def score_details(reference, test):
    aligned, valid, warp, _ = register_ecc(reference, test)
    maps, safe = difference_maps(reference, aligned, valid, 11, 11)
    foreground, _, _, _ = foreground_region(reference, aligned, valid)
    roi = safe & foreground
    anomaly_map = maps["smoothed"]
    values = anomaly_map[roi]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("没有有效前景分数。")
    # 小缺陷可能被全图平均稀释；先固定使用最高1%像素的平均值。
    score = top_fraction_mean(values)
    return float(score), values, aligned, anomaly_map, roi, warp


def image_score(reference, test):
    return score_details(reference, test)[0]

