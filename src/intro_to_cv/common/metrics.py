"""像素指标；PRO按每个8连通真实区域等权，包含正常图背景像素。"""

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def image_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("图像标签和分数应非空、同形状且分数有限。")
    tn, fp, fn, tp = confusion_matrix(labels, scores > threshold, labels=[0, 1]).ravel()
    both = np.unique(labels).size == 2
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)) if both else None,
        "image_AP": float(average_precision_score(labels, scores)) if both else None,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "FPR": float(fp / (fp + tn)) if fp + tn else None,
        "FNR": float(fn / (fn + tp)) if fn + tp else None,
    }


def localization_metrics(masks, maps, pro_limit=0.3):
    labels, scores, weights = [], [], []
    regions = 0
    for mask, anomaly_map in zip(masks, maps, strict=True):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != anomaly_map.shape or not np.isfinite(anomaly_map).all():
            raise ValueError("像素标签和分数需同尺寸，且分数全部有限。")
        count, component = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        area = np.bincount(component.ravel())
        lookup = np.ones(count, dtype=np.float64)
        # 每个真实区域全部像素的权重和=1，从而PRO是区域覆盖率的等权平均。
        lookup[1:] = 1.0 / area[1:]
        weights.append(lookup[component].ravel())
        labels.append(mask.ravel())
        scores.append(anomaly_map.ravel())
        regions += count - 1
    y, s, w = np.concatenate(labels), np.concatenate(scores), np.concatenate(weights)
    if np.unique(y).size != 2:
        return {"pixel_AUROC": None, "pixel_AP": None, "AUPRO": None,
                "pro_limit": pro_limit, "regions": regions}
    fpr, pro, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    # 在FPR上限处线性插值并截断，再除以上限，得到归一化AUPRO。
    keep = fpr < pro_limit
    x = np.r_[fpr[keep], pro_limit]
    z = np.r_[pro[keep], np.interp(pro_limit, fpr, pro)]
    return {"pixel_AUROC": float(roc_auc_score(y, s)),
            "pixel_AP": float(average_precision_score(y, s)),
            "AUPRO": float(np.trapezoid(z, x) / pro_limit),
            "pro_limit": pro_limit, "regions": regions,
            "connectivity": 8, "normal_test_images_included": True}
