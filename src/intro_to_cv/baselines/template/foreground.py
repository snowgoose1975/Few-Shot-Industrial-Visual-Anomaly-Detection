"""亮腰果、暗背景样例的前景分割；不是通用产品分割器。"""

import cv2
import numpy as np


def foreground_region(reference, aligned, valid):
    # 分割阈值只由正常模板的亮度直方图决定，不使用缺陷标注。
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    threshold, _ = cv2.threshold(ref_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    def object_mask(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        binary = (cv2.GaussianBlur(gray, (5, 5), 0) > threshold).astype(np.uint8)
        binary[~valid] = 0
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("未找到亮前景；当前分割假设可能不适用。")
        # 单个腰果对应最大的外轮廓。填充内部，保留暗斑点作为检测区域。
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        return filled.astype(bool)

    ref_mask, test_mask = object_mask(reference), object_mask(aligned)
    # 用并集：缺损部位若不在测试前景内，正常模板仍能保留该部位。
    # 保守外扩正常物体框最长边的10%，保留轮廓、阴影和邻近缺损。
    # 本轮是探索性示例；正式实验应在正常验证集固定该规则。
    union = (ref_mask | test_mask).astype(np.uint8)
    _, _, width, height = cv2.boundingRect(ref_mask.astype(np.uint8))
    margin = max(1, int(np.ceil(0.1 * max(width, height))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    region = cv2.dilate(union, kernel).astype(bool) & valid
    return region, ref_mask, test_mask, float(threshold)
