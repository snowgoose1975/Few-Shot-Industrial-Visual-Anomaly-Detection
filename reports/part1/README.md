# Part 1 报告与Presentation素材

当前保留的是腰果类别开发阶段的统一128×128对照。原生分辨率与旧148张子集的教学成绩未混入结果表。

## 结果文件

- `results/metrics.csv`：可直接用于报告表格。
- `results/metrics.json`：完整指标、混淆矩阵、光照误报与本机时间记录。
- `results/protocol.json`：参考/校准清单、预算、尺寸、区域评价、ECC失败回退。
- `results/thresholds.json`：两条方法各自的正常校准阈值，均适用于统一128评分。
- `results/predictions.csv`：每个方法150张测试图的预测，保留FP/FN供失败分析。
- `results/lighting_probe.csv`：50张正常测试图固定0.8倍亮度下的记录。
- `report_notes.md`：结果解释与限制，可作为正文草稿的依据。

## 图像

- `figures/comparison_000.png`、`comparison_001.png`：两条方法的输入/真实mask/连续热图/二值预测。它们是开发阶段展示例，不能用于证明未见样例效果。
- `figures/ae_training_loss.png`：正常重建训练损失。
- `figures/ae_normal_reconstruction.png`、`ae_anomaly_reconstruction.png`：正常/异常图片的输入、重建、误差；用于解释重建模糊与定位局限。

## Presentation建议（3页）

1. **两条baseline如何工作**：正常模板→ECC→差分；正常图片→AE训练→重建误差。强调正常训练与独立正常校准。
2. **检测与定位要分开看**：结果表配对照图；本实验AE图像检测较好，差分定位指标较好，不能把一个指标的提升当全面改善。
3. **失败与主线动机**：轮廓/背景误差、细缺陷分辨率、重建模糊、光照变化；自然引出Part 2冻结特征与正常patch检索。不要提前宣称DINO已解决这些问题。

模型与训练/校准元数据保存在 `artifacts/part1/`。旧教学中间文件和重复check结果已移到本地 `.local_archive/`，不上传到GitHub。新实验使用 `outputs/`，重新生成后再选择需要发布的报告快照。
