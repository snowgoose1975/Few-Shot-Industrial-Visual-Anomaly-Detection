"""Meaningful shared contracts: scoring, region weighting, shapes and data leakage."""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from intro_to_cv.baselines.autoencoder.model import Autoencoder
from intro_to_cv.common.metrics import image_metrics, localization_metrics
from intro_to_cv.common.scoring import normal_threshold, top_fraction_mean


class BaselineContracts(unittest.TestCase):
    def test_image_scoring_and_strict_threshold(self):
        self.assertEqual(top_fraction_mean(np.arange(100)), 99.0)
        threshold = normal_threshold([0.1, 0.2, 0.3, 0.4], 0.25)
        self.assertEqual(threshold, 0.4)
        result = image_metrics([0, 1], [threshold, threshold + 0.1], threshold)
        self.assertEqual(result["TN"], 1)
        self.assertEqual(result["TP"], 1)

    def test_perfect_pixel_prediction_including_normal_image(self):
        mask = np.zeros((20, 20), bool)
        mask[1:3, 1:3] = True
        mask[8:13, 8:13] = True
        result = localization_metrics([mask, np.zeros_like(mask)],
                                      [mask.astype(float), np.zeros_like(mask, dtype=float)])
        self.assertEqual(result["regions"], 2)
        for metric in ["pixel_AUROC", "pixel_AP", "AUPRO"]:
            self.assertAlmostEqual(result[metric], 1)

    def test_pro_weights_regions_equally_not_by_area(self):
        mask = np.zeros((20, 20), bool)
        mask[1, 1] = True
        mask[8:11, 8:11] = True
        scores = np.full(mask.shape, 0.5)
        scores[8:11, 8:11] = 1
        scores[1, 1] = 0
        # Before any background false positives, only the larger of two regions is found.
        # Region coverage is 1/2, not 9/10.
        result = localization_metrics([mask], [scores])
        self.assertAlmostEqual(result["AUPRO"], 0.5)

    def test_constant_scores_are_not_perfect(self):
        mask = np.zeros((10, 10), bool)
        mask[2:4, 2:4] = True
        result = localization_metrics([mask], [np.full(mask.shape, 0.5)])
        self.assertAlmostEqual(result["pixel_AUROC"], 0.5)
        self.assertAlmostEqual(result["AUPRO"], 0.15)

    def test_autoencoder_shape_and_range(self):
        model = Autoencoder().eval()
        with torch.no_grad():
            result = model(torch.zeros(2, 3, 128, 128))
        self.assertEqual(tuple(result.shape), (2, 3, 128, 128))
        self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))

    def test_saved_training_and_calibration_do_not_overlap(self):
        artifact = Path(__file__).resolve().parents[1] / "artifacts/part1"
        training = json.loads((artifact / "autoencoder/training.json").read_text())
        calibration = json.loads((artifact / "normal_calibration/calibration_records.json").read_text())
        train = set(training["train_images"])
        normal = {row["image"] for row in calibration["normal_calibration"]}
        self.assertEqual(len(train), 407)
        self.assertEqual(len(normal), 40)
        self.assertFalse(train & normal)
        self.assertTrue(all("/Normal/" in path for path in train | normal))


if __name__ == "__main__":
    unittest.main()
