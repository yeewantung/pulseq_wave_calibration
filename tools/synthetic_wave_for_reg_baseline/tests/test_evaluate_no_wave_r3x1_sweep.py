"""Tests for no-Wave R3x1 sweep metric summaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_no_wave_r3x1_sweep import metric_leaders  # noqa: E402


class NoWaveSweepEvaluationTests(unittest.TestCase):
    def test_metric_leaders_keep_objectives_separate(self) -> None:
        records = [
            {
                "lambda": 1e-4,
                "nrmse_brain": 0.05,
                "ssim_3d_brain_bbox": 0.91,
                "gradient_ncc_brain_edge": 0.99,
                "edge_preservation_ratio": 1.02,
            },
            {
                "lambda": 1e-3,
                "nrmse_brain": 0.04,
                "ssim_3d_brain_bbox": 0.95,
                "gradient_ncc_brain_edge": 0.98,
                "edge_preservation_ratio": 1.005,
            },
        ]

        leaders = metric_leaders(records)

        self.assertEqual(leaders["nrmse_brain"]["lambda"], 1e-3)
        self.assertEqual(leaders["ssim_3d_brain_bbox"]["lambda"], 1e-3)
        self.assertEqual(leaders["gradient_ncc_brain_edge"]["lambda"], 1e-4)
        self.assertEqual(leaders["edge_preservation_ratio"]["lambda"], 1e-3)


if __name__ == "__main__":
    unittest.main()
