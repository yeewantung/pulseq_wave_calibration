"""Focused tests for the ESPIRiT intensity-correction pilot comparison."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_ecalib_intensity_pilot import _agreement  # noqa: E402


class AgreementTests(unittest.TestCase):
    def test_positive_lsq_removes_global_scale_only(self) -> None:
        reference = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
        mask = np.ones(reference.shape, dtype=bool)
        result = _agreement(reference, reference / 7.0, mask)
        self.assertAlmostEqual(result["scale"], 7.0)
        self.assertAlmostEqual(result["ncc"], 1.0)
        self.assertAlmostEqual(result["nrmse"], 0.0)

    def test_spatial_bias_is_not_removed_by_global_scale(self) -> None:
        reference = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
        biased = reference.copy()
        biased[1:] *= 2.0
        result = _agreement(reference, biased, np.ones(reference.shape, dtype=bool))
        self.assertLess(result["ncc"], 1.0)
        self.assertGreater(result["nrmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
