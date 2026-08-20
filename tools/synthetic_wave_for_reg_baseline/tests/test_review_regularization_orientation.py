"""Tests for the explicit orientation-review helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from review_regularization_orientation import (  # noqa: E402
    _lsq_scale,
    _ncc,
    _plane_slice,
    _signed_axis_search,
)


class OrientationHelperTests(unittest.TestCase):
    def test_plane_slices_follow_canonical_ras_axes(self) -> None:
        data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        indices = (1, 2, 3)
        np.testing.assert_array_equal(_plane_slice(data, "sagittal", indices), data[1, :, :].T)
        np.testing.assert_array_equal(_plane_slice(data, "coronal", indices), data[:, 2, :].T)
        np.testing.assert_array_equal(_plane_slice(data, "axial", indices), data[:, :, 3].T)

    def test_lsq_scale_and_ncc_are_intensity_invariant(self) -> None:
        reference = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
        candidate = reference / 4.0
        mask = np.ones(reference.shape, dtype=bool)
        self.assertAlmostEqual(_lsq_scale(reference, candidate, mask), 4.0)
        self.assertAlmostEqual(_ncc(reference, candidate, mask), 1.0)

    def test_signed_axis_search_recovers_known_two_axis_flip(self) -> None:
        rng = np.random.default_rng(123)
        reference = rng.normal(size=(8, 8, 8)).astype(np.float32)
        candidate = reference[::-1, :, ::-1]
        results = _signed_axis_search(
            reference,
            candidate,
            np.ones(reference.shape, dtype=bool),
            downsample_factor=1,
        )
        self.assertEqual(results[0]["permutation"], [0, 1, 2])
        self.assertEqual(results[0]["flips_ras_grid_axes"], [True, False, True])
        self.assertAlmostEqual(results[0]["ncc"], 1.0)


if __name__ == "__main__":
    unittest.main()
