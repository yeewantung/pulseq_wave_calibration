"""Focused tests for shared registration and whole-volume metric helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_regularization_volume import (  # noqa: E402
    _json_default,
    compute_metrics,
    rigid_resample,
    rotation_matrix_xyz_degrees,
    signed_axis_transform,
)


class OrientationAndRigidTests(unittest.TestCase):
    def test_json_default_converts_numpy_scalars(self) -> None:
        self.assertEqual(_json_default(np.float32(1.25)), 1.25)
        self.assertEqual(_json_default(np.int64(4)), 4)

    def test_approved_two_axis_flip_is_proper_rotation(self) -> None:
        data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        transformed = signed_axis_transform(data, [0, 1, 2], [True, False, True])
        np.testing.assert_array_equal(transformed, data[::-1, :, ::-1])
        self.assertGreater(np.linalg.det(np.diag([-1.0, 1.0, -1.0])), 0.0)

    def test_rotation_matrix_remains_proper(self) -> None:
        rotation = rotation_matrix_xyz_degrees([1.2, -0.5, 2.0])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_zero_rigid_parameters_are_identity(self) -> None:
        data = np.arange(9 * 9 * 9, dtype=np.float32).reshape(9, 9, 9)
        np.testing.assert_allclose(rigid_resample(data, np.zeros(6)), data)


class MetricTests(unittest.TestCase):
    def test_identical_volume_has_ideal_primary_metrics(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.uniform(1.0, 100.0, size=(24, 24, 24)).astype(np.float32)
        foreground = np.ones(reference.shape, dtype=bool)
        background = np.zeros(reference.shape, dtype=bool)
        background[0] = True
        foreground[0] = False
        edge = foreground.copy()
        metadata = {
            "reference_positive_p99": float(np.percentile(reference, 99)),
            "foreground_threshold": 0.0,
        }
        metrics, scaled = compute_metrics(
            reference,
            reference,
            {"foreground": foreground, "background": background, "edge": edge},
            metadata,
        )
        np.testing.assert_allclose(scaled, reference)
        self.assertAlmostEqual(metrics["ncc_foreground"], 1.0)
        self.assertAlmostEqual(metrics["nrmse_foreground"], 0.0)
        self.assertAlmostEqual(metrics["ssim_3d_bbox"], 1.0)
        self.assertAlmostEqual(metrics["gradient_ncc_edge"], 1.0)


if __name__ == "__main__":
    unittest.main()
