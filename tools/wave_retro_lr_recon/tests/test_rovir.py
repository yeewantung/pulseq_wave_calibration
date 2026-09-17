"""Focused tests for the nonexecuting BART ROVir integration frame."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir import (  # noqa: E402
    bart_ccapply_command,
    bart_rovir_command,
    build_rovir_command_plan,
    masked_coil_images,
    region_correlation_diagnostics,
    region_energy_curve,
    rovir_matrix_view,
    validate_region_masks,
    validate_rovir_transform,
)


class RovirMaskTests(unittest.TestCase):
    """Validate the conservative disjoint-region contract."""

    def test_masks_and_masked_coil_images(self) -> None:
        """Apply signal and nuisance masks without changing array axes.

        Returns:
            None.
        """
        images = np.arange(24, dtype=np.float32).reshape(2, 3, 4).astype(np.complex64)
        signal = np.zeros((2, 3), dtype=np.float32)
        signal[0] = 1
        interference = np.zeros((2, 3), dtype=np.float32)
        interference[1] = 0.5
        report = validate_region_masks(signal, interference, (2, 3))
        self.assertEqual(report["signal_support_voxels"], 3)
        self.assertEqual(report["interference_support_voxels"], 3)
        positive, negative = masked_coil_images(images, signal, interference)
        np.testing.assert_array_equal(positive[0], images[0])
        np.testing.assert_array_equal(positive[1], 0)
        np.testing.assert_array_equal(negative[0], 0)
        np.testing.assert_array_equal(negative[1], images[1] * 0.5)

    def test_overlapping_masks_are_rejected(self) -> None:
        """Reject ambiguous voxels assigned to both scientific regions.

        Returns:
            None.
        """
        signal = np.array([1.0, 0.0])
        interference = np.array([1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_region_masks(signal, interference, signal.shape)

    def test_nonfinite_and_empty_masks_are_rejected(self) -> None:
        """Reject masks that cannot define valid correlation matrices.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_region_masks(np.array([np.nan, 1.0]), np.array([1.0, 0.0]), (2,))
        with self.assertRaisesRegex(ValueError, "no positive support"):
            validate_region_masks(np.zeros(2), np.ones(2), (2,))

    def test_singular_interference_correlation_is_rejected(self) -> None:
        """Stop before BART when the generalized eigenproblem is singular.

        Returns:
            None.
        """
        images = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [2.0, 0.0]],
            ],
            dtype=np.complex64,
        )
        signal = np.array([[1.0, 1.0], [0.0, 0.0]])
        interference = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "too singular"):
            region_correlation_diagnostics(images, signal, interference)

    def test_full_rank_interference_correlation_is_reported(self) -> None:
        """Report a well-conditioned two-coil nuisance correlation matrix.

        Returns:
            None.
        """
        images = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.complex64,
        )
        signal = np.array([[1.0, 1.0], [0.0, 0.0]])
        interference = np.array([[0.0, 0.0], [1.0, 1.0]])
        report = region_correlation_diagnostics(images, signal, interference)
        self.assertEqual(report["interference_rank_at_floor"], 2)
        self.assertAlmostEqual(report["interference_condition_number"], 4.0)


class RovirTransformTests(unittest.TestCase):
    """Validate BART transform shape and orthogonality conventions."""

    def test_bart_shaped_transform_is_accepted(self) -> None:
        """Interpret BART coil and maps dimensions as matrix rows and columns.

        Returns:
            None.
        """
        transform = np.zeros((1, 1, 1, 2, 2, 1), dtype=np.complex64)
        transform[0, 0, 0, :, :, 0] = np.eye(2, dtype=np.complex64)
        np.testing.assert_array_equal(rovir_matrix_view(transform), np.eye(2))
        report = validate_rovir_transform(transform)
        self.assertEqual(report["physical_coils"], 2)
        self.assertEqual(report["maximum_gram_residual"], 0.0)

    def test_nonorthogonal_transform_is_rejected(self) -> None:
        """Reject a transform that would alter coil-domain noise scaling.

        Returns:
            None.
        """
        transform = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "not orthonormal"):
            validate_rovir_transform(transform)

    def test_region_curve_reports_requested_counts_without_selection(self) -> None:
        """Report the cumulative tradeoff for all requested dimensions.

        Returns:
            None.
        """
        images = np.array(
            [
                [[2.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 3.0]],
            ],
            dtype=np.complex64,
        )
        signal = np.array([[1.0, 0.0], [1.0, 0.0]])
        interference = np.array([[0.0, 1.0], [0.0, 1.0]])
        report = region_energy_curve(
            images,
            np.eye(2, dtype=np.complex64),
            signal,
            interference,
            (1, 2),
            voxel_chunk=1,
        )
        entries = report["channel_counts"]
        self.assertEqual([entry["virtual_coils"] for entry in entries], [1, 2])
        self.assertAlmostEqual(entries[0]["signal_retention_fraction"], 1.0)
        self.assertAlmostEqual(entries[0]["interference_remaining_fraction"], 0.0)
        self.assertAlmostEqual(entries[1]["signal_retention_fraction"], 1.0)
        self.assertAlmostEqual(entries[1]["interference_remaining_fraction"], 1.0)

    def test_region_curve_matches_bart_ccapply_complex_convention(self) -> None:
        """Conjugate the stored transform exactly as BART forward ccapply does.

        Returns:
            None.
        """
        images = np.array(
            [
                [[1.0 + 2.0j, 3.0 - 1.0j], [0.5 - 1.0j, -2.0 + 3.0j]],
                [[-2.0 + 1.0j, 0.5 + 4.0j], [2.0 + 0.5j, 1.0 - 2.0j]],
            ],
            dtype=np.complex64,
        )
        scale = np.float32(1.0 / np.sqrt(2.0))
        transform = np.array(
            [[scale, 1.0j * scale], [1.0j * scale, scale]],
            dtype=np.complex64,
        )
        signal = np.array([[1.0, 0.0], [1.0, 0.0]])
        interference = np.array([[0.0, 1.0], [0.0, 1.0]])
        report = region_energy_curve(
            images,
            transform,
            signal,
            interference,
            (1, 2),
            voxel_chunk=1,
        )

        projected = images.reshape(-1, 2) @ transform.conj()
        signal_support = signal.reshape(-1).astype(bool)
        interference_support = interference.reshape(-1).astype(bool)
        expected_signal = np.sum(np.abs(projected[signal_support]) ** 2, axis=0)
        expected_interference = np.sum(
            np.abs(projected[interference_support]) ** 2,
            axis=0,
        )
        np.testing.assert_allclose(
            report["signal_energy_by_ordered_virtual_coil"],
            expected_signal,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            report["interference_energy_by_ordered_virtual_coil"],
            expected_interference,
            rtol=1e-6,
        )


class RovirCommandTests(unittest.TestCase):
    """Keep the BART backend explicit and nonexecuting."""

    def test_commands_use_native_bart_rovir_and_one_shared_transform(self) -> None:
        """Build commands for transform estimation and matched projections.

        Returns:
            None.
        """
        self.assertEqual(
            bart_rovir_command("positive.cfl", "negative.hdr", "transform"),
            ("bart", "rovir", "positive", "negative", "transform"),
        )
        self.assertEqual(
            bart_ccapply_command("image", "transform", "image_rovir", 16),
            ("bart", "ccapply", "-p", "16", "image", "transform", "image_rovir"),
        )
        plan = build_rovir_command_plan(
            "positive",
            "negative",
            "transform",
            "image",
            "acs",
            "image_rovir",
            "acs_rovir",
            20,
        )
        self.assertEqual(plan["project_image_kspace"][5], "transform")
        self.assertEqual(plan["project_calibration_kspace"][5], "transform")

    def test_module_does_not_launch_external_processes(self) -> None:
        """Keep reviewed execution in a future explicit shell entry point.

        Returns:
            None.
        """
        import wave_retro_lr.rovir as rovir

        source = inspect.getsource(rovir)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("Popen", source)


if __name__ == "__main__":
    unittest.main()
