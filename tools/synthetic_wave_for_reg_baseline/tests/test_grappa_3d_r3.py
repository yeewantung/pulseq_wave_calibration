"""Tests for local joint-coil 5×5×Kz R=3 GRAPPA primitives."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from grappa_3d_r3 import (  # noqa: E402
    accumulate_normal_equations_3d,
    apply_grappa_3d_block,
    calibration_matrices_3d,
    pe2_offsets,
    solve_weights_3d,
    source_matrix_for_targets_3d,
)
from reconstruct_no_wave_grappa_3d import _validate_resume_pair  # noqa: E402


class GeometryTests(unittest.TestCase):
    """Verify feature counts and kz-halo source ordering."""

    def test_calibration_matrix_uses_thirty_spatial_sources(self) -> None:
        calibration = np.ones((8, 7, 5, 3), dtype=np.complex64)
        source, target = calibration_matrices_3d(calibration, [1, 2, 3], 1)
        self.assertEqual(source.shape, (8 * 7 * 3, 30 * 3))
        self.assertEqual(target.shape, (8 * 7 * 3, 3))

    def test_five_partition_kernel_uses_fifty_spatial_sources(self) -> None:
        """A 5×5×5 kernel has 5 RO × 2 acquired PE1 × 5 PE2 locations."""
        calibration = np.ones((8, 7, 7, 3), dtype=np.complex64)
        source, target = calibration_matrices_3d(
            calibration, [2, 3, 4], 1, pe2_kernel_size=5
        )
        self.assertEqual(source.shape, (8 * 7 * 3, 50 * 3))
        self.assertEqual(target.shape, (8 * 7 * 3, 3))

    def test_pe2_kernel_must_be_positive_and_odd(self) -> None:
        """Even or empty PE2 kernels cannot have a unique centered target."""
        self.assertEqual(pe2_offsets(5), (-2, -1, 0, 1, 2))
        for invalid in (0, 2, 4):
            with self.assertRaisesRegex(ValueError, "positive odd"):
                pe2_offsets(invalid)

    def test_application_source_rows_include_three_pe2_locations(self) -> None:
        block = np.zeros((7, 9, 5, 1), dtype=np.complex64)
        for x in range(7):
            for y in range(9):
                for z in range(5):
                    block[x, y, z, 0] = 1000 * x + 10 * y + z
        source = source_matrix_for_targets_3d(block, [4], [2], 1)
        expected_first_row = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-1, 2):
                for dz in (-1, 0, 1):
                    x = dx
                    expected_first_row.append(
                        0 if x < 0 else 1000 * x + 10 * (4 + dy) + 2 + dz
                    )
        np.testing.assert_array_equal(source[0], expected_first_row)

    def test_application_source_rows_include_five_pe2_locations(self) -> None:
        """The configurable source collector preserves spatial-major ordering."""
        block = np.zeros((7, 9, 7, 1), dtype=np.complex64)
        for x in range(7):
            for y in range(9):
                for z in range(7):
                    block[x, y, z, 0] = 1000 * x + 10 * y + z
        source = source_matrix_for_targets_3d(
            block, [4], [3], 1, pe2_kernel_size=5
        )
        expected_first_row = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    expected_first_row.append(
                        0 if dx < 0 else 1000 * dx + 10 * (4 + dy) + 3 + dz
                    )
        np.testing.assert_array_equal(source[0], expected_first_row)


class ReconstructionTests(unittest.TestCase):
    """Verify joint calibration, output shape, and data consistency."""

    def test_reconstruction_is_finite_and_preserves_acquired_samples(self) -> None:
        rng = np.random.default_rng(31)
        calibration = (
            rng.standard_normal((18, 15, 5, 3))
            + 1j * rng.standard_normal((18, 15, 5, 3))
        ).astype(np.complex64)
        equations = accumulate_normal_equations_3d(calibration, [0, 1, 2, 3, 4])
        weights = solve_weights_3d(equations)
        mask = np.arange(15) % 3 == 1
        undersampled = calibration.copy()
        undersampled[:, ~mask, :, :] = 0

        output = apply_grappa_3d_block(
            undersampled, [1, 2, 3], mask, weights, acquired_residue=1
        )
        self.assertEqual(output.shape, (18, 15, 3, 3))
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(
            output[:, mask, :, :], undersampled[:, mask, :, :][:, :, 1:4, :]
        )

    def test_batched_core_matches_one_partition_calls(self) -> None:
        rng = np.random.default_rng(37)
        calibration = (
            rng.standard_normal((16, 15, 5, 2))
            + 1j * rng.standard_normal((16, 15, 5, 2))
        ).astype(np.complex64)
        weights = solve_weights_3d(
            accumulate_normal_equations_3d(calibration, range(5))
        )
        mask = np.arange(15) % 3 == 1
        undersampled = calibration.copy()
        undersampled[:, ~mask, :, :] = 0
        batched = apply_grappa_3d_block(undersampled, [1, 2, 3], mask, weights)
        separate = np.concatenate(
            [
                apply_grappa_3d_block(undersampled, [partition], mask, weights)
                for partition in (1, 2, 3)
            ],
            axis=2,
        )
        np.testing.assert_allclose(batched, separate, rtol=1e-6, atol=1e-6)

    def test_five_partition_reconstruction_preserves_acquired_samples(self) -> None:
        """The 5×5×5 path returns finite predictions and exact measured values."""
        rng = np.random.default_rng(41)
        calibration = (
            rng.standard_normal((14, 15, 7, 2))
            + 1j * rng.standard_normal((14, 15, 7, 2))
        ).astype(np.complex64)
        equations = accumulate_normal_equations_3d(
            calibration, range(7), pe2_kernel_size=5
        )
        weights = solve_weights_3d(equations)
        mask = np.arange(15) % 3 == 1
        undersampled = calibration.copy()
        undersampled[:, ~mask, :, :] = 0
        output = apply_grappa_3d_block(
            undersampled,
            [2, 3, 4],
            mask,
            weights,
            pe2_kernel_size=5,
        )
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(
            output[:, mask, :, :], undersampled[:, mask, :, :][:, :, 2:5, :]
        )


class ResumeTests(unittest.TestCase):
    """Ensure incomplete checkpoint pairs cannot skip unwritten output."""

    def test_resume_rejects_data_without_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "output.npy"
            progress = Path(directory) / "progress.json"
            data.touch()
            with self.assertRaisesRegex(ValueError, "both checkpoint files"):
                _validate_resume_pair(data, progress, resume=True)

    def test_fresh_or_complete_pair_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "output.npy"
            progress = Path(directory) / "progress.json"
            _validate_resume_pair(data, progress, resume=True)
            data.touch()
            progress.touch()
            _validate_resume_pair(data, progress, resume=True)


if __name__ == "__main__":
    unittest.main()
