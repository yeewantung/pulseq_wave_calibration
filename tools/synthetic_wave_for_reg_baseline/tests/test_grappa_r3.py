"""Tests for explicit R=3 GRAPPA geometry and invariants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from grappa_r3 import (  # noqa: E402
    SOURCE_PE1_OFFSETS,
    accumulate_normal_equations,
    apply_grappa_plane,
    apply_grappa_volume,
    apply_grappa_volume_partitionwise,
    calibration_matrices,
    solve_weights,
    source_matrix_for_targets,
)
from export_multicoil_nifti import centered_ifft3, output_basename  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_source_pe1_offsets_match_pygrappa_r3_patterns(self) -> None:
        self.assertEqual(SOURCE_PE1_OFFSETS[1], (-1, 2))
        self.assertEqual(SOURCE_PE1_OFFSETS[2], (-2, 1))

    def test_source_order_is_readout_then_pe1_then_coil(self) -> None:
        plane = np.zeros((7, 7, 1), dtype=np.complex64)
        for x in range(7):
            for y in range(7):
                plane[x, y, 0] = 100 * x + y

        source = source_matrix_for_targets(plane, [3], target_offset=1)
        expected = []
        for x in range(7):
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-1, 2):
                    sx = x + dx
                    expected.append(0 if sx < 0 or sx >= 7 else 100 * sx + 3 + dy)
        np.testing.assert_array_equal(source[:, 0::1].ravel(), expected)

    def test_calibration_matrix_sizes_include_zero_padded_edges(self) -> None:
        calibration = np.ones((8, 6, 2, 3), dtype=np.complex64)
        source, target = calibration_matrices(calibration, target_offset=1)
        self.assertEqual(source.shape, (8 * 6 * 2, 10 * 3))
        self.assertEqual(target.shape, (8 * 6 * 2, 3))


class ReconstructionTests(unittest.TestCase):
    def test_centered_ifft3_round_trip(self) -> None:
        rng = np.random.default_rng(19)
        image = (
            rng.standard_normal((8, 7, 6)) + 1j * rng.standard_normal((8, 7, 6))
        ).astype(np.complex64)
        kspace = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(image), norm="ortho")
        )
        np.testing.assert_allclose(centered_ifft3(kspace), image, rtol=1e-6, atol=1e-6)

    def test_nifti_name_encodes_echo_coil_and_part(self) -> None:
        self.assertEqual(
            output_basename("product", 1, 24, "phase"),
            "sub-product_echo-01_coil-24_part-phase_GRAPPA",
        )

    def test_batched_volume_matches_plane_application(self) -> None:
        rng = np.random.default_rng(11)
        calibration = (
            rng.standard_normal((20, 18, 3, 4))
            + 1j * rng.standard_normal((20, 18, 3, 4))
        ).astype(np.complex64)
        equations = accumulate_normal_equations(calibration)
        weights = solve_weights(equations, max_ncc=4, ncc=4, regularization=0.01)
        mask = np.arange(calibration.shape[1]) % 3 == 1
        undersampled = calibration.copy()
        undersampled[:, ~mask, :, :] = 0

        batched = apply_grappa_volume(undersampled, mask, weights)
        planes = np.stack(
            [apply_grappa_plane(undersampled[:, :, index, :], mask, weights) for index in range(3)],
            axis=2,
        )
        np.testing.assert_allclose(batched, planes, rtol=1e-6, atol=1e-6)

    def test_acquired_samples_remain_bitwise_identical(self) -> None:
        rng = np.random.default_rng(1)
        calibration = (
            rng.standard_normal((20, 18, 2, 4))
            + 1j * rng.standard_normal((20, 18, 2, 4))
        ).astype(np.complex64)
        equations = accumulate_normal_equations(calibration)
        weights = solve_weights(equations, max_ncc=4, ncc=4, regularization=0.01)

        plane = calibration[:, :, 0].copy()
        mask = np.arange(plane.shape[1]) % 3 == 1
        plane[:, ~mask] = 0
        reconstructed = apply_grappa_plane(plane, mask, weights)
        np.testing.assert_array_equal(reconstructed[:, mask], plane[:, mask])

    def test_partitionwise_mode_matches_independent_plane_calibrations(self) -> None:
        """Each PE2 plane must use its own joint-multicoil ACS equations."""
        rng = np.random.default_rng(23)
        calibration = (
            rng.standard_normal((20, 18, 3, 4))
            + 1j * rng.standard_normal((20, 18, 3, 4))
        ).astype(np.complex64)
        mask = np.arange(calibration.shape[1]) % 3 == 1
        undersampled = calibration.copy()
        undersampled[:, ~mask, :, :] = 0

        reconstructed = apply_grappa_volume_partitionwise(
            undersampled,
            calibration,
            mask,
        )
        expected = []
        for partition in range(calibration.shape[2]):
            equations = accumulate_normal_equations(
                calibration[:, :, partition : partition + 1, :]
            )
            weights = solve_weights(
                equations, max_ncc=4, ncc=4, regularization=0.01
            )
            expected.append(
                apply_grappa_plane(undersampled[:, :, partition, :], mask, weights)
            )
        np.testing.assert_allclose(
            reconstructed, np.stack(expected, axis=2), rtol=1e-6, atol=1e-6
        )

    def test_interior_matches_pygrappa_reference(self) -> None:
        try:
            from pygrappa import grappa as pygrappa_grappa
        except ImportError as exc:  # pragma: no cover - optional reference dependency
            self.skipTest(f"pygrappa reference unavailable: {exc}")

        rng = np.random.default_rng(7)
        calibration = (
            rng.standard_normal((28, 24, 3))
            + 1j * rng.standard_normal((28, 24, 3))
        ).astype(np.complex64)
        undersampled = calibration.copy()
        mask = np.arange(24) % 3 == 1
        undersampled[:, ~mask] = 0

        equations = accumulate_normal_equations(calibration[:, :, None, :])
        weights = solve_weights(equations, max_ncc=3, ncc=3, regularization=0.01)
        local = apply_grappa_plane(undersampled, mask, weights)
        reference = pygrappa_grappa(
            undersampled,
            calibration,
            kernel_size=(5, 5),
            coil_axis=-1,
            lamda=0.01,
            silent=True,
        )

        valid_y = []
        for y in np.flatnonzero(~mask):
            offset = (y - 1) % 3
            if offset in (1, 2) and all(
                0 <= y + dy < 24 for dy in SOURCE_PE1_OFFSETS[offset]
            ):
                valid_y.append(y)
        np.testing.assert_allclose(
            local[2:-2, valid_y], reference[2:-2, valid_y], rtol=2e-4, atol=2e-4
        )


if __name__ == "__main__":
    unittest.main()
