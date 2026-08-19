"""Unit tests for coil-compression math and chunk conventions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from estimate_coil_compression import (  # noqa: E402
    accumulate_coil_covariance,
    apply_coil_compression_coillast,
    coil_basis_from_covariance,
)


class CoilCompressionTests(unittest.TestCase):
    def test_chunked_covariance_matches_direct_calculation(self) -> None:
        rng = np.random.default_rng(42)
        chunks = []
        arrays = []
        for index in range(2):
            data = (
                rng.standard_normal((8, 3, 2, 4))
                + 1j * rng.standard_normal((8, 3, 2, 4))
            ).astype(np.complex64)
            chunks.append((2 * index, 2 * index + 2, data))
            arrays.append(data)

        covariance, info = accumulate_coil_covariance(
            iter(chunks), ncoil=4, readout_step=2
        )
        design = np.concatenate(
            [np.ascontiguousarray(data[::2]).reshape(-1, 4) for data in arrays], axis=0
        )
        expected = design.conj().T @ design
        expected = 0.5 * (expected + expected.conj().T)

        np.testing.assert_allclose(covariance, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(info["chunk_count"], 2)
        self.assertEqual(info["sample_rows_considered"], design.shape[0])

    def test_basis_is_ordered_and_orthonormal(self) -> None:
        covariance = np.diag([1.0, 9.0, 4.0]).astype(np.complex128)
        basis, singular_values, energy = coil_basis_from_covariance(covariance, max_ncc=2)

        np.testing.assert_allclose(singular_values, [3.0, 2.0, 1.0])
        np.testing.assert_allclose(energy, [9 / 14, 13 / 14, 1.0])
        np.testing.assert_allclose(basis.conj().T @ basis, np.eye(2), atol=1e-7)

    def test_apply_uses_last_axis_as_coil(self) -> None:
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4).astype(np.complex64)
        basis = np.eye(4, dtype=np.complex64)[:, :2]
        compressed = apply_coil_compression_coillast(data, basis)
        np.testing.assert_array_equal(compressed, data[..., :2])

    def test_zero_rows_are_removed(self) -> None:
        data = np.zeros((4, 2, 1, 3), dtype=np.complex64)
        data[0, 0, 0] = [1, 2, 3]
        _, info = accumulate_coil_covariance(
            iter([(0, 1, data)]), ncoil=3, readout_step=1
        )
        self.assertEqual(info["nonzero_sample_rows"], 1)
        self.assertEqual(info["zero_sample_rows_removed"], 7)


if __name__ == "__main__":
    unittest.main()
