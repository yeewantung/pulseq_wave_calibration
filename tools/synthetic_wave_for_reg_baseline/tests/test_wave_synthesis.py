"""Tests for extended-readout theoretical Wave synthesis helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from wave_synthesis import (  # noqa: E402
    apply_wave_forward,
    build_theoretical_psf,
    center_embed_readout,
    centered_fftn,
    logical_array_sha256,
    logical_bart_cfl_sha256,
)


class ExtendedReadoutTests(unittest.TestCase):
    def test_center_embedding_preserves_image_and_zero_exterior(self) -> None:
        image = np.arange(8 * 3 * 2, dtype=np.float32).reshape(8, 3, 2).astype(np.complex64)
        extended, support = center_embed_readout(image, 32)
        self.assertEqual(support, slice(12, 20))
        np.testing.assert_array_equal(extended[support], image)
        self.assertFalse(np.any(extended[:12]))
        self.assertFalse(np.any(extended[20:]))

    def test_centered_fft_round_trip(self) -> None:
        rng = np.random.default_rng(31)
        image = (
            rng.standard_normal((12, 7, 6)) + 1j * rng.standard_normal((12, 7, 6))
        ).astype(np.complex64)
        kspace = centered_fftn(image, axes=(0, 1, 2))
        recovered = centered_fftn(kspace, axes=(0, 1, 2), inverse=True)
        np.testing.assert_allclose(recovered, image, rtol=2e-6, atol=2e-6)


class TheoreticalPsfTests(unittest.TestCase):
    def test_bart_fortran_payload_has_same_logical_hash(self) -> None:
        array = np.arange(8 * 5 * 4, dtype=np.float32).reshape(8, 5, 4).astype(np.complex64)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "psf"
            np.ravel(array, order="F").tofile(base.with_suffix(".cfl"))
            self.assertEqual(
                logical_bart_cfl_sha256(base, array.shape),
                logical_array_sha256(array),
            )

    def test_psf_shape_and_unit_magnitude(self) -> None:
        delta_y = np.linspace(-2.0, 2.0, 16)
        delta_z = np.linspace(1.5, -1.5, 16)
        psf = build_theoretical_psf(delta_y, delta_z, ny=7, nz=6)
        self.assertEqual(psf.shape, (16, 7, 6))
        self.assertEqual(psf.dtype, np.complex64)
        np.testing.assert_allclose(np.abs(psf), 1.0, rtol=2e-7, atol=2e-7)

    def test_unity_psf_matches_full_centered_fft(self) -> None:
        rng = np.random.default_rng(41)
        image = (
            rng.standard_normal((16, 7, 6)) + 1j * rng.standard_normal((16, 7, 6))
        ).astype(np.complex64)
        local = apply_wave_forward(image, np.ones_like(image))
        expected = centered_fftn(image, axes=(0, 1, 2))
        np.testing.assert_allclose(local, expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
