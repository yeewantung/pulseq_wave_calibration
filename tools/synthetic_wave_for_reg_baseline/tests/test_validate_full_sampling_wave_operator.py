"""Tests for full-sampling Wave operator acceptance gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_full_sampling_wave_operator import (  # noqa: E402
    complex_error_metrics,
    validate_coil_operator,
)
from wave_synthesis import (  # noqa: E402
    SPATIAL_AXES,
    apply_wave_forward,
    center_embed_readout,
    centered_fftn,
)


class OperatorValidationTests(unittest.TestCase):
    def test_real_data_gate_math_is_exact_on_synthetic_arrays(self) -> None:
        rng = np.random.default_rng(91)
        source_kspace = (
            rng.standard_normal((8, 5, 4)) + 1j * rng.standard_normal((8, 5, 4))
        ).astype(np.complex64)
        source_image = centered_fftn(
            source_kspace, axes=SPATIAL_AXES, inverse=True
        )
        extended, _ = center_embed_readout(source_image, 16)
        phase = rng.uniform(-np.pi, np.pi, extended.shape)
        psf = np.exp(1j * phase).astype(np.complex64)
        full_wave = apply_wave_forward(extended, psf)

        metrics = validate_coil_operator(
            source_kspace, full_wave, psf, workers=1
        )
        self.assertLess(
            metrics["psf_one_no_wave_identity"]["relative_complex_l2"], 2e-6
        )
        self.assertLess(
            metrics["full_sampling_wave_inverse"]["relative_complex_l2"], 3e-6
        )
        self.assertLess(
            metrics["full_sampling_wave_inverse"][
                "recovered_exterior_energy_fraction"
            ],
            1e-12,
        )

    def test_complex_error_metrics_reports_known_relative_error(self) -> None:
        reference = np.ones((6, 3, 2), dtype=np.complex64)
        candidate = reference * np.complex64(1.0 + 1e-3j)
        metrics = complex_error_metrics(reference, candidate, x_chunk=2)
        self.assertAlmostEqual(metrics["relative_complex_l2"], 1e-3, places=7)
        self.assertAlmostEqual(
            metrics["relative_maximum_complex_error"], 1e-3, places=7
        )


if __name__ == "__main__":
    unittest.main()
