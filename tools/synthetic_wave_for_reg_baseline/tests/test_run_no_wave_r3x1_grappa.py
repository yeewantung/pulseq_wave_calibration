"""Tests for retrospective no-Wave R3x1 GRAPPA orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_no_wave_r3x1_grappa import (  # noqa: E402
    reconstruct_retrospective_grappa,
    rss_magnitude_with_sensitivity_phase,
)


class RetrospectiveGrappaTests(unittest.TestCase):
    def test_reconstruction_preserves_acquired_samples_and_resumes(self) -> None:
        rng = np.random.default_rng(7)
        source = (
            rng.standard_normal((4, 7, 5, 2))
            + 1j * rng.standard_normal((4, 7, 5, 2))
        ).astype(np.complex64)
        mask = np.asarray([False, True, False, False, True, False, False])

        def fake_apply(block, core, acquired, _weights, **_kwargs):
            result = block[:, :, core, :].copy()
            result[:, ~np.asarray(acquired), :, :] = np.complex64(2 + 3j)
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "completed.npy"
            progress = root / "progress.json"
            with patch("run_no_wave_r3x1_grappa.apply_grappa_3d_block", fake_apply):
                report = reconstruct_retrospective_grappa(
                    source,
                    mask,
                    {},
                    output,
                    progress,
                    chunk_size=2,
                    pe2_kernel_size=5,
                    resume=False,
                )
                resumed = reconstruct_retrospective_grappa(
                    source,
                    mask,
                    {},
                    output,
                    progress,
                    chunk_size=2,
                    pe2_kernel_size=5,
                    resume=True,
                )

            completed = np.load(output)
            np.testing.assert_array_equal(completed[:, mask, :, :], source[:, mask, :, :])
            self.assertTrue(np.all(completed[:, ~mask, :, :] == np.complex64(2 + 3j)))
            self.assertTrue(report["acquired_samples_equal_source_bitwise"])
            self.assertEqual(resumed["predicted_nonzero_count"], report["predicted_nonzero_count"])
            self.assertTrue(json.loads(progress.read_text())["complete"])

    def test_combination_keeps_rss_magnitude_and_aligned_phase(self) -> None:
        shape = (4, 5, 6)
        object_image = np.ones(shape, dtype=np.complex64) * np.exp(0.7j)
        sensitivities = np.empty((*shape, 2), dtype=np.complex64)
        sensitivities[..., 0] = np.complex64(0.6)
        sensitivities[..., 1] = np.complex64(0.8j)
        coil_images = sensitivities * object_image[..., None]
        kspace = np.empty_like(coil_images)
        for coil in range(2):
            kspace[..., coil] = np.fft.fftshift(
                np.fft.fftn(np.fft.ifftshift(coil_images[..., coil]), norm="ortho")
            ).astype(np.complex64)

        combined = rss_magnitude_with_sensitivity_phase(
            kspace,
            sensitivities[..., None],
            map_power_threshold_fraction=1e-8,
        )

        np.testing.assert_allclose(np.abs(combined), 1.0, atol=2e-6)
        np.testing.assert_allclose(np.angle(combined), 0.7, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
