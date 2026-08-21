"""Focused tests for retrospective low-resolution Wave-MPRAGE support."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import create_cfl, open_cfl, read_shape
from wave_retro_lr.core import (
    CaseSpec,
    Geometry,
    apply_wave_forward,
    build_case_mask,
    build_wave_options,
    center_crop_bounds,
    evaluate_psf_phase_planes,
    extract_psf_phase_planes,
    psf_identity_metrics,
    resolve_case,
)
from wave_retro_lr.pipeline import _write_target_calibration, _write_target_maps


def geometry() -> Geometry:
    return Geometry((256.0, 256.0, 256.0), (256, 256, 256))


class GeometryTests(unittest.TestCase):
    def test_center_crop_preserves_python_center(self) -> None:
        for source in (255, 256):
            for target in (127, 128, 129, 130):
                start, stop = center_crop_bounds(source, target)
                self.assertEqual(stop - start, target)
                self.assertEqual(start + target // 2, source // 2)

    def test_planned_physical_resolutions_crop_only_pe(self) -> None:
        expected = {
            (1.5, 1.0, 1.0): (256, 256, 172),
            (1.0, 1.5, 1.0): (256, 172, 256),
            (1.25, 1.25, 1.0): (256, 204, 204),
        }
        for resolution, logical_shape in expected.items():
            case = resolve_case(CaseSpec(resolution, (3, 2)), geometry())
            self.assertEqual(case.target_logical_matrix_ro_lin_par, logical_shape)
            self.assertEqual(case.target_physical_matrix_xyz[2], 256)

    def test_readout_resolution_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical-Z/readout"):
            resolve_case(CaseSpec((1.0, 1.0, 1.5), (3, 2)), geometry())


class SamplingTests(unittest.TestCase):
    def test_crop_preserves_source_lattice_and_fully_sampled_acs(self) -> None:
        source = np.zeros((256, 256), dtype=bool)
        source[1::3, ::2] = True
        source[115:139, :] = True
        case = resolve_case(CaseSpec((1.0, 1.5, 1.0), (3, 2)), geometry())
        target = build_case_mask(source, case, (3, 2))
        self.assertEqual(target.shape, (172, 256))
        shifted_acs = slice(115 - case.crop_bounds_lin[0], 139 - case.crop_bounds_lin[0])
        self.assertTrue(np.all(target[shifted_acs, :]))
        self.assertTrue(target[target.shape[0] // 2, target.shape[1] // 2])

    def test_cannot_change_an_already_accelerated_axis(self) -> None:
        source = np.ones((256, 256), dtype=bool)
        case = resolve_case(CaseSpec((1.0, 1.0, 1.0), (6, 2)), geometry())
        with self.assertRaisesRegex(ValueError, "already accelerated"):
            build_case_mask(source, case, (3, 2))


class PsfAndOperatorTests(unittest.TestCase):
    def test_phase_plane_extraction_is_source_grid_identity(self) -> None:
        # A 256x256 plane exercises the complex128 reduction required by the
        # real source PSF; complex64 accumulation biases these large means.
        alpha = np.array([0.0, 45.0, 90.0])
        beta = np.array([0.0, -20.0, 3.0])
        gamma = np.array([-0.5, 0.0, 0.5])
        psf = evaluate_psf_phase_planes(alpha, beta, gamma, 256, 256)
        found = extract_psf_phase_planes(psf, readout_chunk=3)
        metrics = psf_identity_metrics(psf, *found, readout_chunk=4)
        self.assertLess(metrics["relative_complex_l2"], 2e-7)
        self.assertLess(metrics["maximum_complex_error"], 5e-7)

    def test_crop_after_wave_encoding_is_not_assumed_equivalent(self) -> None:
        rng = np.random.default_rng(42)
        no_wave = (
            rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8))
        ).astype(np.complex64)
        alpha = np.linspace(-5.0, 5.0, 8)
        beta = np.linspace(3.0, -3.0, 8)
        gamma = np.zeros(8)
        full_psf = evaluate_psf_phase_planes(alpha, beta, gamma, 8, 8)
        target_psf = evaluate_psf_phase_planes(alpha, beta, gamma, 6, 6)
        crop = slice(*center_crop_bounds(8, 6))
        crop_first = apply_wave_forward(
            no_wave[:, crop, crop], target_psf, readout_oversampled=8
        )
        encode_first = apply_wave_forward(
            no_wave, full_psf, readout_oversampled=8
        )[:, crop, crop]
        relative = np.linalg.norm(crop_first - encode_first) / np.linalg.norm(crop_first)
        self.assertGreater(relative, 1e-2)

        no_wave_psf = np.ones((8, 8, 8), dtype=np.complex64)
        no_wave_target = np.ones((8, 6, 6), dtype=np.complex64)
        crop_first_no_wave = apply_wave_forward(
            no_wave[:, crop, crop], no_wave_target, readout_oversampled=8
        )
        encode_first_no_wave = apply_wave_forward(
            no_wave, no_wave_psf, readout_oversampled=8
        )[:, crop, crop]
        np.testing.assert_allclose(crop_first_no_wave, encode_first_no_wave, atol=2e-6)


class BartContractTests(unittest.TestCase):
    def test_all_reconstruction_options_require_gpu(self) -> None:
        for regularizer, value in (("none", None), ("wavelet", 1e-4), ("llr", 2e-5)):
            options = build_wave_options(
                regularizer,
                value,
                block_size=8,
                iterations=100,
                tolerance=1e-6,
                maximum_eigenvalue=None,
            )
            self.assertIn("-g", options)
        llr = build_wave_options(
            "llr", 2e-5, block_size=8, iterations=100, tolerance=1e-6, maximum_eigenvalue=None
        )
        self.assertIn("-v", llr)

    def test_cfl_round_trip_and_target_pe_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            maps = create_cfl(root / "maps", (4, 5, 6, 2, 1))
            maps[:] = 1 + 0j
            maps.flush()
            del maps
            calibration = create_cfl(root / "calibration", (4, 5, 6, 2))
            values = np.arange(4 * 5 * 6 * 2, dtype=np.float32).reshape(
                (4, 5, 6, 2), order="F"
            )
            calibration[:] = values
            calibration.flush()
            del calibration

            small_geometry = Geometry((6.0, 5.0, 4.0), (4, 5, 6))
            case = resolve_case(CaseSpec((1.5, 1.25, 1.0), (1, 1)), small_geometry)
            _write_target_maps(root / "maps", root / "target_maps", (4, 4, 4))
            _write_target_calibration(root / "calibration", root / "target_calibration", case)
            self.assertEqual(read_shape(root / "target_maps"), (4, 4, 4, 2, 1))
            self.assertEqual(read_shape(root / "target_calibration"), (4, 4, 4, 2))
            target_maps = open_cfl(root / "target_maps")
            rss = np.sqrt(np.sum(np.abs(target_maps[:, :, :, :, 0]) ** 2, axis=3))
            np.testing.assert_allclose(rss, 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
