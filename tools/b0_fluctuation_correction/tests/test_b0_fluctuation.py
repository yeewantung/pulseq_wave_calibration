"""Unit tests for the Wave B0-fluctuation tools on a small synthetic problem."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from b0_fluctuation.acquisition import LineTiming, timing_from_loop_order  # noqa: E402
from b0_fluctuation.cfl import read_cfl, write_cfl  # noqa: E402
from b0_fluctuation.field_model import FieldModelSettings, run_linear_field_navigator  # noqa: E402
from b0_fluctuation.navigator import (  # noqa: E402
    NavigatorSettings,
    apply_line_phase,
    gaussian_time_kernel,
    run_consistency_navigator,
)
from b0_fluctuation.recon import cg_sense  # noqa: E402
from b0_fluctuation.simulation import FluctuationModel, add_noise, inject_fluctuation  # noqa: E402
from b0_fluctuation.wave_operator import WaveOperator, fftc, ifftc  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _problem(seed: int = 0, coils: int = 8, shape=(16, 24, 16), oversampling: int = 3, acceleration: int = 2):
    """Build a small Wave problem with smooth localized coils and a sine-wave PSF.

    Args:
        seed: Random seed.
        coils: Coil count.
        shape: Image dimensions (X, Y, Z).
        oversampling: Readout oversampling of the Wave grid.
        acceleration: LIN acceleration.

    Returns:
        Operator, ground-truth image, and loop-order timing.
    """
    generator = np.random.default_rng(seed)
    x, y, z = shape
    xe = oversampling * x
    grid = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    phantom = ((grid[0] ** 2 + grid[1] ** 2 / 0.8 + grid[2] ** 2 / 0.7) < 0.7).astype(np.complex64)
    phantom *= 1 + 0.3 * np.sin(4 * grid[1]) * np.cos(3 * grid[2])
    maps = []
    for _ in range(coils):
        centre = generator.uniform(-1.2, 1.2, 3)
        distance = sum((grid[k] - centre[k]) ** 2 for k in range(3))
        maps.append(np.exp(-distance) * np.exp(1j * generator.uniform(0, 2 * np.pi)))
    maps = np.stack(maps).astype(np.complex64)
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    kx = np.linspace(-np.pi, np.pi, xe)[:, None, None]
    yy = np.linspace(-0.5, 0.5, y)[None, :, None]
    zz = np.linspace(-0.5, 0.5, z)[None, None, :]
    psf = np.exp(1j * 3.0 * (np.sin(3 * kx) * yy + np.cos(3 * kx) * zz)).astype(np.complex64)
    mask = np.zeros((y, z), dtype=bool)
    mask[::acceleration] = True
    operator = WaveOperator(
        torch.from_numpy(maps).to(DEVICE), torch.from_numpy(psf).to(DEVICE), torch.from_numpy(mask).to(DEVICE), coil_chunk=4
    )
    timing = timing_from_loop_order(mask, 0.03, outer="lin")
    return operator, torch.from_numpy(phantom).to(DEVICE), timing


class OperatorTests(unittest.TestCase):
    """Transform and operator identities."""

    def test_centred_fft_is_unitary_and_inverse(self) -> None:
        """Round trip and norm preservation of the centred FFTs.

        Returns:
            None.
        """
        values = torch.randn(6, 8, 10, dtype=torch.complex64, device=DEVICE)
        transformed = fftc(values, (0, 2))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(transformed)), float(torch.linalg.vector_norm(values)), places=3)
        torch.testing.assert_close(ifftc(transformed, (0, 2)), values, rtol=1e-4, atol=1e-5)

    def test_adjoint_identity(self) -> None:
        """``<A x, y> = <x, A^H y>`` for the masked multi-coil operator.

        Returns:
            None.
        """
        operator, _, _ = _problem()
        x = torch.randn(operator.image_shape, dtype=torch.complex64, device=DEVICE)
        y = torch.randn(operator.kspace_shape, dtype=torch.complex64, device=DEVICE)
        left = torch.vdot(operator.forward(x).reshape(-1), y.reshape(-1))
        right = torch.vdot(x.reshape(-1), operator.adjoint(y).reshape(-1))
        self.assertLess(float(abs(left - right) / abs(left)), 1e-4)
        torch.testing.assert_close(operator.normal(x), operator.adjoint(operator.forward(x)), rtol=1e-4, atol=1e-4)

    def test_cg_recovers_consistent_data(self) -> None:
        """CG-SENSE reproduces the image from noiseless data.

        Returns:
            None.
        """
        operator, truth, _ = _problem()
        image, history = cg_sense(operator, operator.forward(truth), iterations=60)
        error = torch.linalg.vector_norm(image - truth) / torch.linalg.vector_norm(truth)
        self.assertLess(float(error), 1e-2)
        self.assertLess(history["relative_residual"][-1], history["relative_residual"][0])


class TimingAndIoTests(unittest.TestCase):
    """Timing, smoothing, phase application, and CFL round trips."""

    def test_loop_order_and_subset(self) -> None:
        """LIN-outer order, mask checks, and subsets.

        Returns:
            None.
        """
        mask = np.zeros((6, 4), dtype=bool)
        mask[::2] = True
        timing = timing_from_loop_order(mask, 0.03)
        self.assertEqual(list(zip(timing.lin[:5], timing.par[:5])), [(0, 0), (0, 1), (0, 2), (0, 3), (2, 0)])
        timing.check_mask(mask)
        sub_mask = mask.copy()
        sub_mask[:, 1:] = False
        subset = timing.subset(sub_mask)
        self.assertEqual(subset.count, 3)
        np.testing.assert_allclose(subset.time_s, [0.0, 0.12, 0.24])
        with self.assertRaises(ValueError):
            timing.check_mask(sub_mask)
        with self.assertRaises(ValueError):
            LineTiming(np.array([0, 0]), np.array([1, 1]), np.array([0.0, 1.0]), "duplicate")

    def test_kernel_and_line_phase(self) -> None:
        """Gaussian FWHM and in-place line phase removal.

        Returns:
            None.
        """
        times = np.arange(101) * 0.01
        kernel = gaussian_time_kernel(times, 0.2, DEVICE)
        row = kernel[50].cpu().numpy()
        self.assertAlmostEqual(row[60], 0.5, places=3)  # 0.1 s = half the FWHM
        kspace = torch.ones((2, 3, 4, 2), dtype=torch.complex64, device=DEVICE)
        lin = torch.tensor([0, 3], device=DEVICE)
        par = torch.tensor([1, 0], device=DEVICE)
        phase = torch.tensor([[0.5, -0.25], [1.0, 0.0]], device=DEVICE)
        apply_line_phase(kspace, phase, lin, par)
        self.assertAlmostEqual(float(torch.angle(kspace[0, 0, 0, 1])), -0.5, places=5)
        self.assertAlmostEqual(float(torch.angle(kspace[1, 2, 0, 1])), -1.0, places=5)
        self.assertAlmostEqual(float(torch.angle(kspace[0, 0, 3, 0])), 0.25, places=5)
        self.assertEqual(float(torch.angle(kspace[0, 0, 1, 1])), 0.0)

    def test_cfl_round_trip(self) -> None:
        """Write and memory-map a CFL in Fortran order.

        Returns:
            None.
        """
        values = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4).astype(np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            write_cfl(Path(directory) / "x", values)
            np.testing.assert_array_equal(np.asarray(read_cfl(Path(directory) / "x")), values)
            self.assertEqual(read_cfl(Path(directory) / "x", 5).shape, (2, 3, 4, 1, 1))


class NavigatorTests(unittest.TestCase):
    """End-to-end recovery of an injected fluctuation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Corrupt a small problem with a known uniform + gradient fluctuation.

        The 0.3 Hz waveform is deliberately not locked to the PAR sweep: a
        fluctuation that repeats identically in every sweep is a kz-only phase,
        which is consistent with a z-blurred image and therefore invisible to
        any consistency method.

        Returns:
            None.
        """
        cls.operator, cls.truth, cls.timing = _problem(seed=3, coils=12, shape=(16, 24, 32))
        cls.voxel = (4.0, 4.0, 4.0)
        cls.model = FluctuationModel(uniform_hz=3.0, gradient_hz_per_mm=0.05, frequency_hz=0.3)
        cls.corrupted = inject_fluctuation(
            cls.operator, cls.truth, cls.timing, cls.model, echo_time_s=0.02, voxel_mm_par=cls.voxel[2]
        )
        add_noise(cls.corrupted, 1e-4 * float(cls.corrupted.abs().max()), cls.operator.line_mask)
        cls.uncorrected, _ = cg_sense(cls.operator, cls.corrupted, iterations=40)

    def _error(self, image: torch.Tensor) -> float:
        """Relative complex error after a least-squares complex scale."""
        scale = torch.vdot(image.reshape(-1), self.truth.reshape(-1)) / torch.vdot(image.reshape(-1), image.reshape(-1))
        return float(torch.linalg.vector_norm(scale * image - self.truth) / torch.linalg.vector_norm(self.truth))

    def _magnitude_error(self, image: torch.Tensor) -> float:
        """Relative magnitude error (ignores the unidentifiable static phase ramp)."""
        a, b = image.abs(), self.truth.abs()
        return float(torch.linalg.vector_norm(torch.sum(a * b) / torch.sum(a * a) * a - b) / torch.linalg.vector_norm(b))

    def test_global_navigator_reduces_error(self) -> None:
        """The global consistency navigator removes most of the corruption.

        Returns:
            None.
        """
        settings = NavigatorSettings(mode="global", outer_iterations=25, cg_iterations=30, temporal_fwhm_s=0.05,
                                     tolerance_rad=1e-5, voxel_mm=self.voxel)
        result = run_consistency_navigator(self.operator, self.corrupted, self.timing, settings, log=lambda _: None)
        self.assertLess(self._error(result.image), 0.65 * self._error(self.uncorrected))
        estimate = result.global_hz(0.02)
        truth = self.model.uniform_trace_hz(self.timing.time_s)
        self.assertGreater(np.corrcoef(estimate - estimate.mean(), truth - truth.mean())[0, 1], 0.85)

    def test_linear_field_recovers_gradient(self) -> None:
        """The model-based navigator recovers the gradient trace and the magnitude image.

        Returns:
            None.
        """
        settings = FieldModelSettings(axes=("z",), outer_iterations=25, cg_iterations=30, temporal_fwhm_s=0.05,
                                      gradient_fwhm_s=None, tolerance_rad=1e-5, voxel_mm=self.voxel)
        result = run_linear_field_navigator(self.operator, self.corrupted, self.timing, settings, log=lambda _: None)
        gradient = result.traces(0.02)["gradient_z_hz_per_mm"]
        truth = self.model.gradient_trace_hz_per_mm(self.timing.time_s)
        self.assertGreater(np.corrcoef(gradient - gradient.mean(), truth - truth.mean())[0, 1], 0.6)
        self.assertLess(self._magnitude_error(result.image), 0.5 * self._magnitude_error(self.uncorrected))
        weights = result.line_energy / result.line_energy.sum()
        self.assertAlmostEqual(float(np.sum(weights * result.parameters[1])), 0.0, delta=1e-3 * np.abs(result.parameters[1]).max())
        self.assertTrue(math.isfinite(result.history[-1]["data_residual_before_update"]))

if __name__ == "__main__":
    unittest.main()


class PipelineTreeTests(unittest.TestCase):
    """Write a corrected tree from a tiny fake standard tree."""

    def test_estimate_and_correct_writes_standard_layout(self) -> None:
        """Corrected normal and retrospective inputs keep the standard layout.

        Returns:
            None.
        """
        import json

        from b0_fluctuation.cfl import read_shape
        from b0_fluctuation.pipeline import estimate_and_correct

        operator, truth, timing = _problem(seed=5, coils=6, shape=(8, 12, 8))
        model = FluctuationModel(uniform_hz=2.0, gradient_hz_per_mm=0.0, frequency_hz=0.3)
        kspace = inject_fluctuation(operator, truth, timing, model, echo_time_s=0.02, voxel_mm_par=4.0)
        mask = operator.line_mask.cpu().numpy()
        retro_mask = mask.copy()
        retro_mask[:, 1::2] = False
        with tempfile.TemporaryDirectory() as directory:
            standard = Path(directory) / "vcc6"
            normal = standard / "normal" / "bart_inputs"
            echo = {"echo": 1, "te_s": 0.02, "wave_kspace": "wave_kspace_echo-01", "psf": "psf_echo-01",
                    "wave_kspace_norm": 1.0, "shared_calibration_id": "x", "selected_wavelet_lambda": 0.015}
            geometry = {"case_id": "native_r3x1", "voxel_mm_ro_lin_par": [4.0, 4.0, 4.0], "matrix_ro_lin_par": [8, 12, 8]}
            values = np.moveaxis(kspace.cpu().numpy(), 0, 3)[..., None]
            write_cfl(normal / "wave_kspace_echo-01", values)
            write_cfl(normal / "psf_echo-01", operator.psf.cpu().numpy())
            np.save(normal / "sampling_mask.npy", mask)
            (normal / "manifest.json").write_text(json.dumps({"geometry": geometry, "echoes": [echo], "output_artifacts": {}}))
            write_cfl(standard / "normal" / "bart_output" / "coil_sens", np.moveaxis(operator.maps.cpu().numpy(), 0, 3))
            (standard / "normal" / "bart_output" / "ecalib_command.txt").write_text("bart ecalib -m 1 -c 0.6 a b\n")
            retro = standard / "retro" / "native_r3x3" / "bart_inputs"
            write_cfl(retro / "wave_kspace_echo-01", values * retro_mask[None, :, :, None, None])
            write_cfl(retro / "psf_echo-01", operator.psf.cpu().numpy())
            np.save(retro / "sampling_mask.npy", retro_mask)
            (retro / "manifest.json").write_text(json.dumps({"case": {**geometry, "case_id": "native_r3x3"}, "echoes": [echo]}))
            output = Path(directory) / "corrected"
            manifest = estimate_and_correct(
                standard, output, method="global", repetition_time_s=0.03, retro_cases=("native_r3x3",),
                device=str(DEVICE), log=lambda _: None, outer_iterations=5, cg_iterations=20, temporal_fwhm_s=0.05,
            )
            self.assertEqual(manifest["status"], "b0_fluctuation_corrected_inputs_ready")
            corrected_normal = json.loads((output / "normal" / "bart_inputs" / "manifest.json").read_text())
            self.assertNotIn("output_artifacts", corrected_normal)
            self.assertEqual(corrected_normal["b0_fluctuation_correction"]["method"], "global")
            self.assertEqual(read_shape(output / "normal" / "bart_inputs" / "wave_kspace_echo-01"), values.shape)
            self.assertTrue((output / "normal" / "bart_output" / "coil_sens.cfl").is_file())
            retro_data = np.asarray(read_cfl(output / "retro" / "native_r3x3" / "bart_inputs" / "wave_kspace_echo-01"))
            acquired = np.abs(retro_data).sum(axis=(0, 3, 4)) > 0
            np.testing.assert_array_equal(acquired, retro_mask)
            # A unit-modulus line phase leaves the k-space norm unchanged.
            self.assertAlmostEqual(corrected_normal["echoes"][0]["wave_kspace_norm"], float(np.linalg.norm(values)), places=3)
            self.assertTrue((output / "estimation" / "figures" / "normal_traces.png").is_file())
            self.assertIn("uniform_hz_std", manifest["cases"]["normal"])
