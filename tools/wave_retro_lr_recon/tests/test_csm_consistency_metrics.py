"""Synthetic tests for the phase-free CSM consistency metrics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import csm_consistency_metrics as metrics  # noqa: E402
from wave_retro_lr.csm_consistency_metrics import (  # noqa: E402
    acs_apodization_window,
    acs_block_slices,
    centered_fft,
    centered_ifft,
    coil_images,
    coil_vector_norm,
    csm_coherence,
    dc_centered_hann,
    eigenvalue_arrays,
    eigenvalue_gap,
    eigenvalue_qc,
    empirical_coil_covariance,
    local_coil_rank,
    map_orthonormality_error,
    map_reproduction,
    map_switching_mask,
    modelled_image_noise_covariance,
    normalize_coil_vectors,
    orthonormal_map_basis,
    projection_residual,
    residual_to_noise_ratio,
    signal_to_noise_energy,
    support_mask,
    validate_two_map_outputs,
)


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Draw circular complex Gaussian samples with unit variance.

    Args:
        rng: Seeded random generator.
        shape: Output shape.

    Returns:
        Complex128 samples with ``E|z|^2 = 1``.
    """
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _unit_vectors(rng: np.random.Generator, shape: tuple[int, ...], coils: int) -> np.ndarray:
    """Draw random unit-norm coil vectors.

    Args:
        rng: Seeded random generator.
        shape: Spatial shape.
        coils: Number of coils.

    Returns:
        Complex128 array of shape ``shape + (coils,)`` with unit coil norms.
    """
    values = _complex_normal(rng, shape + (coils,))
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _orthonormal_pair(
    rng: np.random.Generator, shape: tuple[int, ...], coils: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two orthonormal coil vectors per voxel by Gram-Schmidt.

    Args:
        rng: Seeded random generator.
        shape: Spatial shape.
        coils: Number of coils.

    Returns:
        ``(s, s_perp)`` complex128 arrays of shape ``shape + (coils,)``.
    """
    s = _unit_vectors(rng, shape, coils)
    other = _complex_normal(rng, shape + (coils,))
    other = other - s * np.sum(s.conj() * other, axis=-1, keepdims=True)
    return s, other / np.linalg.norm(other, axis=-1, keepdims=True)


def _smooth_orthonormal_fields(
    shape: tuple[int, int, int], coils: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build two smooth, mutually orthogonal unit coil-vector fields.

    Each coil has a broad Gaussian magnitude profile around its own center
    and a slow linear phase ramp. One voxel step is 0.02 length units
    against a profile width near 1, so neighbouring voxels differ only
    slightly (face-neighbour coherence above 0.9999 on the test grids).

    Args:
        shape: Spatial grid ``(RO, LIN, PAR)``.
        coils: Number of coils.

    Returns:
        ``(s1, s2)`` complex128 fields of shape ``shape + (coils,)``.
    """
    axes = [0.02 * (np.arange(size) - (size - 1) / 2.0) for size in shape]
    position = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)[..., np.newaxis, :]
    angles = 2.0 * np.pi * np.arange(coils) / coils
    centers_1 = np.stack(
        [0.8 * np.cos(angles), 0.8 * np.sin(angles), 0.3 * np.cos(2 * angles)], axis=-1
    )
    centers_2 = np.stack(
        [0.6 * np.sin(angles), -0.7 * np.cos(angles), -0.4 * np.sin(3 * angles)], axis=-1
    )
    ramp_x = position[..., 0]
    ramp_y = position[..., 1]
    field_1 = np.exp(-np.sum((position - centers_1) ** 2, axis=-1) / 2.0) * np.exp(
        1j * (angles + 0.5 * ramp_x)
    )
    field_2 = np.exp(-np.sum((position - centers_2) ** 2, axis=-1) / 2.0) * np.exp(
        1j * (1.7 * angles - 0.4 * ramp_y)
    )
    s1 = field_1 / np.linalg.norm(field_1, axis=-1, keepdims=True)
    field_2 = field_2 - s1 * np.sum(s1.conj() * field_2, axis=-1, keepdims=True)
    return s1, field_2 / np.linalg.norm(field_2, axis=-1, keepdims=True)


def _fixed_orthonormal_vectors(rng: np.random.Generator, coils: int, count: int) -> np.ndarray:
    """Draw a fixed set of orthonormal coil vectors.

    Args:
        rng: Seeded random generator.
        coils: Vector length.
        count: Number of orthonormal vectors.

    Returns:
        Complex128 array of shape ``(count, coils)`` with orthonormal rows.
    """
    matrix, _ = np.linalg.qr(_complex_normal(rng, (coils, count)))
    return matrix.T


class ApodizationAndFftTests(unittest.TestCase):
    """Verify the ACS apodization window and centered FFT conventions."""

    def test_dc_centered_hann_is_symmetric_with_zero_edge(self) -> None:
        """Keep the window symmetric about DC, 1 at DC, and 0 at -n/2.

        Returns:
            None.
        """
        for length in (1, 2, 5, 8, 9, 32):
            window = dc_centered_hann(length)
            self.assertEqual(window.shape, (length,))
            self.assertEqual(window.dtype, np.float64)
            self.assertEqual(window[length // 2], 1.0)
            self.assertEqual(int(np.argmax(window)), length // 2)
            self.assertLessEqual(float(window.max()), 1.0)
            for offset in range(1, (length - 1) // 2 + 1):
                self.assertEqual(window[length // 2 - offset], window[length // 2 + offset])
            if length % 2 == 0:
                self.assertEqual(window[0], 0.0)
            else:
                self.assertTrue(np.all(window > 0))
            reference = np.cos(np.pi * (np.arange(length) - length // 2) / length) ** 2
            np.testing.assert_allclose(window, reference, rtol=0, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "positive"):
            dc_centered_hann(0)

    def test_acs_window_places_block_at_grid_center(self) -> None:
        """Embed the separable Hann block at N//2 - n//2 and zero elsewhere.

        Returns:
            None.
        """
        for grid, acs in (((4, 10, 9), 4), ((3, 32, 32), 32), ((2, 11, 12), 5)):
            _, lin, par = grid
            lin_block, par_block = acs_block_slices(lin, par, acs)
            self.assertEqual(lin_block, slice(lin // 2 - acs // 2, lin // 2 - acs // 2 + acs))
            self.assertEqual(par_block, slice(par // 2 - acs // 2, par // 2 - acs // 2 + acs))
            window = acs_apodization_window(grid, acs)
            self.assertEqual(window.shape, (1, lin, par))
            self.assertEqual(window.dtype, np.float64)
            profile = dc_centered_hann(acs)
            np.testing.assert_array_equal(
                window[0, lin_block, par_block], np.outer(profile, profile)
            )
            outside = window.copy()
            outside[0, lin_block, par_block] = 0
            self.assertEqual(np.count_nonzero(outside), 0)
            # The k-space DC sample N//2 carries the peak weight.
            self.assertEqual(window[0, lin // 2, par // 2], 1.0)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            acs_block_slices(8, 6, 7)
        with self.assertRaisesRegex(ValueError, "RO, LIN, PAR"):
            acs_apodization_window((8, 8), 4)

    def test_centered_fft_round_trip_and_dc_delta(self) -> None:
        """Invert exactly and map a DC delta to a 1/sqrt(N) constant image.

        Returns:
            None.
        """
        rng = np.random.default_rng(101)
        values = _complex_normal(rng, (6, 8, 5, 2))
        np.testing.assert_allclose(centered_fft(centered_ifft(values)), values, atol=1e-12)
        np.testing.assert_allclose(centered_ifft(centered_fft(values)), values, atol=1e-12)
        np.testing.assert_allclose(
            centered_ifft(values, axes=(0, 1)),
            np.stack([centered_ifft(values[:, :, index], axes=(0, 1)) for index in range(5)], 2),
            atol=1e-12,
        )
        for shape in ((8, 6, 4), (7, 5, 3)):
            delta = np.zeros(shape, dtype=np.complex128)
            delta[tuple(size // 2 for size in shape)] = 1.0
            image = centered_ifft(delta)
            expected = 1.0 / np.sqrt(np.prod(shape))
            np.testing.assert_allclose(np.abs(image), expected, rtol=1e-12, atol=0)
            np.testing.assert_allclose(image, expected, atol=1e-14)
            self.assertAlmostEqual(float(np.sum(np.abs(image) ** 2)), 1.0, places=12)
        with self.assertRaisesRegex(ValueError, "distinct"):
            centered_ifft(values, axes=(0, 0))
        with self.assertRaisesRegex(ValueError, "invalid"):
            centered_fft(values, axes=(4,))

    def test_coil_images_match_per_coil_reference(self) -> None:
        """Reproduce an explicit per-coil windowed and unwindowed reconstruction.

        Returns:
            None.
        """
        rng = np.random.default_rng(102)
        kspace = _complex_normal(rng, (6, 8, 4, 3)).astype(np.complex64)
        window = acs_apodization_window((6, 8, 4), 4)
        axes = (0, 1, 2)
        for weights in (window, None):
            reference = np.stack(
                [
                    np.fft.fftshift(
                        np.fft.ifftn(
                            np.fft.ifftshift(
                                kspace[..., coil].astype(np.complex128)
                                * (1.0 if weights is None else weights),
                                axes=axes,
                            ),
                            axes=axes,
                            norm="ortho",
                        ),
                        axes=axes,
                    )
                    for coil in range(3)
                ],
                axis=-1,
            )
            exact = coil_images(kspace, weights, dtype=np.complex128)
            self.assertEqual(exact.dtype, np.complex128)
            np.testing.assert_allclose(exact, reference, rtol=0, atol=1e-12)
            default = coil_images(kspace, weights)
            self.assertEqual(default.dtype, np.complex64)
            np.testing.assert_allclose(default, reference, rtol=0, atol=1e-6)
        # The window is applied to a private copy, never to the caller's array.
        exact_input = kspace.astype(np.complex128)
        original = exact_input.copy()
        coil_images(exact_input, window, dtype=np.complex128)
        np.testing.assert_array_equal(exact_input, original)

    def test_invalid_windows_and_kspace_are_rejected(self) -> None:
        """Reject complex, non-broadcasting, or enlarging windows and bad data.

        Returns:
            None.
        """
        kspace = np.ones((6, 8, 4, 2), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            coil_images(kspace, np.ones((1, 8, 4), dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "broadcast"):
            coil_images(kspace, np.ones((2, 8, 4)))
        with self.assertRaisesRegex(ValueError, "enlarges"):
            coil_images(kspace, np.ones((1, 1, 8, 4)))
        with self.assertRaisesRegex(ValueError, "complex"):
            coil_images(kspace, None, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"\(RO, LIN, PAR, Nc\)"):
            coil_images(kspace[..., 0, np.newaxis, np.newaxis], None)
        corrupted = kspace.copy()
        corrupted[0, 0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "channel 1"):
            coil_images(corrupted, None)


class CoilVectorTests(unittest.TestCase):
    """Verify coil-vector normalization, SNR support, and noise covariances."""

    def test_normalize_coil_vectors_marks_outside_and_zero_norm(self) -> None:
        """Return unit vectors in the mask and complex NaN elsewhere.

        Returns:
            None.
        """
        rng = np.random.default_rng(201)
        vectors = _complex_normal(rng, (3, 4, 5)).astype(np.complex64)
        vectors[1, 2] = 0
        mask = np.ones((3, 4), dtype=bool)
        mask[0, 3] = False
        normalized = normalize_coil_vectors(vectors, mask)
        self.assertEqual(normalized.dtype, np.complex128)
        inside = mask.copy()
        inside[1, 2] = False
        np.testing.assert_allclose(coil_vector_norm(normalized[inside]), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            normalized[inside] * coil_vector_norm(vectors)[inside][:, np.newaxis],
            vectors[inside],
            atol=1e-6,
        )
        for index in ((0, 3), (1, 2)):
            self.assertTrue(np.all(np.isnan(normalized[index].real)))
            self.assertTrue(np.all(np.isnan(normalized[index].imag)))
        self.assertEqual(coil_vector_norm(vectors).dtype, np.float64)
        with self.assertRaisesRegex(ValueError, "boolean"):
            normalize_coil_vectors(vectors, mask.astype(np.uint8))

    def test_snr_energy_and_support_mask(self) -> None:
        """Divide coil energy by the noise trace and threshold inclusively.

        Returns:
            None.
        """
        vectors = np.zeros((4, 2), dtype=np.complex64)
        vectors[1, 0] = 1.0
        vectors[2] = [np.sqrt(2.0), np.sqrt(2.0) * 1j]
        vectors[3, 1] = np.nan
        snr = signal_to_noise_energy(vectors, 2.0)
        self.assertEqual(snr.dtype, np.float64)
        np.testing.assert_allclose(snr[:3], [0.0, 0.5, 2.0], rtol=1e-6)
        self.assertTrue(np.isnan(snr[3]))
        np.testing.assert_array_equal(support_mask(snr, 0.5), [False, True, True, False])
        for trace in (0.0, -1.0, np.nan):
            with self.assertRaisesRegex(ValueError, "Noise trace"):
                signal_to_noise_energy(vectors, trace)
        with self.assertRaisesRegex(ValueError, "finite"):
            support_mask(snr, np.inf)

    def test_covariance_matches_parseval_and_white_noise_model(self) -> None:
        """Tie the empirical image covariance to the modelled k-space noise.

        The masked mean of ``d d^H`` over the whole grid equals, by Parseval,
        the windowed k-space second moment divided by the voxel count. Its
        expectation is the white-noise model; with about 2000 effective Hann
        samples the Monte Carlo relative Frobenius error is near 3 %, so a
        10 % tolerance is used.

        Returns:
            None.
        """
        rng = np.random.default_rng(202)
        grid = (32, 32, 32)
        acs = 16
        coils = 3
        mixing = _complex_normal(rng, (coils, coils))
        kspace_covariance = mixing @ mixing.conj().T / coils + 0.3 * np.eye(coils)
        lin_block, par_block = acs_block_slices(grid[1], grid[2], acs)
        kspace = np.zeros(grid + (coils,), dtype=np.complex128)
        kspace[:, lin_block, par_block, :] = _complex_normal(
            rng, (grid[0], acs, acs, coils)
        ) @ np.linalg.cholesky(kspace_covariance).T
        voxels = int(np.prod(grid))
        acquired = grid[0] * acs * acs
        everywhere = np.ones(grid, dtype=bool)
        window = acs_apodization_window(grid, acs)

        images = coil_images(kspace, window, dtype=np.complex128)
        empirical, count = empirical_coil_covariance(images, everywhere)
        self.assertEqual(count, voxels)
        np.testing.assert_allclose(empirical, empirical.conj().T, atol=0)
        weighted = (kspace * window[..., np.newaxis]).reshape(-1, coils)
        parseval = weighted.T @ weighted.conj() / voxels
        np.testing.assert_allclose(empirical, parseval, rtol=0, atol=1e-12)

        model = modelled_image_noise_covariance(kspace_covariance, window, voxels, acquired)
        energy = grid[0] * float(np.sum(window**2))
        np.testing.assert_allclose(model, kspace_covariance * energy / voxels, atol=1e-15)
        relative = np.linalg.norm(empirical - model) / np.linalg.norm(model)
        self.assertLess(relative, 0.10)

        unapodized, _ = empirical_coil_covariance(coil_images(kspace, None), everywhere)
        plain_model = modelled_image_noise_covariance(kspace_covariance, None, voxels, acquired)
        np.testing.assert_allclose(plain_model, kspace_covariance * acquired / voxels)
        self.assertLess(
            np.linalg.norm(unapodized - plain_model) / np.linalg.norm(plain_model), 0.06
        )

        with self.assertRaisesRegex(ValueError, "vanish outside"):
            modelled_image_noise_covariance(
                kspace_covariance, np.ones((1, 32, 32)), voxels, acquired
            )
        with self.assertRaisesRegex(ValueError, "divide"):
            modelled_image_noise_covariance(kspace_covariance, np.ones(7), voxels, acquired)
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            modelled_image_noise_covariance(np.triu(np.ones((3, 3))), None, voxels, acquired)
        few = np.zeros(grid, dtype=bool)
        few[0, 0, :coils] = True
        with self.assertRaisesRegex(ValueError, "more than 3"):
            empirical_coil_covariance(images, few)


class ProjectionTests(unittest.TestCase):
    """Verify the projection residual, RNR, and orthonormal map bases."""

    def test_rank_one_residual_statistics_and_invariance(self) -> None:
        """Match the white-noise residual law and ignore map phase and scale.

        With ``d = s * rho + n``, ``|rho| = 1`` and white noise of total
        energy ``1 / SNR``, the mean of ``rho_1^2`` is ``(Nc - 1) / (Nc SNR)``
        up to a ``1 / (1 + 1 / SNR)`` factor (0.5 % at SNR 200). The sampling
        error over 4096 voxels with seven residual degrees of freedom is about
        0.6 %, so a 5 % tolerance is used; the energy-SNR form
        ``rho_1^2 * ||d||^2 / tr(Psi)`` has mean ``(Nc - 1) / Nc`` and is
        checked within 3 %.

        Returns:
            None.
        """
        rng = np.random.default_rng(301)
        shape = (16, 16, 16)
        coils = 8
        snr = 200.0
        s, s_perp = _orthonormal_pair(rng, shape, coils)
        noise_variance = 1.0 / (coils * snr)
        amplitude = np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        vectors = s * amplitude[..., np.newaxis] + np.sqrt(noise_variance) * _complex_normal(
            rng, shape + (coils,)
        )
        maps = np.stack([s, s_perp], axis=-1)
        mask = np.ones(shape, dtype=bool)
        basis_1, rank_1 = orthonormal_map_basis(maps[..., :1])
        basis_2, rank_2 = orthonormal_map_basis(maps)
        np.testing.assert_array_equal(rank_1, 1)
        np.testing.assert_array_equal(rank_2, 2)
        rho_1 = projection_residual(vectors, basis_1, mask)
        rho_2 = projection_residual(vectors, basis_2, mask)
        self.assertEqual(rho_1.dtype, np.float64)
        expected = (coils - 1) / (coils * snr)
        self.assertLess(abs(float(np.mean(rho_1**2)) / expected - 1.0), 0.05)
        energy_snr = signal_to_noise_energy(vectors, coils * noise_variance)
        self.assertLess(
            abs(float(np.mean(rho_1**2 * energy_snr)) / ((coils - 1) / coils) - 1.0), 0.03
        )
        self.assertTrue(np.all(rho_2 <= rho_1 + 1e-12))

        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, shape + (1, 2)))
        scales = rng.uniform(0.2, 5.0, shape + (1, 2))
        for transformed in (maps * phases, maps * scales, maps * phases * scales):
            np.testing.assert_allclose(
                projection_residual(vectors, orthonormal_map_basis(transformed[..., :1])[0], mask),
                rho_1,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                projection_residual(vectors, orthonormal_map_basis(transformed)[0], mask),
                rho_2,
                rtol=0,
                atol=1e-12,
            )

    def test_rank_two_signal_is_captured_by_two_maps(self) -> None:
        """Leave a clear map-1 residual but no residual for the two-map span.

        Returns:
            None.
        """
        rng = np.random.default_rng(302)
        shape = (6, 5, 4)
        coils = 8
        s1 = _unit_vectors(rng, shape, coils)
        s2 = _unit_vectors(rng, shape, coils)
        a = np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        b = 0.8 * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        vectors = s1 * a[..., np.newaxis] + s2 * b[..., np.newaxis]
        maps = np.stack([s1, s2], axis=-1)
        mask = np.ones(shape, dtype=bool)
        rho_1 = projection_residual(vectors, orthonormal_map_basis(maps[..., :1])[0], mask)
        basis_2, rank_2 = orthonormal_map_basis(maps)
        rho_2 = projection_residual(vectors, basis_2, mask)
        np.testing.assert_array_equal(rank_2, 2)
        self.assertGreater(float(np.min(rho_1)), 0.1)
        self.assertLess(float(np.max(rho_2)), 1e-6)

    def test_residual_to_noise_ratio_with_correlated_noise(self) -> None:
        """Average RNR_1 to 1 for exact rank-1 signal and known correlated noise.

        Per-voxel RNR has a relative spread near 0.5 here, so the mean over
        4096 voxels has a sampling error near 1 % and a 4 % tolerance is used.
        Doubling the covariance halves the ratio, which is why the diagnostic
        is conditional on an independently validated covariance.

        Returns:
            None.
        """
        rng = np.random.default_rng(303)
        shape = (16, 16, 16)
        coils = 6
        mixing = _complex_normal(rng, (coils, coils))
        covariance = mixing @ mixing.conj().T / coils + 0.2 * np.eye(coils)
        self.assertGreater(float(np.max(np.abs(covariance - np.diag(np.diag(covariance))))), 0.1)
        noise = _complex_normal(rng, shape + (coils,)) @ np.linalg.cholesky(covariance).T
        s = _unit_vectors(rng, shape, coils)
        signal = 3.0 * s * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))[..., np.newaxis]
        vectors = signal + noise
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0] = False
        basis, _ = orthonormal_map_basis(s[..., np.newaxis])
        ratio = residual_to_noise_ratio(vectors, basis, covariance, mask)
        self.assertEqual(ratio.dtype, np.float64)
        self.assertTrue(np.isnan(ratio[0, 0, 0]))
        self.assertLess(abs(float(np.nanmean(ratio)) - 1.0), 0.04)
        doubled = residual_to_noise_ratio(vectors, basis, 2.0 * covariance, mask)
        np.testing.assert_allclose(doubled, ratio / 2.0, rtol=1e-12)
        self.assertIn("conditional", residual_to_noise_ratio.__doc__)
        self.assertIn("independent", residual_to_noise_ratio.__doc__)
        with self.assertRaisesRegex(ValueError, "coils"):
            residual_to_noise_ratio(vectors, basis, np.eye(coils + 1), mask)

    def test_undefined_voxels_and_basis_validation(self) -> None:
        """Return NaN for undefined voxels and reject non-orthonormal bases.

        Returns:
            None.
        """
        rng = np.random.default_rng(304)
        s = _unit_vectors(rng, (2, 3, 1), 4)
        basis, _ = orthonormal_map_basis(s[..., np.newaxis])
        vectors = 2.0 * s
        vectors[0, 0, 0] = 0
        basis[1, 2, 0] = 0
        mask = np.ones((2, 3, 1), dtype=bool)
        mask[0, 1, 0] = False
        rho = projection_residual(vectors, basis, mask)
        undefined = np.zeros((2, 3, 1), dtype=bool)
        undefined[0, 0, 0] = undefined[1, 2, 0] = undefined[0, 1, 0] = True
        np.testing.assert_array_equal(np.isnan(rho), undefined)
        np.testing.assert_allclose(rho[~undefined], 0.0, atol=1e-7)
        ratio = residual_to_noise_ratio(vectors, basis, np.eye(4), mask)
        self.assertEqual(ratio[0, 0, 0], 0.0)
        self.assertTrue(np.isnan(ratio[1, 2, 0]) and np.isnan(ratio[0, 1, 0]))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            projection_residual(vectors, 2.0 * basis, mask)
        with self.assertRaisesRegex(ValueError, "trailing map axis"):
            projection_residual(vectors[..., :3], basis, mask)
        with self.assertRaisesRegex(ValueError, "boolean"):
            projection_residual(vectors, basis, mask.astype(float))

    def test_orthonormal_map_basis_rank_and_tolerance(self) -> None:
        """Drop collinear or negligible columns and keep zero maps NaN-free.

        Returns:
            None.
        """
        rng = np.random.default_rng(305)
        shape = (3, 4, 2)
        s, s_perp = _orthonormal_pair(rng, shape, 5)
        basis, rank = orthonormal_map_basis(np.stack([s, (2.0 - 1.0j) * s], axis=-1))
        self.assertEqual(rank.shape, shape)
        np.testing.assert_array_equal(rank, 1)
        np.testing.assert_array_equal(basis[..., 1], 0)
        np.testing.assert_allclose(
            np.abs(np.sum(basis[..., 0].conj() * s, axis=-1)), 1.0, atol=1e-12
        )
        zero_basis, zero_rank = orthonormal_map_basis(np.zeros(shape + (5, 2), np.complex64))
        np.testing.assert_array_equal(zero_rank, 0)
        np.testing.assert_array_equal(zero_basis, 0)
        self.assertFalse(np.isnan(zero_basis).any())
        for scale, tolerance, expected in (
            (1e-4, 1e-3, 1),
            (1e-2, 1e-3, 2),
            (1e-2, 0.1, 1),
            (0.3, 0.4, 1),
            (0.5, 0.4, 2),
        ):
            maps = np.stack([s, scale * s_perp], axis=-1)
            basis, rank = orthonormal_map_basis(maps, tolerance)
            np.testing.assert_array_equal(rank, expected)
            self.assertLess(float(np.max(map_orthonormality_error(basis[..., :expected]))), 1e-12)
        for tolerance in (0.0, -1.0, 1.5, np.nan):
            with self.assertRaisesRegex(ValueError, "Rank tolerance"):
                orthonormal_map_basis(np.stack([s, s_perp], axis=-1), tolerance)
        corrupted = np.stack([s, s_perp], axis=-1)
        corrupted[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            orthonormal_map_basis(corrupted)

    def test_map_orthonormality_error(self) -> None:
        """Report zero for orthonormal maps and the Frobenius error otherwise.

        Returns:
            None.
        """
        rng = np.random.default_rng(306)
        s, s_perp = _orthonormal_pair(rng, (2, 2, 2), 6)
        self.assertLess(float(np.max(map_orthonormality_error(np.stack([s, s_perp], -1)))), 1e-12)
        np.testing.assert_allclose(
            map_orthonormality_error(np.stack([2.0 * s, s_perp], axis=-1)), 3.0, atol=1e-12
        )


class CalibrationOutputTests(unittest.TestCase):
    """Verify two-map layout checks, eigenvalue QC, and map reproduction."""

    def test_two_map_layout_validation(self) -> None:
        """Accept trailing singletons and reject every other layout change.

        Returns:
            None.
        """
        spatial = (3, 4, 2)
        validate_two_map_outputs((3, 4, 2, 5, 2), (3, 4, 2, 1, 2), spatial, 5)
        validate_two_map_outputs((3, 4, 2, 5, 2, 1, 1), (3, 4, 2, 1, 2, 1), spatial, 5)
        failures = (
            (((3, 4, 2, 5, 1), (3, 4, 2, 1, 2)), "MAPS"),
            (((3, 4, 2, 5, 2), (3, 4, 2, 1, 3)), "MAPS"),
            (((3, 4, 3, 5, 2), (3, 4, 2, 1, 2)), "PAR"),
            (((3, 4, 2, 5, 2), (2, 4, 2, 1, 2)), "RO"),
            (((3, 4, 2, 6, 2), (3, 4, 2, 1, 2)), "coil"),
            (((3, 4, 2, 5, 2), (3, 4, 2, 2, 2)), "dimension 3"),
            (((3, 4, 2, 5, 2, 2), (3, 4, 2, 1, 2)), "trailing"),
            (((3, 4, 2, 5), (3, 4, 2, 1, 2)), "at least"),
        )
        for (maps_shape, eigen_shape), message in failures:
            with self.assertRaisesRegex(ValueError, message):
                validate_two_map_outputs(maps_shape, eigen_shape, spatial, 5)

    def test_eigenvalue_arrays_conversion_and_rejection(self) -> None:
        """Drop singleton dimensions and reject large imaginary parts.

        Returns:
            None.
        """
        rng = np.random.default_rng(401)
        real = rng.uniform(0.0, 1.0, (3, 4, 2, 2))
        stored = np.zeros((3, 4, 2, 1, 2, 1), dtype=np.complex64)
        stored[:, :, :, 0, :, 0] = real + 1e-7j
        converted = eigenvalue_arrays(stored)
        self.assertEqual(converted.shape, (3, 4, 2, 2))
        self.assertEqual(converted.dtype, np.float64)
        np.testing.assert_allclose(converted, real, atol=1e-7)
        noisy = stored.copy()
        noisy[1, 1, 1, 0, 1, 0] += 1e-3j
        with self.assertRaisesRegex(ValueError, "imaginary"):
            eigenvalue_arrays(noisy)
        np.testing.assert_allclose(
            eigenvalue_arrays(noisy, imaginary_tolerance=1e-2), real, atol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "dimension 3"):
            eigenvalue_arrays(np.zeros((3, 4, 2, 2, 2), dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "trailing"):
            eigenvalue_arrays(np.zeros((3, 4, 2, 1, 2, 3), dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "at least 5"):
            eigenvalue_arrays(np.zeros((3, 4, 2, 2), dtype=np.complex64))

    def test_eigenvalue_qc_counts(self) -> None:
        """Count |lambda| >= 1, negatives, and ordering violations.

        Returns:
            None.
        """
        rows = np.array(
            [
                [0.9, 0.2],
                [1.0, 0.5],
                [0.3, 0.6],
                [-0.1, -0.2],
                [np.nan, 0.1],
                [1.2, 1.1],
                [0.5, -1.5],
                [0.4, 0.4],
            ]
        )
        eigenvalues = rows.reshape(2, 2, 2, 2)
        report = eigenvalue_qc(eigenvalues)
        self.assertEqual(
            report,
            {
                "map_count": 2,
                "voxels": 8,
                "finite_fraction": 15 / 16,
                "min": -1.5,
                "max": 1.2,
                "count_ge_one": 4,
                "count_negative": 3,
                "descending_order_violations": 1,
            },
        )
        json.dumps(report)
        mask = np.zeros(8, dtype=bool)
        mask[:4] = True
        masked = eigenvalue_qc(eigenvalues, mask.reshape(2, 2, 2))
        self.assertEqual(masked["voxels"], 4)
        self.assertEqual(masked["finite_fraction"], 1.0)
        self.assertEqual((masked["min"], masked["max"]), (-0.2, 1.0))
        self.assertEqual(masked["count_ge_one"], 1)
        self.assertEqual(masked["count_negative"], 2)
        self.assertEqual(masked["descending_order_violations"], 1)
        empty = eigenvalue_qc(eigenvalues, np.zeros((2, 2, 2), dtype=bool))
        self.assertIsNone(empty["finite_fraction"])
        self.assertIsNone(empty["min"])
        json.dumps(empty)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            eigenvalue_qc(eigenvalues.astype(np.complex64))

    def test_eigenvalue_gap(self) -> None:
        """Subtract map-2 from map-1 eigenvalues and require two maps.

        Returns:
            None.
        """
        gap = eigenvalue_gap(np.array([[[[0.9, 0.7], [0.4, 0.5]]]], dtype=np.float32))
        self.assertEqual(gap.dtype, np.float64)
        np.testing.assert_allclose(gap, [[[0.2, -0.1]]], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "two maps"):
            eigenvalue_gap(np.ones((2, 2, 1)))

    def test_map_reproduction_is_phase_invariant(self) -> None:
        """Give 1 for maps equal up to a complex factor and 0 when orthogonal.

        Returns:
            None.
        """
        rng = np.random.default_rng(402)
        shape = (4, 3, 2)
        s, s_perp = _orthonormal_pair(rng, shape, 6)
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0] = False
        factor = rng.uniform(0.3, 3.0, shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        alpha = map_reproduction(s * factor[..., np.newaxis], 2.0 * s, mask)
        self.assertEqual(alpha.dtype, np.float64)
        self.assertTrue(np.isnan(alpha[0, 0, 0]))
        np.testing.assert_allclose(alpha[mask], 1.0, atol=1e-12)
        np.testing.assert_allclose(map_reproduction(s_perp, s, mask)[mask], 0.0, atol=1e-12)
        zero = s.copy()
        zero[1, 1, 1] = 0
        self.assertTrue(np.isnan(map_reproduction(zero, s, mask)[1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "does not match"):
            map_reproduction(s[..., :5], s, mask)


class LocalRankTests(unittest.TestCase):
    """Verify the model-free local coil-vector rank."""

    def test_smooth_rank_one_field_has_unit_e1(self) -> None:
        """Keep e1 near 1 when sensitivities vary slowly along RO.

        Returns:
            None.
        """
        rng = np.random.default_rng(501)
        shape = (24, 4, 3)
        s1, _ = _smooth_orthonormal_fields(shape, 8)
        weights = rng.uniform(0.5, 2.0, shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        vectors = (s1 * weights[..., np.newaxis]).astype(np.complex64)
        result = local_coil_rank(vectors, np.ones(shape, dtype=bool))
        self.assertEqual(set(result), {"e1", "kappa2", "members"})
        self.assertEqual(result["e1"].dtype, np.float64)
        self.assertEqual(result["members"].dtype, np.int32)
        defined = np.isfinite(result["e1"])
        np.testing.assert_array_equal(defined[1:-1], True)
        self.assertGreater(float(np.min(result["e1"][defined])), 0.999)
        self.assertLess(float(np.max(result["kappa2"][defined])), 1e-3)

    def test_two_source_mixture_lowers_e1(self) -> None:
        """Reproduce the analytic spectrum of a mixture rotating along RO.

        With ``d(x) = cos(pi x / 4) s_a + sin(pi x / 4) s_b`` and five RO
        members, the Gram eigenvalues are 3 and 2 for every interior voxel,
        so ``e1 = 3 / 5`` and ``kappa2 = 2 / 3`` independent of per-voxel
        complex weights.

        Returns:
            None.
        """
        rng = np.random.default_rng(502)
        shape = (16, 3, 2)
        coils = 6
        s_a, s_b = _fixed_orthonormal_vectors(rng, coils, 2)
        angle = np.pi * np.arange(shape[0]) / 4.0
        profile = (
            np.cos(angle)[:, np.newaxis] * s_a + np.sin(angle)[:, np.newaxis] * s_b
        )
        weights = rng.uniform(0.5, 2.0, shape) * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        vectors = profile[:, np.newaxis, np.newaxis, :] * weights[..., np.newaxis]
        result = local_coil_rank(vectors, np.ones(shape, dtype=bool))
        np.testing.assert_allclose(result["e1"][2:-2], 0.6, atol=1e-12)
        np.testing.assert_allclose(result["kappa2"][2:-2], 2.0 / 3.0, atol=1e-12)
        s1, _ = _smooth_orthonormal_fields(shape, coils)
        smooth = local_coil_rank(s1, np.ones(shape, dtype=bool))
        self.assertLess(float(np.nanmax(result["e1"])), float(np.nanmin(smooth["e1"])) - 0.2)

    def test_members_at_edges_without_wrap_around(self) -> None:
        """Clip RO neighbourhoods at array edges and require enough members.

        Returns:
            None.
        """
        rng = np.random.default_rng(503)
        shape = (12, 2, 2)
        s_a, s_b = _fixed_orthonormal_vectors(rng, 5, 2)
        vectors = np.broadcast_to(s_a, shape + (5,)).copy()
        vectors[-1] = s_b
        support = np.ones(shape, dtype=bool)
        default = local_coil_rank(vectors, support)
        np.testing.assert_array_equal(
            default["members"][:, 0, 0], [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3]
        )
        self.assertTrue(np.all(np.isnan(default["e1"][[0, -1]])))
        self.assertTrue(np.all(np.isnan(default["kappa2"][[0, -1]])))
        self.assertTrue(np.all(np.isfinite(default["e1"][1:-1])))
        relaxed = local_coil_rank(vectors, support, min_members=3)
        # A wrapped box would mix the s_b plane into RO index 0.
        np.testing.assert_allclose(relaxed["e1"][0], 1.0, atol=1e-12)
        np.testing.assert_allclose(relaxed["e1"][-1], 2.0 / 3.0, atol=1e-12)
        np.testing.assert_allclose(relaxed["kappa2"][-1], 0.5, atol=1e-12)
        np.testing.assert_allclose(relaxed["e1"][-2], 0.75, atol=1e-12)
        np.testing.assert_allclose(relaxed["e1"][-3], 0.8, atol=1e-12)

        sparse = support.copy()
        sparse[4] = False
        holes = local_coil_rank(vectors, sparse)
        np.testing.assert_array_equal(holes["members"][4], 0)
        self.assertTrue(np.all(np.isnan(holes["e1"][4])))
        np.testing.assert_array_equal(
            holes["members"][:, 0, 0], [3, 4, 4, 4, 0, 4, 4, 5, 5, 5, 4, 3]
        )

    def test_three_dimensional_box_on_both_matrix_paths(self) -> None:
        """Handle a (5, 3, 3) box without PAR wrap for small and large Nc.

        With 4 coils the box (45 voxels) exceeds the coil count and the
        ``D D^H`` path is used; with 48 coils the Gram path is used. Both must
        give the analytic PAR-edge spectra.

        Returns:
            None.
        """
        rng = np.random.default_rng(504)
        shape = (9, 5, 6)
        for coils in (4, 48):
            s_a, s_b = _fixed_orthonormal_vectors(rng, coils, 2)
            vectors = np.broadcast_to(s_a, shape + (coils,)).copy()
            vectors[:, :, -1] = s_b
            result = local_coil_rank(
                vectors, np.ones(shape, dtype=bool), neighborhood=(5, 3, 3), par_chunk=2
            )
            self.assertEqual(result["members"][4, 2, 2], 45)
            self.assertEqual(result["members"][0, 0, 0], 12)
            np.testing.assert_allclose(result["e1"][2:-2, 1:-1, 0], 1.0, atol=1e-12)
            np.testing.assert_allclose(result["e1"][2:-2, 1:-1, -2], 2.0 / 3.0, atol=1e-12)
            np.testing.assert_allclose(result["e1"][2:-2, 1:-1, -1], 0.5, atol=1e-12)
            np.testing.assert_allclose(result["kappa2"][2:-2, 1:-1, -1], 1.0, atol=1e-12)

    def test_par_chunk_does_not_change_results(self) -> None:
        """Return identical arrays for every PAR chunk size.

        Returns:
            None.
        """
        rng = np.random.default_rng(505)
        shape = (7, 5, 11)
        vectors = _complex_normal(rng, shape + (6,)).astype(np.complex64)
        support = rng.uniform(size=shape) < 0.8
        for box in ((5, 1, 1), (3, 3, 5)):
            reference = local_coil_rank(vectors, support, neighborhood=box, par_chunk=5)
            for chunk in (1, 2, 3, 8, 64):
                result = local_coil_rank(vectors, support, neighborhood=box, par_chunk=chunk)
                for key in ("e1", "kappa2", "members"):
                    np.testing.assert_array_equal(result[key], reference[key])

    def test_invalid_local_rank_arguments_are_rejected(self) -> None:
        """Reject malformed boxes, member counts, chunks, and support values.

        Returns:
            None.
        """
        vectors = np.ones((6, 2, 2, 3), dtype=np.complex64)
        support = np.ones((6, 2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "odd"):
            local_coil_rank(vectors, support, neighborhood=(4, 1, 1))
        with self.assertRaisesRegex(ValueError, "odd"):
            local_coil_rank(vectors, support, neighborhood=(5, 1))
        with self.assertRaisesRegex(ValueError, "Minimum member count"):
            local_coil_rank(vectors, support, min_members=6)
        with self.assertRaisesRegex(ValueError, "Minimum member count"):
            local_coil_rank(vectors, support, min_members=1)
        with self.assertRaisesRegex(ValueError, "PAR chunk"):
            local_coil_rank(vectors, support, par_chunk=0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            local_coil_rank(vectors, support.astype(np.int8))
        with self.assertRaisesRegex(ValueError, "two coils"):
            local_coil_rank(vectors[..., :1], support)
        corrupted = vectors.copy()
        corrupted[2, 1, 1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            local_coil_rank(corrupted, support)
        outside = support.copy()
        outside[2, 1, 1] = False
        self.assertTrue(np.isnan(local_coil_rank(corrupted, outside)["e1"][2, 1, 1]))


class CoherenceTests(unittest.TestCase):
    """Verify phase-free CSM smoothness and descriptive map-switching labels."""

    def test_smooth_maps_are_coherent_and_phase_free(self) -> None:
        """Keep C1 and C2 near 1 for smooth maps with random per-voxel phases.

        Returns:
            None.
        """
        rng = np.random.default_rng(601)
        shape = (10, 6, 5)
        s1, s2 = _smooth_orthonormal_fields(shape, 8)
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, shape + (1, 2)))
        maps = np.stack([s1, s2], axis=-1) * phases
        basis, rank = orthonormal_map_basis(maps)
        mask = np.ones(shape, dtype=bool)
        mask[5, 2, 2] = False
        c1, c2 = csm_coherence(maps[..., 0], basis, rank, mask)
        self.assertEqual((c1.dtype, c2.dtype), (np.float64, np.float64))
        self.assertTrue(np.isnan(c1[5, 2, 2]) and np.isnan(c2[5, 2, 2]))
        self.assertGreater(float(np.min(c1[mask])), 0.999)
        self.assertGreater(float(np.min(c2[mask])), 0.999)
        self.assertLessEqual(float(np.max(c1[mask])), 1.0)
        lonely = np.zeros(shape, dtype=bool)
        lonely[4, 3, 2] = True
        lonely_c1, lonely_c2 = csm_coherence(maps[..., 0], basis, rank, lonely)
        self.assertTrue(np.all(np.isnan(lonely_c1)) and np.all(np.isnan(lonely_c2)))

    def test_swapped_degenerate_pair_drops_c1_only(self) -> None:
        """Lower C1 at a swap border while C2 stays near 1, and label it.

        Returns:
            None.
        """
        shape = (10, 6, 5)
        s1, s2 = _smooth_orthonormal_fields(shape, 8)
        swapped = slice(3, 7)
        map_1, map_2 = s1.copy(), s2.copy()
        map_1[swapped], map_2[swapped] = s2[swapped], s1[swapped]
        maps = np.stack([map_1, map_2], axis=-1)
        basis, rank = orthonormal_map_basis(maps)
        mask = np.ones(shape, dtype=bool)
        c1, c2 = csm_coherence(maps[..., 0], basis, rank, mask)
        border = np.zeros(shape[0], dtype=bool)
        border[[2, 3, 6, 7]] = True
        self.assertLess(float(np.max(c1[border])), 0.1)
        self.assertGreater(float(np.min(c1[~border])), 0.999)
        self.assertGreater(float(np.min(c2)), 0.999)

        lambda1 = np.full(shape, 0.95)
        lambda2 = np.full(shape, 0.3)
        lambda1[swapped] = lambda2[swapped] = 0.9
        labels = map_switching_mask(c1, c2, lambda1, lambda2)
        self.assertEqual(labels.dtype, np.bool_)
        expected = np.zeros(shape, dtype=bool)
        expected[[3, 6]] = True
        np.testing.assert_array_equal(labels, expected)
        self.assertIn("descriptive", map_switching_mask.__doc__.lower())
        nan_c1 = c1.copy()
        nan_c1[3] = np.nan
        self.assertFalse(map_switching_mask(nan_c1, c2, lambda1, lambda2)[3].any())
        with self.assertRaisesRegex(ValueError, "thresholds"):
            map_switching_mask(c1, c2, lambda1, lambda2, gap_max=-0.1)
        with self.assertRaisesRegex(ValueError, "one shape"):
            map_switching_mask(c1, c2[:-1], lambda1, lambda2)

    def test_edges_are_not_wrapped(self) -> None:
        """Keep edge coherence high when opposite RO edges are orthogonal.

        Map 1 rotates from ``u`` to ``w`` across RO while map 2 stays ``v``;
        a wrapped neighbour would give C1 = 0 and C2 = 0.5 at the edges.

        Returns:
            None.
        """
        rng = np.random.default_rng(603)
        shape = (10, 4, 3)
        u, v, w = _fixed_orthonormal_vectors(rng, 6, 3)
        theta = 0.5 * np.pi * np.arange(shape[0]) / (shape[0] - 1)
        map_1 = np.cos(theta)[:, np.newaxis] * u + np.sin(theta)[:, np.newaxis] * w
        map_1 = np.broadcast_to(map_1[:, np.newaxis, np.newaxis, :], shape + (6,))
        map_2 = np.broadcast_to(v, shape + (6,))
        maps = np.stack([map_1, map_2], axis=-1)
        basis, rank = orthonormal_map_basis(maps)
        c1, c2 = csm_coherence(maps[..., 0], basis, rank, np.ones(shape, dtype=bool))
        step = np.cos(theta[1])
        np.testing.assert_allclose(c1[[0, -1]], step, atol=1e-12)
        np.testing.assert_allclose(c2[[0, -1]], (step**2 + 1.0) / 2.0, atol=1e-12)

    def test_rank_deficient_voxels_have_no_c2(self) -> None:
        """Return NaN C2 where the two-map rank is below 2.

        Returns:
            None.
        """
        shape = (6, 4, 3)
        s1, s2 = _smooth_orthonormal_fields(shape, 8)
        s2[0] = 0
        maps = np.stack([s1, s2], axis=-1)
        basis, rank = orthonormal_map_basis(maps)
        np.testing.assert_array_equal(rank[0], 1)
        c1, c2 = csm_coherence(maps[..., 0], basis, rank, np.ones(shape, dtype=bool))
        self.assertTrue(np.all(np.isnan(c2[0])))
        self.assertTrue(np.all(np.isfinite(c2[1:])))
        self.assertTrue(np.all(np.isfinite(c1)))
        with self.assertRaisesRegex(ValueError, "Rank does not match"):
            csm_coherence(maps[..., 0], basis, np.full(shape, 2), np.ones(shape, dtype=bool))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            csm_coherence(maps[..., 0], 2.0 * basis, rank, np.ones(shape, dtype=bool))


class ModuleContractTests(unittest.TestCase):
    """Verify determinism and the pure-computation module contract."""

    def test_repeated_calls_are_deterministic(self) -> None:
        """Return bitwise identical arrays for repeated identical calls.

        Returns:
            None.
        """
        rng = np.random.default_rng(701)
        shape = (6, 8, 8)
        kspace = _complex_normal(rng, shape + (4,)).astype(np.complex64)
        window = acs_apodization_window(shape, 4)
        images = coil_images(kspace, window)
        np.testing.assert_array_equal(images, coil_images(kspace, window))
        maps = _complex_normal(rng, shape + (4, 2))
        basis, rank = orthonormal_map_basis(maps)
        repeated_basis, repeated_rank = orthonormal_map_basis(maps)
        np.testing.assert_array_equal(basis, repeated_basis)
        np.testing.assert_array_equal(rank, repeated_rank)
        mask = np.ones(shape, dtype=bool)
        np.testing.assert_array_equal(
            projection_residual(images, basis, mask), projection_residual(images, basis, mask)
        )
        covariance = np.eye(4)
        np.testing.assert_array_equal(
            residual_to_noise_ratio(images, basis, covariance, mask),
            residual_to_noise_ratio(images, basis, covariance, mask),
        )
        local = local_coil_rank(images, mask, neighborhood=(5, 3, 3))
        repeated_local = local_coil_rank(images, mask, neighborhood=(5, 3, 3))
        for key in local:
            np.testing.assert_array_equal(local[key], repeated_local[key])
        coherence = csm_coherence(maps[..., 0], basis, rank, mask)
        repeated_coherence = csm_coherence(maps[..., 0], basis, rank, mask)
        for values, repeated in zip(coherence, repeated_coherence):
            np.testing.assert_array_equal(values, repeated)

    def test_internal_chunking_does_not_change_results(self) -> None:
        """Give identical results when the private memory budgets force chunking.

        The synthetic arrays are small enough to fit in one block, so the
        axis-0 block budget, the coherence PAR-chunk budget, and the Gram batch
        budget are reduced to exercise every chunk and halo boundary. Voxelwise
        results must be bitwise identical; the covariance is a sum over voxels
        whose accumulation order changes with the blocks, so it is compared to
        within rounding.

        Returns:
            None.
        """
        rng = np.random.default_rng(702)
        shape = (5, 4, 7)
        coils = 3
        vectors = _complex_normal(rng, shape + (coils,)).astype(np.complex64)
        maps = _complex_normal(rng, shape + (coils, 2)).astype(np.complex64)
        maps[1, 2, 3, :, 1] = 0
        mask = rng.uniform(size=shape) < 0.85
        covariance = np.eye(coils) + 0.1

        def evaluate() -> dict[str, np.ndarray]:
            """Evaluate every chunked function on the fixed synthetic inputs.

            Returns:
                Mapping from a result label to its array.
            """
            basis, rank = orthonormal_map_basis(maps)
            basis_1, _ = orthonormal_map_basis(maps[..., :1])
            c1, c2 = csm_coherence(maps[..., 0], basis, rank, mask)
            local = local_coil_rank(vectors, mask, neighborhood=(3, 3, 3), min_members=3)
            return {
                "basis": basis,
                "rank": rank,
                "norm": coil_vector_norm(vectors),
                "normalized": normalize_coil_vectors(vectors, mask),
                "covariance": empirical_coil_covariance(vectors, mask)[0],
                "rho_1": projection_residual(vectors, basis_1, mask),
                "rho_2": projection_residual(vectors, basis, mask),
                "rnr_2": residual_to_noise_ratio(vectors, basis, covariance, mask),
                "orthonormality": map_orthonormality_error(maps),
                "alpha": map_reproduction(maps[..., 0], maps[..., 1], mask),
                "c1": c1,
                "c2": c2,
                **local,
            }

        reference = evaluate()
        # One PAR plane per coherence chunk and one RO row per axis-0 block.
        with (
            patch.object(metrics, "_CHUNK_ELEMENTS", 4 * 7 * coils * 2),
            patch.object(metrics, "_GRAM_BATCH_BYTES", 1),
        ):
            chunked = evaluate()
        self.assertEqual(set(chunked), set(reference))
        np.testing.assert_allclose(
            chunked.pop("covariance"), reference.pop("covariance"), rtol=1e-13, atol=0
        )
        for key, values in reference.items():
            np.testing.assert_array_equal(chunked[key], values, err_msg=key)
        self.assertTrue(np.isfinite(reference["c2"]).any())
        self.assertTrue(np.isnan(reference["c2"][1, 2, 3]))

    def test_module_source_has_no_external_execution_or_file_access(self) -> None:
        """Keep the metrics module free of process launches and file opens.

        Returns:
            None.
        """
        source = Path(metrics.__file__).read_text(encoding="utf-8")
        for token in ("subprocess", "os.system", "bart ", "open("):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
