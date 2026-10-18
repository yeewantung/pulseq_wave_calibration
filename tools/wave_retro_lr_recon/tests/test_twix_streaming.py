"""Numerical and bounded-read contracts for TWIX IMAGE streaming."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twix_fakes import FakeImageStream, FakeRefscanStream
from wave_retro_lr.twix_streaming import (
    image_reading_identity, iter_compressed_partitions, load_compressed_image,
)


class StreamingTests(unittest.TestCase):
    """Compare partition projection with dense algebra and reject corrupt inputs."""

    def test_dense_equivalence_for_single_and_multiple_echoes(self):
        """Match dense projection with complex whitening and nonzero grid offsets."""
        rng = np.random.default_rng(42)
        for echoes in (1, 3):
            with self.subTest(echoes=echoes):
                values = (rng.normal(size=(4, 5, 3, echoes, 4))
                          + 1j * rng.normal(size=(4, 5, 3, echoes, 4))).astype(np.complex64)
                values[:, 1::2] = 0
                values[:, :, 1] = 0
                # A complex, nonorthogonal product also exercises W @ C order.
                whitening = (np.eye(4) + 0.2j * np.triu(np.ones((4, 4)), 1)).astype(np.complex64)
                basis = (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))).astype(np.complex64)
                combined = whitening @ basis
                mask = np.zeros((8, 5), dtype=bool)
                mask[np.ix_([2, 4, 6], [1, 3])] = True
                stream = FakeImageStream(values, skip_lin=2, skip_par=1)
                target = (4, 8, 5) if echoes == 1 else (4, 8, 5, echoes)
                result = load_compressed_image(stream, combined, output_shape=target, sampling_mask=mask)
                expected = np.zeros((4, 8, 5, echoes, 2), dtype=np.complex64)
                expected[:, 2:7, 1:4] = (values @ whitening) @ basis
                if echoes == 1:
                    expected = expected[:, :, :, 0]
                np.testing.assert_allclose(result.numpy(), expected, atol=3e-6, rtol=3e-6)
                self.assertEqual(result.dtype, torch.complex64)
                self.assertEqual(result.device.type, "cpu")
                self.assertEqual(len(stream.selections), 2 * echoes)
                self.assertEqual({s[3] for s in stream.selections}, {0, 2})
                self.assertEqual({s[7] for s in stream.selections}, set(range(echoes)))
                self.assertTrue(stream.flagRemoveOS)
                self.assertTrue(stream.squeeze)

    def test_singleton_axes_and_named_dimension_order(self):
        """Preserve singleton spatial/coil axes even with reordered named dimensions."""
        stream = FakeImageStream(np.full((2, 1, 1, 1), 2 + 3j))
        order = list(reversed(range(16)))
        stream.values = stream.values.transpose(order)
        stream.fullSize = stream.fullSize[order]
        stream.dataDims = [stream.dataDims[i] for i in order]
        result = load_compressed_image(stream, np.array([[1j]]), output_shape=(2, 1, 1),
                                       sampling_mask=np.ones((1, 1), bool))
        np.testing.assert_array_equal(result.numpy(), np.full((2, 1, 1, 1), -3 + 2j))

    def test_rejects_physical_nullspace_contamination(self):
        """Reject off-mask data even if compression would annihilate it exactly."""
        values = np.ones((2, 3, 2, 2), dtype=np.complex64)
        mask = np.ones((3, 2), dtype=bool)
        mask[1] = False
        stream = FakeImageStream(values)
        with self.assertRaisesRegex(ValueError, "outside the sampling mask"):
            load_compressed_image(stream, np.array([[1], [-1]]),
                                  output_shape=(2, 3, 2), sampling_mask=mask)
        self.assertTrue(stream.flagRemoveOS)
        self.assertTrue(stream.squeeze)

    def test_rejects_nonfinite_physical_data_in_any_echo(self):
        """Reject NaN/Inf before projection, including later GRE echoes."""
        for invalid in (np.nan, np.inf):
            values = np.ones((2, 3, 2, 2, 2), dtype=np.complex64)
            values[0, 0, 1, 1, 0] = invalid
            stream = FakeImageStream(values)
            with self.assertRaisesRegex(ValueError, "physical IMAGE.*nonfinite"):
                load_compressed_image(stream, np.ones((2, 1)), output_shape=(2, 3, 2, 2),
                                      sampling_mask=np.ones((3, 2), bool))
            self.assertTrue(stream.squeeze)

    def test_rejects_metadata_before_reading(self):
        """Reject incompatible geometry, basis, masks, and unsupported axes before I/O."""
        cases = [
            ({"output_shape": (3, 3, 2)}, "readout"),
            ({"output_shape": (2, 3, 2, 2)}, "echo"),
            ({"output_shape": (2, 3.5, 2)}, "positive integers"),
            ({"compression_matrix": np.ones((3, 1))}, "physical-coil"),
            ({"compression_matrix": np.full((2, 1), np.nan)}, "finite"),
            ({"compression_matrix": np.ones((2, 0))}, "two-dimensional"),
            ({"sampling_mask": np.ones((3, 2), float)}, "boolean"),
            ({"sampling_mask": np.zeros((3, 2), bool)}, "nonempty"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                stream = FakeImageStream(np.ones((2, 3, 2, 2)))
                arguments = dict(compression_matrix=np.ones((2, 1)), output_shape=(2, 3, 2),
                                 sampling_mask=np.ones((3, 2), bool))
                arguments.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    load_compressed_image(stream, **arguments)
                self.assertFalse(stream.selections)
        for axis in ("Set", "Rep", "Ave", "Seg", "Sli"):
            stream = FakeImageStream(np.ones((2, 3, 2, 2)))
            stream.fullSize[stream.dataDims.index(axis)] = 2
            with self.assertRaisesRegex(ValueError, "non-singleton"):
                load_compressed_image(stream, np.ones((2, 1)), output_shape=(2, 3, 2),
                                      sampling_mask=np.ones((3, 2), bool))
            self.assertFalse(stream.selections)

    def test_rejects_out_of_bounds_and_incomplete_extents(self):
        """Require stored grid offsets to cover the complete validated mask."""
        for offset, target, message in ((1, (2, 3, 2), "exceed"),
                                        (0, (2, 4, 2), "do not cover")):
            stream = FakeImageStream(np.ones((2, 3, 2, 2)), skip_lin=offset)
            with self.assertRaisesRegex(ValueError, message):
                load_compressed_image(stream, np.ones((2, 1)), output_shape=target,
                                      sampling_mask=np.ones(target[1:], bool))
            self.assertFalse(stream.selections)

    def test_generator_close_restores_flags(self):
        """Restore stream options when a consumer stops after the first partition."""
        stream = FakeImageStream(np.ones((2, 3, 2, 2)))
        blocks = iter_compressed_partitions(stream, np.ones((2, 1)), output_shape=(2, 3, 2),
                                            sampling_mask=np.ones((3, 2), bool))
        next(blocks)
        blocks.close()
        self.assertTrue(stream.flagRemoveOS)
        self.assertTrue(stream.squeeze)
        self.assertEqual(len(stream.selections), 1)

    def test_rejects_partition_or_echo_averaging(self):
        """Prevent mapVBVD averaging from expanding a bounded selection."""
        for axis in ("Lin", "Par", "Eco"):
            stream = FakeImageStream(np.ones((2, 3, 2, 2, 2)))
            stream.average_dim[stream.dataDims.index(axis)] = True
            with self.assertRaisesRegex(ValueError, "averaging"):
                load_compressed_image(stream, np.ones((2, 1)), output_shape=(2, 3, 2, 2),
                                      sampling_mask=np.ones((3, 2), bool))
            self.assertFalse(stream.selections)

    def test_rejects_unexpected_return_layout(self):
        """Reject a rank-changing backend response and restore stream options."""
        stream = FakeImageStream(np.ones((2, 3, 2, 2)))
        with patch.object(FakeImageStream, "__getitem__", return_value=np.ones((2, 2, 3))):
            with self.assertRaisesRegex(RuntimeError, "Unexpected IMAGE partition shape"):
                load_compressed_image(stream, np.ones((2, 1)), output_shape=(2, 3, 2),
                                      sampling_mask=np.ones((3, 2), bool))
        self.assertTrue(stream.flagRemoveOS)
        self.assertTrue(stream.squeeze)

    def test_rejects_projection_overflow(self):
        """Reject nonfinite compressed values even when both inputs were finite."""
        stream = FakeImageStream(np.full((2, 3, 2, 2), 1e30, dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "Compressed IMAGE.*nonfinite"):
            load_compressed_image(stream, np.full((2, 1), 1e30, dtype=np.complex64),
                                  output_shape=(2, 3, 2), sampling_mask=np.ones((3, 2), bool))

    def test_reader_provenance(self):
        """Record the bounded-read policy and the exact local implementation hash."""
        identity = image_reading_identity()
        self.assertEqual(identity["partitions_per_read"], 1)
        self.assertFalse(identity["physical_image_materialized"])
        self.assertEqual(len(identity["source_sha256"]), 64)

    def test_gre_preparation_streams_validated_measurement_and_preserves_echoes(self):
        """Prepare synthetic GRE outputs through streaming with mocked calibration.

        Checks selected measurement, echo order, coil algebra, manifest policy,
        and written BART arrays without a real TWIX, PSF fit, GPU, or BART run.
        """
        from wave_retro_lr import gre
        from wave_retro_lr.bart_io import open_cfl
        from wave_retro_lr.sampling import pure_cartesian_image_lattice_mask

        mask, sampling = pure_cartesian_image_lattice_mask(
            (12, 72), acceleration_lin_par=(3, 1), residue_lin_par=(2, 0)
        )
        physical = np.zeros((1000, 10, 72, 2, 2), dtype=np.complex64)
        physical[:, ::3, :, 0, :] = [1 + 2j, 3 - 1j]
        physical[:, ::3, :, 1, :] = [-2 + 1j, 4 + 3j]
        stream = FakeImageStream(physical, skip_lin=2)
        basis = np.array([[0.6], [0.8j]], dtype=np.complex64)
        cfg = dict(Nx=250, Ny=12, Nz=72, Nx_os=1000, Necho=2,
                   TE_s=[.01, .02], FOVxyz_m=np.array([.220, .0108, .180]),
                   Ncalib1=2, Nacs=2, Nsets=5, ACSSetID=4, os_factor=4, yflip=1, zflip=1)
        coefficients = tuple(np.zeros(1000) for _ in range(3))
        native = SimpleNamespace(
            load_img=Mock(side_effect=AssertionError("Bulk IMAGE read forbidden")),
            load_ref=Mock(return_value=torch.ones((1000, 2, 2, 5, 2), dtype=torch.complex64)),
            _check_integrated_refscan_shape=lambda values, **kwargs: values,
            remove_readout_oversampling_kspace=Mock(
                return_value=torch.ones((250, 2, 2, 2), dtype=torch.complex64)),
            estimate_cc_matrix_coillast=Mock(return_value=(basis, np.array([3., 2.]), np.array([.8, 1.]))),
            apply_cc_coillast_torch=lambda values, matrix, **kwargs: values @ torch.from_numpy(matrix),
            fit_wave_psf_deviation_from_projection=Mock(return_value=(*coefficients, {"projection_quality": {}})),
            _process_psf_coefficients=Mock(return_value=(*coefficients, {"kx_range": None})),
            _echo_theoretical_wave_trajectories=Mock(return_value=[coefficients[:2]] * 2),
        )

        def inspect_twix(path, *, measurement_sink, **kwargs):
            """Return synthetic validated metadata and its selected measurement.

            Args:
                path: Unused synthetic TWIX path.
                measurement_sink: List receiving dummy and selected measurements.
                **kwargs: Unused sequence consistency arguments.

            Returns:
                The validated mask and minimal preparation metadata.
            """
            measurement_sink[:] = [object(), SimpleNamespace(
                image=stream,
                refscan=FakeRefscanStream(np.ones((1000, 2, 2, 5, 2), dtype=np.complex64)),
            )]
            return mask, {"measurement_index": 1, "sampling": sampling}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            twix, sequence = root / "synthetic.dat", root / "synthetic.seq"
            twix.write_bytes(b"synthetic test fixture")
            sequence.write_bytes(b"synthetic test sequence")
            with (
                patch.object(gre, "load_wave_gre_helpers", return_value=native),
                patch.object(gre, "validate_gre_sequence", return_value=(None, cfg, None, None)),
                patch.object(gre, "inspect_gre_twix", side_effect=inspect_twix),
                patch.object(gre, "_write_gre_psf_coefficient_plots"),
                patch.object(gre, "_fit_gre_projection_coefficients",
                             return_value=(*coefficients, {"projection_quality": {}})),
            ):
                manifest = gre.prepare_normal_gre(twix, root / "outputs", sequence, virtual_coils=1)
            native.load_img.assert_not_called()
            self.assertEqual(len(stream.selections), 72 * 2)
            self.assertFalse(manifest["image_reading"]["physical_image_materialized"])
            inputs = root / "outputs/vcc1/normal/bart_inputs"
            for echo in range(2):
                actual = open_cfl(inputs / f"wave_kspace_echo-{echo + 1:02d}")
                expected = np.zeros((1000, 12, 72, 1), dtype=np.complex64)
                expected[:, 2:] = physical[:, :, :, echo] @ basis
                np.testing.assert_allclose(actual[..., 0], expected, atol=2e-6)
                del actual


if __name__ == "__main__":
    unittest.main()
