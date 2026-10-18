"""Selective REFSCAN read contracts and dense-calibration equivalence tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twix_fakes import FakeRefscanStream
from wave_retro_lr.refscan import open_refscan, read_refscan_set, refscan_shape, refscan_reading_identity


class RefscanTests(unittest.TestCase):
    """Require explicit labels, bounded reads, and unchanged fitted results."""

    def test_selected_planes_and_last_acs_match_dense_slices(self):
        """Compare all four projections and the final ACS SET against dense indexing."""
        rng = np.random.default_rng(7)
        values = (rng.normal(size=(4, 6, 6, 6, 3))
                  + 1j * rng.normal(size=(4, 6, 6, 6, 3))).astype(np.complex64)
        stream = FakeRefscanStream(values)
        self.assertEqual(refscan_shape(stream, ncalib=6, nacs=4), values.shape)
        self.assertFalse(stream.selections)
        for index, lin, par in ((0, 6, 1), (1, 6, 1), (2, 1, 6), (3, 1, 6), (-1, 4, 4)):
            with self.subTest(set_index=index):
                stream.selections.clear()
                result = read_refscan_set(stream, index, lin_count=lin, par_count=par)
                np.testing.assert_array_equal(result.numpy(), values[:, :lin, :par, index, :])
                self.assertEqual(len(stream.selections), par)
                self.assertEqual({selection[9] for selection in stream.selections}, {index % 6})
                self.assertEqual({selection[3] for selection in stream.selections}, set(range(par)))
                self.assertTrue(all(selection[2] == slice(0, lin) for selection in stream.selections))
                self.assertTrue(stream.squeeze)
                self.assertTrue(stream.flagRemoveOS)

    def test_named_axes_preserve_singleton_readout_and_coil(self):
        """Retain singleton dimensions when metadata axes are reordered."""
        values = np.full((1, 3, 3, 5, 1), 2 + 3j, dtype=np.complex64)
        stream = FakeRefscanStream(values)
        order = list(reversed(range(16)))
        stream.values = stream.values.transpose(order)
        stream.fullSize = stream.fullSize[order]
        stream.dataDims = [stream.dataDims[index] for index in order]
        result = read_refscan_set(stream, 3, lin_count=1, par_count=3)
        self.assertEqual(tuple(result.shape), (1, 1, 3, 1))
        np.testing.assert_array_equal(result.numpy(), values[:, :1, :, 3, :])

    def test_invalid_regions_and_missing_sets_rejected_before_read(self):
        """Reject invalid labels instead of silently selecting or clipping another SET."""
        stream = FakeRefscanStream(np.ones((4, 6, 6, 5, 2)))
        for index, lin, par in ((5, 2, 2), (-6, 2, 2), (4, 7, 2), (4, 2, 7),
                                (4, 0, 2), (4.2, 2, 2), (True, 2, 2)):
            with self.assertRaises(ValueError):
                read_refscan_set(stream, index, lin_count=lin, par_count=par)
        stream.Set = np.arange(4)
        with self.assertRaisesRegex(ValueError, "no acquired records"):
            read_refscan_set(stream, 4, lin_count=2, par_count=2)
        self.assertFalse(stream.selections)

    def test_metadata_geometry_and_averaging_rejected(self):
        """Keep complete layout validation before selectively reading a small region."""
        for name in ("Lin", "Par", "Set", "Sli", "Eco", "Ave"):
            stream = FakeRefscanStream(np.ones((4, 6, 6, 5, 2)))
            stream.fullSize[stream.dataDims.index(name)] = 2
            with self.assertRaises(ValueError):
                refscan_shape(stream, ncalib=6, nacs=4)
            self.assertFalse(stream.selections)
        stream = FakeRefscanStream(np.ones((4, 6, 6, 5, 2)))
        stream.average_dim[9] = True
        with self.assertRaisesRegex(ValueError, "averaging"):
            read_refscan_set(stream, 4, lin_count=2, par_count=2)
        stream.average_dim[9] = False
        stream.skipLin = 1
        with self.assertRaisesRegex(ValueError, "origins"):
            refscan_shape(stream, ncalib=6, nacs=4)

    def test_invalid_payload_restores_flags(self):
        """Reject nonfinite selected samples and rank changes without retaining stream flags."""
        for invalid in (np.nan, np.inf):
            values = np.ones((4, 6, 6, 5, 2), dtype=np.complex64)
            values[1, 1, 1, 4, 0] = invalid
            stream = FakeRefscanStream(values)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_refscan_set(stream, 4, lin_count=2, par_count=2)
            self.assertTrue(stream.flagRemoveOS)
            self.assertTrue(stream.squeeze)
        stream = FakeRefscanStream(np.ones((4, 6, 6, 5, 2)))
        with patch.object(FakeRefscanStream, "__getitem__", return_value=np.ones((4, 2, 2))):
            with self.assertRaisesRegex(RuntimeError, "Unexpected selected REFSCAN shape"):
                read_refscan_set(stream, 4, lin_count=2, par_count=2)
        self.assertTrue(stream.flagRemoveOS)
        self.assertTrue(stream.squeeze)

    def test_open_preserves_pinned_measurement_selection(self):
        """Select measurement 1, not the last RAID entry, without reading any payload."""
        selected = FakeRefscanStream(np.ones((2, 3, 3, 5, 1)))
        measurements = [{"refscan": object()}, {"refscan": selected}, {"refscan": object()}]
        with patch("mapvbvd.mapVBVD", return_value=measurements):
            self.assertIs(open_refscan("synthetic.dat"), selected)
        self.assertFalse(selected.selections)
        with patch("mapvbvd.mapVBVD", return_value={}):
            with self.assertRaisesRegex(ValueError, "no REFSCAN"):
                open_refscan("synthetic.dat")
        self.assertFalse(refscan_reading_identity()["dense_all_sets_materialized"])

    def test_gre_fitting_matches_pinned_dense_fitter(self):
        """Match actual upstream Fourier/IRLS coefficients, quality, and raw cache arrays."""
        from wave_retro_lr.gre import load_wave_gre_helpers, _fit_gre_projection_coefficients

        native = load_wave_gre_helpers()
        nx, width, coils = 16, 12, 2
        reference = torch.zeros((nx, width, width, 5, coils), dtype=torch.complex64)
        coordinate = (np.arange(width) - width / 2) / width
        a = .1 * np.sin(np.arange(nx) / 3)
        b = .2 * np.cos(np.arange(nx) / 4)
        for first, axis, slope in ((0, 1, a), (2, 2, b)):
            shape = (nx, width, 1, coils) if axis == 1 else (nx, 1, width, coils)
            nowave = torch.ones(shape, dtype=torch.complex64)
            phase = slope[:, None] * coordinate[None, :] + .02
            phase = phase[:, :, None, None] if axis == 1 else phase[:, None, :, None]
            wave = nowave * torch.from_numpy(np.exp(1j * phase).astype(np.complex64))
            if axis == 1:
                reference[:, :, :1, first] = native.fftc_dim(nowave, dim=axis)
                reference[:, :, :1, first + 1] = native.fftc_dim(wave, dim=axis)
            else:
                reference[:, :1, :, first] = native.fftc_dim(nowave, dim=axis)
                reference[:, :1, :, first + 1] = native.fftc_dim(wave, dim=axis)
        cfg = dict(Nx_os=nx, Ncalib1=width, Ncalib2=1, Nacs=4, Nsets=5,
                   FOVxyz_m=[.2, .21, .18], yflip=-1, zflip=1)
        trajectory = np.zeros((3, native._calibration_readout_count(cfg), nx))
        trajectory[1, width:2 * width] = .03 * np.sin(np.arange(nx) / 5)
        trajectory[2, 3 * width:4 * width] = .02 * np.cos(np.arange(nx) / 6)
        stream = FakeRefscanStream(reference.numpy())
        with tempfile.TemporaryDirectory() as directory:
            dense_dir, selected_dir = Path(directory) / "dense", Path(directory) / "selected"
            dense_dir.mkdir()
            selected_dir.mkdir()
            with patch.object(native, "load_ref", return_value=reference):
                dense = native.fit_wave_psf_deviation_from_projection(
                    twix_file=Path("synthetic.dat"), calib_lines=trajectory, cfg=cfg,
                    out_folder=dense_dir, file_tag="shared", return_diagnostics=True)
            selected = _fit_gre_projection_coefficients(
                native=native, reference=stream, calib_lines=trajectory, cfg=cfg,
                out_folder=selected_dir, file_tag="shared", return_diagnostics=True)
            for expected, actual in zip(dense[:3], selected[:3], strict=True):
                self.assertTrue(torch.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
            for projection in ("sin", "cos"):
                for key, expected in dense[3]["projection_quality"][projection].items():
                    np.testing.assert_allclose(selected[3]["projection_quality"][projection][key], expected,
                                               atol=1e-7, rtol=0)
            for cache in dense_dir.glob("*.npy"):
                np.testing.assert_array_equal(np.load(cache), np.load(selected_dir / cache.name))
        self.assertEqual(len(stream.selections), 2 + 2 * width)
        self.assertEqual({selection[9] for selection in stream.selections}, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
