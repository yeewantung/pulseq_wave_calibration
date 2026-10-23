"""Native support and scalar gain preservation on retrospective PE grids."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))

from wave_retro_lr.bart_io import create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.core import centered_fftn  # noqa: E402
from wave_retro_lr.retrospective import (  # noqa: E402
    CSM_RESAMPLING_POLICY, map_native_rss, native_pe_coordinates, resample_sensitivity_maps,
)
from wave_retro_lr.mprage import prepare_retro_sensitivity_maps  # noqa: E402
from wave_retro_lr.gre import prepare_retro_gre_sensitivity_maps  # noqa: E402


def write_maps(base: Path, rss: np.ndarray) -> None:
    """Write two constant-direction complex coils with a requested spatial RSS envelope."""
    maps = create_cfl(base, (*rss.shape, 2, 1))
    maps[..., 0, 0] = rss/np.sqrt(2)
    maps[..., 1, 0] = 1j*rss/np.sqrt(2)
    maps.flush()


def read_rss(base: Path) -> np.ndarray:
    """Return the RSS of one single-map test CFL."""
    values = open_cfl(base)[..., 0]
    return np.sqrt(np.sum(np.abs(values)**2, axis=3))


class NativeSupportTests(unittest.TestCase):
    """Validate physical-grid support, soft gain, phase and immutable derivative reuse."""

    def setUp(self):
        """Create a disposable native source with an asymmetric boundary and internal hole."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.target = self.root/"native", self.root/"low"
        self.rss = np.zeros((2, 12, 10), np.float32)
        self.rss[:, 2:10, 2:9] = 1
        self.rss[:, 4:6, 4:6] = 0
        write_maps(self.source, self.rss)

    def test_hard_support_rejects_fourier_ringing(self):
        """Ringing outside mapped native support stays zero instead of becoming unit RSS."""
        record = resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        lin = np.floor((np.arange(8)-4)*1.5+6+.5).astype(int)
        par = np.floor((np.arange(6)-3)*(10/6)+5+.5).astype(int)
        expected = self.rss[:, lin, :][:, :, par]
        np.testing.assert_allclose(read_rss(self.target), expected, atol=2e-6)
        legacy = centered_fftn(centered_fftn(self.rss, axes=(1, 2))[:, 2:10, 2:8], axes=(1, 2), inverse=True)
        self.assertTrue(np.any((np.abs(legacy) > 1e-8) & (expected == 0)))
        self.assertEqual(record["request"]["policy"], CSM_RESAMPLING_POLICY)
        maps = open_cfl(self.target)
        np.testing.assert_allclose(maps[..., 1, 0], 1j*maps[..., 0, 0], atol=1e-6)

    def test_soft_gain_is_not_renormalized_away(self):
        """Positive scalar weights, including tiny values and values above one, survive transfer."""
        for index, gain in enumerate((.2, .7, 1.5, 1e-12)):
            source, target = self.root/f"source{index}", self.root/f"target{index}"
            write_maps(source, self.rss*gain)
            resample_sensitivity_maps(source, target, target_lin_par=(8, 6))
            expected, _ = map_native_rss(self.rss*gain, (8, 6))
            np.testing.assert_allclose(read_rss(target), expected, rtol=3e-6, atol=gain*1e-6)

    def test_soft_linear_field_and_readout_planes(self):
        """Supported bilinear gain follows native spatial variation without mixing RO planes."""
        rss = np.broadcast_to(.2+.03*np.arange(12)[None, :, None], (2, 12, 10)).copy()
        rss[1] *= .5
        write_maps(self.source, rss)
        resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        expected = np.broadcast_to(.2+.03*np.arange(8)[None, :, None]*1.5, (2, 8, 6)).copy()
        expected[1] *= .5
        np.testing.assert_allclose(read_rss(self.target), expected, rtol=2e-6)

    def test_odd_even_center_convention_and_singleton(self):
        """Centered coordinates handle parity changes and singleton RO/PE without squeeze errors."""
        np.testing.assert_allclose(native_pe_coordinates(5, 4), [-.5, .75, 2, 3.25])
        np.testing.assert_allclose(native_pe_coordinates(6, 3), [1, 3, 5])
        rss = np.ones((1, 5, 7), np.float32)
        rss[:, 0] = 0
        write_maps(self.source, rss)
        resample_sensitivity_maps(self.source, self.target, target_lin_par=(4, 1))
        np.testing.assert_allclose(read_rss(self.target)[0, :, 0], [0, 1, 1, 1], atol=1e-6)

    def test_same_grid_is_bitwise_copy(self):
        """No support or phase change is introduced when no grid reduction is requested."""
        source = open_cfl(self.source, mode="r+")
        source[:] *= np.exp(.13j)*.4
        source.flush()
        record = resample_sensitivity_maps(self.source, self.target, target_lin_par=(12, 10))
        np.testing.assert_array_equal(open_cfl(self.source), open_cfl(self.target))
        self.assertEqual(record["operation"], "exact_same_grid_copy")

    def test_zero_field_remains_zero(self):
        """An absent support must never become background signal."""
        write_maps(self.source, np.zeros_like(self.rss))
        resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        self.assertFalse(np.any(open_cfl(self.target)))

    def test_cancelled_direction_uses_recorded_native_fallback(self):
        """If Fourier cropping cancels coil direction, use a valid native vector, not a new phase."""
        write_maps(self.source, np.ones_like(self.rss)*.3)
        source = open_cfl(self.source, mode="r+")
        source[:, 1::2] *= -1
        source.flush()
        record = resample_sensitivity_maps(self.source, self.target, target_lin_par=(6, 10))
        self.assertGreater(record["fallback_voxels"], 0)
        np.testing.assert_allclose(read_rss(self.target), .3, atol=1e-6)

    def test_hash_bound_resume_and_mutation(self):
        """Exact resume preserves bytes and refuses a changed source or derived output."""
        first = resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        path = self.target.with_name("low_resampling.json")
        before = sha256_file(path)
        second = resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        self.assertEqual(first, second)
        self.assertEqual(before, sha256_file(path))
        source = open_cfl(self.source, mode="r+")
        source[0, 3, 3, 0, 0] *= 2
        source.flush()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        write_maps(self.source, self.rss)
        output = open_cfl(self.target, mode="r+")
        output[0, 0, 0] = 3
        output.flush()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))

    def test_legacy_and_partial_outputs_are_not_overwritten(self):
        """A shape-compatible old map without the new attestation is not labeled fixed."""
        write_maps(self.target, np.ones((2, 8, 6), np.float32))
        before = sha256_file(self.target.with_suffix(".cfl"))
        with self.assertRaisesRegex(ValueError, "legacy/unbound"):
            resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        self.assertEqual(before, sha256_file(self.target.with_suffix(".cfl")))

    def test_invalid_geometry_nonfinite_and_multimap(self):
        """Reject invalid inputs without creating or overwriting the scientific output pair."""
        for shape in ((0, 6), (8., 6), (13, 6)):
            with self.assertRaises(ValueError):
                resample_sensitivity_maps(self.source, self.target, target_lin_par=shape)
        source = open_cfl(self.source, mode="r+")
        source[0, 0, 0, 0, 0] = np.nan
        source.flush()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))
        self.assertFalse(self.target.with_suffix(".cfl").exists())
        create_cfl(self.root/"multi", (2, 12, 10, 2, 2)).flush()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resample_sensitivity_maps(self.root/"multi", self.target, target_lin_par=(8, 6))

    def test_concurrent_writer_lock(self):
        """A second writer cannot race the completion record or map payload."""
        with (self.root/"low_resampling.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                resample_sensitivity_maps(self.source, self.target, target_lin_par=(8, 6))


class StandardIntegrationTests(unittest.TestCase):
    """Exercise actual standard MPRAGE and GRE preparation APIs with tiny native maps."""

    def test_mprage_gre_support_contract_and_resume(self):
        """Both APIs bind the same policy; source maps stay unchanged and GRE echoes share it."""
        for kind in ("mprage", "gre"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                native = root/"vcc2/normal/bart_output/coil_sens"
                gain = np.ones((2, 12, 10), np.float32)*.3
                gain[:, :3] = 0
                write_maps(native, gain)
                native_hash = sha256_file(native.with_suffix(".cfl"))
                case = "lr_y_1p5mm_r3x2" if kind == "mprage" else "lin_low_resolution_r3x2"
                key = "target_logical_matrix_ro_lin_par" if kind == "mprage" else "matrix_ro_lin_par"
                inputs = root/f"vcc2/retro/{case}/bart_inputs"
                inputs.mkdir(parents=True)
                manifest = inputs/"manifest.json"
                manifest.write_text(json.dumps({"case": {key: [2, 8, 10]}, "echoes": [{}, {}, {}]}))
                prepare = prepare_retro_sensitivity_maps if kind == "mprage" else prepare_retro_gre_sensitivity_maps
                prepare(root, virtual_coils=2)
                saved_hash = sha256_file(manifest)
                record = json.loads(manifest.read_text())
                self.assertEqual(record["coil_sens_resampling"]["request"]["policy"], CSM_RESAMPLING_POLICY)
                self.assertEqual(native_hash, sha256_file(native.with_suffix(".cfl")))
                prepare(root, virtual_coils=2)
                self.assertEqual(saved_hash, sha256_file(manifest))
                if kind == "gre":
                    self.assertTrue(record["coil_sens_shared_across_echoes"])
                    self.assertEqual(record["echoes"], [{}, {}, {}])


if __name__ == "__main__":
    unittest.main()
