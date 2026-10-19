"""Automatic metal support, raw bootstrap, calibration-only reuse and exact collection tests."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import fft

from test_metal_csm import synthetic_maps, write_array
from test_metal_pipeline import arguments, pipeline_fixture, fake_cpu, fake_convert
from wave_retro_lr.bart_io import cfl_record, open_cfl, sha256_file
from wave_retro_lr.metal_auto_csm import AUTO_POLICY, automatic_vectors, calibration_rss
from wave_retro_lr.metal_auto_pipeline import automatic_source, select_psf_donor
from wave_retro_lr.metal_pipeline import run_pipeline
from wave_retro_lr.metal_workflow import file_record, read_json, write_json
from wave_retro_lr.mprage import _file_identity
from wave_retro_lr.metal_psf_reuse import PSF_NAMES, accepted_psf, psf_tuple
from wave_retro_lr.nifti_collection import build_mprage_nifti_collection


def noop() -> None:
    """Allow tiny fixture computations without checking production host memory."""


def raw_files(root: Path) -> None:
    """Create fake source identity files; no real scanner data is read by these tests."""
    (root/"raw.dat").write_bytes(b"synthetic raw")
    (root/"scan.seq").write_bytes(b"synthetic sequence")


def automatic_fixture(root: Path, *, keep_original_maps: bool) -> None:
    """Create native/case fixtures, remove manual config and optionally remove all baseline products."""
    config = pipeline_fixture(root)
    config.unlink()
    (root/"vcc2/normal/bart_inputs/sampling_class.txt").write_text("R3x1\n")
    # These temporary fixture images are intentionally absent: auto mode must
    # neither require them nor synthesize a fake completed normal reconstruction.
    shutil.rmtree(root/"vcc2/normal/nifti")
    for path in (root/"vcc2/normal/bart_output").glob("*/image_wave.*"):
        path.unlink()
    if not keep_original_maps:
        shutil.rmtree(root/"vcc2/normal/bart_output")


def fake_ecalib(output, preparation, threads=8, *, operation="wave", gpu=False):
    """Emulate only isolated calibration; no production BART process or device is used."""
    if operation != "ecalib" or gpu:
        raise AssertionError("Expected CPU-only ecalib")
    state = output/"metadata/reconstruction.json"
    if state.exists():
        return read_json(state)
    base = output/"reconstruction/coil_sens"
    write_array(base, synthetic_maps())
    command = ["bart", "ecalib", "-m", "1", "-c", str(preparation["request"]["ecalib_crop"]),
               preparation["request"]["source"]["records"]["calibration"]["base"], str(base)]
    result = {"status": "complete", "output": cfl_record(base), "logs": [],
              "identity": {"gpu": False, "command": command,
                           "preparation_sha256": sha256_file(output/"metadata/preparation.json")}}
    write_json(state, result)
    return result


@contextmanager
def mocked_auto(root: Path, *, fresh=False):
    """Mock raw import, calibration and reconstruction while retaining real orchestration/hash checks."""
    def prepare(*args, **kwargs):
        """Materialize tiny inputs on the first fresh call and otherwise return their manifest."""
        path = root/"vcc2/normal/bart_inputs/manifest.json"
        if not path.exists():
            if not fresh:
                raise AssertionError("Unexpected fresh preparation")
            automatic_fixture(root, keep_original_maps=False)
        return read_json(path)

    with patch("wave_retro_lr.metal_pipeline.memory_guard"), patch("wave_retro_lr.metal_auto_pipeline.memory_guard"), \
         patch("wave_retro_lr.mprage.prepare_normal_mprage", side_effect=prepare) as importer, \
         patch("wave_retro_lr.mprage.prepare_retro_mprage"), patch("wave_retro_lr.mprage.prepare_retro_mprage_r3x3"), \
         patch("wave_retro_lr.metal_auto_pipeline.run_reconstruction", side_effect=fake_ecalib) as calibration, \
         patch("wave_retro_lr.metal_pipeline.run_reconstruction", side_effect=fake_cpu) as wave, \
         patch("scripts.convert_mprage_bart_to_nifti.convert", side_effect=fake_convert):
        yield importer, calibration, wave


class AutomaticNumericsTests(unittest.TestCase):
    """Test local recovery of known synthetic maps and conservative background/no-op behavior."""

    def test_signal_gate_gauge_invariance_and_known_hole(self):
        """Recover a known coil direction in small defects without changing native vectors or background."""
        maps = synthetic_maps()
        rss = np.zeros(maps.shape[:3], np.float32)
        rss[2:-2, 2:-2, 2:-2] = 1
        native, added, vectors, stats, _ = automatic_vectors(maps, rss, (1., 1., 1.), noop)
        self.assertGreater(added.sum(), 0)
        self.assertFalse(np.any(added & native))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        expected = np.array([1, np.exp(1j)], np.complex64)/np.sqrt(2)
        self.assertTrue(np.all(abs(vectors @ expected.conj()) > .999))
        phase = np.exp(1j*np.random.default_rng(12).uniform(-np.pi, np.pi, rss.shape)).astype(np.complex64)
        _, shifted, vectors_shifted, _, _ = automatic_vectors(maps*phase[..., None], rss, (1., 1., 1.), noop)
        np.testing.assert_array_equal(added, shifted)
        self.assertTrue(np.all(abs(np.sum(vectors.conj()*vectors_shifted, axis=1)) > .999))
        self.assertLessEqual(stats["maximum_added_distance_mm"], AUTO_POLICY["maximum_distance_mm"])
        rss[~native] = 0
        _, no_signal, _, stats, _ = automatic_vectors(maps, rss, (1., 1., 1.), noop)
        self.assertFalse(no_signal.any())
        self.assertEqual(stats["status"], "no_op")

    def test_anisotropy_noise_volume_gate_and_invalid_maps(self):
        """Use mm distances, refuse uncontrolled expansion and reject invalid scientific inputs."""
        maps = synthetic_maps()
        rss = np.zeros(maps.shape[:3], np.float32)
        rss[2:-2, 2:-2, 2:-2] = 1
        _, _, _, stats, _ = automatic_vectors(maps, rss, (2., 1.5, 3.), noop)
        self.assertLessEqual(stats["maximum_added_distance_mm"], 8.)
        with patch.dict(AUTO_POLICY, maximum_added_native_fraction=1e-8):
            _, added, _, stats, _ = automatic_vectors(maps, rss, (1.,)*3, noop)
            self.assertFalse(added.any())
            self.assertEqual(stats["reason"], "automatic_safety_volume_cap")
        for broken in (maps*.5, np.full_like(maps, np.nan)):
            with self.assertRaises(ValueError):
                automatic_vectors(broken, rss, (1.,)*3, noop)
        with self.assertRaises(ValueError):
            automatic_vectors(maps, rss, (0., 1., 1.), noop)

    def test_unmasked_calibration_rss_is_not_csm_cropped(self):
        """Recover known RSS from complex ACS images, including a region outside hard support."""
        maps = synthetic_maps()
        images = np.ones_like(maps)
        calibration = fft.fftshift(fft.fftn(fft.ifftshift(images, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho"), axes=(0, 1, 2))
        rss = calibration_rss(calibration, noop)
        np.testing.assert_allclose(rss, np.sqrt(2), atol=1e-6)


class AutomaticPipelineTests(unittest.TestCase):
    """Verify no-config fresh/reuse paths, optional GPU, isolation and collection using tiny fixtures."""

    def test_fresh_normal_retro_collection_without_baseline_or_config(self):
        """Bootstrap inputs and CSM, run paired exports, and resume without manual reference or baseline image."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            with mocked_auto(root, fresh=True) as (importer, calibration, wave):
                destination = run_pipeline(arguments(root, "normal", "-g"))
                self.assertIsNone(importer.call_args.kwargs["accepted_psf_inputs"])
                run_pipeline(arguments(root, "retro", "-g"))
                run_pipeline(arguments(root, "r3x3", "-g"))
                self.assertFalse((root/"metal_csm.local.json").exists())
                self.assertFalse((root/"vcc2/normal/bart_output").exists())
                self.assertTrue(calibration.called)
                native = read_json(destination/"shared_csm/metadata/preparation.json")
                self.assertEqual(native["statistics"]["status"], "no_op")
                np.testing.assert_array_equal(open_cfl(native["candidate"]["base"]), synthetic_maps())
                result = build_mprage_nifti_collection(root, require_retro=True)
                self.assertEqual(len(result["cases"]), 4)
                for case in result["cases"]:
                    self.assertTrue(all(f["collection_source_selection"] == "metal_csm" for f in case["files"]))
                before = {str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file()}
                run_pipeline(arguments(root, "normal", "-g"))
                self.assertEqual(before, {str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file()})
                with self.assertRaisesRegex(ValueError, "request mismatch"):
                    run_pipeline(arguments(root, "normal", "--ecalib-crop", ".5"))

    def test_existing_inputs_and_maps_no_reference_needed(self):
        """Reuse matched baseline CSM even when all baseline images/NIfTIs have been pruned."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            automatic_fixture(root, keep_original_maps=True)
            original = {str(p): sha256_file(p) for p in (root/"vcc2").rglob("*") if p.is_file()}
            with mocked_auto(root) as (_, calibration, wave):
                destination = run_pipeline(arguments(root, "normal", "--metal-prepare-only"))
                calibration.assert_not_called()
                wave.assert_not_called()
                run_pipeline(arguments(root))
                self.assertFalse((destination/"source_reference").exists())
            self.assertEqual(original, {str(p): sha256_file(p) for p in (root/"vcc2").rglob("*") if p.is_file()})

    def test_disabled_missing_explicit_config_and_unbound_csm_rejected(self):
        """Reject missing explicit config, absent opt-in and a CSM with no completed hash binding."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            args = arguments(root)
            args.metal_csm_extrapolation = False
            with self.assertRaisesRegex(ValueError, "disabled"):
                run_pipeline(args)
            with patch("wave_retro_lr.metal_pipeline.memory_guard"), self.assertRaises(FileNotFoundError):
                run_pipeline(arguments(root, "normal", "--metal-csm-config", str(root/"missing.local.json")))
            automatic_fixture(root, keep_original_maps=True)
            for path in (root/"vcc2/normal/bart_output").glob("*/reconstruction_manifest.json"):
                path.unlink()
            with mocked_auto(root), self.assertRaisesRegex(ValueError, "no completed"):
                run_pipeline(arguments(root))

    def test_prepare_only_never_launches_missing_ecalib(self):
        """A prepare-only request may prepare inputs but must not implicitly launch missing calibration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            with mocked_auto(root, fresh=True) as (_, calibration, wave):
                with self.assertRaisesRegex(ValueError, "needs existing"):
                    run_pipeline(arguments(root, "normal", "--metal-prepare-only"))
                calibration.assert_not_called()
                wave.assert_not_called()


def donor_fixture(root: Path) -> Path:
    """Create a tiny ROVir source anchor with all accepted coil-independent PSF artifacts."""
    donor = root/"normal/rovir/source_inputs"
    donor.mkdir(parents=True)
    write_array(donor/"psf", np.ones((8, 6, 4, 1, 1), np.complex64))
    write_array(donor/"wave_trajectory", np.ones((8, 2), np.complex64))
    for name in PSF_NAMES[2:]:
        write_array(donor/name, np.ones((8, 1 if name.endswith("turns") else 3), np.complex64))
    payload = {"status": "mprage_rovir_source_anchor_ready", "source": {
        "twix": _file_identity(root/"raw.dat"), "sequence": _file_identity(root/"scan.seq", include_hash=True)},
        "geometry": {"logical_matrix_ro_lin_par": [4, 6, 4], "physical_fov_mm_xyz": [4., 6., 4.], "readout_oversampling_factor": 2},
        "psf_calibration": {"coefficient_processing": "sine-line", "requested_fit_z_range": [2, 5]},
        "output_artifacts": {name: cfl_record(donor/name) for name in PSF_NAMES}}
    write_json(donor/"manifest.json", payload)
    return donor


class AutomaticPSFReuseTests(unittest.TestCase):
    """Verify ROVir anchor PSF reuse without reinterpreting its coil transform."""

    def test_rovir_donor_exact_vectors_and_source_mismatch(self):
        """Reuse only hash-bound artifacts; reject conflicting bounds, sources and payload tampering."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            donor = donor_fixture(root)
            selected = select_psf_donor(root, arguments(root))
            self.assertEqual(selected, donor)
            record = accepted_psf(donor, root/"raw.dat", root/"scan.seq", {})
            values = psf_tuple(record)
            self.assertEqual(len(values), 10)
            np.testing.assert_array_equal(values[0], np.ones(8))
            with self.assertRaisesRegex(ValueError, "bounds"):
                accepted_psf(donor, root/"raw.dat", root/"scan.seq", {"requested_fit_z_range": [1, 5]})
            (root/"raw.dat").write_bytes(b"other source")
            with self.assertRaisesRegex(ValueError, "different raw"):
                select_psf_donor(root, arguments(root))

    def test_corrupt_donor_does_not_trigger_fresh_fit(self):
        """Refuse an advertised donor with modified PSF rather than silently fitting a new model."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_files(root)
            donor = donor_fixture(root)
            values = open_cfl(donor/"psf", mode="r+")
            values.flat[0] = 2
            values.flush()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                select_psf_donor(root, arguments(root))


if __name__ == "__main__":
    unittest.main()
