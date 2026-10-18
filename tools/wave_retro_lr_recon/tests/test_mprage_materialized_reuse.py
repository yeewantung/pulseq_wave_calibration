"""Exercise custom-PSF resume across software changes using small real CFLs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wave_retro_lr import mprage
from wave_retro_lr.bart_io import create_cfl, open_cfl, sha256_file
from wave_retro_lr.sampling import SamplingPattern
from wave_retro_lr.standard import prepared_artifact_records, write_pca_basis


class MaterializedReuseTests(unittest.TestCase):
    """Validate frozen normal artifacts through the public preparation APIs."""

    def setUp(self) -> None:
        """Create temporary source files and a hash-bound accepted custom PSF."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.twix = self.root / "input.dat"
        self.sequence = self.root / "input.seq"
        self.twix.write_bytes(b"synthetic TWIX identity only")
        self.sequence.write_bytes(b"synthetic sequence identity only")
        self.output = self.root / "output"
        self.inputs = self.output / "vcc2/normal/bart_inputs"
        self.inputs.mkdir(parents=True)
        self.manifest_path = self.inputs / "manifest.json"
        self.request = dict(virtual_coils=2, psf_fit_z_min=1, psf_fit_z_max=3)
        pattern = SamplingPattern(
            name="R3x1", acceleration_lin_par=(3, 1), lin_residue=1,
            matrix_lin_par=(16, 16), acquired_lin=(1, 4, 7, 10, 13),
            acquired_par=tuple(range(16)), measurement_index=0,
        )
        self.names = ("wave_kspace", "kspace_calib", "psf",
                      "wave_trajectory", "psf_coefficients")
        for name, shape in zip(self.names, (
            (8, 16, 16, 2, 1), (4, 16, 16, 2), (8, 16, 16, 1, 1),
            (8, 2), (8, 3),
        )):
            array = create_cfl(self.inputs / name, shape)
            array[...] = 1 if name in {"kspace_calib", "psf"} else 0
            if name == "wave_kspace":
                array[:, pattern.acquired_lin, :, :, :] = 1
            array.flush()
            del array
        basis = write_pca_basis(
            self.inputs / "coil_compression_basis.npy", np.eye(2, dtype=np.complex64)
        )
        self.manifest = {
            "status": "measured_wave_mprage_bart_inputs_ready",
            "standard_pca_variant": "vcc2",
            "source": {
                "twix": mprage._file_identity(self.twix),
                "sequence": mprage._file_identity(self.sequence, include_hash=True),
            },
            "geometry": {
                "logical_matrix_ro_lin_par": [4, 16, 16],
                "physical_fov_mm_xyz": [16, 16, 4],
                "readout_oversampling_factor": 2,
            },
            "sampling": pattern.to_json(),
            "coil_compression": {
                "physical_coils": 2, "virtual_coils": 2,
                "retained_energy": 1.0, "basis": basis,
                "readout_oversampling_removal": {
                    **mprage.COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
                    "oversampling_factor": 2, "input_readout": 8, "output_readout": 4,
                },
            },
            "noise_prewhitening": {
                "request": {"enabled": False}, "applied": False,
            },
            "psf_calibration": {
                **mprage._normalize_psf_coefficient_settings("sine-line", None, None),
                **mprage._normalize_psf_spatial_settings(None, None, 1, 3),
                "fit_kx_range": [1, 7],
                "processing_implementation": {"files_sha256": {
                    path: "0" * 64 for path in mprage.LEGACY_PSF_IMPLEMENTATION_PATHS
                }},
            },
            "output_artifacts": prepared_artifact_records(
                self.inputs, self.names, ("coil_compression_basis.npy",)
            ),
        }
        self._save_manifest()
        # A resume must never attempt to import a reader or rerun calibration.
        reader = patch("wave_retro_lr.mprage.load_wave_mprage_helpers",
                       side_effect=AssertionError("resume attempted raw-data import"))
        reader.start()
        self.addCleanup(reader.stop)
        calibration = patch("wave_retro_lr.mprage._calibrated_psf_inputs",
                            side_effect=AssertionError("resume attempted PSF fitting"))
        calibration.start()
        self.addCleanup(calibration.stop)

    def _save_manifest(self) -> None:
        """Persist the current synthetic manifest for the next validation call."""
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _resume(self, **changes):
        """Run normal reuse with optional request changes and return its manifest."""
        return mprage.prepare_normal_mprage(
            self.twix, self.output, self.sequence, **{**self.request, **changes}
        )

    def test_normal_retro_and_r3x3_preserve_custom_artifacts(self) -> None:
        """Resume every MPRAGE stage twice without changing the frozen normal files."""
        before = {p.name: (sha256_file(p), p.stat().st_mtime_ns)
                  for p in self.inputs.iterdir()}
        for _ in range(2):
            self.assertEqual(self._resume(), self.manifest)
            mprage.prepare_retro_mprage(
                self.twix, self.output, self.sequence, **self.request
            )
            mprage.prepare_retro_mprage_r3x3(
                self.twix, self.output, self.sequence, **self.request
            )
        after = {p.name: (sha256_file(p), p.stat().st_mtime_ns)
                 for p in self.inputs.iterdir()}
        self.assertEqual(before, after)
        attestation = json.loads((
            self.inputs.parent / mprage.NORMAL_REUSE_ATTESTATION_NAME
        ).read_text())
        self.assertFalse(attestation["scientific_artifacts_modified"])
        self.assertEqual(attestation["recorded_psf_request"]["requested_fit_z_range"], [1, 3])
        self.assertNotEqual(attestation["recorded_processing_implementation"],
                            attestation["current_processing_implementation_not_applied"])

    def test_all_manual_axes_must_match(self) -> None:
        """Accept identical kx/y/z requests while rejecting changed or omitted axes."""
        self.manifest["psf_calibration"].update(
            mprage._normalize_psf_coefficient_settings("sine-line", 1, 7)
        )
        self.manifest["psf_calibration"].update(
            mprage._normalize_psf_spatial_settings(1, 3, 1, 3)
        )
        self._save_manifest()
        self.request.update(psf_fit_kx_min=1, psf_fit_kx_max=7,
                            psf_fit_y_min=1, psf_fit_y_max=3)
        self._resume()
        for axis in ("kx", "y", "z"):
            for changes in (
                {f"psf_fit_{axis}_max": 4},
                {f"psf_fit_{axis}_min": None, f"psf_fit_{axis}_max": None},
            ):
                with self.subTest(axis=axis, changes=changes):
                    for retrospective in (False, True):
                        with self.assertRaisesRegex(ValueError, "different sources"):
                            self._resume(**changes, allow_legacy_artifact_reuse=retrospective)

    def test_changed_mode_and_missing_contract_are_rejected(self) -> None:
        """Reject processing-mode changes and unknown custom-fit provenance."""
        with self.assertRaisesRegex(ValueError, "different sources"):
            self._resume(psf_coefficient_processing="smooth")
        original = dict(self.manifest["psf_calibration"])
        for key in ("requested_fit_z_range", "requested_fit_y_range",
                    "fit_spatial_range_convention", "fit_kx_range_convention"):
            with self.subTest(key=key):
                self.manifest["psf_calibration"] = dict(original)
                del self.manifest["psf_calibration"][key]
                self._save_manifest()
                with self.assertRaisesRegex(ValueError, "different sources"):
                    self._resume()

    def test_automatic_selected_range_is_not_a_manual_request(self) -> None:
        """Reject a manual kx request that merely matches a former automatic result."""
        with self.assertRaisesRegex(ValueError, "different sources"):
            self._resume(psf_fit_kx_min=1, psf_fit_kx_max=7)

    def test_normal_requires_explicit_match_but_retro_can_inherit(self) -> None:
        """Keep historical no-override inheritance restricted to retro callers."""
        with self.assertRaisesRegex(ValueError, "different sources"):
            self._resume(psf_fit_z_min=None, psf_fit_z_max=None)
        self.assertEqual(self._resume(
            psf_fit_z_min=None, psf_fit_z_max=None, allow_legacy_artifact_reuse=True
        ), self.manifest)

    def test_fully_recorded_automatic_normal_request_is_reusable(self) -> None:
        """Reuse an unchanged automatic request without invoking the new fitter."""
        self.manifest["psf_calibration"].update(
            mprage._normalize_psf_spatial_settings(None, None, None, None)
        )
        self._save_manifest()
        self.assertEqual(self._resume(psf_fit_z_min=None, psf_fit_z_max=None),
                         self.manifest)

    def test_changed_conventions_and_selection_version_are_rejected(self) -> None:
        """Require the original units and selection contract for custom requests."""
        original = dict(self.manifest["psf_calibration"])
        for key, value in (
            ("fit_spatial_range_convention", "inclusive"),
            ("fit_kx_range_convention", "inclusive"),
            ("spatial_region_selection_version", -1),
        ):
            with self.subTest(key=key):
                self.manifest["psf_calibration"] = {**original, key: value}
                self._save_manifest()
                with self.assertRaisesRegex(ValueError, "different sources"):
                    self._resume()

    def test_sources_prewhitening_and_coil_contract_remain_strict(self) -> None:
        """Reject changed source identities, whitening, and alias-prone calibration."""
        with self.assertRaisesRegex(ValueError, "different sources"):
            self._resume(noise_prewhitening=True)
        for path in (self.twix, self.sequence):
            with self.subTest(path=path.name):
                original = self.manifest["source"]["twix" if path == self.twix else "sequence"]
                path.write_bytes(b"changed input")
                with self.assertRaisesRegex(ValueError, "different sources"):
                    self._resume()
                self.manifest["source"]["twix" if path == self.twix else "sequence"] = (
                    mprage._file_identity(path, include_hash=path == self.sequence)
                )
                self.assertNotEqual(original, self.manifest["source"]["twix" if path == self.twix else "sequence"])
                self._save_manifest()
        self.manifest["coil_compression"]["readout_oversampling_removal"]["version"] = -1
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "different sources"):
            self._resume()

    def test_payload_corruption_is_rejected(self) -> None:
        """Reject modified core data even when sources and custom bounds match."""
        for name in self.names:
            with self.subTest(name=name):
                path = (self.inputs / name).with_suffix(".cfl")
                original = path.read_bytes()
                path.write_bytes(b"\x00\x00\xc0\x7f" + original[4:])
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    self._resume()
                path.write_bytes(original)

    def test_finite_and_geometry_checks_survive_valid_hashes(self) -> None:
        """Reject inconsistent geometry and hash-consistent non-finite PSF data."""
        self.manifest["geometry"]["logical_matrix_ro_lin_par"][1] = 12
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "inconsistent core geometry"):
            self._resume()
        self.manifest["geometry"]["logical_matrix_ro_lin_par"][1] = 16
        array = open_cfl(self.inputs / "psf", mode="r+")
        array.flat[0] = np.nan
        array.flush()
        del array
        self.manifest["output_artifacts"] = prepared_artifact_records(
            self.inputs, self.names, ("coil_compression_basis.npy",)
        )
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._resume()


if __name__ == "__main__":
    unittest.main()
