"""Tests for the portable dataset-manifest contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_manifest import (  # noqa: E402
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
    validate_dataset_manifest,
)


def valid_payload() -> dict:
    return {
        "format_version": 1,
        "dataset_id": "test-r1",
        "subject": "test-r1",
        "inputs": {
            "twix": "inputs/scan.dat",
            "wave_sequence": "inputs/wave.seq",
            "dicom": {
                "directory": "inputs/dicom",
                "required_image_type_tokens": ["ND", "NORM"],
                "excluded_image_type_tokens": ["DIS2D", "DIS3D"],
            },
        },
        "outputs": {
            "root": "outputs",
            "inspection_report": "metadata/report.json",
            "coil_compression_prefix": "calibration/coil_compression",
            "source_reconstruction_prefix": "reconstructions/no_wave/source",
            "wave_synthesis_dir": "synthetic_wave/full_encoding",
            "bart_export_dir": "synthetic_wave/target_sampling",
            "lambda0_reconstruction_dir": "reconstructions/synthetic_wave/lambda0",
        },
        "geometry": {
            "logical_axes": ["readout", "phase_encode_1", "phase_encode_2"],
            "matrix": [256, 240, 192],
            "fov_mm": [256.0, 240.0, 192.0],
        },
        "sampling": {
            "source_acceleration_pe1_pe2": [1, 1],
            "synthetic_wave_acceleration_pe1_pe2": [3, 2],
            "synthetic_wave_residue_pe1_pe2": [1, 0],
            "synthetic_wave_mask_kind": "cartesian_with_full_pe1_acs",
            "synthetic_wave_acs_pe1_start": 115,
            "synthetic_wave_acs_pe1_stop_exclusive": 139,
            "readout_oversampling_factor": 2.0,
            "require_complete_source_grid": True,
            "expected_acs_pe1_pe2": [24, 192],
        },
        "reconstruction": {
            "physical_coils": 32,
            "virtual_coils": 12,
            "coil_compression_source": "image",
            "grappa": {"kernel": [5, 5, 5], "regularization": 0.01},
            "bart": {
                "use_gpu": True,
                "calibration_source": "image",
                "maximum_eigenvalue": None,
                "ecalib_crop": 0.8,
                "ecalib_intensity_correction": False,
                "lambda0_iterations": 300,
                "lambda0_tolerance": 1e-3,
            },
        },
        "wave_synthesis": {
            "extended_readout_samples": 1024,
            "calibration_ncalib1": 72,
            "calibration_nacs": 32,
            "orientation": "SAG",
            "pe1_phase_sign": -1,
            "pe2_phase_sign": -1,
            "diagnostic_coils": [1, 2, 3, 4],
        },
        "evaluation": {
            "ranking_reference": {"kind": "grappa", "path": "references/grappa.nii.gz"},
            "dicom_intensity_ranking_enabled": False,
            "brain_mask": {"usage": "metrics_only", "path": None},
        },
    }


class DatasetManifestTests(unittest.TestCase):
    def test_resolves_inputs_from_manifest_and_report_from_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "config" / "dataset.json"
            manifest_path.parent.mkdir()
            manifest_path.write_text(json.dumps(valid_payload()), encoding="utf-8")

            manifest = load_dataset_manifest(manifest_path)

            self.assertEqual(manifest.input_path("twix"), root / "config" / "inputs" / "scan.dat")
            self.assertEqual(manifest.output_root, root / "config" / "outputs")
            self.assertEqual(
                manifest.inspection_report,
                root / "config" / "outputs" / "metadata" / "report.json",
            )
            self.assertEqual(len(manifest.sha256), 64)

    def test_rejects_cpu_bart_and_nonmetric_mask_use(self) -> None:
        payload = valid_payload()
        payload["reconstruction"]["bart"]["use_gpu"] = False
        payload["reconstruction"]["bart"]["calibration_source"] = "automatic"
        payload["evaluation"]["brain_mask"]["usage"] = "reconstruction"

        with self.assertRaises(DatasetManifestError) as context:
            validate_dataset_manifest(payload)

        self.assertIn("use_gpu must be true", str(context.exception))
        self.assertIn("calibration_source", str(context.exception))
        self.assertIn("must be 'metrics_only'", str(context.exception))

    def test_dicom_reference_and_enable_switch_must_agree(self) -> None:
        payload = valid_payload()
        payload["evaluation"]["ranking_reference"] = {"kind": "dicom"}

        with self.assertRaisesRegex(DatasetManifestError, "true exactly when"):
            validate_dataset_manifest(payload)

        payload["evaluation"]["dicom_intensity_ranking_enabled"] = True
        validate_dataset_manifest(payload)

    def test_disabled_dicom_and_no_ranking_reference_are_valid(self) -> None:
        payload = valid_payload()
        payload["inputs"]["dicom"] = {
            "enabled": False,
            "directory": None,
            "required_image_type_tokens": [],
            "excluded_image_type_tokens": [],
        }
        payload["reconstruction"]["grappa"] = None
        payload["evaluation"]["ranking_reference"] = {
            "kind": "none",
            "path": None,
        }

        validate_dataset_manifest(payload)

    def test_disabled_dicom_rejects_dicom_ranking_and_tokens(self) -> None:
        payload = valid_payload()
        payload["inputs"]["dicom"]["enabled"] = False
        payload["inputs"]["dicom"]["directory"] = None
        payload["evaluation"]["ranking_reference"] = {"kind": "dicom"}
        payload["evaluation"]["dicom_intensity_ranking_enabled"] = True

        with self.assertRaises(DatasetManifestError) as context:
            validate_dataset_manifest(payload)

        self.assertIn("must be empty when DICOM is disabled", str(context.exception))
        self.assertIn("DICOM ranking cannot be selected", str(context.exception))

    def test_rejects_even_grappa_kernel_and_fixed_nonpositive_eigenvalue(self) -> None:
        payload = copy.deepcopy(valid_payload())
        payload["reconstruction"]["grappa"]["kernel"] = [5, 4, 5]
        payload["reconstruction"]["bart"]["maximum_eigenvalue"] = 0

        with self.assertRaises(DatasetManifestError) as context:
            validate_dataset_manifest(payload)

        self.assertIn("kernel values must be odd", str(context.exception))
        self.assertIn("maximum_eigenvalue must be a positive number", str(context.exception))

    def test_inspection_report_cannot_escape_output_root(self) -> None:
        payload = valid_payload()
        payload["outputs"]["inspection_report"] = "../report.json"

        with self.assertRaisesRegex(DatasetManifestError, "contained by outputs.root"):
            validate_dataset_manifest(payload)

    def test_rejects_invalid_target_mask_and_wave_embedding(self) -> None:
        payload = valid_payload()
        payload["sampling"]["synthetic_wave_residue_pe1_pe2"] = [3, 0]
        payload["sampling"]["synthetic_wave_acs_pe1_stop_exclusive"] = 241
        payload["wave_synthesis"]["extended_readout_samples"] = 1023

        with self.assertRaises(DatasetManifestError) as context:
            validate_dataset_manifest(payload)

        message = str(context.exception)
        self.assertIn("residue 0 must be below", message)
        self.assertIn("ACS PE1 bounds", message)
        self.assertIn("must center-embed", message)

    def test_downstream_gate_requires_matching_passed_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "dataset.json"
            manifest_path.write_text(json.dumps(valid_payload()), encoding="utf-8")
            manifest = load_dataset_manifest(manifest_path)
            manifest.inspection_report.parent.mkdir(parents=True)
            manifest.inspection_report.write_text(
                json.dumps(
                    {
                        "dataset_manifest": {"sha256": manifest.sha256},
                        "contract_checks": {"all_passed": True},
                    }
                ),
                encoding="utf-8",
            )

            report = load_passed_inspection(manifest)

            self.assertTrue(report["contract_checks"]["all_passed"])
            report["dataset_manifest"]["sha256"] = "stale"
            manifest.inspection_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(DatasetManifestError, "current manifest"):
                load_passed_inspection(manifest)


if __name__ == "__main__":
    unittest.main()
