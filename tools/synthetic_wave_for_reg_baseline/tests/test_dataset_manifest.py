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
        "outputs": {"root": "outputs", "inspection_report": "metadata/report.json"},
        "geometry": {
            "logical_axes": ["readout", "phase_encode_1", "phase_encode_2"],
            "matrix": [256, 240, 192],
            "fov_mm": [256.0, 240.0, 192.0],
        },
        "sampling": {
            "source_acceleration_pe1_pe2": [1, 1],
            "synthetic_wave_acceleration_pe1_pe2": [3, 2],
            "readout_oversampling_factor": 2.0,
            "require_complete_source_grid": True,
            "expected_acs_pe1_pe2": [24, 192],
        },
        "reconstruction": {
            "physical_coils": 32,
            "virtual_coils": 12,
            "grappa": {"kernel": [5, 5, 5], "regularization": 0.01},
            "bart": {"use_gpu": True, "maximum_eigenvalue": None},
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
        payload["evaluation"]["brain_mask"]["usage"] = "reconstruction"

        with self.assertRaises(DatasetManifestError) as context:
            validate_dataset_manifest(payload)

        self.assertIn("use_gpu must be true", str(context.exception))
        self.assertIn("must be 'metrics_only'", str(context.exception))

    def test_dicom_reference_and_enable_switch_must_agree(self) -> None:
        payload = valid_payload()
        payload["evaluation"]["ranking_reference"] = {"kind": "dicom"}

        with self.assertRaisesRegex(DatasetManifestError, "true exactly when"):
            validate_dataset_manifest(payload)

        payload["evaluation"]["dicom_intensity_ranking_enabled"] = True
        validate_dataset_manifest(payload)

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


if __name__ == "__main__":
    unittest.main()
