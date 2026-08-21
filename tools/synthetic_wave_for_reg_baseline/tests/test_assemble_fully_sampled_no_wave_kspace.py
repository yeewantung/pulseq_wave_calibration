"""Tests for direct, interpolation-free fully sampled source preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from assemble_fully_sampled_no_wave_kspace import (  # noqa: E402
    assemble_fully_sampled_kspace,
    resolve_fully_sampled_inputs,
    validate_image_stream,
)
from dataset_manifest import load_dataset_manifest  # noqa: E402


def _write_direct_manifest(root: Path) -> Path:
    example = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "incoming_r1_dataset.example.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    payload["inputs"]["twix"] = "inputs/scan.dat"
    payload["outputs"]["root"] = "outputs"
    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_dataset_manifest(manifest_path)
    manifest.inspection_report.parent.mkdir(parents=True)
    manifest.inspection_report.write_text(
        json.dumps(
            {
                "dataset_manifest": {"sha256": manifest.sha256},
                "contract_checks": {
                    "all_passed": True,
                    "checks": [
                        {"name": "complete_centered_readout", "passed": True}
                    ],
                },
                "twix": {
                    "selected_measurement_sampling": {
                        "image_unique_coordinate_count": 256 * 256,
                        "image_duplicate_coordinate_count": 0,
                        "image_inferred_pe1_stride": 1,
                        "out_of_range_coordinates": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class FullySampledContractTests(unittest.TestCase):
    def test_resolves_r1_source_paths_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_direct_manifest(root)

            inputs = resolve_fully_sampled_inputs(path)

            self.assertEqual(inputs.twix, root / "inputs" / "scan.dat")
            self.assertEqual(
                inputs.coil_basis,
                root / "outputs" / "calibration" / "coil_compression.npz",
            )
            self.assertEqual(inputs.matrix_rolinpar, (256, 256, 256))
            self.assertEqual(
                inputs.output_path,
                root
                / "outputs"
                / "reconstructions"
                / "no_wave"
                / "source_full_ncc12.npy",
            )

    def test_rejects_incomplete_or_duplicate_pe_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_direct_manifest(Path(temporary))
            manifest = load_dataset_manifest(path)
            report = json.loads(manifest.inspection_report.read_text())
            report["twix"]["selected_measurement_sampling"][
                "image_duplicate_coordinate_count"
            ] = 1
            manifest.inspection_report.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate-free"):
                resolve_fully_sampled_inputs(path)


class DirectAssemblyTests(unittest.TestCase):
    class FakeImage:
        def __init__(self, data: np.ndarray):
            self.data = data
            self.sqzSize = data.shape
            self.skipLin = 0
            self.skipPar = 0
            self.Lin = np.arange(data.shape[2])
            self.Par = np.arange(data.shape[3])

        def __getitem__(self, key):
            return self.data[key]

    def test_stream_validation_requires_zero_origin_and_complete_counters(self) -> None:
        image = self.FakeImage(np.zeros((4, 2, 3, 5), dtype=np.complex64))
        result = validate_image_stream(
            image, matrix_rolinpar=(4, 3, 5), physical_coils=2
        )
        self.assertEqual(result["mapvbvd_shape_ro_coil_pe1_pe2"], [4, 2, 3, 5])

        image.skipLin = 1
        with self.assertRaisesRegex(ValueError, "must start"):
            validate_image_stream(
                image, matrix_rolinpar=(4, 3, 5), physical_coils=2
            )

    def test_identity_basis_preserves_every_measured_sample_and_resumes(self) -> None:
        values = np.arange(4 * 2 * 3 * 5, dtype=np.float32).reshape(4, 2, 3, 5)
        raw = (values + 1j * values[::-1]).astype(np.complex64)
        image = self.FakeImage(raw)
        basis = np.eye(2, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "direct.npy"
            progress = root / "progress.json"

            report = assemble_fully_sampled_kspace(
                image,
                basis,
                output,
                progress,
                matrix_rolinpar=(4, 3, 5),
                pe2_chunk=2,
                resume=False,
                run_signature_sha256="signature",
            )

            expected = np.transpose(raw, (0, 2, 3, 1))
            np.testing.assert_array_equal(np.load(output), expected)
            self.assertFalse(report["grappa_applied"])
            self.assertEqual(report["interpolation"], "none")
            self.assertTrue(json.loads(progress.read_text())["complete"])

            resumed = assemble_fully_sampled_kspace(
                image,
                basis,
                output,
                progress,
                matrix_rolinpar=(4, 3, 5),
                pe2_chunk=2,
                resume=True,
                run_signature_sha256="signature",
            )
            self.assertEqual(resumed["input_energy_this_invocation"], 0.0)
            with self.assertRaisesRegex(ValueError, "different inputs"):
                assemble_fully_sampled_kspace(
                    image,
                    basis,
                    output,
                    progress,
                    matrix_rolinpar=(4, 3, 5),
                    pe2_chunk=2,
                    resume=True,
                    run_signature_sha256="stale",
                )


if __name__ == "__main__":
    unittest.main()
