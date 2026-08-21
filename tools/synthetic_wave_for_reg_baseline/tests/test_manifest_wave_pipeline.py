"""Tests for manifest propagation through Wave synthesis and BART export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_manifest import load_dataset_manifest  # noqa: E402
from export_bart_wave_inputs import _build_parser as export_parser  # noqa: E402
from export_bart_wave_inputs import run as export_bart_inputs  # noqa: E402
from synthesize_wave_kspace import _build_parser as synthesis_parser  # noqa: E402
from synthesize_wave_kspace import (  # noqa: E402
    completed_synthesis_reusable,
    resolve_wave_synthesis_inputs,
)
from wave_synthesis import logical_array_sha256, sha256_file  # noqa: E402


def _write_bart(base: Path, array: np.ndarray) -> None:
    """Write the minimal BART pair needed by the small integration fixture."""
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(value) for value in array.shape) + "\n",
        encoding="utf-8",
    )
    base.with_suffix(".cfl").write_bytes(
        np.asfortranarray(array, dtype=np.complex64).tobytes(order="F")
    )


def _write_fixture(root: Path) -> tuple[Path, object]:
    example = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "incoming_r1_dataset.example.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    payload["inputs"]["twix"] = "inputs/scan.dat"
    payload["inputs"]["wave_sequence"] = "inputs/wave.seq"
    payload["inputs"]["dicom"]["directory"] = "inputs/dicom"
    payload["outputs"]["root"] = "outputs"
    payload["geometry"]["matrix"] = [4, 6, 5]
    payload["geometry"]["fov_mm"] = [4.0, 6.0, 5.0]
    payload["sampling"]["synthetic_wave_acs_pe1_start"] = 2
    payload["sampling"]["synthetic_wave_acs_pe1_stop_exclusive"] = 4
    payload["reconstruction"]["physical_coils"] = 2
    payload["reconstruction"]["virtual_coils"] = 2
    payload["wave_synthesis"].update(
        {
            "extended_readout_samples": 8,
            "calibration_ncalib1": 4,
            "calibration_nacs": 2,
            "diagnostic_coils": [1, 2],
        }
    )
    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = load_dataset_manifest(manifest_path)

    for path in (dataset.input_path("twix"), dataset.input_path("wave_sequence")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    dataset.dicom_directory.mkdir(parents=True)
    dataset.inspection_report.parent.mkdir(parents=True)
    dataset.inspection_report.write_text(
        json.dumps(
            {
                "dataset_manifest": {"sha256": dataset.sha256},
                "contract_checks": {"all_passed": True},
                "twix": {"selected_measurement_index": 0},
            }
        ),
        encoding="utf-8",
    )

    source_prefix = dataset.output_path("source_reconstruction_prefix")
    source_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = source_prefix.with_name(source_prefix.name + "_full_ncc2.npy")
    np.save(source_path, np.ones((4, 6, 5, 2), dtype=np.complex64))
    source_prefix.with_name(source_prefix.name + "_report.json").write_text(
        json.dumps(
            {
                "dataset_manifest": {"sha256": dataset.sha256},
                "assembly": {
                    "output": str(source_path),
                    "shape": [4, 6, 5, 2],
                    "finite": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, dataset


class ManifestWaveSynthesisTests(unittest.TestCase):
    def test_resolves_all_dataset_and_wave_settings_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, dataset = _write_fixture(Path(temporary))
            args = synthesis_parser().parse_args(["--dataset-manifest", str(path)])

            inputs = resolve_wave_synthesis_inputs(args)

            self.assertEqual(inputs.matrix_rolinpar, (4, 6, 5))
            self.assertEqual(inputs.nx_extended, 8)
            self.assertEqual(inputs.virtual_coils, 2)
            self.assertEqual(inputs.output_dir, dataset.output_path("wave_synthesis_dir"))
            self.assertEqual(inputs.measurement_index, 0)

    def test_complete_reuse_requires_current_source_report_and_psf_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, dataset = _write_fixture(Path(temporary))
            args = synthesis_parser().parse_args(["--dataset-manifest", str(path)])
            inputs = resolve_wave_synthesis_inputs(args)
            inputs.output_dir.mkdir(parents=True)
            full_wave = np.ones((8, 6, 5, 2), dtype=np.complex64)
            psf = np.ones((8, 6, 5), dtype=np.complex64)
            full_wave_path = inputs.output_dir / "full_wave_kspace.npy"
            psf_path = inputs.output_dir / "theoretical_psf.npy"
            np.save(full_wave_path, full_wave)
            np.save(psf_path, psf)
            bart_psf_base = inputs.output_dir / "bart_inputs" / "psf"
            _write_bart(bart_psf_base, psf[..., None, None])
            manifest_path = inputs.output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "awaiting_visual_review_before_mask_and_bart",
                        "dataset_manifest": {"sha256": dataset.sha256},
                        "source_reconstruction_report": {
                            "sha256": sha256_file(inputs.source_report)
                        },
                        "full_wave_kspace": {
                            "path": str(full_wave_path),
                            "all_samples_finite": True,
                            "norm": float(np.linalg.norm(full_wave)),
                        },
                        "psf": {
                            "npy": str(psf_path),
                            "bart_base": str(bart_psf_base),
                            "bart_shape": [8, 6, 5, 1, 1],
                            "logical_sha256": logical_array_sha256(psf),
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(completed_synthesis_reusable(manifest_path, inputs))
            psf[0, 0, 0] = 2
            np.save(psf_path, psf)
            self.assertFalse(completed_synthesis_reusable(manifest_path, inputs))


class ManifestBartExportTests(unittest.TestCase):
    def test_applies_target_mask_after_wave_and_exports_separate_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, dataset = _write_fixture(Path(temporary))
            synthesis_dir = dataset.output_path("wave_synthesis_dir")
            synthesis_dir.mkdir(parents=True)
            rng = np.random.default_rng(73)
            full_wave = (
                rng.standard_normal((8, 6, 5, 2))
                + 1j * rng.standard_normal((8, 6, 5, 2))
            ).astype(np.complex64)
            full_wave_path = synthesis_dir / "full_wave_kspace.npy"
            np.save(full_wave_path, full_wave)
            psf = np.ones((8, 6, 5, 1, 1), dtype=np.complex64)
            psf_base = synthesis_dir / "bart_inputs" / "psf"
            _write_bart(psf_base, psf)
            synthesis_manifest_path = synthesis_dir / "manifest.json"
            synthesis_manifest_path.write_text(
                json.dumps(
                    {
                        "status": "awaiting_visual_review_before_mask_and_bart",
                        "dataset_manifest": {"sha256": dataset.sha256},
                        "full_wave_kspace": {
                            "path": str(full_wave_path),
                            "shape": [8, 6, 5, 2],
                            "dtype": "complex64",
                            "sampling_mask_applied": False,
                        },
                        "psf": {
                            "bart_base": str(psf_base),
                            "bart_shape": [8, 6, 5, 1, 1],
                            "logical_sha256": logical_array_sha256(psf),
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_manifest_hash = sha256_file(synthesis_manifest_path)
            args = export_parser().parse_args(
                [
                    "--dataset-manifest",
                    str(path),
                    "--visual-review-approved",
                    "--pe2-chunk",
                    "2",
                ]
            )

            result = export_bart_inputs(args)

            output_dir = dataset.output_path("bart_export_dir")
            mask = np.load(output_dir / "sampling_mask.npy")
            self.assertTrue(np.all(mask[2:4, :]))
            outside_acs = np.array([0, 1, 4, 5])
            expected = (outside_acs[:, None] % 3 == 1) & (
                np.arange(5)[None, :] % 2 == 0
            )
            np.testing.assert_array_equal(mask[outside_acs, :], expected)
            exported = np.memmap(
                output_dir / "bart_inputs" / "wave_kspace.cfl",
                mode="r",
                dtype=np.complex64,
                shape=(8, 6, 5, 2, 1),
                order="F",
            )[..., 0]
            np.testing.assert_array_equal(exported[:, mask, :], full_wave[:, mask, :])
            self.assertFalse(np.any(exported[:, ~mask, :]))
            self.assertTrue(result["sampling_mask"]["applied_after_wave_encoding"])
            self.assertEqual(sha256_file(synthesis_manifest_path), source_manifest_hash)

            resumed_args = export_parser().parse_args(
                [
                    "--dataset-manifest",
                    str(path),
                    "--visual-review-approved",
                    "--resume",
                ]
            )
            resumed = export_bart_inputs(resumed_args)
            self.assertEqual(resumed["status"], "manifest_bart_inputs_ready")


if __name__ == "__main__":
    unittest.main()
