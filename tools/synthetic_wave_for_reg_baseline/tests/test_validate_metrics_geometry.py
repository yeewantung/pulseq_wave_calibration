"""Focused tests for the exact-grid metrics geometry gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from bart_cfl import sha256_file  # noqa: E402
from validate_metrics_geometry import _require_same_geometry, run  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class MetricsGeometryGateTests(unittest.TestCase):
    def _fixture(self, root: Path, *, gpu: bool = True) -> argparse.Namespace:
        affine = np.eye(4)
        reference_path = root / "reference.nii.gz"
        mask_path = root / "mask.nii.gz"
        case_nifti = root / "sweep" / "case" / "magnitude.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), affine),
            reference_path,
        )
        mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
        mask_data[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(mask_data, affine), mask_path)
        case_nifti.parent.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(np.full((4, 4, 4), 2, dtype=np.float32), affine),
            case_nifti,
        )

        dataset = root / "dataset.json"
        source_kspace = root / "source.npy"
        dataset.write_text("{}\n", encoding="utf-8")
        source_kspace.write_bytes(b"fully sampled test k-space")

        mask_manifest = root / "mask_manifest.json"
        _write_json(
            mask_manifest,
            {
                "status": "approved_for_metrics",
                "approval": {
                    "mask_boundary_visually_approved": True,
                    "left_right_orientation_visually_approved": True,
                },
            },
        )
        sidecar = case_nifti.with_suffix("").with_suffix(".json")
        _write_json(sidecar, {"kind": "magnitude"})

        dataset_hash = sha256_file(dataset)
        case_manifest = case_nifti.parent / "manifest.json"
        _write_json(
            case_manifest,
            {
                "status": "complete",
                "config": {
                    "regularizer": "wavelet",
                    "lambda": 0.0,
                    "lambda_label": "0",
                    "block_size": None,
                    "backend": "gpu" if gpu else "cpu",
                    "matrix_rolinpar": [4, 4, 4],
                    "dataset_manifest_sha256": dataset_hash,
                    "lambda_zero_manifest_sha256": "lambda-zero-hash",
                    "bart_input_manifest_sha256": "bart-input-hash",
                },
                "effective_bart_command": ["bart", "wave", "-g"] if gpu else ["bart", "wave"],
                "source_provenance": {
                    "dataset_manifest": {"sha256": dataset_hash}
                },
                "maps": {"cfl_sha256": "maps-hash"},
                "nifti_outputs": [
                    {
                        "part": "mag",
                        "nifti": str(case_nifti),
                        "nifti_sha256": sha256_file(case_nifti),
                        "json": str(sidecar),
                        "json_sha256": sha256_file(sidecar),
                    }
                ],
            },
        )

        metrics = root / "metrics_reference.json"
        _write_json(
            metrics,
            {
                "status": "approved_for_metrics",
                "dataset": {
                    "manifest": str(dataset),
                    "manifest_sha256": dataset_hash,
                },
                "ranking_reference": {
                    "kind": "direct_fft_rss",
                    "path": str(reference_path),
                    "sha256": sha256_file(reference_path),
                    "source_kspace": str(source_kspace),
                    "source_kspace_sha256": sha256_file(source_kspace),
                },
                "brain_mask": {
                    "usage": "metrics_only",
                    "path": str(mask_path),
                    "sha256": sha256_file(mask_path),
                    "manifest": str(mask_manifest),
                    "manifest_sha256": sha256_file(mask_manifest),
                    "voxel_count": 8,
                },
            },
        )
        return argparse.Namespace(
            metrics_reference_manifest=metrics,
            sweep_root=[root / "sweep"],
            expected_case=["wavelet:0"],
            output=root / "geometry" / "report.json",
        )

    def test_passes_exact_grid_with_approved_mask_and_gpu_case(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            args = self._fixture(Path(temporary))
            report = run(args)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["cases"][0]["case_id"], "wavelet:0")
            self.assertFalse(report["geometry_policy"]["registration_performed"])
            self.assertTrue(args.output.is_file())

    def test_rejects_case_without_gpu_flag(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            args = self._fixture(Path(temporary), gpu=False)
            with self.assertRaisesRegex(ValueError, "GPU BART -g"):
                run(args)

    def test_rejects_affine_difference(self) -> None:
        reference = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
        shifted_affine = np.eye(4)
        shifted_affine[0, 3] = 0.01
        candidate = nib.Nifti1Image(np.ones((3, 3, 3)), shifted_affine)
        with self.assertRaisesRegex(ValueError, "affine differs"):
            _require_same_geometry(candidate, reference, "candidate")


if __name__ == "__main__":
    unittest.main()
