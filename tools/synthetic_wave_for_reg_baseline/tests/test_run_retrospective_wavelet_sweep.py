"""Tests for the retrospective low-resolution Wavelet sweep."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
RETRO_TOOL_ROOT = Path(__file__).resolve().parents[2] / "wave_retro_lr_recon"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(RETRO_TOOL_ROOT))

from evaluate_retrospective_wavelet_sweep import (  # noqa: E402
    _finalize_reference_directory,
    direct_fft_rss,
    metric_leaders,
)
from run_retrospective_wavelet_sweep import (  # noqa: E402
    _lambda_from_options,
    build_lambda_config,
    lambda_label,
)
from wave_retro_lr.core import centered_fftn  # noqa: E402


class ReconstructionConfigurationTests(unittest.TestCase):
    def test_lambda_labels_and_bart_options_are_explicit(self) -> None:
        self.assertEqual(lambda_label(0.0), "0")
        self.assertEqual(lambda_label(1.5e-2), "0p015")
        self.assertEqual(
            _lambda_from_options(["-w", "-f", "-r", "0.015", "-g"]),
            0.015,
        )
        with self.assertRaisesRegex(ValueError, "-w -f"):
            _lambda_from_options(["-r", "0.015", "-g"])

    def test_lambda_config_is_a_thin_specialization(self) -> None:
        base = {
            "source": {"subject": "base", "twix": "/placeholder"},
            "output_root": "/old",
            "reconstruction": {
                "regularizer": "wavelet",
                "lambda": 0.0,
                "iterations": 100,
            },
        }
        result = build_lambda_config(
            base,
            lambda_value=0.003,
            output_root=Path("/new"),
            prepared_root=Path("/prepared"),
            subject="sweep",
        )
        self.assertEqual(result["reconstruction"]["lambda"], 0.003)
        self.assertEqual(result["source"]["subject"], "sweep")
        self.assertEqual(result["prepared_cases_root"], "/prepared")
        self.assertEqual(base["reconstruction"]["lambda"], 0.0)


class NativeReferenceTests(unittest.TestCase):
    def test_reference_manifest_geometry_is_json_native_and_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            temporary = root / ".case-temporary"
            nifti_dir = temporary / "nifti" / "sub-test"
            nifti_dir.mkdir(parents=True)
            nifti_path = nifti_dir / "sub-test_part-mag_DirectFFTRSSNativeGrid.nii.gz"
            candidate_path = root / "candidate.nii.gz"
            affine = np.diag([1.5, 1.0, 1.0, 1.0])
            image = nib.Nifti1Image(np.ones((8, 10, 12), dtype=np.float32), affine)
            nib.save(image, nifti_path)
            nib.save(image, candidate_path)
            case = {"case_name": "case", "achieved_resolution_mm_xyz": [1.5, 1.0, 1.0]}
            nifti_path.with_name(nifti_path.name[:-7] + ".json").write_text(
                json.dumps({"RetrospectiveCase": case}), encoding="utf-8"
            )

            record = _finalize_reference_directory(
                temporary_dir=temporary,
                case_dir=root / "case",
                case=case,
                source_path=root / "source.npy",
                source_sha256="source-hash",
                expected_candidate=candidate_path,
            )

            self.assertTrue((root / "case" / "reference_manifest.json").is_file())
            loaded = json.loads((root / "case" / "reference_manifest.json").read_text())
            self.assertEqual(loaded["voxel_size_mm_ras_xyz"], [1.5, 1.0, 1.0])
            self.assertEqual(record["shape_ras_xyz"], [8, 10, 12])

    def test_direct_fft_rss_uses_exact_center_crop(self) -> None:
        rng = np.random.default_rng(12)
        source = (
            rng.standard_normal((4, 4, 4, 2))
            + 1j * rng.standard_normal((4, 4, 4, 2))
        ).astype(np.complex64)
        case = {
            "target_logical_matrix_ro_lin_par": [4, 2, 2],
            "crop_bounds_lin": [1, 3],
            "crop_bounds_par": [1, 3],
        }
        found = direct_fft_rss(source, case, fft_workers=1)
        expected_squared = np.zeros((4, 2, 2), dtype=np.float32)
        for coil in range(2):
            image = centered_fftn(
                source[:, 1:3, 1:3, coil],
                axes=(0, 1, 2),
                inverse=True,
                workers=1,
            )
            expected_squared += np.abs(image).astype(np.float32) ** 2
        np.testing.assert_allclose(found, np.sqrt(expected_squared), rtol=2e-6)

    def test_metric_leaders_remain_per_metric(self) -> None:
        rows = []
        for value, nrmse, ssim, gradient, edge in (
            (1e-3, 0.10, 0.90, 0.95, 0.98),
            (1e-2, 0.08, 0.89, 0.96, 1.02),
        ):
            rows.append(
                {
                    "case_name": "case",
                    "lambda": value,
                    "nrmse_brain": nrmse,
                    "ssim_axial_brain_bbox_mean": ssim,
                    "gradient_ncc_fixed_edge": gradient,
                    "edge_gradient_preservation_ratio": edge,
                }
            )
        leaders = metric_leaders(rows)["case"]
        self.assertEqual(leaders["nrmse_brain"]["lambda"], 1e-2)
        self.assertEqual(leaders["ssim_axial_brain_bbox_mean"]["lambda"], 1e-3)


if __name__ == "__main__":
    unittest.main()
