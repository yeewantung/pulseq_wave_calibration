"""Focused contracts for the one-command Ncc=24 Wavelet sweep launchers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = TOOL_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_direct_fft_regularization import _metric_leaders  # noqa: E402
from evaluate_mprage_vcc24_wavelet_sweep import (  # noqa: E402
    EXPECTED_LAMBDAS,
    validate_one_shot_contract,
)
from gre_synthetic_wave import validate_config_document  # noqa: E402
from run_bart_regularization import canonical_lambda  # noqa: E402


class OneShotVcc24SweepTests(unittest.TestCase):
    """Validate grids, Ncc isolation, native cases, and Wavelet-only behavior."""

    def test_mprage_launcher_has_pure_r3x1_grid_and_no_llr(self) -> None:
        """Keep the MPRAGE one-shot run pure-mask and within 1e-2 to 5e-2."""
        source = (SCRIPT_ROOT / "run_mprage_vcc24_wavelet_sweep.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MPRAGE_LAMBDAS=(0 0.01 0.015", source)
        self.assertIn("0.045 0.05)", source)
        self.assertIn('!= "pure_cartesian_image_lattice"', source)
        self.assertIn('!= [3, 1]', source)
        self.assertIn("--regularizer wavelet", source)
        self.assertNotIn("--regularizer llr", source)
        self.assertIn("evaluate_mprage_vcc24_wavelet_sweep.py", source)
        self.assertIn("--evaluate-only", source)
        self.assertNotIn("record-selection", source)
        inspection_index = source.index("inspect_product_dataset.py")
        validation_index = source.index("validate_dataset_manifest.py")
        self.assertLess(inspection_index, validation_index)

    def test_mprage_evaluation_contract_is_standard_pca_vcc24(self) -> None:
        """Reject incompatible coil or sampling contracts before metric output."""
        payload = {
            "reconstruction": {
                "physical_coils": 64,
                "virtual_coils": 24,
                "coil_compression_source": "image",
            },
            "geometry": {"matrix": [256, 256, 256]},
            "sampling": {
                "synthetic_wave_mask_kind": "pure_cartesian_image_lattice",
                "synthetic_wave_acceleration_pe1_pe2": [3, 1],
                "synthetic_wave_residue_pe1_pe2": [1, 0],
            },
        }
        validate_one_shot_contract(payload)
        payload["reconstruction"]["virtual_coils"] = 12
        with self.assertRaisesRegex(ValueError, "64-to-24"):
            validate_one_shot_contract(payload)

    def test_wavelet_only_metric_leaders_do_not_require_llr(self) -> None:
        """Support descriptive Wavelet curves without inventing an LLR branch."""
        records = [
            {
                "case_id": "wavelet:0.01",
                "regularizer": "wavelet",
                "lambda": 0.01,
                "nrmse_brain": 0.2,
                "ssim_3d_brain_bbox": 0.9,
                "ncc_brain": 0.95,
                "gradient_ncc_brain_edge": 0.8,
                "edge_preservation_ratio": 1.1,
            }
        ]
        leaders = _metric_leaders(records)
        self.assertEqual(set(leaders), {"wavelet"})
        self.assertEqual(
            leaders["wavelet"]["lowest_nrmse_brain"]["lambda"], 0.01
        )

    def test_mprage_evaluation_uses_manifest_lambda_labels(self) -> None:
        """Match scientific lambda values to canonical case-manifest labels."""
        labels = [canonical_lambda(value) for value in EXPECTED_LAMBDAS]
        self.assertEqual(
            labels,
            ["0", "1e-2", "1.5e-2", "2e-2", "2.5e-2", "3e-2", "3.5e-2", "4e-2", "4.5e-2", "5e-2"],
        )

    def test_gre_format_three_is_ncc24_native_wavelet_only(self) -> None:
        """Accept one native GRE case and a shared Wavelet-only Ncc=24 grid."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "format_version": 3,
                "workflow": "synthetic_wave_gre_regularization_sweep",
                "output_parent": str(root),
                "run_name": "gre_vcc24_sweep",
                "case_ids": ["native_r3x1"],
                "geometry": {
                    "source_matrix_ro_lin_par": [256, 256, 72],
                    "native_matrix_ro_lin_par": [250, 250, 72],
                    "fov_mm_ro_lin_par": [220.0, 220.0, 180.0],
                    "extended_wave_readout": 1000,
                    "low_resolution_matrix_ro_lin_par": [250, 148, 72],
                },
                "sampling": {
                    "mask_kind": "pure_cartesian_image_lattice",
                    "residue_lin_par": [2, 0],
                },
                "coil_compression": {
                    "physical_coils": 44,
                    "virtual_coils": 24,
                    "partition_chunk": 4,
                    "readout_step": 4,
                    "covariance": "trace_balanced_across_two_echoes",
                },
                "csm": {
                    "calibration_echo": 1,
                    "calibration_size_ro_lin_par": [250, 32, 32],
                    "ecalib_maps": 1,
                    "ecalib_crop": 0.6,
                    "shared_across_echoes": True,
                },
                "brain_mask": {
                    "source_case": "native_r3x1",
                    "source_echo": 1,
                    "fractional_intensity_threshold": 0.3,
                    "vertical_gradient": 0.0,
                    "robust_center": True,
                    "dilation_voxels": 0,
                },
                "reused_brain_mask": {
                    "path": str(root / "approved_mask_manifest.json"),
                    "sha256": "0" * 64,
                },
                "sweep": {
                    "mode": "wavelet_only",
                    "wavelet_lambdas": [
                        0.005,
                        0.0075,
                        0.01,
                        0.0125,
                        0.015,
                        0.0175,
                        0.02,
                        0.0225,
                        0.025,
                        0.0275,
                        0.03,
                    ],
                    "iterations": 100,
                    "tolerance": 1e-6,
                },
                "runtime": {"backend": "gpu", "bart": "bart", "fft_workers": 4},
            }
            validated = validate_config_document(config)
        self.assertEqual(validated["virtual_coils"], 24)
        self.assertEqual(validated["case_ids"], ["native_r3x1"])
        self.assertEqual(validated["coarse_jobs_per_group"], 12)
        self.assertEqual(validated["coarse_job_count"], 24)
        self.assertEqual(validated["candidate_settings"][0]["method"], "fista_lambda0")
        self.assertTrue(
            all(
                setting["method"] in {"fista_lambda0", "wavelet"}
                for setting in validated["candidate_settings"]
            )
        )

    def test_gre_launcher_runs_shared_echo_review_without_selection(self) -> None:
        """Require delta-B0 review while prohibiting an automatic selection stage."""
        source = (SCRIPT_ROOT / "run_gre_vcc24_wavelet_sweep.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate-shared-lambda", source)
        self.assertIn("plot-shared-lambda", source)
        self.assertIn("No parameter winner was selected", source)
        self.assertNotIn("record-selection", source)

    def test_tracked_examples_are_private_path_free(self) -> None:
        """Keep server paths exclusively in ignored local launch files."""
        paths = (
            SCRIPT_ROOT / "run_mprage_vcc24_wavelet_sweep.example.sh",
            SCRIPT_ROOT / "run_gre_vcc24_wavelet_sweep.example.sh",
        )
        forbidden = ("/autofs/", "/homes/", "yd918", "MID00", "FID")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertFalse(any(value in source for value in forbidden), path)


if __name__ == "__main__":
    unittest.main()
