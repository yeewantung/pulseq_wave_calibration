"""Focused tests for the corrected pure-mask rerun contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
RETRO_ROOT = Path(__file__).resolve().parents[2] / "wave_retro_lr_recon"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(RETRO_ROOT))

from evaluate_pure_mask_sweeps import (  # noqa: E402
    _canonical_mask,
    _plot_metric_curves,
    _validate_refresh_tree,
    metric_leaders,
    scale_candidate_for_display,
)
from build_pure_mask_presentation import _lambda_token, _setting_key  # noqa: E402
from pure_mask_rerun import (  # noqa: E402
    COARSE_LLR_LAMBDAS,
    COARSE_WAVELET_LAMBDAS,
    FINE_LAMBDA_POOL,
    build_wave_command,
    coarse_candidate_settings,
    output_layout,
    validate_bart_artifact,
    validate_csm_rss_normalization,
    validate_manifest_binding,
    validate_psf_unit_magnitude,
    write_masked_wave_cfl,
)
from wave_retro_lr.bart_io import create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.sampling import pure_cartesian_image_lattice_mask  # noqa: E402


class PureMaskPreparationTests(unittest.TestCase):
    """Validate exact sampling, BART export, and immutable input gates."""

    def test_masked_wave_export_preserves_acquired_and_zeros_missing(self) -> None:
        """Verify full sample equality and exact zeros outside a pure mask.

        Returns:
            None.
        """
        rng = np.random.default_rng(91)
        source = (
            rng.standard_normal((8, 7, 6, 2))
            + 1j * rng.standard_normal((8, 7, 6, 2))
        ).astype(np.complex64)
        mask, _metadata = pure_cartesian_image_lattice_mask(
            (7, 6), acceleration_lin_par=(3, 2), residue_lin_par=(2, 1)
        )
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "wave_kspace"
            record = write_masked_wave_cfl(source, mask, base)
            result = np.asarray(open_cfl(base))[..., 0]
            np.testing.assert_array_equal(result[:, mask, :], source[:, mask, :])
            self.assertFalse(np.any(result[:, ~mask, :]))
            self.assertEqual(record["acquired_mismatch_count"], 0)
            self.assertEqual(record["unacquired_nonzero_count"], 0)

    def test_bart_geometry_hash_and_provenance_are_strict(self) -> None:
        """Verify accepted CSM/PSF-like inputs require exact geometry and provenance.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "artifact"
            array = create_cfl(base, (4, 6, 8, 2, 1))
            array[...] = np.complex64(1 / np.sqrt(2))
            array.flush()
            del array
            provenance_path = root / "provenance.json"
            provenance = {
                "dataset": "accepted",
                "geometry": [4, 6, 8],
                "coil_order": "frozen",
                "fov": [8.0, 6.0, 4.0],
            }
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            specification = {
                "base": str(base),
                "header_sha256": sha256_file(base.with_suffix(".hdr")),
                "payload_sha256": sha256_file(base.with_suffix(".cfl")),
                "manifest": {
                    "path": str(provenance_path),
                    "sha256": sha256_file(provenance_path),
                    "assertions": [
                        {"label": key, "json_path": [key], "equals": value}
                        for key, value in provenance.items()
                    ],
                },
            }
            _path, record = validate_bart_artifact(
                specification,
                root,
                expected_shape=(4, 6, 8, 2, 1),
                required_assertion_labels={"dataset", "geometry", "coil_order", "fov"},
                label="accepted CSM",
            )
            self.assertEqual(record["shape"], [4, 6, 8, 2, 1])
            with self.assertRaisesRegex(ValueError, "shape"):
                validate_bart_artifact(
                    specification,
                    root,
                    expected_shape=(4, 8, 6, 2, 1),
                    required_assertion_labels={"dataset", "geometry", "coil_order", "fov"},
                    label="accepted CSM",
                )
            specification["manifest"]["assertions"][0]["equals"] = "changed"
            with self.assertRaisesRegex(ValueError, "provenance assertion"):
                validate_bart_artifact(
                    specification,
                    root,
                    expected_shape=(4, 6, 8, 2, 1),
                    required_assertion_labels={"dataset", "geometry", "coil_order", "fov"},
                    label="accepted CSM",
                )

    def test_provenance_binding_rejects_changed_upstream_manifest(self) -> None:
        """Verify a local binding index remains chained to immutable upstream JSON.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            upstream = root / "upstream.json"
            upstream.write_text(json.dumps({"status": "accepted"}), encoding="utf-8")
            binding = root / "binding.json"
            binding.write_text(
                json.dumps(
                    {
                        "value": 4,
                        "upstream_manifests": [
                            {"path": str(upstream), "sha256": sha256_file(upstream)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specification = {
                "path": str(binding),
                "sha256": sha256_file(binding),
                "assertions": [{"label": "value", "json_path": ["value"], "equals": 4}],
            }
            record = validate_manifest_binding(
                specification,
                root,
                required_assertion_labels={"value"},
                label="accepted binding",
            )
            self.assertEqual(len(record["upstream_manifests"]), 1)
            upstream.write_text(json.dumps({"status": "changed"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "upstream provenance manifest changed"):
                validate_manifest_binding(
                    specification,
                    root,
                    required_assertion_labels={"value"},
                    label="accepted binding",
                )
    def test_csm_and_psf_value_contracts_reject_invalid_inputs(self) -> None:
        """Verify CSM RSS and PSF unit-magnitude gates accept only valid values.

        Returns:
            None.
        """
        csm = np.full((3, 4, 5, 2), 1 / np.sqrt(2), dtype=np.complex64)
        csm_metrics = validate_csm_rss_normalization(
            csm, support_threshold=1e-6, tolerance=1e-5
        )
        self.assertLess(csm_metrics["maximum_absolute_error_from_one"], 1e-5)
        csm[..., 0] *= 2
        with self.assertRaisesRegex(ValueError, "RSS normalization"):
            validate_csm_rss_normalization(
                csm, support_threshold=1e-6, tolerance=1e-5
            )

        psf = np.exp(1j * np.linspace(0, 1, 60)).reshape(3, 4, 5).astype(np.complex64)
        psf_metrics = validate_psf_unit_magnitude(psf, tolerance=1e-6)
        self.assertLess(psf_metrics["maximum_absolute_error_from_one"], 1e-6)
        psf[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "unit magnitude"):
            validate_psf_unit_magnitude(psf, tolerance=1e-6)

    def test_approved_bet_mask_requires_exact_geometry_and_finite_values(self) -> None:
        """Verify the reused BET mask is finite and matches the native FOV/grid.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "approved_mask.nii.gz"
            values = np.ones((4, 6, 8), dtype=np.float32)
            nib.save(nib.Nifti1Image(values, np.diag([2.0, 1.0, 0.5, 1.0])), path)
            _image, mask = _canonical_mask(
                path,
                expected_shape_xyz=(4, 6, 8),
                expected_fov_mm_xyz=(8.0, 6.0, 4.0),
            )
            self.assertEqual(mask.shape, (4, 6, 8))
            with self.assertRaisesRegex(ValueError, "dimension, FOV"):
                _canonical_mask(
                    path,
                    expected_shape_xyz=(4, 8, 6),
                    expected_fov_mm_xyz=(8.0, 6.0, 4.0),
                )
            values[0, 0, 0] = np.nan
            nib.save(nib.Nifti1Image(values, np.diag([2.0, 1.0, 0.5, 1.0])), path)
            with self.assertRaisesRegex(ValueError, "finite-value"):
                _canonical_mask(
                    path,
                    expected_shape_xyz=(4, 6, 8),
                    expected_fov_mm_xyz=(8.0, 6.0, 4.0),
                )

    def test_layout_is_fixed_and_contains_all_cases(self) -> None:
        """Verify preparation, sweep, and evaluation trees are declared up front.

        Returns:
            None.
        """
        layout = output_layout("/path/to/approved-run")
        self.assertEqual(len(layout["cases"]), 5)
        self.assertTrue(layout["sweeps"]["coarse"].endswith("sweeps/coarse"))
        self.assertTrue(layout["evaluation"]["review"].endswith("evaluation/review"))


class PureMaskSweepTests(unittest.TestCase):
    """Validate FISTA, Wavelet, corrected-LLR, and selection contracts."""

    def test_coarse_grid_has_one_control_and_approved_families(self) -> None:
        """Verify one control, seven Wavelet, and 3x5 corrected-LLR settings.

        Returns:
            None.
        """
        settings = coarse_candidate_settings()
        controls = [item for item in settings if item["method"] == "fista_lambda0"]
        wavelets = [item for item in settings if item["method"] == "wavelet"]
        llr = [item for item in settings if item["method"] == "llr"]
        self.assertEqual(len(settings), 23)
        self.assertEqual(controls, [{"method": "fista_lambda0", "lambda": 0.0, "block_size": None}])
        self.assertEqual(tuple(item["lambda"] for item in wavelets), COARSE_WAVELET_LAMBDAS)
        for block in (4, 8, 16):
            self.assertEqual(
                tuple(item["lambda"] for item in llr if item["block_size"] == block),
                COARSE_LLR_LAMBDAS,
            )

    def test_commands_are_gpu_fista_and_corrected_split_complex_llr(self) -> None:
        """Verify exact approved BART option forms for all three methods.

        Returns:
            None.
        """
        common = {
            "bart": "bart",
            "csm_base": "csm",
            "psf_base": "psf",
            "wave_kspace_base": "wave",
            "output_base": "image",
        }
        control = build_wave_command(
            **common, method="fista_lambda0", lambda_value=0.0, block_size=None
        )
        self.assertEqual(control[2:7], ["-g", "-w", "-f", "-r", "0"])
        wavelet = build_wave_command(
            **common, method="wavelet", lambda_value=0.015, block_size=None
        )
        self.assertTrue({"-g", "-w", "-f", "-r"}.issubset(wavelet))
        llr = build_wave_command(
            **common, method="llr", lambda_value=0.01, block_size=16
        )
        self.assertTrue({"-g", "-l", "-v", "-b", "-f", "-r"}.issubset(llr))
        self.assertEqual(llr[llr.index("-b") + 1], "16")

    def test_fine_pool_brackets_the_reviewed_wavelet_optima(self) -> None:
        """Verify the fine pool supports the reviewed native and LR intervals.

        Returns:
            None.
        """
        expected = {0.0175, 0.02, 0.025, 0.0275, 0.0325, 0.035, 0.04}
        self.assertTrue(expected.issubset(FINE_LAMBDA_POOL))

    def test_presentation_keys_normalize_selected_settings(self) -> None:
        """Verify stable presentation lambda tokens and setting identities.

        Returns:
            None.
        """
        setting = {"method": "wavelet", "block_size": None, "lambda": 0.035}
        self.assertEqual(_lambda_token(setting["lambda"]), "0p035")
        self.assertEqual(_setting_key(setting), ("wavelet", None, 0.035))
        with self.assertRaisesRegex(ValueError, "positive"):
            _lambda_token(0.0)

    def test_metric_leaders_remain_separate_without_composite(self) -> None:
        """Verify evaluation reports per-metric leaders instead of one winner.

        Returns:
            None.
        """
        rows = []
        for case_id in (
            "native_r3x1",
            "native_r3x2",
            "lr_x_r3x2",
            "lr_y_r3x2",
            "lr_xy_r3x2",
        ):
            for method, block, value, nrmse, ncc in (
                ("fista_lambda0", None, 0.0, 0.2, 0.95),
                ("wavelet", None, 0.01, 0.1, 0.94),
                ("llr", 4, 0.01, 0.12, 0.97),
                ("llr", 8, 0.01, 0.11, 0.96),
                ("llr", 16, 0.01, 0.13, 0.98),
            ):
                rows.append(
                    {
                        "case_id": case_id,
                        "method": method,
                        "block_size": block,
                        "lambda": value,
                        "nrmse_brain": nrmse,
                        "rmse_brain": nrmse,
                        "mae_brain": nrmse,
                        "ncc_brain": ncc,
                        "ssim_3d_brain_bbox": ncc,
                        "gradient_ncc_fixed_edge": ncc,
                        "edge_gradient_preservation_ratio": 1.0 + nrmse,
                    }
                )
        leaders = metric_leaders(rows)
        self.assertEqual(leaders["native_r3x1"]["wavelet"]["nrmse_brain"]["method"], "wavelet")
        self.assertNotIn("composite", json.dumps(leaders).lower())
        self.assertNotIn("winner", json.dumps(leaders).lower())

    def test_display_scaling_uses_the_recorded_positive_lsq_factor(self) -> None:
        """Verify montage scaling maps candidates into reference intensity units.

        Returns:
            None.
        """
        candidate = np.asarray([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            scale_candidate_for_display(candidate, 3.0),
            np.asarray([[3.0, 6.0]], dtype=np.float32),
        )
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                scale_candidate_for_display(candidate, invalid)

    def test_metric_curve_writes_native_and_matched_series(self) -> None:
        """Verify LR metric curves include regularized values and FISTA controls.

        Returns:
            None.
        """
        rows = []
        for method, lambda_value in (("fista_lambda0", 0.0), ("wavelet", 0.002), ("wavelet", 0.01)):
            row = {
                "method": method,
                "block_size": None,
                "lambda": lambda_value,
            }
            for metric in (
                "nrmse_brain",
                "ssim_3d_brain_bbox",
                "ncc_brain",
                "gradient_ncc_fixed_edge",
                "edge_gradient_preservation_ratio",
                "background_std_normalized_p99_qc",
            ):
                row[metric] = 1.0 + lambda_value
                row[f"matched_1mm_{metric}"] = 2.0 + lambda_value
            rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curves.png"
            _plot_metric_curves(
                path,
                rows,
                family="wavelet",
                include_matched_1mm=True,
                title="test curves",
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_refresh_tree_rejects_unowned_or_changed_files(self) -> None:
        """Verify evaluation refresh is limited to hash-bound evaluator outputs.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "evaluation_manifest.json"
            metric_path = root / "metrics.csv"
            mask_path = root / "case" / "approved_bet_mask_native.npy"
            mask_path.parent.mkdir()
            manifest_path.write_text("{}", encoding="utf-8")
            metric_path.write_text("metric\n1\n", encoding="utf-8")
            np.save(mask_path, np.ones((2, 2, 2), dtype=bool))
            prior = {
                "outputs": [{"path": str(metric_path), "sha256": sha256_file(metric_path)}],
                "derived_native_bet_masks": {
                    "case": {"path": str(mask_path), "sha256": sha256_file(mask_path)}
                },
            }
            _validate_refresh_tree(root, manifest_path, prior)
            unknown = root / "manual_note.txt"
            unknown.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unowned files"):
                _validate_refresh_tree(root, manifest_path, prior)
            unknown.unlink()
            metric_path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file changed"):
                _validate_refresh_tree(root, manifest_path, prior)


if __name__ == "__main__":
    unittest.main()
