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
    _case_specifications,
    _remapped_residue,
    build_wave_command,
    coarse_candidate_settings,
    configured_case_ids,
    output_layout,
    validate_bart_artifact,
    validate_csm_rss_normalization,
    validate_direct_fft_reference,
    validate_config,
    validate_manifest_binding,
    validate_psf_unit_magnitude,
    write_masked_wave_cfl,
)
from wave_retro_lr.core import Geometry, resolve_case  # noqa: E402
from wave_retro_lr.bart_io import (  # noqa: E402
    create_cfl,
    logical_array_sha256,
    open_cfl,
    sha256_file,
)
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

    def test_reused_direct_fft_reference_is_hash_and_source_bound(self) -> None:
        """Verify reusable references require exact array and source provenance.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            reference_path = root / "reference.npy"
            np.save(reference_path, reference)
            source_hash = "1" * 64
            manifest_path = root / "case_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_no_wave_sha256": source_hash,
                        "shape": [2, 3, 4],
                        "reference_sha256": sha256_file(reference_path),
                    }
                ),
                encoding="utf-8",
            )
            specification = {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
                "logical_sha256": logical_array_sha256(reference),
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "assertions": [
                        {
                            "label": "source_no_wave",
                            "json_path": ["source_no_wave_sha256"],
                            "equals": source_hash,
                        },
                        {
                            "label": "dimensions",
                            "json_path": ["shape"],
                            "equals": [2, 3, 4],
                        },
                        {
                            "label": "artifact",
                            "json_path": ["reference_sha256"],
                            "equals": sha256_file(reference_path),
                        },
                    ],
                },
            }
            _path, record = validate_direct_fft_reference(
                specification,
                root,
                expected_shape=(2, 3, 4),
                source_no_wave_sha256=source_hash,
                label="accepted reference",
            )
            self.assertTrue(record["reused"])
            with self.assertRaisesRegex(ValueError, "different no-Wave source"):
                validate_direct_fft_reference(
                    specification,
                    root,
                    expected_shape=(2, 3, 4),
                    source_no_wave_sha256="2" * 64,
                    label="accepted reference",
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

    def test_manifest_defined_native_r3x3_case_has_exact_mask_contract(self) -> None:
        """Verify a single native R3x3 case resolves without legacy case assumptions.

        Returns:
            None.
        """
        config = {
            "format_version": 2,
            "cases": {
                "native_r3x3": {
                    "requested_resolution_mm_xyz": [1.0, 1.0, 1.0],
                    "acceleration_lin_par": [3, 3],
                    "label": "native R3x3",
                }
            },
        }
        geometry = Geometry((256.0, 256.0, 256.0), (256, 256, 256))
        specifications = _case_specifications(config, geometry)
        self.assertEqual(tuple(case_id for case_id, _ in specifications), ("native_r3x3",))
        case = resolve_case(specifications[0][1], geometry)
        residue = _remapped_residue((1, 2), case)
        mask, metadata = pure_cartesian_image_lattice_mask(
            case.target_logical_matrix_ro_lin_par[1:],
            acceleration_lin_par=case.acceleration_ry_rz,
            residue_lin_par=residue,
        )
        self.assertEqual(residue, (1, 2))
        self.assertEqual(int(mask.sum()), 7225)
        self.assertEqual(
            metadata["logical_sha256"],
            "36412ff8771b49c3f60b7b2d6ff766101a99334d73811c75d4b45571b2b536f3",
        )
        layout = output_layout(
            "/path/to/r3x3-run",
            case_ids=("native_r3x3",),
            native_case_ids=("native_r3x3",),
            include_source_materialization=False,
        )
        self.assertEqual(tuple(layout["cases"]), ("native_r3x3",))
        self.assertNotIn("source_materialization", layout)
        self.assertNotIn("full_wave_kspace", layout["cases"]["native_r3x3"])

    def test_format_two_single_case_validates_all_reused_artifacts(self) -> None:
        """Verify one native R3x3 config is fully hash and provenance validated.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            no_wave_path = root / "no_wave.npy"
            full_wave_path = root / "full_wave.npy"
            reference_path = root / "reference.npy"
            np.save(no_wave_path, np.ones((4, 8, 8, 2), dtype=np.complex64))
            np.save(full_wave_path, np.ones((8, 8, 8, 2), dtype=np.complex64))
            reference = np.ones((4, 8, 8), dtype=np.float32)
            np.save(reference_path, reference)
            csm_base = root / "coil_sens"
            csm = create_cfl(csm_base, (4, 8, 8, 2, 1))
            csm[...] = np.complex64(1 / np.sqrt(2))
            csm.flush()
            del csm
            psf_base = root / "psf"
            psf = create_cfl(psf_base, (8, 8, 8, 1, 1))
            psf[...] = np.complex64(1)
            psf.flush()
            del psf
            bet_path = root / "brain_mask.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((8, 8, 4), dtype=np.uint8), np.eye(4)),
                bet_path,
            )
            dataset = "a" * 64
            coil_order = "b" * 64
            trajectory = "c" * 64
            binding = {
                "dataset": dataset,
                "fov": [8.0, 8.0, 4.0],
                "source": {
                    "no_wave_dimensions": [4, 8, 8, 2],
                    "full_wave_dimensions": [8, 8, 8, 2],
                },
                "coil_order": coil_order,
                "trajectory": trajectory,
                "psf_model": "theoretical_sequence_trajectory_without_calibrated_correction",
                "wave_data_origin": "synthetic_from_fully_sampled_no_wave",
                "calibration_samples_merged_into_wave_kspace": False,
                "calibration_source": "fully_sampled_image_kspace",
                "cases": {
                    "native_r3x3": {
                        "csm_dimensions": [4, 8, 8, 2, 1],
                        "psf_dimensions": [8, 8, 8, 1, 1],
                    }
                },
                "brain_mask": {"approved": True, "canonical_ras": True},
                "orientation": {
                    "logical_to_canonical_axis_flips": [False, False, True],
                    "canonical_ras": True,
                },
                "reference": {
                    "source_no_wave_sha256": sha256_file(no_wave_path),
                    "shape": [4, 8, 8],
                    "sha256": sha256_file(reference_path),
                },
            }
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(binding), encoding="utf-8")

            def manifest(assertions: list[dict[str, object]]) -> dict[str, object]:
                """Build one test binding against the shared provenance object.

                Args:
                    assertions: Labeled JSON assertions to bind.

                Returns:
                    Test manifest specification with exact file hash.
                """
                return {
                    "path": str(binding_path),
                    "sha256": sha256_file(binding_path),
                    "assertions": assertions,
                }

            mask, mask_metadata = pure_cartesian_image_lattice_mask(
                (8, 8), acceleration_lin_par=(3, 3), residue_lin_par=(1, 2)
            )
            del mask
            common_manifest = manifest(
                [
                    {"label": "dataset", "json_path": ["dataset"], "equals": dataset},
                    {"label": "fov", "json_path": ["fov"], "equals": [8.0, 8.0, 4.0]},
                    {"label": "coil_order", "json_path": ["coil_order"], "equals": coil_order},
                ]
            )
            config = {
                "format_version": 2,
                "workflow": "synthetic_wave_pure_mask_regularization_rerun",
                "output_root": str(root / "unused_output"),
                "geometry": {
                    "physical_fov_mm_xyz": [8.0, 8.0, 4.0],
                    "native_logical_matrix_ro_lin_par": [4, 8, 8],
                    "extended_wave_readout": 8,
                    "virtual_coils": 2,
                },
                "sampling": {
                    "mask_kind": "pure_cartesian_image_lattice",
                    "native_residue_lin_par": [1, 2],
                },
                "source": {
                    "no_wave_kspace": {
                        "path": str(no_wave_path),
                        "sha256": sha256_file(no_wave_path),
                        "manifest": {
                            **common_manifest,
                            "assertions": common_manifest["assertions"]
                            + [
                                {
                                    "label": "dimensions",
                                    "json_path": ["source", "no_wave_dimensions"],
                                    "equals": [4, 8, 8, 2],
                                }
                            ],
                        },
                    },
                    "native_full_wave_kspace": {
                        "path": str(full_wave_path),
                        "sha256": sha256_file(full_wave_path),
                        "manifest": manifest(
                            [
                                {"label": "dataset", "json_path": ["dataset"], "equals": dataset},
                                {"label": "fov", "json_path": ["fov"], "equals": [8.0, 8.0, 4.0]},
                                {"label": "dimensions", "json_path": ["source", "full_wave_dimensions"], "equals": [8, 8, 8, 2]},
                                {"label": "coil_order", "json_path": ["coil_order"], "equals": coil_order},
                                {"label": "trajectory", "json_path": ["trajectory"], "equals": trajectory},
                                {"label": "psf_model", "json_path": ["psf_model"], "equals": binding["psf_model"]},
                                {"label": "wave_data_origin", "json_path": ["wave_data_origin"], "equals": binding["wave_data_origin"]},
                                {"label": "calibration_samples_merged", "json_path": ["calibration_samples_merged_into_wave_kspace"], "equals": False},
                            ]
                        ),
                    },
                    "approved_bet_mask": {
                        "path": str(bet_path),
                        "sha256": sha256_file(bet_path),
                        "manifest": manifest(
                            [
                                {"label": "approval", "json_path": ["brain_mask", "approved"], "equals": True},
                                {"label": "geometry", "json_path": ["brain_mask", "canonical_ras"], "equals": True},
                            ]
                        ),
                    },
                },
                "cases": {
                    "native_r3x3": {
                        "requested_resolution_mm_xyz": [1.0, 1.0, 1.0],
                        "acceleration_lin_par": [3, 3],
                        "label": "native R3x3",
                        "expected_mask_logical_sha256": mask_metadata["logical_sha256"],
                        "direct_fft_reference": {
                            "path": str(reference_path),
                            "sha256": sha256_file(reference_path),
                            "logical_sha256": logical_array_sha256(reference),
                            "manifest": manifest(
                                [
                                    {"label": "source_no_wave", "json_path": ["reference", "source_no_wave_sha256"], "equals": sha256_file(no_wave_path)},
                                    {"label": "dimensions", "json_path": ["reference", "shape"], "equals": [4, 8, 8]},
                                    {"label": "artifact", "json_path": ["reference", "sha256"], "equals": sha256_file(reference_path)},
                                ]
                            ),
                        },
                        "csm": {
                            "base": str(csm_base),
                            "header_sha256": sha256_file(csm_base.with_suffix(".hdr")),
                            "payload_sha256": sha256_file(csm_base.with_suffix(".cfl")),
                            "manifest": manifest(
                                [
                                    {"label": "dataset", "json_path": ["dataset"], "equals": dataset},
                                    {"label": "fov", "json_path": ["fov"], "equals": [8.0, 8.0, 4.0]},
                                    {"label": "dimensions", "json_path": ["cases", "native_r3x3", "csm_dimensions"], "equals": [4, 8, 8, 2, 1]},
                                    {"label": "coil_order", "json_path": ["coil_order"], "equals": coil_order},
                                    {"label": "calibration_source", "json_path": ["calibration_source"], "equals": "fully_sampled_image_kspace"},
                                ]
                            ),
                        },
                        "psf": {
                            "base": str(psf_base),
                            "header_sha256": sha256_file(psf_base.with_suffix(".hdr")),
                            "payload_sha256": sha256_file(psf_base.with_suffix(".cfl")),
                            "manifest": manifest(
                                [
                                    {"label": "dataset", "json_path": ["dataset"], "equals": dataset},
                                    {"label": "fov", "json_path": ["fov"], "equals": [8.0, 8.0, 4.0]},
                                    {"label": "dimensions", "json_path": ["cases", "native_r3x3", "psf_dimensions"], "equals": [8, 8, 8, 1, 1]},
                                    {"label": "trajectory", "json_path": ["trajectory"], "equals": trajectory},
                                    {"label": "psf_model", "json_path": ["psf_model"], "equals": binding["psf_model"]},
                                    {"label": "wave_data_origin", "json_path": ["wave_data_origin"], "equals": binding["wave_data_origin"]},
                                ]
                            ),
                        },
                    }
                },
                "evaluation": {
                    "logical_to_canonical_axis_order": [2, 1, 0],
                    "logical_to_canonical_axis_flips": [False, False, True],
                    "orientation_manifest": manifest(
                        [
                            {"label": "orientation", "json_path": ["orientation", "logical_to_canonical_axis_flips"], "equals": [False, False, True]},
                            {"label": "canonical_ras", "json_path": ["orientation", "canonical_ras"], "equals": True},
                        ]
                    ),
                },
            }
            config_path = root / "config.local.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            validated = validate_config(config_path)
            self.assertEqual(configured_case_ids(validated), ("native_r3x3",))
            self.assertTrue(validated["cases"][0].direct_fft_reference["reused"])


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
        """Verify the fine pool supports reviewed intervals and R3x3 extension.

        Returns:
            None.
        """
        expected = {
            0.0175,
            0.02,
            0.025,
            0.0275,
            0.0325,
            0.035,
            0.04,
            0.045,
            0.055,
            0.06,
            0.065,
            0.07,
            0.08,
            0.09,
            0.1,
        }
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
