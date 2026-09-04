"""Focused tests for the two-echo synthetic-Wave GRE scientific contracts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gre_synthetic_wave import (  # noqa: E402
    COARSE_LAMBDAS,
    LLR_BLOCK_SIZES,
    NATIVE_MATRIX,
    apply_sampling_mask,
    bart_wave_restoration_factor,
    build_case_mask,
    build_wave_command,
    case_definitions,
    circular_phase_metrics,
    coarse_candidate_settings,
    completed_manifest_reusable,
    crop_native_for_case,
    crop_source_to_native,
    expected_mask_records,
    fit_shared_echo1_scale,
    inter_echo_metrics,
    json_sha256,
    refinement_points,
    restore_bart_normalization,
    theoretical_psf,
    validate_config_document,
    validate_echo_counters,
    validate_geometry_contract,
)
from gre_synthetic_wave_sweep import (  # noqa: E402
    GRE_AFFINE_AXIS_FLIPS,
    GRE_BART_OUTPUT_CONVENTION_VERSION,
    GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION,
    GRE_LOGICAL_AXIS_ROLES,
    GRE_NIFTI_EXPORT_VERSION,
    GRE_PRIOR_MASK_ANTERIOR_D1_CANDIDATE_ID,
    GRE_PRIOR_MASK_CANDIDATE_ID,
    _apply_orientation,
    _candidate_nifti_arrays,
    _dilate_mask_anterior,
    _fsl_runtime_environment,
    _global_echo_metrics,
    _invert_full_wave_encoding,
    _map_mask_to_lr,
    _normalize_magnitude_for_display,
    _scaling_matches_current_bart_convention,
    _validated_explicit_refinement_settings,
    build_parser,
)
from wave_retro_lr.core import apply_wave_forward  # noqa: E402


class GreGeometryAndSamplingTests(unittest.TestCase):
    """Verify exact native/LR geometry, crops, masks, and hashes."""

    def test_geometry_and_centered_crop_contracts(self) -> None:
        """Require measured-Wave native geometry and the approved LIN crop."""

        geometry = validate_geometry_contract()
        self.assertEqual(geometry["source_to_native_crop_bounds"], [[3, 253], [3, 253], [0, 72]])
        self.assertEqual(geometry["native_matrix_ro_lin_par"], [250, 250, 72])
        low = geometry["cases"]["lin_low_resolution_r3x2"]
        self.assertEqual(low["matrix_ro_lin_par"], [250, 148, 72])
        self.assertEqual(low["crop_bounds_from_native"][1], [51, 199])
        self.assertAlmostEqual(low["voxel_mm_ro_lin_par"][1], 220 / 148)

    def test_source_and_lr_crops_change_kspace_without_interpolation(self) -> None:
        """Require exact NumPy slicing for both logical k-space reductions."""

        source = np.arange(256 * 256 * 72, dtype=np.float32).reshape(256, 256, 72)
        native = crop_source_to_native(source)
        self.assertEqual(native.shape, NATIVE_MATRIX)
        self.assertEqual(native[0, 0, 0], source[3, 3, 0])
        low = crop_native_for_case(native, case_definitions()["lin_low_resolution_r3x2"])
        self.assertEqual(low.shape, (250, 148, 72))
        self.assertEqual(low[0, 0, 0], source[3, 54, 0])

    def test_echo_extraction_binds_counters_to_ten_and_twenty_ms(self) -> None:
        """Separate identical PE coordinate grids using the TWIX Eco counter."""

        coordinates = [(line, partition) for line in range(5) for partition in range(3)]
        lines = [value[0] for value in coordinates] * 2
        partitions = [value[1] for value in coordinates] * 2
        echoes = [0] * len(coordinates) + [1] * len(coordinates)
        records = validate_echo_counters(
            lines,
            partitions,
            echoes,
            matrix_lin_par=(5, 3),
            echo_times_s=(0.010, 0.020),
        )
        self.assertEqual([record["te_s"] for record in records], [0.010, 0.020])
        echoes[-1] = 0
        with self.assertRaisesRegex(ValueError, "duplicate-free"):
            validate_echo_counters(
                lines,
                partitions,
                echoes,
                matrix_lin_par=(5, 3),
                echo_times_s=(0.010, 0.020),
            )

    def test_pure_mask_counts_and_hashes_are_exact(self) -> None:
        """Bind all case masks to exact counts and canonical logical hashes."""

        records = expected_mask_records()
        expected = {
            "native_r3x1": (5976, "5cdabbf2c6c10fff65201052a145604db143e3afd8a261f09e560b8f49d9b32b"),
            "native_r3x2": (2988, "1449e149fe9c38b1530712f4b124f7ff47c128d62573ab68ad74e4676f7726cd"),
            "lin_low_resolution_r3x2": (1764, "2d508bf3349b39e7649dda9f7d141c4444f5ca428a2e8d9293beb900b556f0da"),
        }
        for case_id, (count, digest) in expected.items():
            self.assertEqual(records[case_id]["acquired_coordinate_count"], count)
            self.assertEqual(records[case_id]["logical_sha256"], digest)
            self.assertFalse(records[case_id]["acs_coordinates_included"])

    def test_acs_union_metadata_is_rejected(self) -> None:
        """Reject historical reconstruction masks containing calibration coordinates."""

        case = case_definitions()["native_r3x1"]
        mask, metadata = build_case_mask(case)
        metadata["mask_kind"] = "cartesian_with_full_pe1_acs"
        from wave_retro_lr.sampling import validate_pure_cartesian_image_lattice

        with self.assertRaisesRegex(ValueError, "Historical ACS-union"):
            validate_pure_cartesian_image_lattice(mask, metadata)

    def test_mask_application_preserves_only_acquired_samples(self) -> None:
        """Verify bitwise acquired equality, outside zeros, and finite output."""

        rng = np.random.default_rng(7)
        full = (rng.normal(size=(9, 10, 6, 2)) + 1j * rng.normal(size=(9, 10, 6, 2))).astype(np.complex64)
        mask = np.zeros((10, 6), dtype=bool)
        mask[2::3, ::2] = True
        output, checks = apply_sampling_mask(full, mask)
        self.assertTrue(all(checks.values()))
        np.testing.assert_array_equal(output[:, mask], full[:, mask])
        self.assertEqual(np.count_nonzero(output[:, ~mask]), 0)


class GreOperatorAndCommandTests(unittest.TestCase):
    """Verify echo-specific theoretical PSFs and explicit BART command syntax."""

    def test_theoretical_psf_has_target_geometry_and_unit_magnitude(self) -> None:
        """Evaluate a deterministic 1000-sample trajectory on native and LR grids."""

        ky = np.linspace(-2, 2, 1000)
        kz = np.linspace(1, -1, 1000)
        native = theoretical_psf(ky, kz, nlin=250, npar=72, yflip=1, zflip=1)
        low = theoretical_psf(ky, kz, nlin=148, npar=72, yflip=1, zflip=1)
        self.assertEqual(native.shape, (1000, 250, 72))
        self.assertEqual(low.shape, (1000, 148, 72))
        self.assertLess(float(np.max(np.abs(np.abs(native) - 1))), 2e-7)

    def test_full_sampling_operator_roundtrip(self) -> None:
        """Require exact inversion for identity and nontrivial unit-magnitude PSFs."""

        rng = np.random.default_rng(12)
        no_wave = (rng.normal(size=(8, 6, 4)) + 1j * rng.normal(size=(8, 6, 4))).astype(np.complex64)
        psf = np.exp(1j * rng.normal(size=(12, 6, 4))).astype(np.complex64)
        encoded = apply_wave_forward(no_wave, psf, readout_oversampled=12)
        recovered = _invert_full_wave_encoding(encoded, psf, logical_readout=8, workers=1)
        self.assertLess(float(np.linalg.norm(recovered - no_wave) / np.linalg.norm(no_wave)), 1e-6)

    def test_coarse_grid_has_one_control_and_explicit_llr_blocks(self) -> None:
        """Require 21 settings per group and no implicit LLR block size."""

        settings = coarse_candidate_settings()
        self.assertEqual(len(settings), 21)
        self.assertEqual(settings[0], {"method": "fista_lambda0", "lambda": 0.0, "block_size": None})
        llr = [value for value in settings if value["method"] == "llr"]
        self.assertEqual({value["block_size"] for value in llr}, set(LLR_BLOCK_SIZES))
        self.assertEqual({value["lambda"] for value in llr}, set(COARSE_LAMBDAS))

    def test_commands_are_explicit_gpu_fista_wavelet_and_split_llr(self) -> None:
        """Bind commands to mandatory GPU, 100 iterations, and 1e-6 tolerance."""

        common = {"maps": "maps", "psf": "psf", "kspace": "kspace", "output": "out"}
        control = build_wave_command("bart", coarse_candidate_settings()[0], **common)
        wavelet = build_wave_command("bart", {"method": "wavelet", "lambda": 1e-4, "block_size": None}, **common)
        llr = build_wave_command("bart", {"method": "llr", "lambda": 1e-3, "block_size": 16}, **common)
        self.assertEqual(control[2:8], ["-g", "-w", "-f", "-r", "0", "-i"])
        self.assertIn("-g", wavelet)
        self.assertEqual(llr[2:10], ["-g", "-l", "-v", "-b", "16", "-f", "-r", "0.001"])
        for command in (control, wavelet, llr):
            self.assertIn("100", command)
            self.assertIn("1e-6", command)

    def test_refinement_uses_logarithmic_trisection(self) -> None:
        """Require two ordered interior points in one adjacent log bracket."""

        first, second = refinement_points(1e-4, 1e-3)
        self.assertAlmostEqual(first, 1e-4 * 10 ** (1 / 3))
        self.assertAlmostEqual(second, 1e-4 * 10 ** (2 / 3))

    def test_explicit_wavelet_refinement_accepts_extended_increasing_grid(self) -> None:
        """Allow a reviewed fine grid to extend beyond the coarse maximum."""

        lambdas = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
        settings = _validated_explicit_refinement_settings(
            [{"method": "wavelet", "block_size": None, "lambdas": lambdas}],
            case_id="native_r3x1",
            echo_id="echo-01",
        )
        self.assertEqual([setting["lambda"] for setting in settings], lambdas)
        self.assertTrue(all(setting["method"] == "wavelet" for setting in settings))
        with self.assertRaisesRegex(ValueError, "unique, and increasing"):
            _validated_explicit_refinement_settings(
                [{"method": "wavelet", "block_size": None, "lambdas": [0.01, 0.005]}],
                case_id="native_r3x1",
                echo_id="echo-01",
            )


class GreScalingPhaseAndManifestTests(unittest.TestCase):
    """Verify scaling, phase, inter-echo, resume, and local-config contracts."""

    def test_bart_norm_restoration_and_shared_scale(self) -> None:
        """Restore the fixed BART scale/phase and fit a separate test scalar."""

        normalized = np.full((2, 4, 2), -1j, dtype=np.complex64)
        encoding_shape = (8, 4, 2)
        factor = bart_wave_restoration_factor(normalized.shape, 3.0, encoding_shape)
        self.assertEqual(GRE_BART_OUTPUT_CONVENTION_VERSION, 2)
        self.assertEqual(factor, 24j)
        restored = restore_bart_normalization(normalized, 3.0, encoding_shape)
        np.testing.assert_allclose(restored, np.full(normalized.shape, 24.0))
        reference = np.ones((3, 3, 3), dtype=np.complex64) * 6
        candidate = np.ones_like(reference) * 2
        self.assertAlmostEqual(fit_shared_echo1_scale(reference, candidate), 3.0)

    def test_scaling_audit_accepts_only_exact_pre_marker_records(self) -> None:
        """Accept legacy provenance only when every scientific field is unchanged."""

        expected = {
            "convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
            "policy": "fixed",
            "kspace_l2_norm": 0.25,
            "factor_real": 0.0,
            "factor_imaginary": -100.0,
            "candidate_specific_fit": False,
        }
        current_match, current_basis = _scaling_matches_current_bart_convention(expected, expected)
        self.assertTrue(current_match)
        self.assertEqual(current_basis, "explicit convention marker")
        legacy = dict(expected)
        legacy.pop("convention_version")
        legacy_match, legacy_basis = _scaling_matches_current_bart_convention(legacy, expected)
        self.assertTrue(legacy_match)
        self.assertEqual(legacy_basis, "exact legacy scaling audit")
        legacy["factor_imaginary"] = -99.0
        stale_match, stale_basis = _scaling_matches_current_bart_convention(legacy, expected)
        self.assertFalse(stale_match)
        self.assertEqual(stale_basis, "mismatch")

    def test_mprage_style_display_normalization_maps_positive_p99_to_one(self) -> None:
        """Make viewable magnitude data while recording reversible p99 scaling."""

        magnitude = np.arange(1, 101, dtype=np.float32).reshape(4, 5, 5)
        normalized, record = _normalize_magnitude_for_display(magnitude)
        self.assertAlmostEqual(float(np.percentile(normalized[normalized > 0], 99)), 1.0, places=6)
        self.assertEqual(record["Method"], "positive-finite-percentile")
        self.assertEqual(record["Percentile"], 99.0)
        np.testing.assert_allclose(normalized * record["InputPercentileValue"], magnitude)

    def test_phase_and_delta_b0_metrics_preserve_known_error(self) -> None:
        """Recover a known global phase and wrapped inter-echo frequency error."""

        shape = (4, 4, 4)
        support = np.ones(shape, dtype=bool)
        reference1 = np.ones(shape, dtype=np.complex64)
        reference2 = np.ones(shape, dtype=np.complex64)
        candidate1 = reference1 * np.exp(1j * 0.2)
        candidate2 = reference2 * np.exp(1j * (0.2 + 2 * np.pi * 0.01 * 5))
        phase = circular_phase_metrics(reference1, candidate1, support)
        pair = inter_echo_metrics(reference1, reference2, candidate1, candidate2, support, delta_te_s=0.01)
        self.assertAlmostEqual(phase["circular_mean_error_rad"], 0.2, places=6)
        self.assertAlmostEqual(pair["wrapped_delta_b0_bias_hz"], 5.0, places=5)

    def test_global_echo_nrmse_pools_energy_before_division(self) -> None:
        """Require joint echo errors and reference energies before NRMSE division."""

        coordinates = np.indices((4, 4, 4)).sum(axis=0).astype(np.float32)
        reference1 = (1.0 + 0.1 * coordinates).astype(np.complex64)
        reference2 = (2.0 + 0.2 * coordinates).astype(np.complex64)
        candidate1 = reference1 * np.complex64(1.1 * np.exp(0.1j))
        candidate2 = reference2 * np.complex64(0.9 * np.exp(0.2j))
        mask = np.ones(reference1.shape, dtype=bool)
        metrics = _global_echo_metrics(
            [reference1, reference2],
            [candidate1, candidate2],
            [mask, mask],
            [mask, mask],
            [
                {"ssim_brain": 0.91, "normalized_acquired_data_residual": 0.1},
                {"ssim_brain": 0.89, "normalized_acquired_data_residual": 0.2},
            ],
            [3.0, 4.0],
        )
        self.assertAlmostEqual(metrics["global_magnitude_nrmse_brain"], 0.1, places=6)
        self.assertAlmostEqual(metrics["mean_echo_ssim_brain"], 0.9, places=6)
        self.assertAlmostEqual(
            metrics["global_normalized_acquired_data_residual"],
            np.sqrt(((0.1 * 3.0) ** 2 + (0.2 * 4.0) ** 2) / 25.0),
            places=6,
        )
        self.assertAlmostEqual(metrics["global_p95_absolute_phase_error_rad"], 0.2, places=6)

    def test_resume_requires_status_signature_and_output_hash(self) -> None:
        """Reject stale or merely existing reconstruction outputs."""

        manifest = {"status": "complete", "signature_sha256": "abc", "output": {"payload_sha256": "def"}}
        self.assertTrue(completed_manifest_reusable(manifest, "abc", "def"))
        self.assertFalse(completed_manifest_reusable(manifest, "changed", "def"))
        self.assertFalse(completed_manifest_reusable(manifest, "abc", "changed"))

    def test_brain_mask_mapping_preserves_physical_center_and_binary_values(self) -> None:
        """Map the approved native mask to 148 LIN samples with nearest neighbors."""

        import nibabel as nib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "native.nii.gz"
            target_path = root / "low.nii.gz"
            affine = np.diag([0.88, 0.88, 2.5, 1.0])
            mask = np.zeros((250, 250, 72), dtype=np.uint8)
            mask[50:200, 40:210, 10:65] = 1
            nib.save(nib.Nifti1Image(mask, affine), source_path)
            record = _map_mask_to_lr(source_path, target_path)
            mapped = nib.load(str(target_path))
            self.assertEqual(mapped.shape, (250, 148, 72))
            self.assertEqual(set(np.unique(np.asarray(mapped.dataobj))), {0, 1})
            self.assertEqual(record["voxel_count"], int(np.count_nonzero(mapped.dataobj)))

    def test_fsl_runtime_is_resolved_from_site_bet_wrapper(self) -> None:
        """Resolve a complete FSLDIR instead of invoking BET with an empty root."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fsl"
            for relative in (
                "bin/bet",
                "bin/remove_ext",
                "bin/bet2",
                "bin/fslstats",
                "bin/fast",
                "bin/fslmaths",
                "bin/standard_space_roi",
                "etc/fslconf/fsl.sh",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            (root / "data/standard").mkdir(parents=True)
            wrapper = Path(temporary) / "site/share/fsl/bin/bet"
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text(f'#!/usr/bin/env bash\n{root}/bin/bet "$@"\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {"FSLDIR": ""}):
                environment, provenance = _fsl_runtime_environment(str(wrapper))
            self.assertEqual(environment["FSLDIR"], str(root))
            self.assertEqual(environment["FSLOUTPUTTYPE"], "NIFTI_GZ")
            self.assertEqual(provenance["fsldir"], str(root))
            self.assertIn(str(root / "bin"), environment["PATH"])

    def test_gre_orientation_is_lossless_and_sequence_specific(self) -> None:
        """Keep GRE direction correction separate and invertible in logical space."""

        import nibabel as nib

        self.assertEqual(GRE_LOGICAL_AXIS_ROLES, ("readout", "phase", "slice"))
        self.assertEqual(GRE_AFFINE_AXIS_FLIPS, (False, True, False))
        logical_affine = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        logical_orientation = nib.orientations.io_orientation(logical_affine)
        ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
        forward = nib.orientations.ornt_transform(logical_orientation, ras_orientation)
        inverse = nib.orientations.ornt_transform(ras_orientation, logical_orientation)
        logical = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        canonical = _apply_orientation(logical, forward)
        restored = _apply_orientation(canonical, inverse)
        self.assertEqual(canonical.shape, (4, 3, 5))
        np.testing.assert_array_equal(restored, logical)

    def test_brain_mask_grid_and_approval_are_explicit(self) -> None:
        """Require an approved prior-mask source and explicit GRE approval."""

        self.assertEqual(GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION, 4)
        self.assertEqual(GRE_PRIOR_MASK_CANDIDATE_ID, "prior_approved_mprage_f0p59_d1")
        self.assertEqual(
            GRE_PRIOR_MASK_ANTERIOR_D1_CANDIDATE_ID,
            "prior_approved_mprage_f0p59_d1_anterior-d1",
        )
        arguments = build_parser().parse_args(
            [
                "--config",
                "local.json",
                "approve-brain-mask",
                "--brain-mask-candidate",
                GRE_PRIOR_MASK_CANDIDATE_ID,
            ]
        )
        self.assertEqual(arguments.brain_mask_candidate, GRE_PRIOR_MASK_CANDIDATE_ID)
        source_arguments = build_parser().parse_args(
            [
                "--config",
                "local.json",
                "prepare-brain-mask",
                "--brain-mask-source-manifest",
                "approved-mask/manifest.json",
            ]
        )
        self.assertEqual(
            source_arguments.brain_mask_source_manifest,
            Path("approved-mask/manifest.json"),
        )

    def test_anterior_mask_dilation_changes_only_positive_ras_y(self) -> None:
        """Expand one voxel only toward anterior on a canonical-RAS grid."""

        mask = np.zeros((5, 6, 7), dtype=bool)
        mask[2, 3, 4] = True
        expanded = _dilate_mask_anterior(mask)
        expected = mask.copy()
        expected[2, 4, 4] = True
        np.testing.assert_array_equal(expanded, expected)
        self.assertEqual(int(expanded.sum()), 2)

    def test_candidate_nifti_export_is_canonical_float32_magnitude_and_phase(self) -> None:
        """Export restored logical complex data with the recorded RAS transform."""

        logical = np.array(
            [
                [[1 + 0j, 1j], [-1 + 0j, -1j]],
                [[2 + 0j, 2j], [-2 + 0j, -2j]],
                [[3 + 0j, 3j], [-3 + 0j, -3j]],
            ],
            dtype=np.complex64,
        )
        transform = np.array([[1.0, 1.0], [0.0, 1.0], [2.0, -1.0]])
        expected = _apply_orientation(logical, transform)
        orientation = {
            "logical_to_canonical_ras_transform": transform.tolist(),
            "canonical_ras_shape": list(expected.shape),
        }
        magnitude, phase = _candidate_nifti_arrays(logical, orientation)
        self.assertEqual(GRE_NIFTI_EXPORT_VERSION, 3)
        self.assertEqual(magnitude.dtype, np.float32)
        self.assertEqual(phase.dtype, np.float32)
        np.testing.assert_allclose(magnitude, np.abs(expected))
        np.testing.assert_allclose(phase, np.angle(expected))
        arguments = build_parser().parse_args(
            ["--config", "local.json", "export-nifti", "--sweep", "coarse"]
        )
        self.assertEqual(arguments.operation, "export-nifti")
        self.assertEqual(arguments.sweep, "coarse")

    def test_local_config_contract_and_job_count(self) -> None:
        """Validate the approved immutable geometry, GPU, and sweep settings."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "format_version": 1,
                "workflow": "synthetic_wave_gre_regularization_sweep",
                "output_parent": str(root),
                "run_name": "gre_sweep",
                "geometry": {
                    "source_matrix_ro_lin_par": [256, 256, 72],
                    "native_matrix_ro_lin_par": [250, 250, 72],
                    "fov_mm_ro_lin_par": [220.0, 220.0, 180.0],
                    "extended_wave_readout": 1000,
                    "low_resolution_matrix_ro_lin_par": [250, 148, 72],
                },
                "sampling": {"mask_kind": "pure_cartesian_image_lattice", "residue_lin_par": [2, 0]},
                "coil_compression": {
                    "physical_coils": 44,
                    "virtual_coils": 12,
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
                "sweep": {
                    "wavelet_lambdas": list(COARSE_LAMBDAS),
                    "llr_lambdas": list(COARSE_LAMBDAS),
                    "llr_blocks": list(LLR_BLOCK_SIZES),
                    "iterations": 100,
                    "tolerance": 1e-6,
                },
                "runtime": {"backend": "gpu", "fft_workers": 4},
            }
            validated = validate_config_document(config)
            self.assertEqual(validated["coarse_job_count"], 126)
            config["runtime"]["backend"] = "cpu"
            with self.assertRaisesRegex(ValueError, "fixed to GPU"):
                validate_config_document(config)

    def test_tracked_examples_do_not_contain_private_paths(self) -> None:
        """Require placeholder-only tracked GRE configuration and shell examples."""

        examples = [
            SCRIPT_ROOT.parent / "configs" / "gre_synthetic_wave_sweep.example.json",
            SCRIPT_ROOT / "run_gre_synthetic_wave_sweep.example.sh",
        ]
        forbidden = ("/homes/", "/autofs/", "yd918", "MID00", "FID")
        for path in examples:
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(token in text for token in forbidden), path)

    def test_json_signature_is_order_independent(self) -> None:
        """Canonicalize manifest signature mappings before hashing."""

        self.assertEqual(json_sha256({"a": 1, "b": 2}), json_sha256({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
