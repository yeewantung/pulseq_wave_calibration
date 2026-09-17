"""Tests for the staged, review-gated MPRAGE ROVir feasibility workflow."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.rovir_feasibility import (  # noqa: E402
    _validate_four_region_masks,
    approve_region_mask_candidate,
    derive_region_mask_candidates,
    derive_ro_partition_mask_candidate,
    export_manual_roi_annotation_nifti,
    export_mprage_physical_calibration,
    prepare_masked_rovir_inputs,
    record_calibration_images,
    validate_manual_roi_annotation,
    write_rovir_transform_qc,
)


class RovirFeasibilityTests(unittest.TestCase):
    """Exercise the non-ranking workflow with small synthetic BART arrays."""

    def test_manual_roi_nifti_export_and_review_validation(self) -> None:
        """Round-trip reviewed labels without resampling or affine drift.

        Returns:
            None.
        """
        import nibabel as nib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_calibration_inputs(root)
            version = root / "bart_version.txt"
            version.write_text("v1.0-test\n", encoding="utf-8")
            record_calibration_images(root, version)
            source_affine = np.asarray(
                [
                    [0.0, 0.0, -2.0, 7.0],
                    [0.0, 3.0, 0.0, -9.0],
                    [1.0, 0.0, 0.0, -5.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            def apply_flips(images: object, flips: object) -> list[np.ndarray]:
                """Apply the fixture's requested logical-axis flips.

                Args:
                    images: Iterable of matched arrays.
                    flips: Iterable of axis flip flags.

                Returns:
                    Contiguous flipped arrays.
                """
                outputs = []
                for image in images:
                    corrected = np.asarray(image)
                    for axis, should_flip in enumerate(flips):
                        if should_flip:
                            corrected = np.flip(corrected, axis=axis)
                    outputs.append(np.ascontiguousarray(corrected))
                return outputs

            def canonicalize(
                images: object, affine: np.ndarray
            ) -> tuple[list[np.ndarray], np.ndarray, list[list[float]]]:
                """Canonicalize fixture arrays through nibabel orientation APIs.

                Args:
                    images: Iterable of matched arrays.
                    affine: Source voxel-to-RAS affine.

                Returns:
                    Canonical arrays, affine, and orientation transform.
                """
                source = nib.orientations.io_orientation(affine)
                target = nib.orientations.axcodes2ornt(("R", "A", "S"))
                transform = nib.orientations.ornt_transform(source, target)
                arrays = [
                    np.ascontiguousarray(
                        nib.orientations.apply_orientation(image, transform)
                    )
                    for image in images
                ]
                canonical_affine = affine @ nib.orientations.inv_ornt_aff(
                    transform, np.asarray(next(iter(images))).shape
                )
                return arrays, canonical_affine, transform.tolist()

            helper = SimpleNamespace(
                make_nifti_affine_from_twix=lambda **_: (
                    source_affine,
                    (1.0, 3.0, 2.0),
                    {"fixture": True},
                ),
                apply_array_axis_flips=apply_flips,
                canonicalize_arrays_to_ras=canonicalize,
            )
            with patch(
                "wave_retro_lr.mprage.load_wave_mprage_helpers",
                return_value=helper,
            ):
                manifest = export_manual_roi_annotation_nifti(root)
            self.assertFalse(manifest["automatic_roi_detection"])
            reference = nib.load(manifest["reference_nifti"]["path"])
            template = nib.load(manifest["label_template_nifti"]["path"])
            self.assertEqual(reference.shape, template.shape)
            np.testing.assert_allclose(reference.affine, template.affine)
            self.assertEqual(nib.aff2axcodes(reference.affine), ("R", "A", "S"))

            labels = np.zeros(template.shape, dtype=np.uint8)
            labels.flat[:8] = 1
            labels.flat[8:16] = 2
            reviewed = (
                root
                / "masks"
                / "manual_annotation"
                / "reviewed"
                / "rovir_roi_labels_reviewed.nii.gz"
            )
            reviewed.parent.mkdir()
            nib.save(nib.Nifti1Image(labels, template.affine), reviewed)
            validation = validate_manual_roi_annotation(root)
            self.assertEqual(validation["label_counts"]["1"], 8)
            self.assertEqual(validation["label_counts"]["2"], 8)
            self.assertEqual(
                validation["source_bart_shape_ro_lin_par"], [12, 8, 8]
            )
            self.assertFalse(validation["inverse_orientation_used_resampling"])
            self.assertFalse(validation["approved"])

            bad_affine = template.affine.copy()
            bad_affine[0, 3] += 1.0
            bad = reviewed.with_name("bad_affine.nii.gz")
            nib.save(nib.Nifti1Image(labels, bad_affine), bad)
            with self.assertRaisesRegex(ValueError, "affine"):
                validate_manual_roi_annotation(root, bad)

    def test_four_region_contract_rejects_ambiguous_estimation_support(self) -> None:
        """Keep mixed in-head signal out of both ROVir estimation inputs.

        Returns:
            None.
        """
        preservation = np.zeros((5, 5, 5), dtype=np.float32)
        preservation[1:4, 1:4, 1:4] = 1
        positive = np.zeros_like(preservation)
        positive[1, 1, 1] = 1
        negative = np.zeros_like(preservation)
        negative[4, 4, 4] = 1
        holdout = np.zeros_like(preservation)
        holdout[3, 3, 3] = 1
        validation = _validate_four_region_masks(
            preservation, positive, negative, holdout
        )
        self.assertTrue(validation["negative_outside_preservation"])
        self.assertTrue(validation["holdout_excluded_from_estimation"])

        invalid_negative = negative.copy()
        invalid_negative[2, 2, 2] = 1
        with self.assertRaisesRegex(ValueError, "outside preservation"):
            _validate_four_region_masks(
                preservation, positive, invalid_negative, holdout
            )

        invalid_positive = positive.copy()
        invalid_positive[3, 3, 3] = 1
        with self.assertRaisesRegex(ValueError, "holdout must be disjoint"):
            _validate_four_region_masks(
                preservation, invalid_positive, negative, holdout
            )

    def test_exact_ro_partition_supports_empty_holdout_and_two_region_qc(self) -> None:
        """Bind an inclusive RO slab to complementary solver regions.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_calibration_inputs(root)
            version = root / "logs" / "bart_version.txt"
            version.parent.mkdir()
            version.write_text("v1.0.00-test\n", encoding="utf-8")
            record_calibration_images(root, version)
            candidates = derive_ro_partition_mask_candidate(root, 2)
            self.assertEqual(candidates["format_version"], 2)
            self.assertEqual(len(candidates["candidates"]), 1)
            candidate = candidates["candidates"][0]
            self.assertEqual(candidate["candidate_id"], "negative_ro000_002")
            candidate_root = root / "masks" / "candidates" / candidate["candidate_id"]
            negative = np.asarray(
                open_cfl(candidate_root / "negative_estimation_mask")
            ).real
            positive = np.asarray(
                open_cfl(candidate_root / "positive_estimation_mask")
            ).real
            holdout = np.asarray(
                open_cfl(candidate_root / "contaminated_holdout_mask")
            ).real
            self.assertTrue(np.all(negative[:3] == 1))
            self.assertTrue(np.all(negative[3:] == 0))
            np.testing.assert_array_equal(positive, 1 - negative)
            self.assertEqual(np.count_nonzero(holdout), 0)
            self.assertEqual(candidate["validation"]["contaminated_holdout_voxels"], 0)

            approve_region_mask_candidate(root, candidate["candidate_id"])
            prepare_masked_rovir_inputs(root, readout_chunk=2)
            transform_directory = root / "transforms" / "rovir_full"
            transform_directory.mkdir(parents=True)
            transform = create_cfl(
                transform_directory / "transform", (1, 1, 1, 2, 2)
            )
            transform[...] = 0
            transform[0, 0, 0, :, :] = np.eye(2, dtype=np.complex64)
            transform.flush()
            del transform
            qc = write_rovir_transform_qc(root, version)
            self.assertIsNone(
                qc["region_curves"]["contaminated_holdout_vs_pure_negative"]
            )
            self.assertTrue(
                (root / "diagnostics" / "region_curves" / "rovir_two_region_curves.csv").is_file()
            )
            self.assertTrue(
                (root / "diagnostics" / "region_curves" / "rovir_two_region_curves.png").is_file()
            )

    def test_reviewed_masks_inputs_and_transform_qc(self) -> None:
        """Require explicit approval and retain BART-only solver provenance.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_calibration_inputs(root)
            version = root / "logs" / "bart_version.txt"
            version.parent.mkdir()
            version.write_text("v1.0.00-test\n", encoding="utf-8")
            images_manifest = record_calibration_images(root, version)
            self.assertEqual(
                images_manifest["status"], "mprage_rovir_calibration_images_ready"
            )
            self.assertEqual(
                len(images_manifest["physical_set4_rss_slice_montages"]), 3
            )

            config = root / "mask_candidates.local.json"
            config.write_text(
                json.dumps(
                    {
                        "scientific_status": "ready_for_visual_review",
                        "candidates": [
                            {
                                "candidate_id": "separated_regions",
                                "minimum_safety_gap_voxels": 2,
                                "preservation": {
                                    "minimum_relative_rss": 0,
                                    "ellipsoids": [
                                        {
                                            "center_normalized": [-0.6, 0, 0],
                                            "radii_normalized": [0.3, 0.45, 0.45],
                                        },
                                        {
                                            "center_normalized": [0, 0, 0],
                                            "radii_normalized": [0.2, 0.3, 0.3],
                                        }
                                    ],
                                },
                                "positive_estimation": {
                                    "minimum_relative_rss": 0,
                                    "ellipsoids": [
                                        {
                                            "center_normalized": [-0.6, 0, 0],
                                            "radii_normalized": [0.3, 0.45, 0.45],
                                        }
                                    ],
                                },
                                "negative_estimation": {
                                    "minimum_relative_rss": 0,
                                    "ellipsoids": [
                                        {
                                            "center_normalized": [0.75, 0, 0],
                                            "radii_normalized": [0.15, 0.3, 0.3],
                                        }
                                    ],
                                },
                                "contaminated_holdout": {
                                    "minimum_relative_rss": 0,
                                    "ellipsoids": [
                                        {
                                            "center_normalized": [0, 0, 0],
                                            "radii_normalized": [0.2, 0.3, 0.3],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidates = derive_region_mask_candidates(root, config)
            self.assertFalse(candidates["automatic_selection"])
            candidate = candidates["candidates"][0]
            self.assertEqual(len(candidate["review_slice_montages"]), 3)
            for name in (
                "preservation_mask",
                "positive_estimation_mask",
                "negative_estimation_mask",
                "contaminated_holdout_mask",
            ):
                self.assertTrue(Path(candidate[name]["base"] + ".cfl").is_file())
            approval = approve_region_mask_candidate(root, "separated_regions")
            self.assertEqual(approval["candidate_id"], "separated_regions")

            inputs = prepare_masked_rovir_inputs(root, readout_chunk=2)
            self.assertEqual(inputs["solver_backend"], "bart rovir only")
            self.assertFalse(inputs["bart_launched"])

            transform_directory = root / "transforms" / "rovir_full"
            transform_directory.mkdir(parents=True)
            transform = create_cfl(
                transform_directory / "transform", (1, 1, 1, 2, 2)
            )
            transform[...] = 0
            transform[0, 0, 0, :, :] = np.eye(2, dtype=np.complex64)
            transform.flush()
            del transform
            qc = write_rovir_transform_qc(root, version)
            self.assertEqual(qc["solver_backend"], "bart rovir only")
            self.assertIsNone(qc["selected_virtual_coils"])
            self.assertFalse(qc["automatic_selection"])
            self.assertEqual(
                len(
                    qc["region_curves"][
                        "solver_clean_positive_vs_pure_negative"
                    ]["channel_counts"]
                ),
                2,
            )
            self.assertTrue(
                qc["region_curves"]["holdout_energy_is_not_anatomy_specific"]
            )
            self.assertTrue(
                (
                    root
                    / "diagnostics"
                    / "region_curves"
                    / "rovir_four_region_curves.csv"
                ).is_file()
            )

    def test_unreviewed_mask_template_is_rejected(self) -> None:
        """Prevent the tracked illustrative mask template from being executed.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_calibration_inputs(root)
            version = root / "bart_version.txt"
            version.write_text("v1.0-test\n", encoding="utf-8")
            record_calibration_images(root, version)
            config = root / "template.json"
            config.write_text(
                json.dumps(
                    {
                        "scientific_status": "template_only_do_not_run",
                        "candidates": [{}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "template"):
                derive_region_mask_candidates(root, config)

    def test_set4_acs_is_centered_without_changing_acquired_samples(self) -> None:
        """Export only packed set 4 and preserve it during center embedding.

        Returns:
            None.
        """
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            twix = base / "source.dat"
            sequence = base / "source.seq"
            twix.write_bytes(b"small synthetic twix identity")
            sequence.write_text("synthetic sequence identity\n", encoding="utf-8")
            normal = base / "normal_root"
            manifest_path = normal / "normal" / "bart_inputs" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            twix_stat = twix.stat()
            sequence_stat = sequence.stat()
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": {
                            "twix": {
                                "path": str(twix),
                                "size_bytes": twix_stat.st_size,
                                "mtime_ns": twix_stat.st_mtime_ns,
                            },
                            "sequence": {
                                "path": str(sequence),
                                "size_bytes": sequence_stat.st_size,
                                "mtime_ns": sequence_stat.st_mtime_ns,
                                "sha256": sha256_file(sequence),
                            },
                        },
                        "geometry": {
                            "logical_matrix_ro_lin_par": [4, 8, 6],
                            "readout_oversampling_factor": 2,
                            "physical_fov_mm_xyz": [4, 8, 6],
                        },
                        "coil_compression": {
                            "physical_coils": 2,
                            "virtual_coils": 2,
                            "method": "fixture",
                        },
                        "psf_calibration": {"ncalib": 4, "nacs": 2},
                    }
                ),
                encoding="utf-8",
            )
            reference = torch.zeros((8, 4, 4, 5, 2), dtype=torch.complex64)
            packed = torch.arange(8 * 2 * 2 * 2, dtype=torch.float32).reshape(
                8, 2, 2, 2
            )
            reference[:, :2, :2, 4, :] = packed.to(torch.complex64) + 1j
            output = base / "feasibility"
            calls: list[tuple[tuple[int, ...], int, int]] = []

            def remove_readout_oversampling_kspace(
                values: object, factor: int, axis: int = 0
            ) -> object:
                """Return a distinguishable logical-grid fixture.

                Args:
                    values: Oversampled physical-coil ACS tensor.
                    factor: Required readout oversampling factor.
                    axis: Required readout axis.

                Returns:
                    Logical-grid tensor used to verify exact center embedding.
                """
                tensor = values
                calls.append((tuple(tensor.shape), factor, axis))
                return tensor[1::factor].contiguous()

            helper = SimpleNamespace(
                load_ref=lambda _: reference.clone(),
                remove_readout_oversampling_kspace=(
                    remove_readout_oversampling_kspace
                ),
            )
            with patch(
                "wave_retro_lr.mprage.load_wave_mprage_helpers",
                return_value=helper,
            ):
                manifest = export_mprage_physical_calibration(
                    twix, sequence, normal, output
                )
            exported = np.asarray(
                open_cfl(
                    output
                    / "inputs"
                    / "physical_calibration"
                    / "physical_set4_kspace"
                )
            )
            expected = reference[1::2, :2, :2, 4, :].numpy()
            np.testing.assert_array_equal(exported[:, 1:3, 1:3, :], expected)
            outside = exported.copy()
            outside[:, 1:3, 1:3, :] = 0
            self.assertEqual(np.count_nonzero(outside), 0)
            contract = manifest["refscan_contract"]
            self.assertEqual(calls, [((8, 2, 2, 2), 2, 0)])
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(
                contract["readout_oversampling_removal"],
                {
                    "method": "centered-image-domain-crop",
                    "version": 1,
                    "fft_normalization": "ortho",
                    "oversampling_factor": 2,
                    "input_readout": 8,
                    "output_readout": 4,
                },
            )
            self.assertNotIn("readout_oversampling_removed_by_stride", contract)
            self.assertTrue(contract["acquired_sample_equality"])
            self.assertTrue(contract["zero_outside_centered_acs"])

            legacy = dict(manifest)
            legacy["format_version"] = 1
            legacy_contract = dict(contract)
            legacy_contract.pop("readout_oversampling_removal")
            legacy_contract["readout_oversampling_removed_by_stride"] = 2
            legacy["refscan_contract"] = legacy_contract
            (output / "manifests" / "physical_calibration.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            with patch(
                "wave_retro_lr.mprage.load_wave_mprage_helpers",
                return_value=helper,
            ):
                with self.assertRaisesRegex(ValueError, "legacy or unversioned"):
                    export_mprage_physical_calibration(
                        twix, sequence, normal, output
                    )

    @staticmethod
    def _write_calibration_inputs(root: Path) -> None:
        """Create finite full-rank physical calibration fixtures.

        Args:
            root: Temporary feasibility root.

        Returns:
            None.
        """
        directory = root / "inputs" / "physical_calibration"
        directory.mkdir(parents=True)
        shape = (12, 8, 8, 2)
        rng = np.random.default_rng(917)
        values = (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ).astype(np.complex64)
        for name, data in (
            ("physical_set4_kspace", values),
            ("physical_set4_coil_images", values),
        ):
            output = create_cfl(directory / name, shape)
            output[...] = data
            output.flush()
            del output
        rss_values = np.sqrt(np.sum(np.abs(values) ** 2, axis=3)).astype(np.float32)
        rss = create_cfl(directory / "physical_set4_rss", rss_values.shape)
        rss[...] = rss_values
        rss.flush()
        del rss
        manifests = root / "manifests"
        manifests.mkdir()
        twix = root / "source.dat"
        twix.write_bytes(b"synthetic source identity")
        (manifests / "physical_calibration.json").write_text(
            json.dumps(
                {
                    "status": "mprage_rovir_physical_calibration_ready",
                    "source": {
                        "twix": {"path": str(twix)},
                        "geometry": {
                            "logical_matrix_ro_lin_par": [12, 16, 10],
                            "physical_fov_mm_xyz": [16, 24, 12],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
