"""Tests for the CSM-consistency five-label ROI contract and ROI statistics."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import patch

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import csm_consistency_roi as roi  # noqa: E402
from wave_retro_lr.csm_consistency_roi import (  # noqa: E402
    FIXED_DISPLAY_WINDOWS,
    alias_partner_mask,
    alias_partner_shift,
    apply_display_window,
    dilate_along_axis,
    display_scale,
    edge_matched_control,
    label_masks,
    labels_from_boxes,
    map_mask_to_grid,
    non_partner_null,
    orientation_round_trip,
    parse_roi_box,
    partner_overlap,
    partner_test,
    representative_slices,
    roi_geometry_record,
    snr_matched_mask,
    summarize_metric,
    summary_rows,
    to_bart_orientation,
    to_stored_orientation,
    validate_label_volume,
    validate_roi_geometry,
    window_for,
)

ROVIR_FIXTURE_AFFINE = np.asarray(
    [
        [0.0, 0.0, -2.0, 7.0],
        [0.0, 3.0, 0.0, -9.0],
        [1.0, 0.0, 0.0, -5.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
ROVIR_FIXTURE_FLIPS = (False, False, True)


def _fixture_helpers() -> SimpleNamespace:
    """Return orientation helpers with the upstream semantics of the ROVir test.

    Returns:
        Namespace with ``apply_array_axis_flips`` and
        ``canonicalize_arrays_to_ras`` callables.
    """

    def apply_flips(images: Sequence[np.ndarray], flips: Sequence[bool]) -> list[np.ndarray]:
        """Apply the requested logical-axis flips.

        Args:
            images: Matched arrays.
            flips: Axis flip flags.

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
        images: Sequence[np.ndarray], affine: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, list[list[float]]]:
        """Canonicalize arrays through nibabel orientation APIs.

        Args:
            images: Matched arrays.
            affine: Source voxel-to-RAS affine.

        Returns:
            Canonical arrays, affine, and orientation transform.
        """
        source = nib.orientations.io_orientation(affine)
        target = nib.orientations.axcodes2ornt(("R", "A", "S"))
        transform = nib.orientations.ornt_transform(source, target)
        arrays = [
            np.ascontiguousarray(nib.orientations.apply_orientation(image, transform))
            for image in images
        ]
        canonical_affine = affine @ nib.orientations.inv_ornt_aff(
            transform, np.asarray(images[0]).shape
        )
        return arrays, canonical_affine, transform.tolist()

    return SimpleNamespace(
        apply_array_axis_flips=apply_flips,
        canonicalize_arrays_to_ras=canonicalize,
    )


def _orientation_cases() -> list[tuple[str, np.ndarray, tuple[bool, bool, bool]]]:
    """Return named source affines and physical flips for orientation tests.

    Returns:
        ``(name, affine, flips)`` cases including the ROVir fixture affine,
        diagonal, permuted, and oblique geometries.
    """
    angle = np.deg2rad(12.0)
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    oblique = np.eye(4)
    oblique[:3, :3] = rotation @ np.asarray(
        [[0.0, 0.0, -1.1], [0.0, 0.9, 0.0], [1.3, 0.0, 0.0]]
    )
    oblique[:3, 3] = (97.3, -121.7, 33.9)
    ras = np.diag([1.0, 1.2, 1.5, 1.0])
    ras[:3, 3] = (-3.0, 4.0, 5.25)
    lps = np.diag([-1.0, -1.0, 1.3, 1.0])
    lps[:3, 3] = (60.0, 80.0, -40.0)
    permuted = np.asarray(
        [
            [0.0, -0.9, 0.0, 12.0],
            [0.0, 0.0, 1.1, -4.0],
            [-1.3, 0.0, 0.0, 30.25],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return [
        ("rovir_fixture", ROVIR_FIXTURE_AFFINE, ROVIR_FIXTURE_FLIPS),
        ("ras_diagonal", ras, (False, False, False)),
        ("lps_diagonal", lps, (True, False, True)),
        ("permuted_mixed_signs", permuted, (True, True, True)),
        ("oblique_sagittal", oblique, (False, True, False)),
    ]


def _digest(text: str) -> str:
    """Return a deterministic SHA-256 test digest.

    Args:
        text: Seed text.

    Returns:
        Hexadecimal SHA-256 digest of the text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _direct_symmetric_mask(
    dilated_source: np.ndarray, shift: int, tolerance: int
) -> np.ndarray:
    """Build the symmetric shifted source mask explicitly for comparison.

    Args:
        dilated_source: RO-dilated source mask.
        shift: LIN shift ``s``.
        tolerance: LIN tolerance of every copy.

    Returns:
        Union of copies rolled by ``+s + d`` and ``-s + d`` along LIN axis 1.
    """
    mask = np.zeros(dilated_source.shape, dtype=bool)
    for sign in (1, -1):
        for offset in range(-tolerance, tolerance + 1):
            mask |= np.roll(dilated_source, sign * shift + offset, axis=1)
    return mask


class CsmConsistencyRoiTests(unittest.TestCase):
    """Exercise the ROI contract and descriptive statistics on synthetic arrays."""

    def test_label_volume_validation(self) -> None:
        """Accept a valid five-label volume and reject contract violations.

        Returns:
            None.
        """
        shape = (6, 5, 4)
        labels = np.zeros(shape, dtype=np.uint8)
        labels[0, 0, 0] = 1
        labels[0, 0, 1] = 2
        labels[1] = 3
        labels[2, :2] = 4
        labels[5, 0, 0] = 5
        record = validate_label_volume(labels, shape)
        self.assertEqual(
            record["label_counts"],
            {
                "unassigned": 89,
                "metal_void": 1,
                "metal_pileup": 1,
                "fringe": 20,
                "preserved_anatomy": 8,
                "background_air": 1,
            },
        )
        self.assertEqual(record["shape"], [6, 5, 4])
        self.assertEqual(record["required"], ["fringe", "preserved_anatomy", "background_air"])
        self.assertTrue(record["require_metal"])
        json.dumps(record, allow_nan=False)
        self.assertEqual(
            validate_label_volume(labels.astype(np.float32), shape)["label_counts"],
            record["label_counts"],
        )

        unsupported = labels.copy()
        unsupported[3, 3, 3] = 6
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_label_volume(unsupported, shape)
        negative = labels.astype(np.int16)
        negative[3, 3, 3] = -1
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_label_volume(negative, shape)
        fractional = labels.astype(np.float32)
        fractional[3, 3, 3] = 2.5
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_label_volume(fractional, shape)
        for bad_value in (np.nan, np.inf):
            nonfinite = labels.astype(np.float64)
            nonfinite[3, 3, 3] = bad_value
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_label_volume(nonfinite, shape)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_label_volume(labels, (6, 5, 3))
        with self.assertRaisesRegex(ValueError, "background_air"):
            validate_label_volume(np.where(labels == 5, 0, labels), shape)
        without_metal = np.where(np.isin(labels, (1, 2)), 0, labels)
        with self.assertRaisesRegex(ValueError, "metal"):
            validate_label_volume(without_metal, shape)
        relaxed = validate_label_volume(without_metal, shape, require_metal=False)
        self.assertEqual(relaxed["label_counts"]["metal_void"], 0)
        with self.assertRaisesRegex(ValueError, "Unknown ROI label"):
            validate_label_volume(labels, shape, required=("metal",))
        with self.assertRaises(ValueError):
            validate_label_volume(labels.astype(bool), shape)
        with self.assertRaises(ValueError):
            validate_label_volume(labels.astype(np.complex64), shape)

        masks = label_masks(labels)
        self.assertEqual(
            list(masks),
            [
                "metal_void",
                "metal_pileup",
                "fringe",
                "preserved_anatomy",
                "background_air",
                "metal",
            ],
        )
        self.assertTrue(all(mask.dtype == np.bool_ for mask in masks.values()))
        np.testing.assert_array_equal(masks["metal"], np.isin(labels, (1, 2)))
        np.testing.assert_array_equal(masks["fringe"], labels == 3)

    def test_roi_boxes_are_inclusive_order_independent_and_disjoint(self) -> None:
        """Parse labeled boxes and build reproducible, conflict-free labels.

        Returns:
            None.
        """
        shape = (10, 12, 8)
        self.assertEqual(
            parse_roi_box("fringe=ro=1:3,lin=2:4,par=all", shape),
            (3, {"ro": [1, 3], "lin": [2, 4], "par": [0, 7]}),
        )
        self.assertEqual(
            parse_roi_box("4=ro=0:0,lin=0:1,par=5:6", shape),
            (4, {"ro": [0, 0], "lin": [0, 1], "par": [5, 6]}),
        )
        self.assertEqual(
            parse_roi_box(" Preserved_Anatomy = ro=0:0,lin=0:1,par=5:6", shape)[0], 4
        )
        for invalid in (
            "0=ro=0:1,lin=0:1,par=0:1",
            "unassigned=ro=0:1,lin=0:1,par=0:1",
            "6=ro=0:1,lin=0:1,par=0:1",
            "air=ro=0:1,lin=0:1,par=0:1",
            "ro=0:1,lin=0:1,par=0:1",
            "fringe",
            "fringe=ro=0:10,lin=0:1,par=0:1",
            "fringe=ro=3:1,lin=0:1,par=0:1",
            "fringe=ro=0:1,lin=0:1",
        ):
            with self.subTest(specification=invalid):
                with self.assertRaises(ValueError):
                    parse_roi_box(invalid, shape)
        with self.assertRaises(ValueError):
            parse_roi_box(3, shape)  # type: ignore[arg-type]

        inclusive, _ = labels_from_boxes(["fringe=ro=1:3,lin=2:4,par=0:0"], shape)
        self.assertEqual(int(np.count_nonzero(inclusive == 3)), 9)
        self.assertEqual(int(inclusive[3, 4, 0]), 3)
        self.assertEqual(int(inclusive[4, 4, 0]), 0)
        self.assertEqual(int(inclusive[1, 2, 0]), 3)

        specifications = [
            "metal_void=ro=0:1,lin=0:1,par=0:1",
            "fringe=ro=4:6,lin=2:5,par=1:3",
            "3=ro=5:7,lin=4:6,par=2:4",
            "background_air=ro=9:9,lin=all,par=all",
            "preserved_anatomy=ro=2:3,lin=8:11,par=0:7",
            "fringe=ro=4:6,lin=2:5,par=1:3",
        ]
        labels, record = labels_from_boxes(specifications, shape)
        self.assertEqual(labels.dtype, np.uint8)
        self.assertEqual(record["label_counts"]["fringe"], 36 + 27 - 8)
        self.assertEqual(record["label_counts"]["metal_void"], 8)
        self.assertEqual(record["label_counts"]["background_air"], 96)
        self.assertEqual(record["submitted_box_count"], 6)
        self.assertEqual(record["canonical_box_count"], 5)
        self.assertEqual(record["duplicate_boxes_removed"], 1)
        keys = [
            (box["label"], box["ro"], box["lin"], box["par"])
            for box in record["canonical_boxes"]
        ]
        self.assertEqual(keys, sorted(keys))
        json.dumps(record, allow_nan=False)
        for reordered in (
            list(reversed(specifications)),
            specifications[2:] + specifications[:2],
        ):
            other_labels, other_record = labels_from_boxes(reordered, shape)
            np.testing.assert_array_equal(other_labels, labels)
            self.assertEqual(other_record, record)
            self.assertEqual(
                other_record["canonical_boxes_sha256"], record["canonical_boxes_sha256"]
            )
        _, changed = labels_from_boxes(specifications[:-2], shape)
        self.assertNotEqual(changed["canonical_boxes_sha256"], record["canonical_boxes_sha256"])
        _, merged = labels_from_boxes(
            ["fringe=ro=0:0,lin=0:0,par=0:0", "3=ro=0:0,lin=0:0,par=0:0"], shape
        )
        self.assertEqual(merged["canonical_box_count"], 1)

        conflicting = [
            "fringe=ro=0:2,lin=0:2,par=0:2",
            "preserved_anatomy=ro=2:3,lin=2:3,par=2:3",
        ]
        for ordering in (conflicting, list(reversed(conflicting))):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                labels_from_boxes(ordering, shape)
        adjacent, _ = labels_from_boxes(
            ["fringe=ro=0:1,lin=0:2,par=0:2", "preserved_anatomy=ro=2:3,lin=0:2,par=0:2"],
            shape,
        )
        self.assertEqual(int(np.count_nonzero(adjacent == 4)), 18)
        with self.assertRaises(ValueError):
            labels_from_boxes(["fringe=ro=0:1,lin=0:12,par=0:1"], shape)
        with self.assertRaises(ValueError):
            labels_from_boxes([], shape)
        with self.assertRaises(ValueError):
            labels_from_boxes("fringe=ro=0:1,lin=0:1,par=0:1", shape)  # type: ignore[arg-type]

    def test_orientation_round_trip_with_fixture_helpers(self) -> None:
        """Round-trip several geometries through the ROVir fixture helpers.

        Returns:
            None.
        """
        self._check_orientation_cases(_fixture_helpers())

    def test_orientation_round_trip_with_pinned_upstream_helpers(self) -> None:
        """Round-trip several geometries through the pinned upstream helpers.

        Returns:
            None.
        """
        try:
            from wave_retro_lr.mprage import load_wave_mprage_helpers

            helpers = load_wave_mprage_helpers()
        except (ImportError, OSError) as exc:
            self.skipTest(f"Pinned upstream Wave-MPRAGE helpers could not be imported: {exc}")
        self._check_orientation_cases(helpers)

    def test_orientation_default_helpers_and_lossy_helper_rejection(self) -> None:
        """Load default helpers lazily and refuse helpers that lose the flips.

        Returns:
            None.
        """
        shape = (5, 4, 3)
        identifiers = np.arange(60, dtype=np.int32).reshape(shape)
        with patch(
            "wave_retro_lr.mprage.load_wave_mprage_helpers",
            return_value=_fixture_helpers(),
        ) as loader:
            stored, _, _ = to_stored_orientation(
                identifiers, ROVIR_FIXTURE_FLIPS, ROVIR_FIXTURE_AFFINE
            )
            report = orientation_round_trip(shape, ROVIR_FIXTURE_FLIPS, ROVIR_FIXTURE_AFFINE)
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(report["identity"])
        np.testing.assert_array_equal(
            to_bart_orientation(stored, ROVIR_FIXTURE_FLIPS, ROVIR_FIXTURE_AFFINE), identifiers
        )

        fixture = _fixture_helpers()
        lossy = SimpleNamespace(
            apply_array_axis_flips=lambda images, flips: [np.asarray(image) for image in images],
            canonicalize_arrays_to_ras=fixture.canonicalize_arrays_to_ras,
        )
        with self.assertRaisesRegex(ValueError, "not inverted exactly"):
            to_stored_orientation(identifiers, ROVIR_FIXTURE_FLIPS, ROVIR_FIXTURE_AFFINE, lossy)
        self.assertFalse(
            orientation_round_trip(shape, ROVIR_FIXTURE_FLIPS, ROVIR_FIXTURE_AFFINE, lossy)[
                "identity"
            ]
        )
        with self.assertRaisesRegex(ValueError, "array_flips"):
            to_stored_orientation(identifiers, (0, 0, 1), ROVIR_FIXTURE_AFFINE, fixture)
        with self.assertRaisesRegex(ValueError, "4x4"):
            to_bart_orientation(identifiers, ROVIR_FIXTURE_FLIPS, np.eye(3))

    def test_roi_geometry_validation_rejects_drift(self) -> None:
        """Reject shape drift, affine drift, and non-RAS axis codes.

        Returns:
            None.
        """
        shape = (7, 5, 4)
        stored, canonical, transform = to_stored_orientation(
            np.zeros(shape, dtype=np.uint8),
            ROVIR_FIXTURE_FLIPS,
            ROVIR_FIXTURE_AFFINE,
            _fixture_helpers(),
        )
        arguments: dict[str, Any] = {
            "bart_shape": shape,
            "stored_shape": stored.shape,
            "stored_affine": canonical,
            "source_affine": ROVIR_FIXTURE_AFFINE,
            "array_flips": ROVIR_FIXTURE_FLIPS,
            "orientation_transform": transform,
            "reference_sha256": _digest("reference"),
            "source_manifest_sha256": _digest("manifest").upper(),
        }
        record = json.loads(json.dumps(roi_geometry_record(**arguments), allow_nan=False))
        self.assertEqual(record["stored_axis_codes"], ["R", "A", "S"])
        self.assertEqual(record["array_flips"], [False, False, True])
        self.assertEqual(record["bart_shape"], [7, 5, 4])
        self.assertEqual(record["source_manifest_sha256"], _digest("manifest"))
        self.assertFalse(record["canonicalization_used_resampling"])
        validate_roi_geometry(stored.shape, canonical, record)
        within = canonical.copy()
        within[0, 3] += 5e-7
        validate_roi_geometry(stored.shape, within, record)

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_roi_geometry((stored.shape[0] + 1, *stored.shape[1:]), canonical, record)
        drifted = canonical.copy()
        drifted[0, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "affine"):
            validate_roi_geometry(stored.shape, drifted, record)
        mirrored = canonical.copy()
        mirrored[:, 0] *= -1.0
        with self.assertRaisesRegex(ValueError, "axis codes"):
            validate_roi_geometry(stored.shape, mirrored, record)
        with self.assertRaisesRegex(ValueError, "axis codes"):
            validate_roi_geometry(
                stored.shape, canonical, {**record, "stored_axis_codes": ["L", "A", "S"]}
            )
        with self.assertRaisesRegex(ValueError, "axis codes"):
            validate_roi_geometry(
                stored.shape, mirrored, {**record, "stored_affine": mirrored.tolist()}
            )
        with self.assertRaisesRegex(ValueError, "stored_affine"):
            validate_roi_geometry(
                stored.shape,
                canonical,
                {key: value for key, value in record.items() if key != "stored_affine"},
            )

        name, oblique_affine, oblique_flips = _orientation_cases()[-1]
        self.assertEqual(name, "oblique_sagittal")
        oblique_stored, oblique_canonical, oblique_transform = to_stored_orientation(
            np.zeros(shape, dtype=np.uint8), oblique_flips, oblique_affine, _fixture_helpers()
        )
        oblique_record = roi_geometry_record(
            **{
                **arguments,
                "stored_shape": oblique_stored.shape,
                "stored_affine": oblique_canonical,
                "source_affine": oblique_affine,
                "array_flips": oblique_flips,
                "orientation_transform": oblique_transform,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oblique_labels_ras.nii.gz"
            image = nib.Nifti1Image(oblique_stored, oblique_canonical)
            image.set_qform(oblique_canonical, code=1)
            image.set_sform(oblique_canonical, code=1)
            nib.save(image, str(path))
            loaded_affine = nib.load(str(path)).affine
        # The float32 NIfTI header rounds this affine by more than atol, so
        # only the float32-aware comparison accepts the saved file.
        self.assertGreater(float(np.max(np.abs(loaded_affine - oblique_canonical))), 1e-6)
        validate_roi_geometry(oblique_stored.shape, loaded_affine, oblique_record)
        shifted = loaded_affine.copy()
        shifted[2, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "affine"):
            validate_roi_geometry(oblique_stored.shape, shifted, oblique_record)

        for key, value, message in (
            ("stored_shape", tuple(reversed(stored.shape)), "stored_shape"),
            (
                "orientation_transform",
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                "orientation_transform",
            ),
            ("stored_affine", drifted, "stored_affine"),
            ("stored_affine", mirrored, "axis codes"),
            ("reference_sha256", "abc", "SHA-256"),
            ("array_flips", (0, 0, 1), "array_flips"),
        ):
            with self.subTest(field=key, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    roi_geometry_record(**{**arguments, key: value})

    def test_map_mask_to_grid_uses_exact_centered_interval_overlap(self) -> None:
        """Map masks between centered grids with exact coverage fractions.

        Returns:
            None.
        """
        generator = np.random.default_rng(2)
        mask = generator.random((6, 5, 4)) < 0.4
        for coverage in (0.25, 0.5, 1.0):
            mapped, fraction = map_mask_to_grid(mask, mask.shape, coverage=coverage)
            np.testing.assert_array_equal(mapped, mask)
            np.testing.assert_array_equal(fraction, mask.astype(np.float64))

        fine = np.zeros((8, 1, 1), dtype=bool)
        fine[2:6] = True
        mapped, fraction = map_mask_to_grid(fine, (4, 1, 1))
        np.testing.assert_array_equal(fraction.ravel(), [0.0, 0.75, 1.0, 0.25])
        np.testing.assert_array_equal(mapped.ravel(), [False, True, True, False])
        np.testing.assert_array_equal(
            map_mask_to_grid(fine, (4, 1, 1), coverage=0.8)[0].ravel(),
            [False, False, True, False],
        )
        np.testing.assert_array_equal(
            map_mask_to_grid(fine, (4, 1, 1), coverage=0.25)[0].ravel(),
            [False, True, True, True],
        )
        full = np.ones((8, 1, 1), dtype=bool)
        np.testing.assert_array_equal(
            map_mask_to_grid(full, (4, 1, 1))[1].ravel(), [0.75, 1.0, 1.0, 1.0]
        )
        tie = np.zeros((8, 1, 1), dtype=bool)
        tie[2] = True
        tie[4] = True
        mapped, fraction = map_mask_to_grid(tie, (4, 1, 1))
        np.testing.assert_array_equal(fraction.ravel(), [0.0, 0.5, 0.5, 0.0])
        np.testing.assert_array_equal(mapped.ravel(), [False, True, True, False])

        readout = np.zeros((256, 1, 1), dtype=bool)
        readout[128] = True
        fraction = map_mask_to_grid(readout, (32, 1, 1))[1].ravel()
        self.assertEqual(fraction[16], 0.125)
        self.assertEqual(float(fraction.sum()), 0.125)
        readout[:] = False
        readout[132] = True
        fraction = map_mask_to_grid(readout, (32, 1, 1))[1].ravel()
        self.assertEqual((fraction[16], fraction[17]), (0.0625, 0.0625))
        readout[:] = False
        readout[124:132] = True
        mapped, fraction = map_mask_to_grid(readout, (32, 1, 1))
        self.assertEqual((fraction[0, 0, 0], fraction[15, 0, 0]), (0.0, 0.0625))
        self.assertEqual(fraction[16, 0, 0], 0.9375)
        self.assertEqual(np.flatnonzero(mapped.ravel()).tolist(), [16])
        edge = map_mask_to_grid(np.ones((256, 1, 1), dtype=bool), (32, 1, 1))[1].ravel()
        self.assertEqual(edge[0], 0.5625)
        np.testing.assert_array_equal(edge[1:], np.ones(31))

        coarse = np.zeros((4, 1, 1), dtype=bool)
        coarse[2] = True
        np.testing.assert_array_equal(
            map_mask_to_grid(coarse, (8, 1, 1))[1].ravel(),
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],
        )
        separable = np.zeros((8, 8, 2), dtype=bool)
        separable[2, 3, 0] = True
        fraction = map_mask_to_grid(separable, (4, 4, 2))[1]
        expected = np.zeros((4, 4, 2))
        expected[1, 1, 0] = 0.125
        expected[1, 2, 0] = 0.125
        np.testing.assert_array_equal(fraction, expected)
        native = map_mask_to_grid(np.ones((16, 24, 12), dtype=bool), (16, 4, 2))[1]
        self.assertEqual(native.shape, (16, 4, 2))
        self.assertTrue(np.all((native > 0) & (native <= 1)))

        for bad_coverage in (0.0, -0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                map_mask_to_grid(fine, (4, 1, 1), coverage=bad_coverage)
        with self.assertRaises(ValueError):
            map_mask_to_grid(fine, (4, 1))
        with self.assertRaises(ValueError):
            map_mask_to_grid(fine.astype(np.float32) * 2, (4, 1, 1))

    def test_alias_partner_shift_mask_and_dilation(self) -> None:
        """Shift sources circularly along LIN and dilate along RO without wrap.

        Returns:
            None.
        """
        self.assertEqual(alias_partner_shift(256), 85)
        self.assertEqual(alias_partner_shift(256, 3), 85)
        self.assertEqual(alias_partner_shift(255), 85)
        self.assertEqual(alias_partner_shift(257), 86)
        self.assertEqual(alias_partner_shift(32), 11)
        self.assertEqual(alias_partner_shift(256, 2), 128)
        for arguments in ((256, 1), (2, 3), (0, 3), (True, 3), (256.0, 3)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    alias_partner_shift(*arguments)  # type: ignore[arg-type]

        source = np.zeros((4, 256, 3), dtype=bool)
        source[1, 200, 2] = True
        for tolerance, lin_values in (
            (0, {29, 115}),
            (1, {28, 29, 30, 114, 115, 116}),
            (2, {27, 28, 29, 30, 31, 113, 114, 115, 116, 117}),
        ):
            partner = alias_partner_mask(source, tolerance=tolerance)
            positions = {tuple(int(value) for value in row) for row in np.argwhere(partner)}
            self.assertEqual(positions, {(1, value, 2) for value in lin_values})
        low = np.zeros((4, 256, 3), dtype=bool)
        low[0, 0, 0] = True
        self.assertEqual(
            np.flatnonzero(alias_partner_mask(low, tolerance=0)[0, :, 0]).tolist(), [85, 171]
        )
        with self.assertRaises(ValueError):
            alias_partner_mask(source, tolerance=85)
        with self.assertRaises(ValueError):
            alias_partner_mask(source, lin_axis=3)

        mask = np.zeros((10, 3, 2), dtype=bool)
        mask[1, 1, 0] = True
        dilated = dilate_along_axis(mask, axis=0, half_width=2)
        self.assertEqual(np.flatnonzero(dilated[:, 1, 0]).tolist(), [0, 1, 2, 3])
        self.assertEqual(int(np.count_nonzero(dilated)), 4)
        edge = np.zeros((10, 3, 2), dtype=bool)
        edge[9, 1, 0] = True
        self.assertEqual(
            np.flatnonzero(dilate_along_axis(edge, half_width=3)[:, 1, 0]).tolist(),
            [6, 7, 8, 9],
        )
        full = dilate_along_axis(mask, half_width=None)
        self.assertEqual(np.flatnonzero(full[:, 1, 0]).tolist(), list(range(10)))
        self.assertEqual(int(np.count_nonzero(full)), 10)
        unchanged = dilate_along_axis(mask, half_width=0)
        np.testing.assert_array_equal(unchanged, mask)
        self.assertIsNot(unchanged, mask)
        for bad_width in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                dilate_along_axis(mask, half_width=bad_width)  # type: ignore[arg-type]

    def test_partner_overlap_null_and_partner_test(self) -> None:
        """Report partner overlaps descriptively against non-partner shifts.

        Returns:
            None.
        """
        source = np.zeros((8, 256, 4), dtype=bool)
        source[2:4, 10:13, 1] = True
        for sign in (1, -1):
            fringe = np.roll(source, sign * 85, axis=1)
            self.assertEqual(partner_overlap(fringe, alias_partner_mask(source)), 1.0)
            record = non_partner_null(fringe, source)
            self.assertEqual(record["partner_shift"], 85)
            self.assertEqual(record["observed_overlap"], 1.0)
            self.assertEqual(record["exceedance_fraction"], 0.0)
            self.assertLess(max(record["null_overlaps"]), 1.0)
        self.assertEqual(len(record["null_shifts"]), 256 - 15)
        self.assertEqual(record["null_shifts"], sorted(record["null_shifts"]))
        self.assertGreaterEqual(min(record["null_shifts"]), -128)
        self.assertLessEqual(max(record["null_shifts"]), 127)
        for shift in record["null_shifts"]:
            for center in (85, -85, 0):
                difference = (shift - center) % 256
                self.assertGreater(min(difference, 256 - difference), 2)
        self.assertIn("not a p-value", record["interpretation"])
        self.assertIn("descriptive", record["interpretation"].lower())
        json.dumps(record, allow_nan=False)

        elsewhere = np.roll(source, 40, axis=1)
        record = non_partner_null(elsewhere, source)
        self.assertEqual(record["observed_overlap"], 0.0)
        self.assertEqual(record["exceedance_fraction"], 1.0)
        self.assertEqual(record["null_overlaps"][record["null_shifts"].index(40)], 1.0)
        self.assertEqual(record["null_overlaps"][record["null_shifts"].index(-40)], 1.0)

        empty = np.zeros_like(source)
        self.assertTrue(math.isnan(partner_overlap(empty, alias_partner_mask(source))))
        record = non_partner_null(empty, source)
        self.assertIsNone(record["observed_overlap"])
        self.assertIsNone(record["exceedance_fraction"])
        self.assertTrue(all(value is None for value in record["null_overlaps"]))
        self.assertEqual(record["target_voxels"], 0)
        json.dumps(record, allow_nan=False)

        generator = np.random.default_rng(7)
        target = generator.random((6, 40, 3)) < 0.1
        random_source = generator.random((6, 40, 3)) < 0.05
        for half_width in (0, 2, None):
            record = non_partner_null(target, random_source, ro_half_width=half_width)
            dilated = dilate_along_axis(random_source, axis=0, half_width=half_width)
            self.assertEqual(record["partner_shift"], 13)
            self.assertEqual(record["ro_half_width"], half_width)
            for shift, value in zip(record["null_shifts"], record["null_overlaps"]):
                self.assertEqual(
                    value, partner_overlap(target, _direct_symmetric_mask(dilated, shift, 1))
                )
            observed = partner_overlap(
                target,
                dilate_along_axis(alias_partner_mask(random_source), axis=0, half_width=half_width),
            )
            self.assertEqual(record["observed_overlap"], observed)
            self.assertEqual(
                record["observed_overlap"], partner_overlap(target, alias_partner_mask(dilated))
            )
            self.assertEqual(
                record["exceedance_fraction"],
                float(np.mean(np.asarray(record["null_overlaps"]) >= observed)),
            )

        wide_source = np.zeros((64, 256, 4), dtype=bool)
        wide_source[10:12, 10:13, 1] = True
        displaced_fringe = np.zeros_like(wide_source)
        displaced_fringe[30:32, 95:98, 1] = True
        records = partner_test(displaced_fringe, wide_source)
        self.assertEqual([item["ro_half_width"] for item in records], [0, 16, 32, 64, None])
        self.assertEqual([item["partner_shift"] for item in records], [85] * 5)
        self.assertEqual(
            [item["observed_overlap"] for item in records], [0.0, 0.0, 1.0, 1.0, 1.0]
        )
        self.assertEqual(
            [item["exceedance_fraction"] for item in records], [1.0, 1.0, 0.0, 0.0, 0.0]
        )
        json.dumps(records, allow_nan=False)

        with self.assertRaisesRegex(ValueError, "source_mask"):
            non_partner_null(source, np.zeros_like(source))
        with self.assertRaisesRegex(ValueError, "RO axis"):
            non_partner_null(source, source, lin_axis=0)
        with self.assertRaises(ValueError):
            non_partner_null(source, source, tolerance=85)
        with self.assertRaises(ValueError):
            partner_test(source, source, ro_dilations=())
        with self.assertRaises(ValueError):
            partner_overlap(source, source[:, :, :2])

    def test_edge_matched_control_uses_physical_distances(self) -> None:
        """Select only head-boundary voxels outside the metal exclusion radius.

        Returns:
            None.
        """
        from scipy.spatial import cKDTree

        shape = (40, 36, 30)
        voxel_size = (1.0, 1.5, 2.0)
        grid = np.indices(shape).astype(np.float64)
        coordinates = np.stack(
            [(grid[axis] - (shape[axis] - 1) / 2.0) * voxel_size[axis] for axis in range(3)],
            axis=-1,
        )
        head = (
            (coordinates[..., 0] / 17.0) ** 2
            + (coordinates[..., 1] / 24.0) ** 2
            + (coordinates[..., 2] / 26.0) ** 2
        ) <= 1.0
        metal = (
            head
            & (coordinates[..., 0] > 12.0)
            & (np.abs(coordinates[..., 1]) < 4.0)
            & (np.abs(coordinates[..., 2]) < 4.0)
        )
        self.assertTrue(metal.any())
        control = edge_matched_control(head, metal, voxel_size, band_mm=3.0, exclusion_mm=12.0)

        head_points = coordinates[head]
        air_distance = cKDTree(coordinates[~head]).query(head_points)[0]
        metal_distance = cKDTree(coordinates[metal]).query(head_points)[0]
        expected = np.zeros(shape, dtype=bool)
        expected[head] = (air_distance <= 3.0) & (metal_distance > 12.0)
        np.testing.assert_array_equal(control, expected)
        self.assertTrue(control.any())
        self.assertFalse(np.any(control & ~head))
        control_points = coordinates[control]
        self.assertTrue(np.all(cKDTree(coordinates[~head]).query(control_points)[0] <= 3.0))
        self.assertTrue(np.all(cKDTree(coordinates[metal]).query(control_points)[0] > 12.0))
        interior = np.zeros(shape, dtype=bool)
        interior[head] = air_distance > 3.0
        self.assertTrue(interior.any())
        self.assertFalse(np.any(control & interior))

        isotropic = edge_matched_control(head, metal, (1.0, 1.0, 1.0), exclusion_mm=12.0)
        self.assertFalse(np.array_equal(isotropic, control))
        # Every head voxel of this small phantom lies within the default 40 mm.
        self.assertLess(float(metal_distance.max()), 40.0)
        self.assertFalse(edge_matched_control(head, metal, voxel_size).any())
        unexcluded = edge_matched_control(head, np.zeros_like(head), voxel_size)
        band_only = np.zeros(shape, dtype=bool)
        band_only[head] = air_distance <= 3.0
        np.testing.assert_array_equal(unexcluded, band_only)
        self.assertFalse(edge_matched_control(np.ones(shape, bool), metal, voxel_size).any())
        with self.assertRaises(ValueError):
            edge_matched_control(np.zeros(shape, bool), metal, voxel_size)
        with self.assertRaises(ValueError):
            edge_matched_control(head, metal, (1.0, 1.5))
        with self.assertRaises(ValueError):
            edge_matched_control(head, metal, (1.0, -1.5, 2.0))
        with self.assertRaises(ValueError):
            edge_matched_control(head, metal, voxel_size, band_mm=0.0)

    def test_snr_matched_mask_is_deterministic_and_matches_bins(self) -> None:
        """Draw reproducible SNR-matched subsamples with exact per-bin counts.

        Returns:
            None.
        """
        generator = np.random.default_rng(11)
        shape = (20, 20, 10)
        snr = generator.gamma(2.0, 3.0, size=shape)
        order = generator.permutation(snr.size)
        target = np.zeros(snr.size, dtype=bool)
        target[order[:150]] = True
        target = target.reshape(shape)
        candidate = np.zeros(snr.size, dtype=bool)
        candidate[order[150:1400]] = True
        candidate = candidate.reshape(shape)

        mask, record = snr_matched_mask(candidate, target, snr, bins=5, seed=3)
        repeat_mask, repeat_record = snr_matched_mask(candidate, target, snr, bins=5, seed=3)
        np.testing.assert_array_equal(mask, repeat_mask)
        self.assertEqual(record, repeat_record)
        json.dumps(record, allow_nan=False)
        edges = np.asarray(record["bin_edges"])

        def bin_counts(values: np.ndarray) -> list[int]:
            """Count values per recorded quantile bin.

            Args:
                values: SNR values inside the recorded range.

            Returns:
                Per-bin counts.
            """
            index = np.searchsorted(edges, values, side="right") - 1
            index[values == edges[-1]] = edges.size - 2
            return np.bincount(index, minlength=edges.size - 1).tolist()

        self.assertTrue(record["complete_match"])
        self.assertEqual(record["bins_effective"], 5)
        self.assertEqual(record["target_counts"], bin_counts(snr[target]))
        self.assertEqual(record["selected_counts"], record["target_counts"])
        self.assertEqual(bin_counts(snr[mask]), record["target_counts"])
        self.assertEqual(int(np.count_nonzero(mask)), 150)
        self.assertFalse(np.any(mask & ~candidate))
        self.assertFalse(np.any(mask & target))

        other_mask, other_record = snr_matched_mask(candidate, target, snr, bins=5, seed=4)
        self.assertFalse(np.array_equal(other_mask, mask))
        self.assertEqual(bin_counts(snr[other_mask]), record["target_counts"])
        self.assertEqual(other_record["selected_counts"], record["selected_counts"])

        scarce = np.zeros(snr.size, dtype=bool)
        scarce[order[150:210]] = True
        scarce = scarce.reshape(shape)
        scarce_mask, scarce_record = snr_matched_mask(scarce, target, snr, bins=5, seed=3)
        self.assertFalse(scarce_record["complete_match"])
        ratio = min(
            Fraction(available, wanted)
            for available, wanted in zip(
                scarce_record["available_counts"], scarce_record["target_counts"]
            )
            if wanted > 0
        )
        self.assertLess(ratio, 1)
        self.assertEqual(scarce_record["scale_factor"], float(ratio))
        self.assertEqual(
            scarce_record["selected_counts"],
            [
                (wanted * ratio.numerator) // ratio.denominator
                for wanted in scarce_record["target_counts"]
            ],
        )
        self.assertTrue(
            all(
                selected <= available
                for selected, available in zip(
                    scarce_record["selected_counts"], scarce_record["available_counts"]
                )
            )
        )
        self.assertEqual(bin_counts(snr[scarce_mask]), scarce_record["selected_counts"])

        overlapping = candidate | target
        overlap_mask, overlap_record = snr_matched_mask(overlapping, target, snr, bins=5, seed=3)
        self.assertFalse(np.any(overlap_mask & target))
        self.assertEqual(overlap_record["candidate_target_overlap_excluded"], 150)
        with_nan = snr.copy()
        with_nan[candidate] = np.nan
        nan_mask, nan_record = snr_matched_mask(candidate, target, with_nan, bins=5, seed=3)
        self.assertFalse(nan_mask.any())
        self.assertEqual(nan_record["candidate_nonfinite_excluded"], 1250)
        with self.assertRaisesRegex(ValueError, "finite SNR"):
            snr_matched_mask(candidate, np.zeros(shape, bool), snr)
        with self.assertRaises(ValueError):
            snr_matched_mask(candidate, target, snr, bins=0)

    def test_display_scale_windows_and_clipping(self) -> None:
        """Use one fixed magnitude scale and fixed metric windows.

        Returns:
            None.
        """
        generator = np.random.default_rng(5)
        magnitude = (generator.random((10, 10, 10)) * 100.0 + 1.0).astype(np.float32)
        magnitude[0, 0, 0] = np.nan
        preserved = np.zeros(magnitude.shape, dtype=bool)
        preserved[2:6, 2:6, 2:6] = True
        record = display_scale(magnitude, preserved)
        self.assertEqual(record["source"], "preserved_anatomy_mask")
        self.assertEqual(record["voxels"], 64)
        self.assertEqual(record["percentile"], 99.5)
        self.assertEqual(
            record["scale"], float(np.percentile(magnitude[preserved].astype(np.float64), 99.5))
        )
        json.dumps(record, allow_nan=False)
        for fallback in (
            display_scale(magnitude, np.zeros_like(preserved)),
            display_scale(magnitude),
        ):
            self.assertEqual(fallback["source"], "all_positive_voxels")
            self.assertEqual(fallback["voxels"], 999)
            self.assertEqual(
                fallback["scale"],
                float(np.percentile(magnitude[np.isfinite(magnitude)].astype(np.float64), 99.5)),
            )
        complex_record = display_scale(magnitude * np.exp(1j * 0.3), preserved)
        self.assertAlmostEqual(complex_record["scale"], record["scale"], places=4)
        with self.assertRaises(ValueError):
            display_scale(np.zeros((4, 4, 4)))
        with self.assertRaises(ValueError):
            display_scale(magnitude, preserved, percentile=0.0)

        self.assertEqual(
            FIXED_DISPLAY_WINDOWS,
            {
                "lambda": (0.0, 1.0),
                "rho": (0.0, 0.5),
                "log10_rnr": (-0.5, 2.0),
                "e1": (0.5, 1.0),
                "coherence": (0.9, 1.0),
            },
        )
        for name, window in FIXED_DISPLAY_WINDOWS.items():
            self.assertEqual(window_for(name), window)
        with self.assertRaises(ValueError):
            window_for("magnitude_autoscaled")

        values = np.asarray([-1.0, 0.2, np.nan, 5.0, np.inf, -np.inf])
        clipped = apply_display_window(values, window_for("lambda"))
        self.assertEqual(clipped.dtype, np.float32)
        np.testing.assert_array_equal(
            clipped, np.asarray([0.0, 0.2, np.nan, 1.0, 1.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(np.isnan(values[2]))
        self.assertEqual(values[0], -1.0)
        coherence = apply_display_window(np.asarray([0.5, 0.95, 1.2]), window_for("coherence"))
        np.testing.assert_array_equal(
            coherence, np.asarray([0.9, 0.95, 1.0], dtype=np.float32)
        )
        for bad_window in ((1.0, 0.0), (0.0, np.nan), (0.0,), (0.0, 1.0, 2.0)):
            with self.assertRaises(ValueError):
                apply_display_window(values, bad_window)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            apply_display_window(values.astype(np.complex64), (0.0, 1.0))

    def test_metric_summaries_rows_and_representative_slices(self) -> None:
        """Summarize metrics deterministically and choose representative slices.

        Returns:
            None.
        """
        metric = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        metric[0, 0, 1] = np.nan
        alpha = np.zeros(metric.shape, dtype=bool)
        alpha[0, 0, :] = True
        masks = {
            "zeta": metric >= 12.0,
            "alpha": alpha,
            "empty": np.zeros(metric.shape, dtype=bool),
        }
        summary = summarize_metric(metric, masks, thresholds=(5.0, 2.0))
        self.assertEqual(list(summary), ["alpha", "empty", "zeta"])
        finite = np.asarray([0.0, 2.0, 3.0])
        levels = (0.05, 0.25, 0.5, 0.75, 0.95)
        self.assertEqual(summary["alpha"]["voxels"], 4)
        self.assertEqual(summary["alpha"]["finite"], 3)
        self.assertEqual(
            summary["alpha"]["quantiles"],
            {str(level): float(np.quantile(finite, level)) for level in levels},
        )
        self.assertEqual(summary["alpha"]["fraction_ge"], {"5.0": 0.0, "2.0": 2.0 / 3.0})
        self.assertEqual(
            summary["empty"],
            {
                "voxels": 0,
                "finite": 0,
                "quantiles": {str(level): None for level in levels},
                "fraction_ge": {"5.0": None, "2.0": None},
            },
        )
        self.assertEqual(summary["zeta"]["voxels"], 12)
        self.assertEqual(summary["zeta"]["fraction_ge"]["5.0"], 1.0)
        reordered = summarize_metric(
            metric, dict(reversed(list(masks.items()))), thresholds=(5.0, 2.0)
        )
        self.assertEqual(
            json.dumps(reordered, allow_nan=False), json.dumps(summary, allow_nan=False)
        )

        rows = summary_rows("lambda1", summary)
        self.assertEqual([row["mask"] for row in rows], ["alpha", "empty", "zeta"])
        self.assertEqual(
            list(rows[0]),
            [
                "metric",
                "mask",
                "voxels",
                "finite",
                "quantile_0.05",
                "quantile_0.25",
                "quantile_0.5",
                "quantile_0.75",
                "quantile_0.95",
                "fraction_ge_2.0",
                "fraction_ge_5.0",
            ],
        )
        self.assertTrue(all(list(row) == list(rows[0]) for row in rows))
        self.assertEqual(rows[0]["metric"], "lambda1")
        self.assertEqual(rows[0]["fraction_ge_2.0"], 2.0 / 3.0)
        self.assertIsNone(rows[1]["quantile_0.5"])
        self.assertEqual(rows[1]["voxels"], 0)
        self.assertEqual(summary_rows("lambda1", reordered), rows)
        self.assertEqual(summary_rows("lambda1", json.loads(json.dumps(summary))), rows)
        with self.assertRaises(ValueError):
            summary_rows("lambda1", {"alpha": {"voxels": 1}})

        with self.assertRaises(ValueError):
            summarize_metric(metric, masks, quantiles=(0.5, 0.5))
        with self.assertRaises(ValueError):
            summarize_metric(metric, masks, quantiles=(1.5,))
        with self.assertRaises(ValueError):
            summarize_metric(metric, {"bad": np.zeros((2, 3), bool)})
        with self.assertRaises(ValueError):
            summarize_metric(metric.astype(np.complex64), masks)

        extent = np.zeros((3, 50, 3), dtype=bool)
        extent[:, 10:30, 1] = True
        self.assertEqual(representative_slices(extent, axis=1), [11, 15, 19, 23, 27])
        self.assertEqual(representative_slices(extent, axis=1, count=1), [19])
        self.assertEqual(representative_slices(extent, axis=-1, count=3), [1])
        gapped = np.zeros((2, 50), dtype=bool)
        gapped[0, [2, 3, 40, 41]] = True
        self.assertEqual(representative_slices(gapped, axis=1, count=3), [3, 40])
        self.assertEqual(representative_slices(np.zeros((4, 9, 6), bool), axis=1), [4])
        with self.assertRaises(ValueError):
            representative_slices(extent, axis=3)
        with self.assertRaises(ValueError):
            representative_slices(extent, axis=1, count=0)

    def test_module_source_avoids_process_launches(self) -> None:
        """Keep the ROI module free of process launches and BART commands.

        Returns:
            None.
        """
        source = Path(roi.__file__).read_text(encoding="utf-8")
        for forbidden in ("subprocess", "os.system", "bart "):
            self.assertFalse(forbidden in source, f"ROI module source contains {forbidden!r}.")

    def _check_orientation_cases(self, helpers: Any) -> None:
        """Verify identity, RAS codes, world coordinates, and NIfTI storage.

        Args:
            helpers: Namespace with upstream-compatible orientation helpers.

        Returns:
            None.
        """
        shape = (7, 5, 4)
        identifiers = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
        for name, source_affine, flips in _orientation_cases():
            with self.subTest(case=name):
                stored, canonical, transform = to_stored_orientation(
                    identifiers, flips, source_affine, helpers
                )
                self.assertEqual(tuple(nib.aff2axcodes(canonical)), ("R", "A", "S"))
                self.assertEqual(stored.dtype, identifiers.dtype)
                restored = to_bart_orientation(stored, flips, source_affine)
                self.assertEqual(restored.dtype, identifiers.dtype)
                np.testing.assert_array_equal(restored, identifiers)
                self._assert_world_coordinates_match(
                    identifiers, stored, flips, source_affine, canonical
                )
                report = orientation_round_trip(shape, flips, source_affine, helpers)
                self.assertTrue(report["identity"])
                self.assertEqual(report["stored_axis_codes"], ["R", "A", "S"])
                self.assertEqual(report["stored_shape"], list(stored.shape))
                self.assertEqual(report["orientation_transform"], transform)
                np.testing.assert_allclose(report["canonical_affine"], canonical, rtol=0, atol=0)
                json.dumps(report, allow_nan=False)

                metric = np.linspace(0.0, 1.0, identifiers.size, dtype=np.float32).reshape(shape)
                metric[0, 1, 2] = np.nan
                stored_metric, _, _ = to_stored_orientation(metric, flips, source_affine, helpers)
                np.testing.assert_array_equal(
                    to_bart_orientation(stored_metric, flips, source_affine), metric
                )
                self._assert_nifti_label_round_trip(
                    (identifiers % 6).astype(np.uint8), flips, source_affine, helpers
                )

    def _assert_world_coordinates_match(
        self,
        identifiers: np.ndarray,
        stored: np.ndarray,
        flips: tuple[bool, bool, bool],
        source_affine: np.ndarray,
        canonical_affine: np.ndarray,
    ) -> None:
        """Check that every voxel keeps its physical position after storage.

        Args:
            identifiers: Unique-ID BART volume.
            stored: Canonical-RAS stored volume of the same IDs.
            flips: Physical array flips applied before canonicalization.
            source_affine: Affine of the flipped BART array.
            canonical_affine: Affine of the stored array.

        Returns:
            None.
        """
        shape = identifiers.shape
        bart_index = np.stack(np.unravel_index(np.arange(identifiers.size), shape), axis=1)
        flipped_index = bart_index.copy()
        for axis, should_flip in enumerate(flips):
            if should_flip:
                flipped_index[:, axis] = shape[axis] - 1 - flipped_index[:, axis]
        stored_index = np.empty_like(bart_index)
        stored_index[stored.ravel()] = np.stack(
            np.unravel_index(np.arange(stored.size), stored.shape), axis=1
        )
        np.testing.assert_allclose(
            nib.affines.apply_affine(canonical_affine, stored_index),
            nib.affines.apply_affine(source_affine, flipped_index),
            rtol=0.0,
            atol=1e-9,
        )

    def _assert_nifti_label_round_trip(
        self,
        labels: np.ndarray,
        flips: tuple[bool, bool, bool],
        source_affine: np.ndarray,
        helpers: Any,
    ) -> None:
        """Save stored labels as NIfTI, validate geometry, and invert exactly.

        Args:
            labels: uint8 label volume on the BART grid.
            flips: Physical array flips applied before canonicalization.
            source_affine: Affine of the flipped BART array.
            helpers: Upstream-compatible orientation helpers.

        Returns:
            None.

        Side Effects:
            Writes one NIfTI file inside a temporary directory.
        """
        stored, canonical, transform = to_stored_orientation(
            labels, flips, source_affine, helpers
        )
        record = json.loads(
            json.dumps(
                roi_geometry_record(
                    bart_shape=labels.shape,
                    stored_shape=stored.shape,
                    stored_affine=canonical,
                    source_affine=source_affine,
                    array_flips=flips,
                    orientation_transform=transform,
                    reference_sha256=_digest("reference"),
                    source_manifest_sha256=_digest("manifest"),
                ),
                allow_nan=False,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels_ras.nii.gz"
            image = nib.Nifti1Image(stored, canonical)
            image.set_qform(canonical, code=1)
            image.set_sform(canonical, code=1)
            nib.save(image, str(path))
            loaded = nib.load(str(path))
            values = np.asanyarray(loaded.dataobj)
            self.assertEqual(tuple(nib.aff2axcodes(loaded.affine)), ("R", "A", "S"))
            validate_roi_geometry(values.shape, loaded.affine, record)
            restored = to_bart_orientation(
                np.rint(values).astype(np.uint8),
                record["array_flips"],
                np.asarray(record["source_affine"], dtype=np.float64),
            )
        np.testing.assert_array_equal(restored, labels)


if __name__ == "__main__":
    unittest.main()
