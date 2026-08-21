"""Tests for physical-coordinate retrospective-resolution visual review."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from review_retrospective_low_resolution import (  # noqa: E402
    ReviewVolume,
    _positive_percentile,
    geometry_record,
    native_plane,
    run,
    validate_shared_physical_geometry,
    world_slice_index,
)


def _image(
    shape: tuple[int, int, int],
    zooms: tuple[float, float, float],
    center_mm: tuple[float, float, float] = (4.0, -2.0, 7.0),
) -> nib.Nifti1Image:
    diagonal = np.asarray(zooms, dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = np.diag(diagonal)
    affine[:3, 3] = np.asarray(center_mm) - diagonal * (np.asarray(shape) - 1.0) / 2.0
    grid = np.indices(shape, dtype=np.float32)
    data = 1.0 + grid[0] + 2.0 * grid[1] + 3.0 * grid[2]
    return nib.Nifti1Image(data, affine)


def _volume(key: str, image: nib.Nifti1Image) -> ReviewVolume:
    data = np.asarray(image.dataobj, dtype=np.float32)
    return ReviewVolume(
        key=key,
        title=key,
        path=Path(f"/{key}.nii.gz"),
        image=image,
        data=data,
        display_scale=_positive_percentile(data, 99.5),
    )


class PhysicalGeometryTests(unittest.TestCase):
    def test_different_matrices_share_world_center_and_fov(self) -> None:
        full = _volume("full", _image((8, 8, 8), (1.0, 1.0, 1.0)))
        lower_x = _volume("lower_x", _image((4, 8, 8), (2.0, 1.0, 1.0)))
        lower_y = _volume("lower_y", _image((8, 4, 8), (1.0, 2.0, 1.0)))
        validate_shared_physical_geometry([full, lower_x, lower_y])
        self.assertEqual(geometry_record(lower_x.image)["fov_mm_xyz"], [8.0, 8.0, 8.0])
        x_index, x_mm = world_slice_index(lower_x.image, "sagittal", (4.0, -2.0, 7.0))
        self.assertIn(x_index, (1, 2))
        self.assertAlmostEqual(abs(x_mm - 4.0), 1.0)

    def test_native_plane_uses_physical_extent(self) -> None:
        full = _volume("full", _image((8, 8, 8), (1.0, 1.0, 1.0)))
        lower_x = _volume("lower_x", _image((4, 8, 8), (2.0, 1.0, 1.0)))
        full_pixels, full_extent, _, _ = native_plane(full, "coronal", (4.0, -2.0, 7.0))
        low_pixels, low_extent, _, _ = native_plane(lower_x, "coronal", (4.0, -2.0, 7.0))
        self.assertEqual(full_pixels.shape, (8, 8))
        self.assertEqual(low_pixels.shape, (8, 4))
        np.testing.assert_allclose(full_extent, low_extent)

    def test_mismatched_center_is_rejected(self) -> None:
        full = _volume("full", _image((8, 8, 8), (1.0, 1.0, 1.0)))
        shifted = _volume(
            "shifted", _image((4, 8, 8), (2.0, 1.0, 1.0), center_mm=(4.1, -2.0, 7.0))
        )
        with self.assertRaisesRegex(ValueError, "grid center differs"):
            validate_shared_physical_geometry([full, shifted])


class ReviewRunTests(unittest.TestCase):
    def test_complete_review_records_no_mask_or_dicom(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full_path = root / "full.nii.gz"
            grappa_path = root / "grappa.nii.gz"
            nib.save(_image((8, 8, 8), (1.0, 1.0, 1.0)), full_path)
            nib.save(_image((8, 8, 8), (1.0, 1.0, 1.0)), grappa_path)
            full_path.with_name("full.json").write_text(
                json.dumps(
                    {
                        "NIfTICanonicalRAS": True,
                        "NIfTIAffineAxisFlips": [True, False, True],
                    }
                ),
                encoding="utf-8",
            )

            cases = [
                ("lower_x", (4, 8, 8), (2.0, 1.0, 1.0)),
                ("lower_y", (8, 4, 8), (1.0, 2.0, 1.0)),
                ("lower_xy", (4, 4, 8), (2.0, 2.0, 1.0)),
            ]
            case_manifests = []
            for key, shape, zooms in cases:
                magnitude = root / f"{key}_part-mag_test.nii.gz"
                nib.save(_image(shape, zooms), magnitude)
                magnitude.with_name(magnitude.name.removesuffix(".nii.gz") + ".json").write_text(
                    json.dumps(
                        {
                            "NIfTICanonicalRAS": True,
                            "NIfTIAffineAxisFlips": [True, False, True],
                        }
                    ),
                    encoding="utf-8",
                )
                manifest = root / f"{key}_manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "case": {
                                "case_name": key,
                                "achieved_resolution_mm_xyz": list(zooms),
                            },
                            "reconstruction": {"nifti_outputs": [str(magnitude)]},
                        }
                    ),
                    encoding="utf-8",
                )
                case_manifests.append(str(manifest))
            batch_manifest = root / "batch_manifest.json"
            batch_manifest.write_text(
                json.dumps({"status": "complete", "case_manifests": case_manifests}),
                encoding="utf-8",
            )
            output = root / "review"
            manifest = run(
                argparse.Namespace(
                    batch_manifest=batch_manifest,
                    grappa_nifti=grappa_path,
                    full_resolution_nifti=full_path,
                    output_dir=output,
                    slice_world_mm=None,
                    display_percentile=99.5,
                )
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["scientific_scope"]["bet_mask_used"])
            self.assertFalse(manifest["scientific_scope"]["dicom_used"])
            self.assertTrue((output / "native_grid_comparison.png").is_file())
            self.assertTrue((output / "matched_1mm_grid_comparison.png").is_file())
            self.assertTrue((output / "input_geometry.csv").is_file())
            self.assertTrue((output / "review_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
