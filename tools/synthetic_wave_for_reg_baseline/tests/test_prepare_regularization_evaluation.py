"""Tests for regularization evaluation input preparation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_regularization_evaluation import (  # noqa: E402
    nifti_sidecar,
    select_dicom_series,
)


def fake_dicom(instance: int, *, description: str = "t1_mprage_sag_p2_ND"):
    return SimpleNamespace(
        SeriesDescription=description,
        ImageType=["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
        Rows=256,
        Columns=256,
        BitsAllocated=16,
        BitsStored=12,
        PixelRepresentation=0,
        InstanceNumber=instance,
        SOPInstanceUID=f"1.2.3.{instance}",
    )


class PathTests(unittest.TestCase):
    def test_nifti_sidecar_handles_compressed_name(self) -> None:
        self.assertEqual(
            nifti_sidecar(Path("image.nii.gz")),
            Path("image.json"),
        )


class DicomSelectionTests(unittest.TestCase):
    def test_selects_exact_uid_and_sorts_instances(self) -> None:
        groups = {
            "good": [(Path("two.dcm"), fake_dicom(2)), (Path("one.dcm"), fake_dicom(1))],
            "display": [
                (
                    Path("display.dcm"),
                    SimpleNamespace(
                        SeriesDescription="t1_mprage_sag_p2",
                        ImageType=["ORIGINAL", "PRIMARY", "M", "DIS3D", "DIS2D"],
                    ),
                )
            ],
        }
        selected, summary = select_dicom_series(groups, "good", "t1_mprage_sag_p2_ND", 2)
        self.assertEqual([record[1].InstanceNumber for record in selected], [1, 2])
        self.assertEqual(len(summary), 2)

    def test_rejects_display_filtered_image_type(self) -> None:
        dataset = fake_dicom(1)
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "M", "NORM", "DIS3D", "DIS2D"]
        with self.assertRaisesRegex(ValueError, "not normalized, unfiltered ND"):
            select_dicom_series({"uid": [(Path("bad.dcm"), dataset)]}, "uid", "t1_mprage_sag_p2_ND", 1)

    def test_rejects_unfiltered_but_non_normalized_series(self) -> None:
        dataset = fake_dicom(1)
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
        with self.assertRaisesRegex(ValueError, "not normalized, unfiltered ND"):
            select_dicom_series(
                {"uid": [(Path("bad.dcm"), dataset)]},
                "uid",
                "t1_mprage_sag_p2_ND",
                1,
            )


if __name__ == "__main__":
    unittest.main()
