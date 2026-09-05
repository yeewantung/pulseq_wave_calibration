"""Tests for the unmasked GRE magnitude/phase NIfTI collection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TOOL_ROOT / "scripts"
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.gre import gre_wavelet_selection_provenance  # noqa: E402
from wave_retro_lr.gre_nifti_collection import (  # noqa: E402
    CASE_LOCATIONS,
    RECONSTRUCTION_BRANCHES,
    build_gre_nifti_collection,
)


class GreNiftiCollectionTests(unittest.TestCase):
    """Verify complete, hash-identical collection without mask products."""

    def test_complete_collection_copies_all_echo_parts_without_masking(self) -> None:
        """Collect three geometries and two branches with exact source hashes."""

        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            source_root = temporary / "reconstruction"
            self._write_complete_source(source_root)
            # Older GRE exports omitted EchoNumber/EchoTime from sidecars; the
            # conversion manifest remains their authoritative echo record.
            legacy_sidecar = next(
                path
                for path in (source_root / "normal" / "nifti" / "fista_r0").glob(
                    "*part-mag*.json"
                )
            )
            legacy_metadata = json.loads(legacy_sidecar.read_text(encoding="utf-8"))
            legacy_metadata.pop("EchoNumber")
            legacy_metadata.pop("EchoTime")
            legacy_sidecar.write_text(json.dumps(legacy_metadata), encoding="utf-8")
            destination = source_root / "nifti_collection"
            manifest = build_gre_nifti_collection(source_root, require_retro=True)

            self.assertEqual(manifest["case_branch_count"], 6)
            self.assertEqual(manifest["nifti_count"], 24)
            self.assertFalse(manifest["scientific_scope"]["masking_applied"])
            self.assertFalse(
                manifest["scientific_scope"]["masked_derivatives_generated"]
            )
            self.assertTrue(
                manifest["scientific_scope"]["nifti_and_sidecars_copied_byte_for_byte"]
            )
            self.assertNotIn("head_mask", json.dumps(manifest).lower())
            self.assertFalse(any("mask" in path.name.lower() for path in destination.rglob("*")))

            for case in manifest["cases"]:
                self.assertEqual(case["echo_count"], 2)
                self.assertEqual(
                    {(item["echo"], item["image_part"]) for item in case["files"]},
                    {(1, "mag"), (1, "phase"), (2, "mag"), (2, "phase")},
                )
                for item in case["files"]:
                    source_nifti = source_root / item["source_nifti"]
                    copied_nifti = destination / item["collection_nifti"]
                    source_sidecar = source_root / item["source_sidecar"]
                    copied_sidecar = destination / item["collection_sidecar"]
                    self.assertEqual(sha256_file(source_nifti), sha256_file(copied_nifti))
                    self.assertEqual(sha256_file(source_sidecar), sha256_file(copied_sidecar))

            self.assertTrue((destination / "manifest.json").is_file())
            self.assertFalse((destination / "masks").exists())
            self.assertFalse((destination / "head_masked_nifti").exists())

    def test_normal_only_and_owned_refresh_safety(self) -> None:
        """Allow normal-only collection and refresh only an intact owned tree."""

        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            source_root = temporary / "reconstruction"
            self._write_case(source_root, "native_r3x1", Path("normal"))
            destination = source_root / "nifti_collection"
            manifest = build_gre_nifti_collection(source_root)
            self.assertEqual(manifest["case_branch_count"], 2)
            self.assertEqual(manifest["nifti_count"], 8)
            refreshed = build_gre_nifti_collection(source_root)
            self.assertEqual(refreshed["nifti_count"], 8)
            with self.assertRaisesRegex(FileNotFoundError, "native_r3x2"):
                build_gre_nifti_collection(source_root, require_retro=True)
            (destination / "user_file.txt").write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "added or missing"):
                build_gre_nifti_collection(source_root)
            self.assertEqual(
                (destination / "user_file.txt").read_text(encoding="utf-8"),
                "preserve\n",
            )

    def test_incomplete_echo_part_and_masked_source_are_rejected(self) -> None:
        """Reject source drift, missing phase data, and masked sidecars."""

        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            source_root = temporary / "reconstruction"
            self._write_case(source_root, "native_r3x1", Path("normal"))
            phase = next((source_root / "normal" / "nifti" / "fista_r0").rglob("*part-phase*.nii.gz"))
            phase.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                build_gre_nifti_collection(source_root)

        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            source_root = temporary / "reconstruction"
            self._write_case(source_root, "native_r3x1", Path("normal"))
            sidecar = next((source_root / "normal" / "nifti" / "fista_r0").rglob("*.json"))
            if sidecar.name == "conversion_manifest.json":
                sidecar = next(
                    path
                    for path in (source_root / "normal" / "nifti" / "fista_r0").rglob("*.json")
                    if path.name != "conversion_manifest.json"
                )
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["PresentationMaskApplied"] = True
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an unmasked source"):
                build_gre_nifti_collection(source_root)

    def test_cli_and_sample_shell_are_path_explicit_and_bart_free(self) -> None:
        """Expose explicit source/destination help without launching BART."""

        python_script = SCRIPTS / "build_gre_nifti_collection.py"
        shell_script = SCRIPTS / "sample_gre_nifti_collection.sh"
        subprocess.run(["bash", "-n", str(shell_script)], check=True)
        shell_help = subprocess.run(
            ["bash", str(shell_script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        python_help = subprocess.run(
            [sys.executable, str(python_script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(shell_help.returncode, 0)
        self.assertEqual(python_help.returncode, 0)
        self.assertIn("OUTPUT_ROOT [--require-retro]", shell_help.stdout)
        self.assertIn("output_root", python_help.stdout)
        for path in (
            python_script,
            shell_script,
            TOOL_ROOT / "wave_retro_lr" / "gre_nifti_collection.py",
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("bart wave", source)
            self.assertNotIn("bart ecalib", source)
            self.assertNotIn("gre_head_mask", source)
        converter = (SCRIPTS / "convert_gre_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"EchoNumber": echo_index + 1', converter)
        self.assertIn('"EchoTime": float(echo_times[echo_index])', converter)

    def _write_complete_source(self, source_root: Path) -> None:
        """Write all normal and retrospective collection fixtures.

        Args:
            source_root: Temporary reconstruction root to populate.

        Returns:
            None.
        """

        for geometry_id, case_location in CASE_LOCATIONS:
            self._write_case(source_root, geometry_id, case_location)

    def _write_case(
        self, source_root: Path, geometry_id: str, case_location: Path
    ) -> None:
        """Write both branches for one synthetic GRE geometry.

        Args:
            source_root: Temporary reconstruction root to populate.
            geometry_id: Stable GRE geometry identifier.
            case_location: Relative normal or retrospective case directory.

        Returns:
            None.
        """

        for branch in RECONSTRUCTION_BRANCHES:
            directory = source_root / case_location / "nifti" / branch
            directory.mkdir(parents=True, exist_ok=True)
            nifti_records = []
            for echo_number, echo_time in ((1, 0.010), (2, 0.020)):
                outputs = []
                for image_part in ("mag", "phase"):
                    basename = (
                        f"sub-{geometry_id}_echo-{echo_number:02d}_acq-wave_"
                        f"part-{image_part}_{branch}"
                    )
                    nifti_path = directory / f"{basename}.nii.gz"
                    sidecar_path = directory / f"{basename}.json"
                    value = echo_number if image_part == "mag" else echo_number * 0.1
                    nib.save(
                        nib.Nifti1Image(
                            np.full((5, 6, 7), value, dtype=np.float32), np.eye(4)
                        ),
                        str(nifti_path),
                    )
                    sidecar_path.write_text(
                        json.dumps(
                            {
                                "ImagePart": image_part,
                                "EchoNumber": echo_number,
                                "EchoTime": echo_time,
                                "CaseID": geometry_id,
                                "PresentationMaskApplied": False,
                                "GRESharedWaveletSelection": gre_wavelet_selection_provenance(
                                    geometry_id, 2
                                ),
                                "GRESelectedWaveletLambda": 0.015,
                                "OrientationPolicy": {
                                    "canonical_coordinate_system": "RAS",
                                    "interpolation": False,
                                },
                                "CanonicalRASReorientation": {
                                    "StoredAxisCodes": ["R", "A", "S"],
                                    "Interpolation": False,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    outputs.append({"nifti": str(nifti_path), "json": str(sidecar_path)})
                nifti_records.append({"echo": echo_number, "outputs": outputs})

            regularization = "0" if branch == "fista_r0" else "0.015"
            wave_commands = [
                (
                    f"bart wave -w -f -r {regularization} -i 100 -t 1e-6 maps "
                    f"psf_echo-{echo:02d} kspace_echo-{echo:02d} output_echo-{echo:02d}"
                )
                for echo in (1, 2)
            ]
            conversion = {
                "format_version": 1,
                "status": "multi_echo_gre_nifti_export_complete",
                "echo_count": 2,
                "echo_times_s": [0.010, 0.020],
                "case_id": geometry_id,
                "bart_commands": {"ecalib": "bart ecalib", "wave_by_echo": wave_commands},
                "wavelet_selection": gre_wavelet_selection_provenance(geometry_id, 2),
                "orientation": {
                    "canonical_coordinate_system": "RAS",
                    "interpolation": False,
                },
                "presentation_mask_applied": False,
                "nifti": nifti_records,
            }
            (directory / "conversion_manifest.json").write_text(
                json.dumps(conversion), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
