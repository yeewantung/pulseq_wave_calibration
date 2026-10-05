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

from wave_retro_lr.bart_io import create_cfl, sha256_file  # noqa: E402
from wave_retro_lr.gre import gre_wavelet_selection_provenance  # noqa: E402
from wave_retro_lr.gre_nifti_collection import (  # noqa: E402
    CASE_LOCATIONS,
    RECONSTRUCTION_BRANCHES,
    build_gre_nifti_collection,
)
from wave_retro_lr.reconstruction_state import (  # noqa: E402
    record_completed_reconstruction,
)
from wave_retro_lr.standard import (  # noqa: E402
    prepared_artifact_records,
    write_pca_basis,
)


class GreNiftiCollectionTests(unittest.TestCase):
    """Verify complete, hash-identical collection without mask products."""

    def test_complete_collection_copies_all_echo_parts_without_masking(self) -> None:
        """Collect four geometries and two branches with exact source hashes."""

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

            self.assertEqual(manifest["case_branch_count"], 8)
            self.assertEqual(manifest["nifti_count"], 32)
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

    def test_legacy_collection_appends_r3x3_and_never_silently_removes_it(self) -> None:
        """Atomically add R3x3 to an old collection and protect prior groups."""

        with tempfile.TemporaryDirectory() as folder:
            source_root = Path(folder) / "reconstruction"
            for geometry_id, case_location in CASE_LOCATIONS[:-1]:
                self._write_case(source_root, geometry_id, case_location)
            initial = build_gre_nifti_collection(source_root, require_retro=True)
            self.assertEqual(initial["case_branch_count"], 6)
            self.assertEqual(
                initial["synchronization"]["mode"], "initial_build"
            )

            geometry_id, case_location = CASE_LOCATIONS[-1]
            self.assertEqual(geometry_id, "native_r3x3")
            self._write_case(source_root, geometry_id, case_location)
            appended = build_gre_nifti_collection(source_root, require_retro=True)
            self.assertEqual(appended["case_branch_count"], 8)
            self.assertEqual(
                appended["synchronization"]["added_case_groups"],
                [
                    "legacy_vcc12:fista_r0:native_r3x3",
                    "legacy_vcc12:selected_wavelet:native_r3x3",
                ],
            )
            self.assertEqual(
                len(appended["synchronization"]["retained_case_groups"]), 6
            )

            for path in (source_root / case_location / "nifti").rglob("*"):
                if path.is_file():
                    path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "refusing to remove"):
                build_gre_nifti_collection(source_root)

    def test_require_retro_accepts_legacy_layout_and_asymmetric_r3x3(self) -> None:
        """Keep legacy cases valid and allow one complete R3x3 branch."""

        with tempfile.TemporaryDirectory() as folder:
            source_root = Path(folder) / "reconstruction"
            for geometry_id, case_location in CASE_LOCATIONS[:-1]:
                self._write_case(source_root, geometry_id, case_location)
            manifest = build_gre_nifti_collection(source_root, require_retro=True)
            self.assertEqual(manifest["case_branch_count"], 6)

        with tempfile.TemporaryDirectory() as folder:
            source_root = Path(folder) / "reconstruction"
            for geometry_id, case_location in CASE_LOCATIONS[:-1]:
                self._write_case(source_root, geometry_id, case_location)
            geometry_id, case_location = CASE_LOCATIONS[-1]
            self._write_case(
                source_root,
                geometry_id,
                case_location,
            )
            selected = source_root / case_location / "nifti" / "selected_wavelet"
            for path in selected.rglob("*"):
                if path.is_file():
                    path.unlink()
            selected.rmdir()
            asymmetric = build_gre_nifti_collection(
                source_root, require_retro=True
            )
            self.assertIn(
                ("fista_r0", "native_r3x3"),
                {
                    (entry["branch"], entry["geometry_id"])
                    for entry in asymmetric["cases"]
                },
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

    def test_vcc24_selected_normal_and_fista_retro_collection_layout(self) -> None:
        """Collect the reviewed normal branch with asymmetric FISTA retro cases."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "reconstruction"
            variant = root / "vcc24"
            self._write_standard_normal_manifest(variant)
            self._write_case(
                variant,
                "native_r3x1",
                Path("normal"),
                branches=("wavelet_selected_vcc24",),
            )
            self._write_standard_run(
                variant,
                "native_r3x1",
                Path("normal"),
                "wavelet_selected_vcc24",
            )
            for geometry_id, case_location in CASE_LOCATIONS[1:]:
                self._write_case(
                    variant,
                    geometry_id,
                    case_location,
                    branches=("fista_r0",),
                )
                self._write_standard_run(
                    variant,
                    geometry_id,
                    case_location,
                    "fista_r0",
                )

            manifest = build_gre_nifti_collection(root, require_retro=True)
            self.assertEqual(
                [record["collection_id"] for record in manifest["variants"]],
                ["vcc24"],
            )
            groups = {
                (record["branch"], record["geometry_id"])
                for record in manifest["cases"]
            }
            self.assertEqual(
                groups,
                {
                    ("wavelet_selected_vcc24", "native_r3x1"),
                    ("fista_r0", "native_r3x2"),
                    ("fista_r0", "lin_low_resolution_r3x2"),
                    ("fista_r0", "native_r3x3"),
                },
            )
            selected = next(
                record
                for record in manifest["cases"]
                if record["branch"] == "wavelet_selected_vcc24"
            )
            self.assertEqual(selected["reconstruction_manifest"]["lambda"], 0.015)
            self.assertTrue(
                (
                    root
                    / "nifti_collection"
                    / "vcc24"
                    / "original_nifti"
                    / "wavelet_selected_vcc24"
                    / "normal"
                ).is_dir()
            )
            self.assertTrue(
                (
                    root
                    / "nifti_collection"
                    / "vcc24"
                    / "original_nifti"
                    / "fista_r0"
                    / "retro"
                    / "native_r3x3"
                ).is_dir()
            )

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
        self,
        source_root: Path,
        geometry_id: str,
        case_location: Path,
        *,
        branches: tuple[str, ...] = RECONSTRUCTION_BRANCHES,
    ) -> None:
        """Write both branches for one synthetic GRE geometry.

        Args:
            source_root: Temporary reconstruction root to populate.
            geometry_id: Stable GRE geometry identifier.
            case_location: Relative normal or retrospective case directory.

        Returns:
            None.
        """

        for branch in branches:
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

    def _write_standard_normal_manifest(self, variant: Path) -> None:
        """Write a compact hash-bound VCC24 normal preparation fixture.

        Args:
            variant: Count-specific reconstruction root.

        Returns:
            None.
        """

        inputs = variant / "normal" / "bart_inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        basis = write_pca_basis(
            inputs / "coil_compression_basis.npy",
            np.eye(32, 24, dtype=np.complex64),
        )
        names = ["kspace_calib"]
        calibration = create_cfl(inputs / "kspace_calib", (4, 4, 4, 24))
        calibration[:] = 1
        calibration.flush()
        del calibration
        for echo_number in (1, 2):
            for prefix in ("psf", "wave_kspace"):
                name = f"{prefix}_echo-{echo_number:02d}"
                values = create_cfl(inputs / name, (4, 4, 4, 24, 1))
                values[:] = 1
                values.flush()
                del values
                names.append(name)
        payload = {
            "format_version": 1,
            "status": "fixture_ready",
            "standard_pca_variant": "vcc24",
            "source": {"fixture": "gre-vcc24"},
            "geometry": {"matrix": [4, 4, 4]},
            "sampling": {"name": "R3x1"},
            "coil_compression": {
                "physical_coils": 32,
                "virtual_coils": 24,
                "retained_energy": 0.99,
                "basis": basis,
            },
        }
        payload["output_artifacts"] = prepared_artifact_records(
            inputs,
            tuple(names),
            ("coil_compression_basis.npy",),
        )
        (inputs / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_standard_run(
        self,
        variant: Path,
        geometry_id: str,
        case_location: Path,
        branch: str,
    ) -> None:
        """Write one completed standard-PCA GRE run contract for collection.

        Args:
            variant: Count-specific reconstruction root.
            geometry_id: Stable normal or retrospective geometry identifier.
            case_location: Relative normal or retrospective case path.
            branch: Reconstruction branch to bind.

        Returns:
            None.
        """

        normal_inputs = variant / "normal" / "bart_inputs"
        normal_manifest = normal_inputs / "manifest.json"
        inputs = variant / case_location / "bart_inputs"
        if case_location == Path("normal"):
            prepared_manifest = normal_manifest
        else:
            inputs.mkdir(parents=True, exist_ok=True)
            names = []
            for echo_number in (1, 2):
                for prefix in ("psf", "wave_kspace"):
                    name = f"{prefix}_echo-{echo_number:02d}"
                    values = create_cfl(inputs / name, (4, 4, 4, 24, 1))
                    values[:] = 1
                    values.flush()
                    del values
                    names.append(name)
            payload = {
                "format_version": 1,
                "status": "fixture_ready",
                "standard_pca_variant": "vcc24",
                "virtual_coils": 24,
                "source": {"fixture": "gre-vcc24"},
                "case": {"case_id": geometry_id, "matrix": [4, 4, 4]},
                "sampling": {"acceleration": [3, 2]},
                "source_normal_manifest_identity": {
                    "path": str(normal_manifest),
                    "sha256": sha256_file(normal_manifest),
                },
            }
            payload["output_artifacts"] = prepared_artifact_records(
                inputs, tuple(names)
            )
            prepared_manifest = inputs / "manifest.json"
            prepared_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        maps = variant / "normal" / "bart_output" / "coil_sens"
        if not maps.with_suffix(".hdr").is_file():
            values = create_cfl(maps, (4, 4, 4, 24))
            values[:] = 1
            values.flush()
            del values
        branch_root = variant / case_location / "bart_output" / branch
        regularization = 0.0 if branch == "fista_r0" else 0.015
        images = []
        records = []
        commands = []
        psfs = []
        kspaces = []
        for echo_number in (1, 2):
            echo_label = f"echo-{echo_number:02d}"
            image = branch_root / echo_label / "image_wave"
            values = create_cfl(image, (4, 4, 4, 1))
            values[:] = 1
            values.flush()
            del values
            command = (
                f"bart wave -w -f -r {regularization:g} -i 100 -t 1e-6 maps "
                f"psf_{echo_label} kspace_{echo_label} output_{echo_label}"
            )
            record = branch_root / echo_label / "wave_command.txt"
            record.write_text(command + "\n", encoding="utf-8")
            images.append(image)
            records.append(record)
            commands.append(command)
            psfs.append(inputs / f"psf_{echo_label}")
            kspaces.append(inputs / f"wave_kspace_{echo_label}")
        record_completed_reconstruction(
            branch_root / "reconstruction_manifest.json",
            prepared_manifest=prepared_manifest,
            normal_manifest=normal_manifest,
            profile=("fista-only" if branch == "fista_r0" else "wavelet-only"),
            case=("normal" if case_location == Path("normal") else geometry_id),
            branch=branch,
            method=("fista" if branch == "fista_r0" else "wavelet"),
            regularization=regularization,
            maps=maps,
            psfs=psfs,
            kspaces=kspaces,
            images=images,
            command_records=records,
            expected_commands=commands,
            nifti_directory=variant / case_location / "nifti" / branch,
        )


if __name__ == "__main__":
    unittest.main()
