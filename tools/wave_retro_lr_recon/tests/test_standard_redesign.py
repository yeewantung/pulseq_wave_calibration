"""Focused contracts for count-specific standard-PCA reconstruction."""

from __future__ import annotations

import inspect
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
from wave_retro_lr.mprage import (  # noqa: E402
    DEFAULT_RETRO_CASE_IDS,
    prepare_normal_mprage,
    prepare_retro_mprage,
)
from wave_retro_lr.gre import prepare_normal_gre, prepare_retro_gre  # noqa: E402
from wave_retro_lr.nifti_collection import (  # noqa: E402
    MASK_BRANCH_PREFERENCE,
    build_mprage_nifti_collection,
)
from wave_retro_lr.reconstruction_state import (  # noqa: E402
    record_completed_reconstruction,
    reconstruction_status,
)
from wave_retro_lr.standard import (  # noqa: E402
    DEFAULT_VIRTUAL_COILS,
    prepared_artifact_records,
    profile_branches,
    retained_energy_diagnostics,
    standard_variant_root,
    validate_virtual_coils,
    write_pca_basis,
)


class StandardPcaContractTests(unittest.TestCase):
    """Validate public defaults, bounds, layouts, and profile selection."""

    def test_api_defaults_and_reduced_mprage_cases(self) -> None:
        """Expose Ncc=24 and only native plus LR-Y R3x2 by default."""

        self.assertEqual(DEFAULT_VIRTUAL_COILS, 24)
        self.assertEqual(DEFAULT_RETRO_CASE_IDS, ("native_r3x2", "lr_y_1p5mm_r3x2"))
        for function in (
            prepare_normal_mprage,
            prepare_retro_mprage,
            prepare_normal_gre,
            prepare_retro_gre,
        ):
            self.assertEqual(
                inspect.signature(function).parameters["virtual_coils"].default,
                24,
            )
        self.assertEqual(
            inspect.signature(prepare_retro_mprage).parameters["case_ids"].default,
            DEFAULT_RETRO_CASE_IDS,
        )

    def test_count_bounds_paths_energy_and_profiles(self) -> None:
        """Validate physical bounds, explicit vccN roots, energy, and profiles."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(standard_variant_root(root, 12), root / "vcc12")
            self.assertEqual(standard_variant_root(root, 24), root / "vcc24")
        self.assertEqual(validate_virtual_coils(24, 32), 24)
        with self.assertRaisesRegex(ValueError, "only 12 physical"):
            validate_virtual_coils(24, 12)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_virtual_coils(0, 32)
        energy = retained_energy_diagnostics(np.linspace(0.1, 1.0, 32), 24)
        self.assertEqual(set(energy["cumulative_at_fixed_counts"]), {"12", "20", "24"})
        self.assertEqual(profile_branches("reg-full"), ("fista_r0", "wavelet"))
        self.assertEqual(profile_branches("wavelet-only"), ("wavelet",))
        self.assertEqual(profile_branches("fista-only"), ("fista_r0",))

    def test_shell_profile_exclusion_and_rovir_isolation(self) -> None:
        """Reject conflicting profiles and standard Ncc overrides for ROVir."""

        launchers = (
            "sample_mprage_normal_recon.sh",
            "sample_mprage_retro_lr_recon.sh",
            "sample_gre_normal_recon.sh",
            "sample_gre_retro_lr_recon.sh",
        )
        for name in launchers:
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / name),
                    "input.dat",
                    "output",
                    "input.seq",
                    "--fista-only",
                    "--wavelet-only",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2, name)
            self.assertIn("mutually exclusive", completed.stderr)
        rovir = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "sample_mprage_retro_lr_recon.sh"),
                "input.dat",
                "output",
                "input.seq",
                "--rovir",
                "--virtual-coils",
                "24",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rovir.returncode, 2)
        self.assertIn("incompatible with --rovir", rovir.stderr)


class StandardPcaStateAndCollectionTests(unittest.TestCase):
    """Validate strict resume state and additive count-specific collections."""

    def test_resume_rejects_regularization_or_input_mismatch(self) -> None:
        """Reuse an exact completed branch and reject changed lambda or PSF."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = self._write_variant_branch(
                root, 24, case="normal", branch="wavelet_selected_vcc24"
            )
            run_manifest = request.pop("run_manifest")
            record_completed_reconstruction(run_manifest, **request)
            self.assertEqual(
                reconstruction_status(run_manifest, **request), "complete"
            )
            mismatched = {**request, "regularization": 0.04}
            with self.assertRaisesRegex(ValueError, "differs from completed"):
                reconstruction_status(run_manifest, **mismatched)
            psf = request["psfs"][0]
            values = create_cfl(psf, (4, 4, 4, 1, 1))
            values[:] = 2
            values.flush()
            del values
            with self.assertRaisesRegex(ValueError, "changed|differs from completed"):
                reconstruction_status(run_manifest, **request)

    def test_r3x3_legacy_normal_identity_schema_remains_hash_bound(self) -> None:
        """Accept the early R3x3 identity field only with its exact normal hash."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = self._write_variant_branch(
                root, 24, case="native_r3x3", branch="fista_r0"
            )
            run_manifest = request.pop("run_manifest")
            prepared_manifest = request["prepared_manifest"]
            payload = json.loads(prepared_manifest.read_text(encoding="utf-8"))
            payload["source_normal_manifest"] = payload.pop(
                "source_normal_manifest_identity"
            )
            prepared_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            record_completed_reconstruction(run_manifest, **request)
            self.assertEqual(
                reconstruction_status(run_manifest, **request), "complete"
            )

            payload["source_normal_manifest"]["sha256"] = "0" * 64
            prepared_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                reconstruction_status(
                    run_manifest.with_name("invalid_reconstruction_manifest.json"),
                    **request,
                )

    def test_two_vcc_variants_and_asymmetric_branches_are_additive(self) -> None:
        """Collect vcc12 and vcc24 with normal-Wavelet/retro-FISTA asymmetry."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for count in (12, 24):
                normal = self._write_variant_branch(
                    root,
                    count,
                    case="normal",
                    branch=(
                        "wavelet_selected_vcc24"
                        if count == 24
                        else "wavelet_transferred_vcc12"
                    ),
                )
                record_completed_reconstruction(
                    normal.pop("run_manifest"), **normal
                )
                retro = self._write_variant_branch(
                    root,
                    count,
                    case="native_r3x2",
                    branch="fista_r0",
                )
                record_completed_reconstruction(retro.pop("run_manifest"), **retro)

            manifest = build_mprage_nifti_collection(root)
            self.assertEqual(
                [item["collection_id"] for item in manifest["variants"]],
                ["vcc12", "vcc24"],
            )
            groups = {
                (item["collection_id"], item["branch"], item["case"])
                for item in manifest["cases"]
            }
            expected_normal_branches = {
                12: "wavelet_transferred_vcc12",
                24: "wavelet_selected_vcc24",
            }
            for count, normal_branch in expected_normal_branches.items():
                variant = f"vcc{count}"
                self.assertIn((variant, normal_branch, "normal"), groups)
                self.assertIn((variant, "fista_r0", "native_r3x2"), groups)
                self.assertTrue((root / "nifti_collection" / variant / "manifest.json").is_file())
            self.assertFalse((root / "vcc12" / "nifti_collection").exists())
            self.assertFalse((root / "vcc24" / "nifti_collection").exists())
            self.assertLess(
                MASK_BRANCH_PREFERENCE.index("wavelet_selected_vcc24"),
                MASK_BRANCH_PREFERENCE.index("wavelet_transferred_vcc12"),
            )
            selected = next(
                item
                for item in manifest["cases"]
                if item["collection_id"] == "vcc24"
                and item["branch"] == "wavelet_selected_vcc24"
                and item["case"] == "normal"
            )
            self.assertEqual(
                selected["reconstruction_manifest"]["lambda"], 0.03
            )
            for variant in manifest["variants"]:
                contract = variant["source_contract"]
                self.assertEqual(
                    contract["compression_basis"]["file_sha256"],
                    sha256_file(
                        root
                        / variant["collection_id"]
                        / "normal"
                        / "bart_inputs"
                        / "coil_compression_basis.npy"
                    ),
                )
                self.assertEqual(
                    contract["source_manifest_sha256"],
                    sha256_file(
                        root
                        / variant["collection_id"]
                        / "normal"
                        / "bart_inputs"
                        / "manifest.json"
                    ),
                )

    def _write_variant_branch(
        self, root: Path, count: int, *, case: str, branch: str
    ) -> dict[str, object]:
        """Write one small hash-bound preparation and reconstruction fixture.

        Args:
            root: Temporary top-level reconstruction root.
            count: Standard-PCA virtual-coil count.
            case: ``normal`` or one retrospective case identifier.
            branch: Reconstruction branch name.

        Returns:
            Keyword request for the reconstruction-state API plus its manifest.
        """

        variant = root / f"vcc{count}"
        normal_inputs = variant / "normal" / "bart_inputs"
        normal_manifest = normal_inputs / "manifest.json"
        if not normal_manifest.is_file():
            normal_inputs.mkdir(parents=True)
            basis = write_pca_basis(
                normal_inputs / "coil_compression_basis.npy",
                np.eye(32, count, dtype=np.complex64),
            )
            calibration = create_cfl(
                normal_inputs / "kspace_calib", (4, 4, 4, count)
            )
            calibration[:] = 1
            calibration.flush()
            del calibration
            normal_payload = {
                "format_version": 1,
                "status": "fixture_ready",
                "standard_pca_variant": f"vcc{count}",
                "source": {"fixture": f"source-{count}"},
                "geometry": {"matrix": [4, 4, 4]},
                "sampling": {"name": "R3x1"},
                "coil_compression": {
                    "physical_coils": 32,
                    "virtual_coils": count,
                    "retained_energy": 0.99,
                    "basis": basis,
                },
            }
            normal_payload["output_artifacts"] = prepared_artifact_records(
                normal_inputs,
                ("kspace_calib",),
                ("coil_compression_basis.npy",),
            )
            normal_manifest.write_text(
                json.dumps(normal_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (normal_inputs / "sampling_class.txt").write_text(
                "R3x1\n", encoding="utf-8"
            )

        case_root = (
            variant / "normal"
            if case == "normal"
            else variant / "retro" / case
        )
        inputs = case_root / "bart_inputs"
        prepared_manifest = inputs / "manifest.json"
        if case == "normal":
            prepared_manifest = normal_manifest
            inputs = normal_inputs
        else:
            inputs.mkdir(parents=True, exist_ok=True)
            for name in ("wave_kspace", "psf"):
                values = create_cfl(inputs / name, (4, 4, 4, count, 1))
                values[:] = 1
                values.flush()
                del values
            payload = {
                "status": "fixture_ready",
                "standard_pca_variant": f"vcc{count}",
                "virtual_coils": count,
                "source": {"fixture": f"source-{count}"},
                "case": {"case_id": case, "matrix": [4, 4, 4]},
                "sampling": {"acceleration": [3, 2]},
                "source_normal_manifest_identity": {
                    "path": str(normal_manifest),
                    "sha256": sha256_file(normal_manifest),
                },
                "output_artifacts": prepared_artifact_records(
                    inputs, ("wave_kspace", "psf")
                ),
            }
            prepared_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        psf = inputs / "psf"
        kspace = inputs / "wave_kspace"
        if case == "normal":
            for name in ("psf", "wave_kspace"):
                if not (inputs / name).with_suffix(".hdr").is_file():
                    values = create_cfl(inputs / name, (4, 4, 4, count, 1))
                    values[:] = 1
                    values.flush()
                    del values
            payload = json.loads(normal_manifest.read_text(encoding="utf-8"))
            payload["output_artifacts"] = prepared_artifact_records(
                inputs,
                ("kspace_calib", "wave_kspace", "psf"),
                ("coil_compression_basis.npy",),
            )
            normal_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        maps = variant / "normal" / "bart_output" / "coil_sens"
        if not maps.with_suffix(".hdr").is_file():
            values = create_cfl(maps, (4, 4, 4, count))
            values[:] = 1
            values.flush()
            del values
        branch_root = case_root / "bart_output" / branch
        image = branch_root / "image_wave"
        values = create_cfl(image, (8, 8, 8, 1))
        values[:] = 1
        values.flush()
        del values
        command_record = branch_root / "wave_command.txt"
        expected = f"bart wave fixture vcc{count} {case} {branch}"
        command_record.write_text(expected + "\n", encoding="utf-8")
        nifti_directory = case_root / "nifti" / branch
        nifti_directory.mkdir(parents=True)
        grid = np.indices((32, 32, 32))
        magnitude = (
            (grid[0] - 15.5) ** 2
            + (grid[1] - 15.5) ** 2
            + (grid[2] - 15.5) ** 2
            <= 100
        ).astype(np.float32)
        nifti = nifti_directory / f"sub-fixture_part-mag_{case}.nii.gz"
        nib.save(nib.Nifti1Image(magnitude, np.eye(4)), str(nifti))
        nifti.with_name(nifti.name[: -len(".nii.gz")] + ".json").write_text(
            json.dumps({"Part": "mag", "Case": case}) + "\n",
            encoding="utf-8",
        )
        method = "fista" if branch == "fista_r0" else "wavelet"
        if method == "fista":
            regularization = 0.0
        elif branch == "wavelet_selected_vcc24":
            regularization = 0.03
        else:
            regularization = 0.035
        profile = "fista-only" if method == "fista" else "wavelet-only"
        return {
            "run_manifest": branch_root / "reconstruction_manifest.json",
            "prepared_manifest": prepared_manifest,
            "normal_manifest": normal_manifest,
            "profile": profile,
            "case": case,
            "branch": branch,
            "method": method,
            "regularization": regularization,
            "maps": maps,
            "psfs": [psf],
            "kspaces": [kspace],
            "images": [image],
            "command_records": [command_record],
            "expected_commands": [expected],
            "nifti_directory": nifti_directory,
        }


if __name__ == "__main__":
    unittest.main()
