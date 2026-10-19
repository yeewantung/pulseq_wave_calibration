"""Public MPRAGE launcher integration, CSM lineage, isolation and metal resume tests."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))

from scripts.mprage_metal_pipeline import parser
from wave_retro_lr.bart_io import cfl_record, open_cfl, sha256_file
from wave_retro_lr.metal_pipeline import (
    branch_settings, load_configuration, locked_directory, prepare_case_maps,
    run_pipeline, validate_options, validate_invocation,
)
from wave_retro_lr.metal_workflow import read_json, write_json, file_record
from wave_retro_lr.standard import prepared_artifact_records
from test_metal_csm import fixture, parameters, write_array


def pipeline_fixture(root: Path) -> Path:
    """Create a reviewed native source, case-local measured inputs and private metal config."""
    manifest, reference = fixture(root)
    native = root/"vcc2/normal/bart_inputs"
    record = root/"vcc2/normal/bart_output/ecalib_command.txt"
    record.write_text(shlex.join(["bart", "ecalib", "-m", "1", "-c", ".6",
                                 str(native/"kspace_calib"), str(record.parent/"coil_sens")])+"\n")
    for case, lin in (("native_r3x2", 24), ("lr_y_1p5mm_r3x2", 16), ("native_r3x3", 24)):
        inputs = root/"vcc2/retro"/case/"bart_inputs"
        inputs.mkdir(parents=True)
        write_array(inputs/"psf", np.ones((48, lin, 20, 1, 1), np.complex64))
        write_array(inputs/"wave_kspace", np.ones((48, lin, 20, 2, 1), np.complex64))
        write_json(inputs/"manifest.json", {
            "standard_pca_variant": "vcc2", "virtual_coils": 2,
            "source_normal_manifest_identity": {"sha256": sha256_file(native/"manifest.json")},
            "case": {"target_logical_matrix_ro_lin_par": [24, lin, 20],
                     "achieved_resolution_mm_xyz": [1, 24/lin, 1]},
            "echoes": [{"wave_kspace_norm": 1.}],
            "output_artifacts": prepared_artifact_records(inputs, ("psf", "wave_kspace")),
        })
    config = root/"metal_csm.local.json"
    write_json(config, {"source_manifest": str(manifest), "reference_nifti": str(reference),
                        "parameters": parameters().to_dict()})
    return config


def arguments(root: Path, mode="normal", *flags):
    """Parse representative public CLI inputs without opening raw data or running BART."""
    return parser().parse_args(["--mode", mode, str(root/"raw.dat"), str(root), str(root/"scan.seq"),
                                "--virtual-coils", "2", "--metal-csm-extrapolation", *flags])


def fake_cpu(output: Path, preparation: dict, threads: int, *, gpu: bool = False) -> dict:
    """Stand in for either BART backend with tiny finite images and a bound execution record."""
    state = output/"metadata/reconstruction.json"
    if state.exists():
        return read_json(state)
    dims = preparation["candidate"]["shape"][:3]
    write_array(output/"reconstruction/image_wave", np.ones(dims, np.complex64))
    from wave_retro_lr.metal_runner import wave_command
    bart = file_record("/bin/true")
    result = {"status": "complete", "output": cfl_record(output/"reconstruction/image_wave"),
              "identity": {"gpu": gpu, "bart": bart, "command": wave_command(preparation, output, bart["path"], gpu=gpu),
                           "preparation_sha256": sha256_file(output/"metadata/preparation.json")}, "logs": []}
    write_json(state, result)
    return result


def fake_convert(inputs, image, twix, sequence, output, suffix, **kwargs):
    """Model the real converter's nested subject directory without reading patient data."""
    output = output/"sub-fixture"
    output.mkdir(parents=True, exist_ok=True)
    values = np.asarray(open_cfl(image)).squeeze().transpose(2, 1, 0)
    manifest = read_json(inputs/"manifest.json")
    spacing = manifest.get("case", {}).get("achieved_resolution_mm_xyz", [1., 1., 1.])
    for part in ("mag", "phase"):
        base = output/f"sub-fixture_part-{part}_{suffix}"
        nib.save(nib.Nifti1Image(abs(values), np.diag([*spacing, 1.])), str(base)+".nii.gz")
        write_json(Path(str(base)+".json"), {"Part": part})


class MetalPipelineTests(unittest.TestCase):
    """Exercise single-switch entry points with no external scientific jobs."""

    def test_profiles_cases_and_transferred_lambdas(self):
        """Keep normal Wavelet and the three retrospective FISTA cases as defaults."""
        self.assertEqual(branch_settings("normal", None, 24, None),
                         [("normal", "wavelet_metal_unvalidated", "wavelet", .03, "wavelet-only")])
        result = branch_settings("retro", None, 16, None)
        self.assertEqual([r[0] for r in result], ["native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3"])
        self.assertTrue(all(r[2:4] == ("fista", 0.) for r in result))
        self.assertEqual(len(branch_settings("retro", "reg-full", 24, None)), 6)
        self.assertEqual(branch_settings("normal", None, 16, .02)[0][3], .02)

    def test_invalid_flags_fail_without_outputs(self):
        """Accept explicit GPU but reject ROVir, N4, invalid bounds and conflicting profiles."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_options(arguments(root, "normal", "-g"))
            for flag in ("--rovir", "--include-legacy-lr", "--n4-bias-correction"):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    validate_options(arguments(root, "normal", flag))
            args = arguments(root)
            args.metal_csm_extrapolation = False
            with self.assertRaisesRegex(ValueError, "disabled"):
                validate_options(args)
            args = arguments(root, "retro", "--r3-lambda", ".02")
            with self.assertRaises(ValueError):
                validate_options(args)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(root, "normal", "--reg-full", "--fista-only")
            self.assertFalse(list(root.iterdir()))

    def test_configuration_binds_source_count_and_flat_output(self):
        """Bind the completed source and private config; reject cross-count reuse and symlinks."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = pipeline_fixture(root)
            _, source, destination = load_configuration(root, config, 2)
            self.assertEqual(destination, root/"vcc2_metal_csm")
            self.assertEqual(source["virtual_coils"], 2)
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(ValueError, "different dataset or VCC"):
                load_configuration(root, config, 3)
            destination.symlink_to(root/"vcc2", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_configuration(root, config, 2)

    def test_normal_retro_and_r3x3_resume_share_native_maps(self):
        """Run mocked branches end-to-end, checking original isolation and LR map lineage."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline_fixture(root)
            before = {str(p): sha256_file(p) for p in (root/"vcc2").rglob("*") if p.is_file()}
            with patch("wave_retro_lr.metal_runner.memory_guard"), patch("wave_retro_lr.metal_pipeline.memory_guard"), \
                 patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={}), \
                 patch("wave_retro_lr.mprage.prepare_normal_mprage"), \
                 patch("wave_retro_lr.mprage.prepare_retro_mprage"), \
                 patch("wave_retro_lr.mprage.prepare_retro_mprage_r3x3"), \
                 patch("wave_retro_lr.metal_pipeline.run_reconstruction", side_effect=fake_cpu), \
                 patch("scripts.convert_mprage_bart_to_nifti.convert", side_effect=fake_convert) as converter:
                destination = run_pipeline(arguments(root))
                run_pipeline(arguments(root, "retro"))
                count = converter.call_count
                self.assertEqual(count, 4)
                run_pipeline(arguments(root))
                run_pipeline(arguments(root, "retro"))
                run_pipeline(arguments(root, "r3x3"))
                self.assertEqual(converter.call_count, count)
                native = read_json(destination/"shared_csm/metadata/preparation.json")
                for case in ("native_r3x2", "native_r3x3"):
                    request = read_json(destination/f"retro/{case}/bart_output/fista_r0/metadata/request.json")
                    self.assertEqual(request["standard_request"]["inputs"]["maps"],
                                     dict(native["candidate"], finite=True))
                lr_maps = read_json(destination/"retro/lr_y_1p5mm_r3x2/maps/metadata/maps.json")
                self.assertEqual(lr_maps["request"]["native_candidate"], native["candidate"])
                self.assertEqual(lr_maps["candidate"]["shape"][:3], [24, 16, 20])
                values = np.asarray(open_cfl(lr_maps["candidate"]["base"])).squeeze()
                rss = np.linalg.norm(values, axis=3)
                np.testing.assert_allclose(rss[rss > 0], 1, atol=1e-5)
                sidecar = next((destination/"retro/native_r3x3").rglob("*part-mag*.json"))
                self.assertFalse(read_json(sidecar)["MetalCSM"]["RegularizationOptimizedForMetalCSM"])
                sidecar.write_text("{}")
                with self.assertRaisesRegex(ValueError, "NIfTI output changed"):
                    run_pipeline(arguments(root, "r3x3"))
            self.assertEqual(before, {str(p): sha256_file(p) for p in (root/"vcc2").rglob("*") if p.is_file()})

    def test_invocation_is_read_only_and_rejects_source_processing_mismatch(self):
        """Use the strict native matcher and refuse metadata migrations before source preparation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = pipeline_fixture(root)
            _, source, _ = load_configuration(root, config, 2)
            before = sha256_file(source["normal_manifest"]["path"])
            with patch("wave_retro_lr.mprage._native_manifest_matches", return_value=False), \
                 patch("wave_retro_lr.mprage._native_manifest_matches_reusable_artifact", return_value=False):
                with self.assertRaisesRegex(ValueError, "sources, prewhitening or PSF"):
                    validate_invocation(source, arguments(root))
            with patch("wave_retro_lr.mprage._native_manifest_matches", return_value=False), \
                 patch("wave_retro_lr.mprage._native_manifest_matches_reusable_artifact", return_value=True):
                record = validate_invocation(source, arguments(root, "retro"))
                self.assertEqual(record["policy"], "accepted_materialized_artifacts_no_refit")
            with patch("wave_retro_lr.mprage._native_manifest_matches", return_value=True) as matcher:
                validate_invocation(source, arguments(root, "normal", "--noise-prewhitening"))
                self.assertEqual(matcher.call_args.args[-2:], (2, True))
                with self.assertRaisesRegex(ValueError, "refusing mutation"):
                    validate_invocation(source, arguments(root, "retro"))
            self.assertEqual(sha256_file(source["normal_manifest"]["path"]), before)
            self.assertFalse((root/"vcc2_metal_csm").exists())

    def test_incomplete_metal_variants_are_not_collection_or_rovir_sources(self):
        """Ignore unfinished metal exports and never reinterpret them as canonical ROVir sources."""
        from wave_retro_lr.nifti_collection import _discover_collection_variants
        from wave_retro_lr.standard import resolve_standard_normal_inputs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root/"vcc24_metal_csm/normal/bart_inputs"
            inputs.mkdir(parents=True)
            write_json(inputs/"manifest.json", {"must_not_be_read": True})
            (inputs.parent/"nifti").mkdir()
            with self.assertRaises(FileNotFoundError):
                _discover_collection_variants(root)
            with self.assertRaises(FileNotFoundError):
                resolve_standard_normal_inputs(root)

    def test_prepare_only_config_change_and_partial_rejection(self):
        """Prepare without BART; reject changed ROI and partial/unknown output states."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = pipeline_fixture(root)
            args = arguments(root, "normal", "--metal-prepare-only")
            with patch("wave_retro_lr.metal_runner.memory_guard"), patch("wave_retro_lr.metal_pipeline.memory_guard"), \
                 patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={}), \
                 patch("wave_retro_lr.mprage.prepare_normal_mprage"), \
                 patch("wave_retro_lr.metal_pipeline.run_reconstruction") as runner:
                destination = run_pipeline(args)
                runner.assert_not_called()
                payload = read_json(config)
                payload["parameters"]["max_distance_mm"] = 5
                config.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "request mismatch"):
                    run_pipeline(args)
            partial = root/"partial"
            partial.mkdir()
            (partial/"orphan").touch()
            with self.assertRaisesRegex(ValueError, "Unbound"):
                with locked_directory(partial, {}):
                    self.fail("Unbound output must not be adopted")

    def test_launchers_dispatch_and_keep_default_help(self):
        """Run actual shell entry points through early argument rejection, never BART."""
        for name in ("normal", "retro_lr", "retro_r3x3"):
            script = TOOL/f"scripts/sample_mprage_{name}_recon.sh"
            with self.subTest(name=name):
                help_result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
                self.assertEqual(help_result.returncode, 0)
                self.assertIn("--metal-csm-extrapolation", help_result.stdout)
                result = subprocess.run(["bash", str(script), "missing.dat", "missing-output", "missing.seq",
                                         "--metal-csm-extrapolation", "-g", "--rovir"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("standard PCA only", result.stderr)


if __name__ == "__main__":
    unittest.main()
