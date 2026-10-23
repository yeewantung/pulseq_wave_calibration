"""Custom-map locality, source/basis isolation, pipeline dispatch and resume tests."""

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_metal_csm import fixture, synthetic_maps, write_array
from test_metal_pipeline import fake_cpu, fake_convert
from wave_retro_lr.bart_io import cfl_record, open_cfl, sha256_file
from wave_retro_lr.csm_cache import file_record, prepared_source, read_json, write_json
from wave_retro_lr.core import centered_fftn
from wave_retro_lr.sampling import SamplingPattern
from wave_retro_lr.custom_csm import SCHEMA, prepare_local_csm, load_custom_csm
from wave_retro_lr.custom_pipeline import run_custom_pipeline, validate_native, prepare_case
from scripts.mprage_custom_pipeline import parser


def custom_fixture(root: Path) -> tuple[Path, Path]:
    """Build a small native PCA cache with a real local continuation configuration."""
    fixture(root)
    normal = root/"vcc2/normal/bart_inputs/manifest.json"
    doc = read_json(normal)
    doc["sampling"] = SamplingPattern(name="R3x1", acceleration_lin_par=(3, 1), lin_residue=1,
        matrix_lin_par=(24, 20), acquired_lin=tuple(range(1, 24, 3)), acquired_par=tuple(range(20)),
        measurement_index=0, skip_lin_par=(0, 0)).to_json()
    write_array(normal.parent/"kspace_calib", centered_fftn(synthetic_maps(), axes=(0, 1, 2)))
    doc["output_artifacts"]["kspace_calib"] = cfl_record(normal.parent/"kspace_calib")
    for label, filename in (("twix", "raw.dat"), ("sequence", "scan.seq")):
        path = root/filename
        path.write_bytes(b"synthetic raw identity")
        doc["source"][label] = dict(file_record(path), mtime_ns=path.stat().st_mtime_ns)
    normal.write_text(json.dumps(doc))
    command = normal.parent.parent/"bart_output/ecalib_command.txt"
    command.write_text("bart ecalib -m 1 -c .6 synthetic_acs synthetic_maps\n")
    maps = synthetic_maps()
    support = np.linalg.norm(maps, axis=-1) > 0
    uncropped = np.ones_like(maps)
    uncropped[support] = maps[support]
    cache = root/"crop_cache"
    (cache/"metadata").mkdir(parents=True)
    write_array(cache/"calibration/maps", uncropped)
    write_array(cache/"calibration/eigen", np.where(support, .9, .3).astype(np.complex64))
    write_json(cache/"metadata/calibration.json", {
        "identity": {"source": prepared_source(normal), "maps": 1, "calibration_crop": 0},
        "maps": cfl_record(cache/"calibration/maps"), "eigenvalues": cfl_record(cache/"calibration/eigen")})
    config = root/"candidate.local.json"
    write_json(config, {"diagnostic_root": str(cache), "baseline_csm": str(normal.parent.parent/"bart_output/coil_sens"),
                        "parameters": {"roi_voxels": [[6, 16], [6, 16], [6, 15]], "envelope_axis": 2,
                                       "base_distance_mm": 2., "max_distance_mm": 4., "near_sigma_mm": .8,
                                       "far_sigma_mm": 1.2, "edge_ramp_mm": 1., "global_base_extension": False}})
    return normal, config


class CustomTests(unittest.TestCase):
    """Exercise new opt-in paths on tiny temporary fixtures, without BART execution."""

    def setUp(self):
        """Create one small immutable input and a bounded local custom derivative."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.normal, config = custom_fixture(self.root)
        with patch("wave_retro_lr.custom_csm.memory_guard"):
            self.custom = prepare_local_csm(config, self.root/"custom")
        self.config = config

    def args(self, mode="normal", *flags):
        """Return parsed public options for this synthetic acquisition and CSM."""
        return parser().parse_args([str(self.root/"raw.dat"), str(self.root), str(self.root/"scan.seq"),
                                   "--custom-csm", str(self.custom), "--mode", mode, *flags])

    def native(self):
        """Validate real source/hash/basis checks, mocking only legacy PSF invocation matching."""
        with patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={"fixture": True}):
            return validate_native(self.args(), load_custom_csm(self.custom))

    def test_locality_phase_hashes_and_resume(self):
        """Preserve all native/non-ROI vectors exactly and reuse completed outputs unchanged."""
        record = load_custom_csm(self.custom, self.normal)
        masks = np.load(self.root/"custom/masks/support.npz")
        added, roi = masks["added"], masks["roi"]
        self.assertTrue(added.any())
        self.assertFalse(np.any(added & ~roi))
        result = open_cfl(record["maps"]["base"])
        original = open_cfl(self.normal.parent.parent/"bart_output/coil_sens")
        np.testing.assert_array_equal(result[~added], original[~added])
        self.assertEqual(record["statistics"]["unchanged_vectors_max_error"], 0)
        before = sha256_file(self.custom)
        self.assertEqual(prepare_local_csm(self.config, self.root/"custom"), self.custom)
        self.assertEqual(before, sha256_file(self.custom))
        with self.assertRaisesRegex(ValueError, "another prepared"):
            load_custom_csm(self.custom, self.root/"different/manifest.json")
        altered = open_cfl(record["maps"]["base"], mode="r+")
        altered[0, 0, 0, 0] = 1
        altered.flush()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_custom_csm(self.custom)

    def test_parser_and_basis_isolation(self):
        """Default-off dispatch, profile exclusivity and explicit basis/count mismatch fail closed."""
        self.assertFalse(self.args().g)
        self.assertIsNone(self.args().virtual_coils)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("normal", "--wavelet-only", "--fista-only")
        for flags in (("--rovir",), ("--virtual-coils", "3"), ("--noise-prewhitening",)):
            with self.assertRaises(ValueError):
                validate_native(self.args("normal", *flags), load_custom_csm(self.custom))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("normal", "--metal-csm-extrapolation")

    def test_native_and_lr_case_lineage(self):
        """Native accelerations use exact maps; LR maps preserve selected support and provenance."""
        native, custom = self.native(), load_custom_csm(self.custom)
        for case in ("native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3"):
            result = prepare_case(self.root/"variant", case, native, custom)
            if case != "lr_y_1p5mm_r3x2":
                self.assertEqual(result["maps"], custom["maps"])
            else:
                self.assertEqual(result["maps"]["shape"][:3], [24, 16, 20])
                self.assertTrue(any("resampling" in p["path"] for p in result["provenance_files"]))
            self.assertEqual(prepare_case(self.root/"variant", case, native, custom), result)
        self.assertFalse((self.root/"vcc2/retro").exists())

    def test_isolated_pipeline_export_and_resume(self):
        """Both profiles/backends bind the candidate and exports, without running a real solver."""
        with patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={"fixture": True}), \
                patch("wave_retro_lr.custom_pipeline.run_reconstruction", side_effect=fake_cpu) as solver, \
                patch("scripts.convert_mprage_bart_to_nifti.convert", side_effect=fake_convert), \
                patch("wave_retro_lr.custom_pipeline.memory_guard"):
            output = run_custom_pipeline(self.args("normal", "-g"))
            self.assertEqual(output.parent.name, "vcc2_custom_csm")
            self.assertTrue(solver.call_args.kwargs["gpu"])
            first = list(output.rglob("export.json"))
            self.assertEqual(len(first), 1)
            digest = sha256_file(first[0])
            run_custom_pipeline(self.args("normal", "-g"))
            self.assertEqual(digest, sha256_file(first[0]))
            with self.assertRaisesRegex(ValueError, "request mismatch"):
                run_custom_pipeline(self.args("normal"))
            run_custom_pipeline(self.args("retro", "--custom-prepare-only"))
            self.assertEqual(len(list(output.rglob("export.json"))), 1)
            cases = [p.parents[3].name for p in (output/"retro").rglob("preparation.json")]
            self.assertEqual(set(cases), {"native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3"})

    def test_rovir_real_contract_and_independent_outputs(self):
        """Validate a separate ROVir contract through normal/retro exports without PCA relabeling."""
        from test_rovir_retro import RovirRetroTests
        root = self.root/"rovir_fixture"
        RovirRetroTests()._write_fixture(root)
        normal_path = root/"normal/rovir/bart_inputs/manifest.json"
        doc = read_json(normal_path)
        anchor = read_json(root/"normal/bart_inputs/manifest.json")
        doc.update(status="measured_wave_mprage_rovir_control_ready", geometry=anchor["geometry"])
        doc["rovir"]["basis_applied_identically_to_image_and_acs"] = True
        normal_path.write_text(json.dumps(doc))
        contract_path = root/"normal/rovir/manifest.json"
        contract = read_json(contract_path)
        contract["artifacts"]["prepared_inputs_manifest"] = file_record(normal_path)
        contract_path.write_text(json.dumps(contract))
        custom_path = root/"reviewed_custom.json"
        write_json(custom_path, {"schema": SCHEMA, "status": "complete", "source": prepared_source(normal_path),
                               "maps": cfl_record(root/"normal/rovir/bart_output/coil_sens")})
        args = parser().parse_args([doc["source"]["twix"]["path"], str(root), doc["source"]["sequence"]["path"],
                                   "--custom-csm", str(custom_path), "--rovir", "--mode", "normal", "--fista-only"])
        old_contract = sha256_file(contract_path)
        with patch("wave_retro_lr.custom_pipeline.run_reconstruction", side_effect=fake_cpu), \
                patch("scripts.convert_mprage_bart_to_nifti.convert", side_effect=fake_convert), \
                patch("wave_retro_lr.custom_pipeline.memory_guard"):
            output = run_custom_pipeline(args)
            self.assertEqual(output.parent.name, "rovir_custom_csm")
            args.mode = "retro"
            run_custom_pipeline(args)
            self.assertEqual(len(list(output.rglob("export.json"))), 4)
            self.assertFalse((root/"retro").exists())
            self.assertEqual(sha256_file(contract_path), old_contract)
            sidecar = next((output/"normal").rglob("*part-mag*.json"))
            self.assertEqual(read_json(sidecar)["CoilProcessing"], "rovir")
            args.rovir = False
            with self.assertRaisesRegex(ValueError, "basis"):
                run_custom_pipeline(args)

    def test_provenance_and_partial_rejection(self):
        """Changed configuration or partial derivative files never authorize overwriting."""
        config = read_json(self.config)
        config["parameters"]["max_distance_mm"] = 5
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "provenance file"):
            load_custom_csm(self.custom)
        with self.assertRaisesRegex(ValueError, "request mismatch"):
            prepare_local_csm(self.config, self.root/"custom")

    def test_launcher_dispatch_is_explicit(self):
        """All five MPRAGE entry points dispatch only explicit custom flags; GRE is unchanged."""
        scripts = Path(__file__).resolve().parents[1]/"scripts"
        for name in ("normal", "retro_lr", "retro_r3x3", "rovir", "rovir_retro"):
            content = (scripts/f"sample_mprage_{name}_recon.sh").read_text()
            self.assertIn('--custom-csm', content)
            self.assertIn('mprage_custom_pipeline.py', content)
        self.assertNotIn("mprage_custom_pipeline", (scripts/"sample_gre_normal_recon.sh").read_text())

    def test_check_only_creates_no_output_and_never_runs_bart(self):
        """Read-only preflight validates both modes without creating fingerprint directories."""
        with patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={"fixture": True}), \
                patch("wave_retro_lr.custom_pipeline.run_reconstruction") as solver:
            for mode in ("normal", "retro"):
                output = run_custom_pipeline(self.args(mode, "--custom-check-only"))
                self.assertFalse(output.parent.exists())
            solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
