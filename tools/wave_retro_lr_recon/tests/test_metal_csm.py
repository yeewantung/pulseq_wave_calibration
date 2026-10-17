"""Opt-in metal-CSM numerics, isolation, provenance, resume and CPU safety tests."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))

from scripts.mprage_metal_recon import parser  # noqa: E402
from wave_retro_lr.bart_io import cfl_record, create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.metal_csm import (  # noqa: E402
    MetalCSMParameters, continuation_vectors, convex_envelope, inverse_orient, orient, write_candidate,
)
from wave_retro_lr.metal_runner import run_metal_workflow, run_cpu_reconstruction  # noqa: E402
from wave_retro_lr.metal_workflow import file_record, read_json, source_contract, write_json  # noqa: E402
from wave_retro_lr.standard import prepared_artifact_records, write_pca_basis  # noqa: E402
from wave_retro_lr.mprage import COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL  # noqa: E402


def parameters(**overrides) -> MetalCSMParameters:
    """Return small synthetic-grid parameters with optional explicit test overrides."""
    values = dict(roi_voxels=[[2, 18], [2, 22], [2, 22]], envelope_axis=2,
                  base_distance_mm=2., max_distance_mm=4., near_sigma_mm=.8,
                  far_sigma_mm=1.2, edge_ramp_mm=1., global_base_extension=False)
    values.update(overrides)
    return MetalCSMParameters.from_dict(values)


def synthetic_maps(count: int = 2) -> np.ndarray:
    """Return normalized phase-varying coil maps with two bounded support defects."""
    values = np.zeros((24, 24, 20, count), np.complex64)
    coordinate = np.indices(values.shape[:3])[0]
    for coil in range(count):
        values[2:-2, 2:-2, 2:-2, coil] = np.exp(1j*(coil+.01*coordinate[2:-2, 2:-2, 2:-2]))/np.sqrt(count)
    values[8:13, 8:13, 8:12] = 0
    values[17:20, 14:17, 4:7] = 0
    return values


def write_array(base: Path, values: np.ndarray) -> None:
    """Write a small synthetic CFL fixture and flush it before hash validation."""
    mapped = create_cfl(base, values.shape)
    mapped[:] = values
    mapped.flush()


def fixture(root: Path, count: int = 2) -> tuple[Path, Path]:
    """Create a hash-bound completed standard-PCA MPRAGE fixture and its magnitude export."""
    normal = root/f"vcc{count}/normal"
    inputs = normal/"bart_inputs"
    inputs.mkdir(parents=True)
    maps = synthetic_maps(count)
    write_array(normal/"bart_output/coil_sens", maps)
    write_array(inputs/"wave_kspace", np.ones((48, 24, 20, count, 1), np.complex64))
    write_array(inputs/"psf", np.ones((48, 24, 20, 1, 1), np.complex64))
    write_array(inputs/"kspace_calib", np.ones(maps.shape, np.complex64))
    basis = write_pca_basis(inputs/"coil_compression_basis.npy", np.eye(4, count, dtype=np.complex64))
    normal_payload = {"status": "measured_wave_mprage_bart_inputs_ready", "standard_pca_variant": f"vcc{count}",
                      "geometry": {"logical_matrix_ro_lin_par": [24, 24, 20], "physical_fov_mm_xyz": [20., 24., 24.],
                                   "readout_oversampling_factor": 2},
                      "sampling": {"name": "R3x1", "acceleration_lin_par": [3, 1]},
                      "source": {"fixture": True}, "echoes": [{"wave_kspace_norm": 1.}],
                      "coil_compression": {"physical_coils": 4, "virtual_coils": count, "retained_energy": .99, "basis": basis}}
    normal_payload["coil_compression"]["readout_oversampling_removal"] = {
        **COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
        "oversampling_factor": 2, "input_readout": 48, "output_readout": 24,
    }
    normal_payload["output_artifacts"] = prepared_artifact_records(inputs, ("wave_kspace", "psf", "kspace_calib"), ("coil_compression_basis.npy",))
    write_json(inputs/"manifest.json", normal_payload)
    x, y, z = np.indices(maps.shape[:3])
    image = ((.5+x/100+y/200+z/300)*(abs(maps[..., 0]) > 0)).astype(np.complex64)
    image_base = normal/"bart_output/wavelet_selected/image_wave"
    write_array(image_base, image)
    ref = normal/"nifti/wavelet_selected/sub-fixture_part-mag_normal.nii.gz"
    ref.parent.mkdir(parents=True)
    nifti = nib.Nifti1Image(orient(abs(image), (True, False, False)), np.diag([1., 1., 1., 1.]))
    nifti.set_qform(nifti.affine, 1)
    nifti.set_sform(nifti.affine, 2)
    nib.save(nifti, ref)
    side = ref.with_name(ref.name.replace(".nii.gz", ".json"))
    write_json(side, {"MagnitudeNormalization": {"InputPercentileValue": 1.}})
    maps_base = normal/"bart_output/coil_sens"
    command = ["bart", "wave", "-g", "-w", "-f", "-r", "0.03", "-i", "100", "-t", "1e-6",
               str(maps_base), str(inputs/"psf"), str(inputs/"wave_kspace"), str(image_base)]
    command_record = image_base.parent/"wave_command.txt"
    command_record.write_text(shlex.join(command)+"\n")
    identity = {"path": str(inputs/"manifest.json"), "sha256": sha256_file(inputs/"manifest.json")}
    request = {k: normal_payload[k] for k in ("source", "geometry", "sampling", "standard_pca_variant")}
    request.update(case="normal", virtual_coils=count, normal_manifest=identity, prepared_manifest=identity,
                   compression_basis=basis, regularization=.03, method="wavelet", expected_commands=[shlex.join(command)],
                   command_records=[str(command_record)],
                   inputs={"maps": cfl_record(maps_base), "psf_by_echo": [cfl_record(inputs/"psf")],
                           "kspace_by_echo": [cfl_record(inputs/"wave_kspace")]})
    run = {"status": "standard_pca_reconstruction_branch_complete", "request": request,
           "outputs": [cfl_record(image_base)], "nifti_outputs": [file_record(ref), file_record(side)]}
    manifest = image_base.parent/"reconstruction_manifest.json"
    write_json(manifest, run)
    return manifest, ref


class MetalNumericsTests(unittest.TestCase):
    """Verify spatial bounds, finite gates, physical distances and exact original-vector preservation."""

    def test_explicit_parameters_and_validation(self):
        """Reject missing, unknown, nonfinite, inverted and ambiguous spatial settings."""
        with self.assertRaises(ValueError):
            MetalCSMParameters.from_dict({})
        for change in ({"max_distance_mm": float("nan")}, {"near_sigma_mm": 0}, {"global_base_extension": "false"},
                       {"envelope_axis": True}, {"roi_voxels": [[1, 0]]*3}, {"extra": 1},
                       {"minimum_weight": 2}, {"max_distance_mm": .1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parameters(**change)

    def test_empty_collinear_hulls_and_orientation(self):
        """Handle degenerate geometry and round-trip orientation without interpolation."""
        plane = np.zeros((12, 12), bool)
        self.assertFalse(convex_envelope(plane).any())
        plane[4, 3:8] = True
        np.testing.assert_array_equal(convex_envelope(plane), plane)
        image = np.arange(120).reshape(4, 5, 6)
        for flips in ((True, False, False), (False, True, True)):
            np.testing.assert_array_equal(inverse_orient(orient(image, flips), flips), image)

    def test_local_preservation_unit_rss_and_anisotropic_extent(self):
        """Preserve all non-added values bitwise; count physical distances using voxel sizes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = synthetic_maps()
            write_array(root/"original", maps)
            before = sha256_file(root/"original.cfl")
            config = parameters(roi_voxels=[[6, 15], [6, 16], [6, 17]])
            stats, native, added = write_candidate(root/"original", root/"candidate", (True, False, False), (2., 1., 1.), config)
            candidate = open_cfl(root/"candidate")
            np.testing.assert_array_equal(candidate[~added], maps[~added])
            np.testing.assert_allclose(np.linalg.norm(candidate, axis=-1), native | added, atol=1e-5)
            self.assertGreater(stats["added_voxels"], 0)
            self.assertEqual(stats["added_outside_roi"], 0)
            self.assertLessEqual(stats["maximum_added_distance_mm"], config.max_distance_mm)
            self.assertEqual(before, sha256_file(root/"original.cfl"))
            with self.assertRaises(FileExistsError):
                write_candidate(root/"original", root/"candidate", (True, False, False), (1.,)*3, config)

    def test_global_base_is_separately_explicit(self):
        """Require explicit global-base permission to alter support outside the ROI."""
        maps = synthetic_maps()
        local = parameters(roi_voxels=[[6, 15], [6, 16], [6, 17]])
        _, _, _, stats = continuation_vectors(maps, (True, False, False), (1.,)*3, local)
        self.assertEqual(stats["added_outside_roi"], 0)
        _, _, _, global_stats = continuation_vectors(maps, (True, False, False), (1.,)*3, dataclasses.replace(local, global_base_extension=True))
        self.assertGreater(global_stats["added_outside_roi"], 0)

    def test_nonfinite_soft_crop_and_roi_bounds_rejected(self):
        """Reject incompatible maps and out-of-grid ROIs instead of silently reinterpreting them."""
        for maps in (synthetic_maps()*.5, np.full((24, 24, 20, 2), np.nan, np.complex64)):
            with self.assertRaises(ValueError):
                continuation_vectors(maps, (True, False, False), (1.,)*3, parameters())
        with self.assertRaisesRegex(ValueError, "ROI exceeds"):
            continuation_vectors(synthetic_maps(), (True, False, False), (1.,)*3, parameters(roi_voxels=[[0, 99], [0, 10], [0, 10]]))

    def test_no_op_does_not_write_a_candidate(self):
        """Reject an already fully supported grid without creating arrays or requesting reconstruction."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = np.ones((24, 24, 20, 2), np.complex64)/np.sqrt(2)
            write_array(root/"original", maps.astype(np.complex64))
            with self.assertRaisesRegex(ValueError, "No eligible"):
                write_candidate(root/"original", root/"candidate", (True, False, False), (1.,)*3, parameters())
            self.assertFalse((root/"candidate.cfl").exists())
            self.assertFalse((root/"candidate.hdr").exists())


class MetalWorkflowTests(unittest.TestCase):
    """Exercise small-array real preparation/export with mocked BART and resource measurements."""

    def test_default_off_cli_and_api(self):
        """Default-off calls fail before reading input or creating an output directory."""
        args = parser().parse_args(["--source-manifest", "missing", "--reference-nifti", "missing",
                                    "--output", "missing", "--config", "missing"])
        self.assertFalse(args.metal_csm_extrapolation)
        self.assertEqual(args.cpu_threads, 8)
        with patch("wave_retro_lr.metal_runner.source_contract") as read:
            with self.assertRaisesRegex(ValueError, "disabled by default"):
                run_metal_workflow("missing", "missing", "missing", parameters())
            read.assert_not_called()
        result = subprocess.run([sys.executable, str(TOOL/"scripts/mprage_metal_recon.py"),
                                 "--source-manifest", "missing", "--reference-nifti", "missing",
                                 "--output", "missing", "--config", "missing"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Disabled by default", result.stderr)

    def test_collection_copy_and_strict_source_binding(self):
        """Accept an exact retained copy after original export removal; reject changed source bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ref = fixture(root)
            copy = root/"nifti_collection/vcc2/original_nifti"/ref.name
            copy.parent.mkdir(parents=True)
            shutil.copy2(ref, copy)
            side = ref.with_name(ref.name.replace(".nii.gz", ".json"))
            shutil.copy2(side, copy.with_name(side.name))
            ref.unlink()
            side.unlink()
            source = source_contract(manifest, copy)
            self.assertEqual(source["orientation_flips"], [True, False, False])
            self.assertEqual(source["virtual_coils"], 2)
            maps = open_cfl(source["records"]["maps"]["base"], mode="r+")
            maps[0, 0, 0, 0] = .1
            maps.flush()
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                source_contract(manifest, copy)

    def test_unsupported_sources_and_reference_rejected(self):
        """Reject retrospective/ROVir/GRE contracts and a reference with altered geometry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ref = fixture(root)
            original = manifest.read_text()
            for field, value in (("case", "native_r3x3"), ("virtual_coils", 5)):
                payload = json.loads(original)
                payload["request"][field] = value
                manifest.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    source_contract(manifest, ref)
            manifest.write_text(original)
            payload = json.loads(original)
            payload["status"] = "rovir_complete"
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                source_contract(manifest, ref)
            manifest.write_text(original)
            image = nib.load(ref)
            nib.save(nib.Nifti1Image(image.get_fdata(), np.diag([2., 1., 1., 1.])), ref)
            with self.assertRaisesRegex(ValueError, "exact original"):
                source_contract(manifest, ref)

    def test_prepare_reuse_mismatch_and_protected_destinations(self):
        """Reuse identical preparations, but isolate changed parameters and frozen output trees."""
        with tempfile.TemporaryDirectory() as directory, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(directory)
            manifest, ref = fixture(root)
            out = root/"diagnostics/metal"
            with patch("wave_retro_lr.metal_runner.subprocess.Popen") as launch:
                first = run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
                second = run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
                self.assertEqual(first, second)
                launch.assert_not_called()
            with self.assertRaisesRegex(ValueError, "request mismatch"):
                run_metal_workflow(manifest, ref, out, parameters(max_distance_mm=5), enabled=True, prepare_only=True)
            for forbidden in (manifest.parent/"metal", root/"nifti_collection/metal", root):
                with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                    run_metal_workflow(manifest, ref, forbidden, parameters(), enabled=True, prepare_only=True)
            (out/"intermediate/support.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Support archive mismatch"):
                run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)

    def test_physical_bounds_alias_free_and_gre_validation(self):
        """Retain physical-coil and alias-free calibration requirements; never reinterpret GRE inputs."""
        for mutation, message in (("physical", "only 1 physical"), ("alias_free", "alias-free"), ("gre", "MPRAGE")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, ref = fixture(root)
                run = read_json(manifest)
                normal_path = Path(run["request"]["normal_manifest"]["path"])
                normal = read_json(normal_path)
                if mutation == "physical":
                    normal["coil_compression"]["physical_coils"] = 1
                elif mutation == "alias_free":
                    del normal["coil_compression"]["readout_oversampling_removal"]
                else:
                    normal["status"] = "measured_wave_gre_bart_inputs_ready"
                normal_path.write_text(json.dumps(normal))
                for identity in ("normal_manifest", "prepared_manifest"):
                    run["request"][identity]["sha256"] = sha256_file(normal_path)
                manifest.write_text(json.dumps(run))
                with self.assertRaisesRegex(ValueError, message):
                    source_contract(manifest, ref)

    def test_tracked_feature_contains_no_private_paths_or_subjects(self):
        """Keep real scan paths and case identifiers out of source, templates and documentation."""
        files = [*TOOL.glob("wave_retro_lr/metal_*.py"), *TOOL.glob("scripts/*metal*.py"),
                 *TOOL.glob("scripts/*metal*.sh"), TOOL/"docs/mprage_metal_csm.md",
                 TOOL/"configs/mprage_metal_csm.example.json", Path(__file__)]
        for path in files:
            content = path.read_text()
            for private in ("/"+"autofs/", "/"+"homes/", "bch_"+"FMP_", "meas_"+"MID"):
                self.assertNotIn(private, content, str(path))

    def test_output_lock_excludes_concurrent_writer(self):
        """Refuse a second writer without changing an already prepared output tree."""
        with tempfile.TemporaryDirectory() as directory, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(directory)
            manifest, ref = fixture(root)
            out = root/"diagnostics/metal"
            run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
            before = sha256_file(out/"metadata/preparation.json")
            with (out/".metal.lock").open("r+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
            self.assertEqual(before, sha256_file(out/"metadata/preparation.json"))

    def test_full_mocked_cpu_run_export_and_resume(self):
        """Mock only BART; verify native preservation, exact export provenance and no second launch."""
        # Initialize font discovery before mocking subprocess creation for the BART child.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

        with tempfile.TemporaryDirectory() as directory, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(directory)
            manifest, ref = fixture(root, count=3)
            out = root/"diagnostics/metal"
            baseline = read_json(manifest)["outputs"][0]["base"]
            source_hash = sha256_file(Path(baseline).with_suffix(".cfl"))

            def fake_bart(command, **kwargs):
                """Emulate one successful CPU process by copying the tiny fixture baseline output."""
                self.assertNotIn("-g", command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "8")
                self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
                write_array(Path(command[-1]), np.asarray(open_cfl(baseline)))
                process = MagicMock(pid=12345, returncode=0)
                process.poll.return_value = 0
                process.wait.return_value = 0
                return process

            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), patch("wave_retro_lr.metal_runner.subprocess.Popen", side_effect=fake_bart) as launch:
                with patch("wave_retro_lr.metal_runner.export_result", side_effect=RuntimeError("export not started")):
                    with self.assertRaisesRegex(RuntimeError, "export not started"):
                        run_metal_workflow(manifest, ref, out, parameters(), enabled=True)
                result = run_metal_workflow(manifest, ref, out, parameters(), enabled=True)
                reused = run_metal_workflow(manifest, ref, out, parameters(), enabled=True)
                self.assertEqual(launch.call_count, 1)
            self.assertEqual(result, reused)
            self.assertEqual(source_hash, sha256_file(Path(baseline).with_suffix(".cfl")))
            output = nib.load(out/"nifti/metal_csm_magnitude.nii.gz")
            reference = nib.load(ref)
            np.testing.assert_array_equal(output.affine, reference.affine)
            np.testing.assert_array_equal(output.get_fdata(), reference.get_fdata())
            self.assertEqual(output.get_qform(coded=True)[1], reference.get_qform(coded=True)[1])
            sidecar = read_json(out/"nifti/metal_csm_magnitude.json")
            self.assertFalse(sidecar["DefaultEnabled"])
            self.assertFalse(sidecar["IndependentlyNormalized"])
            self.assertEqual(sidecar["VirtualCoils"], 3)
            self.assertEqual(sidecar["SourceReference"]["sha256"], sha256_file(ref))
            for record in result["outputs"]:
                self.assertEqual(file_record(record["path"]), record)
            (out/"figures/comparison.png").write_bytes(b"changed")
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), self.assertRaisesRegex(ValueError, "output hash mismatch"):
                run_metal_workflow(manifest, ref, out, parameters(), enabled=True)
            (out/"metadata/comparison.json").unlink()
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), self.assertRaisesRegex(ValueError, "Partial export"):
                run_metal_workflow(manifest, ref, out, parameters(), enabled=True)

    def test_memory_admission_and_partial_attempt_rejection(self):
        """Do not launch under memory pressure or silently overwrite an interrupted reconstruction."""
        with tempfile.TemporaryDirectory() as directory, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(directory)
            manifest, ref = fixture(root)
            out = root/"diagnostics/metal"
            prep = run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), patch("wave_retro_lr.metal_runner.available_gib", return_value=100), patch("wave_retro_lr.metal_runner.subprocess.Popen") as launch:
                with self.assertRaises(MemoryError):
                    run_cpu_reconstruction(out, prep)
                launch.assert_not_called()
            (out/"reconstruction/image_wave.cfl").write_bytes(b"partial")
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), self.assertRaisesRegex(ValueError, "partial"):
                run_cpu_reconstruction(out, prep)

    def test_memory_watchdog_terminates_only_its_child(self):
        """Simulate pressure after launch; require child-only termination and durable failure status."""
        with tempfile.TemporaryDirectory() as directory, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(directory)
            manifest, ref = fixture(root)
            out = root/"diagnostics/metal"
            prep = run_metal_workflow(manifest, ref, out, parameters(), enabled=True, prepare_only=True)
            process = MagicMock(pid=12345, returncode=-15)
            process.poll.return_value = None
            original_read = Path.read_text

            def proc_read(path, *args, **kwargs):
                """Supply synthetic RSS only for the simulated child; retain real JSON reads."""
                return "VmRSS: 1024 kB\n" if str(path) == "/proc/12345/status" else original_read(path, *args, **kwargs)

            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), patch("wave_retro_lr.metal_runner.subprocess.Popen", return_value=process), patch("wave_retro_lr.metal_runner.available_gib", side_effect=[200, 200, 200, 32]), patch.object(Path, "read_text", proc_read):
                with self.assertRaises(MemoryError):
                    run_cpu_reconstruction(out, prep)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=15)
            self.assertEqual(read_json(out/"metadata/reconstruction.json")["status"], "failed_or_stopped")


if __name__ == "__main__":
    unittest.main()
