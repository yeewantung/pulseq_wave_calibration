"""Uncropped calibration cache contracts and small synthetic BART parity checks."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))

from wave_retro_lr.bart_io import cfl_record, create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.csm_cache import prepared_source, run_diagnostics, run_ecalib  # noqa: E402
from wave_retro_lr.csm_crop import crop_weights, validate_map_shapes  # noqa: E402
from wave_retro_lr.standard import write_pca_basis  # noqa: E402
from scripts.diagnose_csm_crop import parser  # noqa: E402


def write_array(path: Path, values: np.ndarray) -> None:
    """Write and flush a synthetic complex64 CFL fixture."""
    output = create_cfl(path, values.shape)
    output[:] = values
    output.flush()


def fixture(root: Path, kind: str = "mprage") -> Path:
    """Return a tiny hash-bound PCA MPRAGE/GRE or independent ROVir prepared manifest."""
    inputs = root/"source/bart_inputs"
    inputs.mkdir(parents=True)
    write_array(inputs/"kspace_calib", np.ones((8, 6, 4, 2), np.complex64))
    basis = write_pca_basis(inputs/"basis.npy", np.eye(4, 2, dtype=np.complex64))
    geometry_key = "matrix_ro_lin_par" if kind == "gre" else "logical_matrix_ro_lin_par"
    fov_key = "fov_mm_ro_lin_par" if kind == "gre" else "physical_fov_mm_xyz"
    status = "measured_multi_echo_wave_gre_bart_inputs_ready" if kind == "gre" else "measured_wave_mprage_bart_inputs_ready"
    manifest = {"status": status, "source": {"synthetic_fixture": True},
                "geometry": {geometry_key: [8, 6, 4], fov_key: [4., 6., 8.]},
                "echoes": [{"index": e} for e in range(3 if kind == "gre" else 1)],
                "standard_pca_variant": "vcc2",
                "coil_compression": {"physical_coils": 4, "virtual_coils": 2,
                                     "retained_energy": .9, "basis": basis},
                "output_artifacts": {"kspace_calib": cfl_record(inputs/"kspace_calib")}}
    if kind == "rovir":
        manifest["status"] = "measured_wave_mprage_rovir_control_ready"
        manifest.pop("standard_pca_variant")
        manifest.pop("coil_compression")
        manifest["artifacts"] = manifest.pop("output_artifacts")
        manifest["rovir"] = {"physical_coils": 4, "virtual_coils": 2,
                             "basis_applied_identically_to_image_and_acs": True,
                             "projection_basis_file": "basis.npy",
                             "projection_basis_sha256": basis["file_sha256"]}
    path = inputs/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def fake_ecalib(command: list[str], log: Path, threads: int) -> None:
    """Emulate only the uncropped ecalib outputs for resumability tests, not BART numerics."""
    shape = open_cfl(command[-3]).shape
    maps = np.ones(shape, np.complex64)*np.exp(.3j)/np.sqrt(shape[3])
    eigen = np.linspace(0, 1, np.prod(shape[:3]), dtype=np.float32).reshape(shape[:3])
    write_array(Path(command[-2]), maps)
    write_array(Path(command[-1]), eigen.astype(np.complex64))
    log.write_text("synthetic mocked calibration\n")


class CropNumerics(unittest.TestCase):
    """Validate literal single-map crop semantics independently of BART execution."""

    def test_inclusive_threshold_phase_and_bounds(self):
        """Hard threshold is inclusive and ignores eigenvalue phase, but rejects invalid inputs."""
        values = np.array([0, .59, .6, -.6, .7j], np.complex64)
        np.testing.assert_array_equal(crop_weights(values, .6), [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(crop_weights(values, 0), np.ones(5))
        for crop in (-.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                crop_weights(values, crop)
        for values in ([np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                crop_weights(values, .6)

    def test_soft_boundary_is_not_silently_clipped(self):
        """Record the BART 1.0 endpoint behavior instead of substituting another smoothstep."""
        weights = crop_weights(np.array([0, .36, 1., 1.01]), .6, "soft_bart_1_0")
        self.assertEqual(weights[0], 0)
        self.assertAlmostEqual(weights[1], .5, places=6)
        self.assertGreater(weights[-1], 1)
        with self.assertRaises(ValueError):
            crop_weights(np.ones(1), 1, "soft_bart_1_0")
        with self.assertRaises(ValueError):
            crop_weights(np.ones(1), .6, "soft_sense")

    def test_reject_multimap_and_mismatched_geometry(self):
        """Only shared single-map native geometry is supported."""
        for maps, eigen in (((4, 4, 4, 2, 2), (4, 4, 4, 1, 2)),
                            ((4, 4, 4, 2), (4, 4, 5)), ((4, 4, 4, 2), (4, 4, 4, 2))):
            with self.assertRaises(ValueError):
                validate_map_shapes(maps, eigen)

    def test_cli_defaults(self):
        """Diagnostics default to nominal hard .6, one CPU thread and no full CSM copies."""
        args = parser().parse_args(["--prepared-manifest", "manifest.json", "--output", "diagnostics/test"])
        self.assertEqual(args.hard_crops, [.6])
        self.assertEqual(args.soft_crops, [])
        self.assertEqual(args.cpu_threads, 1)
        self.assertFalse(args.materialize)


class ResourceGuards(unittest.TestCase):
    """Verify guard behavior without launching any real process."""

    def test_low_ram_admission_creates_no_log(self):
        """Busy hosts reject our calibration before Popen or file creation."""
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/"ecalib.log"
            with patch("wave_retro_lr.csm_cache.available_gib", return_value=100), \
                    patch("wave_retro_lr.csm_cache.subprocess.Popen") as launch:
                with self.assertRaises(MemoryError):
                    run_ecalib(["bart", "ecalib"], log, 1)
                launch.assert_not_called()
                self.assertFalse(log.exists())

    def test_cpu_environment_and_idle_setup(self):
        """Only our CPU child is launched with constrained threading and CUDA disabled."""
        child = Mock()
        child.poll.return_value = 0
        child.returncode = 0
        with tempfile.TemporaryDirectory() as temp, \
                patch("wave_retro_lr.csm_cache.available_gib", return_value=200), \
                patch("wave_retro_lr.csm_cache.subprocess.Popen", return_value=child) as launch:
            run_ecalib(["bart", "ecalib"], Path(temp)/"ecalib.log", 2)
            env = launch.call_args.kwargs["env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(launch.call_args.kwargs["preexec_fn"].__name__, "idle_child")
            child.terminate.assert_not_called()


class CacheContracts(unittest.TestCase):
    """Exercise isolated filesystem state using mocked tiny calibration outputs."""

    def setUp(self):
        """Create temporary source, fake executable and diagnostic destination fixtures."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = fixture(self.root)
        self.output = self.root/"diagnostics/crop_test"
        self.bart = self.root/"bart"
        self.bart.write_bytes(b"synthetic executable identity")
        which = patch("wave_retro_lr.csm_cache.shutil.which", return_value=str(self.bart))
        which.start()
        self.addCleanup(which.stop)
        runner = patch("wave_retro_lr.csm_cache.run_ecalib", side_effect=fake_ecalib)
        self.runner = runner.start()
        self.addCleanup(runner.stop)

    def run_case(self, variants=None, **kwargs):
        """Execute the fixture workflow and return its JSON-ready completion record."""
        return run_diagnostics(self.manifest, self.output, variants or [("hard", .6)], **kwargs)

    def test_check_only_leaves_no_outputs(self):
        """Read-only inspection must not create even the diagnostic destination."""
        result = self.run_case(check_only=True)
        self.assertTrue(result["check_only"])
        self.assertFalse(self.output.exists())
        self.runner.assert_not_called()

    def test_resume_additive_variants_materialize_and_figures(self):
        """Completed maps resume; new crops and later full CSM exports do not recalibrate."""
        source_hash = sha256_file(self.manifest)
        first = self.run_case()
        self.assertTrue(Path(first["figure"]).is_file())
        self.run_case()
        result = self.run_case([("hard", .6), ("soft_bart_1_0", .5)], materialize=True)
        self.assertEqual(self.runner.call_count, 1)
        self.assertEqual(sha256_file(self.manifest), source_hash)
        for variant in result["variants"]:
            folder = Path(variant["directory"])
            maps = open_cfl(self.output/"calibration/coil_sens_uncropped")
            weights = open_cfl(folder/"weights")
            derived = open_cfl(folder/"coil_sens")
            np.testing.assert_array_equal(derived, maps*weights[..., None])
            active = weights.real > 0
            np.testing.assert_allclose(np.angle(derived[active]), np.angle(maps[active]), atol=1e-6)
        self.run_case(materialize=True)
        self.assertEqual(self.runner.call_count, 1)

    def test_corruption_and_tool_identity_reject(self):
        """Mutated executable or stored weights cannot be silently reused."""
        result = self.run_case()
        self.bart.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_case()
        self.bart.write_bytes(b"synthetic executable identity")
        weights = open_cfl(Path(result["variants"][0]["directory"])/"weights", mode="r+")
        weights[0, 0, 0] = 13
        weights.flush()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_case()

    def test_partial_cache_rejects(self):
        """A calibration with no completion attestation is preserved, never overwritten."""
        path = self.output/"calibration"
        path.mkdir(parents=True)
        (path/"interrupted.hdr").write_text("partial")
        with self.assertRaisesRegex(ValueError, "Incomplete calibration"):
            self.run_case()
        self.runner.assert_not_called()

    def test_source_hash_count_geometry_and_finite_checks(self):
        """Reject all source changes before creating the output root."""
        original = json.loads(self.manifest.read_text())
        for field, value in (("virtual_coils", 5), ("physical_coils", 1)):
            changed = json.loads(json.dumps(original))
            changed["coil_compression"][field] = value
            self.manifest.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.run_case()
        self.manifest.write_text(json.dumps(original))
        source = open_cfl(self.manifest.parent/"kspace_calib", mode="r+")
        source[0, 0, 0, 0] = np.nan
        source.flush()
        original["output_artifacts"]["kspace_calib"] = cfl_record(self.manifest.parent/"kspace_calib")
        self.manifest.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.run_case()
        self.assertFalse(self.output.exists())

    def test_invalid_slices_and_destination_are_read_only(self):
        """Invalid comparison coordinates and production-root destinations fail before BART."""
        with self.assertRaises(ValueError):
            self.run_case(slices=[8, 0, 0])
        with self.assertRaises(ValueError):
            run_diagnostics(self.manifest, self.manifest.parent, [("hard", .6)])
        self.assertFalse(self.output.exists())
        self.runner.assert_not_called()

    def test_gre_shared_echoes_and_rovir_isolation(self):
        """Accept both source contracts without reinterpreting a ROVir basis as PCA."""
        for kind in ("gre", "rovir"):
            manifest = fixture(self.root/kind, kind)
            source = prepared_source(manifest)
            self.assertEqual(source["kind"], "rovir" if kind == "rovir" else "standard_pca")
            if kind == "gre":
                self.assertEqual(len(source["manifest_content"]["echoes"]), 3)
            else:
                original = json.loads(manifest.read_text())
                original["standard_pca_variant"] = "vcc2"
                manifest.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    prepared_source(manifest)

    def test_concurrent_output_lock(self):
        """Another writer prevents any calibration or derivative writes."""
        self.output.mkdir(parents=True)
        with (self.output/".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_case()
        self.runner.assert_not_called()

    def test_basis_geometry_and_failed_preparation_reject(self):
        """Basis corruption, altered spatial dimensions and incomplete preparation fail closed."""
        original = self.manifest.read_text()
        for key, value in (("status", "in_progress"),
                           ("geometry", {"logical_matrix_ro_lin_par": [8, 6, 5],
                                         "physical_fov_mm_xyz": [4., 6., 8.]})):
            payload = json.loads(original)
            payload[key] = value
            self.manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                self.run_case()
        self.manifest.write_text(original)
        np.save(self.manifest.parent/"basis.npy", np.zeros((4, 2), np.complex64))
        with self.assertRaisesRegex(ValueError, "basis file hash mismatch"):
            self.run_case()
        self.assertFalse(self.output.exists())

    def test_partial_variant_and_materialization_reject(self):
        """Never overwrite half-written weights or a failed full-map derivative."""
        self.run_case()
        partial = self.output/"csm_variants/hard_0p5"
        partial.mkdir()
        (partial/"weights.hdr").write_text("partial")
        with self.assertRaisesRegex(ValueError, "Partial CSM variant"):
            self.run_case([("hard", .5)])
        (self.output/"csm_variants/hard_0p6/coil_sens.hdr").write_text("partial")
        with self.assertRaisesRegex(ValueError, "Partial materialized"):
            self.run_case(materialize=True)
        self.assertEqual(self.runner.call_count, 1)


@unittest.skipUnless(os.environ.get("CSM_TEST_BART") == "1" and shutil.which("bart"),
                     "Set CSM_TEST_BART=1 after host BART startup for tiny CPU parity test")
class BartParity(unittest.TestCase):
    """Validate postprocessing against actual ecalib on a temporary 16-cubed phantom."""

    def test_direct_hard_and_soft(self):
        """Derived maps match direct BART calibration; eigenmaps remain identical."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x, y, z = np.meshgrid(*([np.linspace(-1, 1, 16)]*3), indexing="ij")
            body = (x*x + y*y + z*z < .7).astype(np.float32)
            coil = np.stack([(1+.2*x)*np.exp(.2j*y), (1-.2*x)*np.exp(-.2j*y),
                             (1+.2*y)*np.exp(.2j*z), (1-.2*y)*np.exp(-.2j*z)], axis=-1)
            image = body[..., None]*coil
            acs = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=(0, 1, 2)),
                                            axes=(0, 1, 2), norm="ortho"), axes=(0, 1, 2))
            write_array(root/"acs", acs.astype(np.complex64))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            for tag, extra in (("zero", ["-c", "0"]), ("hard", ["-c", ".6"]),
                               ("soft", ["-c", ".6", "-S"])):
                command = [shutil.which("bart"), "ecalib", "-m", "1", "-r", "12", "-k", "4",
                           *extra, str(root/"acs"), str(root/tag), str(root/(tag+"_eig"))]
                subprocess.run(command, env=env, check=True, capture_output=True, timeout=60)
            base = open_cfl(root/"zero")
            eigen = open_cfl(root/"zero_eig")
            for tag, mode in (("hard", "hard"), ("soft", "soft_bart_1_0")):
                np.testing.assert_array_equal(eigen, open_cfl(root/(tag+"_eig")))
                derived = base*crop_weights(eigen, .6, mode)
                np.testing.assert_allclose(derived, open_cfl(root/tag), atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
