"""Exercise real retrospective preparation/reuse with pre-whitening-era manifests."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_metal_auto import automatic_fixture
from test_metal_csm import write_array
from wave_retro_lr.bart_io import sha256_file, cfl_record
from wave_retro_lr.mprage import (
    _retro_prewhitening_matches, prepare_retro_mprage, prepare_retro_mprage_r3x3,
)
from wave_retro_lr.relocation import read_json
from wave_retro_lr.sampling import SamplingPattern


def upgrade_retro_fixture(root: Path) -> None:
    """Replace mocked retro inputs in a temporary fixture with real three-case preparation.

    Use tiny measured-lattice arrays, then remove whitening keys to model old
    software. Update fixture hashes before any relocation receipt is created.
    No BART or scanner reader runs.
    """
    normal_path = root / "vcc2/normal/bart_inputs/manifest.json"
    normal = read_json(normal_path)
    pattern = SamplingPattern("R3x1", (3, 1), 1, (24, 20), tuple(range(1, 24, 3)), tuple(range(20)))
    normal["sampling"] = pattern.to_json()
    wave = np.broadcast_to(pattern.mask()[None, :, :, None, None], (48, 24, 20, 2, 1)).astype(np.complex64)
    write_array(normal_path.parent / "wave_kspace", wave)
    for name, count in (("wave_trajectory", 2), ("psf_coefficients", 3)):
        write_array(normal_path.parent / name, np.zeros((48, count), np.complex64))
        normal["output_artifacts"][name] = cfl_record(normal_path.parent / name)
    normal["output_artifacts"]["wave_kspace"] = cfl_record(normal_path.parent / "wave_kspace")
    normal_path.write_text(json.dumps(normal))
    shutil.rmtree(root / "vcc2/retro")
    with patch("wave_retro_lr.mprage.prepare_normal_mprage", return_value=normal):
        prepare_retro_mprage(root / "raw.dat", root, root / "scan.seq", virtual_coils=2)
        prepare_retro_mprage_r3x3(root / "raw.dat", root, root / "scan.seq", virtual_coils=2)
    for path in (root / "vcc2/retro").rglob("manifest.json"):
        payload = read_json(path)
        payload.pop("noise_prewhitening")
        path.write_text(json.dumps(payload))
    branch_path = root / "vcc2/normal/bart_output/wavelet_selected/reconstruction_manifest.json"
    branch = read_json(branch_path)
    for key in ("normal_manifest", "prepared_manifest"):
        branch["request"][key]["sha256"] = sha256_file(normal_path)
    branch["request"]["sampling"] = normal["sampling"]
    branch["request"]["inputs"]["kspace_by_echo"] = [cfl_record(normal_path.parent / "wave_kspace")]
    branch_path.write_text(json.dumps(branch))


class LegacyRetroTests(unittest.TestCase):
    """Cover missing legacy keys without ignoring changed metadata or data hashes."""

    def test_whitening_records_are_not_generically_equivalent(self):
        """Only absent/absent and the exact legacy-parent sentinel are compatible."""
        legacy = {"version": 0, "status": "legacy_unwhitened_source", "applied": False,
                  "request": {"enabled": False}}
        self.assertTrue(_retro_prewhitening_matches({}, {}))
        self.assertTrue(_retro_prewhitening_matches({"noise_prewhitening": legacy}, {}))
        for record in (None, {}, {"applied": False}, {"applied": True},
                       {"applied": False, "request": {"enabled": True}}):
            self.assertFalse(_retro_prewhitening_matches({}, {"noise_prewhitening": record}))
            self.assertFalse(_retro_prewhitening_matches({"noise_prewhitening": record}, {}))
        self.assertTrue(_retro_prewhitening_matches({"noise_prewhitening": legacy}, {"noise_prewhitening": legacy}))

    def test_real_three_case_reuse_and_strict_rejection(self):
        """Exercise both actual reuse APIs, rejecting altered case/source/count/whitening/hash."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            automatic_fixture(root, keep_original_maps=True)
            upgrade_retro_fixture(root)
            normal = read_json(root / "vcc2/normal/bart_inputs/manifest.json")
            with patch("wave_retro_lr.mprage.prepare_normal_mprage", return_value=normal):
                for case in ("native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3"):
                    kwargs = {"virtual_coils": 2}
                    prepare = prepare_retro_mprage_r3x3 if case == "native_r3x3" else prepare_retro_mprage
                    if case != "native_r3x3":
                        kwargs["case_ids"] = (case,)
                    path = root / "vcc2/retro" / case / "bart_inputs/manifest.json"
                    original = path.read_bytes()
                    prepare(root / "raw.dat", root, root / "scan.seq", **kwargs)
                    self.assertEqual(path.read_bytes(), original)
                    mutations = {"noise_prewhitening": {"applied": True}, "source": {},
                                 "source_normal_manifest_identity": {"sha256": "wrong"},
                                 "case": {}, "virtual_coils": 3}
                    for key, value in mutations.items():
                        with self.subTest(case=case, field=key):
                            payload = json.loads(original)
                            payload[key] = value
                            path.write_text(json.dumps(payload))
                            with self.assertRaises(ValueError):
                                prepare(root / "raw.dat", root, root / "scan.seq", **kwargs)
                    payload = json.loads(original)
                    payload["output_artifacts"]["wave_kspace"]["payload_sha256"] = "wrong"
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, "artifact changed"):
                        prepare(root / "raw.dat", root, root / "scan.seq", **kwargs)
                    path.write_bytes(original)
