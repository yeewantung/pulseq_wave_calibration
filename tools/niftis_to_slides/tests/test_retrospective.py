"""Standalone decks must pair accelerations without mixing echoes or models."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from pptx import Presentation

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT / "scripts"))

from build_retrospective_slides import build, canonical_magnitude, discover, physical_strip


def write_volume(root, variant, *, echo="01", model="uncond", gain=1):
    directory = root / "FMP_1/SWI_preGad/vcc24_original/fista_r0/retro" / variant
    directory.mkdir(parents=True, exist_ok=True)
    algorithm_variant = variant if variant.endswith("2") else "NativeR3x3"
    path = directory / (f"sub-{variant}_echo-{echo}_part-mag_"
                        f"BARTWaveGRE{algorithm_variant}FISTAR0_{model}.nii.gz")
    data = np.arange(12 * 14 * 16, dtype=np.float32).reshape(12, 14, 16) + 1
    nib.save(nib.Nifti1Image(data * gain, np.diag([-1., 1., 2., 1.])), path)
    return path


def test_groups_pair_accelerations_and_separate_echoes_and_models(tmp_path):
    for variant in ("native_r3x2", "native_r3x3"):
        for echo, model in (("01", "uncond"), ("02", "uncond"), ("01", "shd")):
            write_volume(tmp_path, variant, echo=echo, model=model)
    groups = {}
    for volume in discover(tmp_path):
        groups.setdefault(volume.group, []).append(volume.variant)
    assert len(groups) == 3
    assert all(sorted(group) == ["native_r3x2", "native_r3x3"]
               for group in groups.values())


def test_build_covers_inputs_with_shared_windows_and_physical_orientation(tmp_path):
    root = tmp_path / "input"
    first = write_volume(root, "native_r3x2")
    write_volume(root, "native_r3x3", gain=2)
    canonical = canonical_magnitude(first)
    assert nib.aff2axcodes(canonical.affine) == ("R", "A", "S")
    assert canonical.get_fdata()[0, 0, 0] > canonical.get_fdata()[-1, 0, 0]
    strip = physical_strip(canonical.get_fdata(), (1, 1, 2))
    assert strip.shape == (32, 38)  # physical height doubles along z
    output = tmp_path / "output/comparison.pptx"
    assert build(root, output, template=TOOL_ROOT / "assets/slide_template.pptx",
                 reference_variant="native_r3x2") == (1, 2)
    with (output.parent / "comparison_assets/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["display_max"] == rows[1]["display_max"]
    assert float(rows[1]["intensity_scale"]) == .5
    assert rows[0]["display_reference"] == rows[1]["display_reference"]
    slides = Presentation(output).slides
    assert len(slides) == 3
    assert "R3x2" in slides[1].shapes.title.text
    assert "R3x3" in slides[2].shapes.title.text
    assert "echo 1, uncond" in slides[2].shapes.title.text
