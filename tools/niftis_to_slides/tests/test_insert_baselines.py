"""Baseline insertion must preserve reviewed pages and match the correct echo."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import pytest
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT / "scripts"))

from build_retrospective_slides import build
from insert_baseline_slides import find_baseline, insert
from test_retrospective import write_volume


def write_baseline(reference_root, source, echo, *, container="vcc24_original"):
    directory = (reference_root / "FMP_1/SWI_preGad" / container /
                 "wavelet_selected_vcc24/normal")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"sub-native_r3x1_echo-{echo}_part-mag_"
                        "BARTWaveGRENormalWaveletSelectedVCC24.nii.gz")
    image = nib.load(source)
    nib.save(nib.Nifti1Image(image.get_fdata() * 4, image.affine), path)
    return path


def pictures(slide):
    return [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]


def test_insert_preserves_original_pages_and_matches_echo_and_container(tmp_path):
    root, references = tmp_path / "input", tmp_path / "references"
    for echo in ("01", "02"):
        source = write_volume(root, "native_r3x2", echo=echo)
        write_volume(root, "native_r3x3", echo=echo)
        write_baseline(references, source, echo)
        # A duplicate in another container must not be selected.
        write_baseline(references, source, echo, container="original_nifti")
    template = TOOL_ROOT / "assets/slide_template.pptx"
    original_path = tmp_path / "comparison.pptx"
    build(root, original_path, template=template, reference_variant="native_r3x2")
    original = Presentation(original_path)
    out = tmp_path / "updated/comparison.pptx"
    assert insert(root, references, original_path, out, template=template) == 2
    updated = Presentation(out)
    assert len(updated.slides) == 7
    assert updated.slides[1].shapes.title.text == "Wave R3x1 – wavelet, echo 1"
    assert updated.slides[4].shapes.title.text == "Wave R3x1 – wavelet, echo 2"
    for old_slide, new_index in zip(original.slides, (0, 2, 3, 5, 6)):
        new_slide = updated.slides[new_index]
        assert old_slide._element.xml == new_slide._element.xml
        assert [p.image.blob for p in pictures(old_slide)] == [p.image.blob for p in pictures(new_slide)]
    for index in (1, 4):
        baseline, comparison = pictures(updated.slides[index])[0], pictures(updated.slides[index + 1])[0]
        assert (baseline.left, baseline.top, baseline.width, baseline.height) == (
            comparison.left, comparison.top, comparison.width, comparison.height)
        assert baseline.image.blob == comparison.image.blob  # gain 1/4 exactly matches
    with (out.parent / "comparison_assets/baseline_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["intensity_scale"]) for row in rows] == [.25, .25]
    assert all("vcc24_original" in row["baseline_source"] for row in rows)


def test_find_baseline_refuses_missing_or_ambiguous_matches(tmp_path):
    root, references = tmp_path / "input", tmp_path / "references"
    source = write_volume(root, "native_r3x2")
    baseline = write_baseline(references, source, "02")
    row = dict(subject="FMP_1", contrast="SWI_preGad", source=str(source.relative_to(root)))
    with pytest.raises(ValueError, match="found 0"):
        find_baseline(references, row)
    matched = write_baseline(references, source, "01")
    matched.with_name(matched.name.replace(".nii.gz", "_duplicate.nii.gz")).write_bytes(matched.read_bytes())
    with pytest.raises(ValueError, match="found 2"):
        find_baseline(references, row)
