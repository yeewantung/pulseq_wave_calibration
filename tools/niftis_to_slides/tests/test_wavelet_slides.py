"""All-subject decks select a unique wavelet baseline and contain only images."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pptx import Presentation

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT / "scripts"))

from build_wavelet_slides import build, discover
from test_insert_baselines import pictures


def write(root, subject, contrast, container, branch, *, echo=None, nested=False):
    directory = root / subject
    if nested:
        directory /= subject
    directory = directory / contrast / container / branch / "normal"
    directory.mkdir(parents=True, exist_ok=True)
    token = f"native_r3x1_echo-{echo}_acq-wave" if echo else "normal"
    algorithm = "WaveletSelectedVCC24" if container == "vcc24_original" else "OptimalWavelet"
    path = directory / f"sub-{token}_part-mag_BARTWaveNormal{algorithm}.nii.gz"
    data = np.arange(12 * 14 * 16, dtype=np.float32).reshape(12, 14, 16) + 1
    nib.save(nib.Nifti1Image(data, np.diag([1., 1., 2., 1.])), path)
    return path


def collection(root):
    original = write(root, "FMP_1", "MPRAGE_preGad", "original_nifti", "optimal_wavelet")
    write(root, "FMP_1", "MPRAGE_preGad", "head_masked_nifti", "optimal_wavelet")
    rovir = write(root, "FMP_1", "MPRAGE_preGad", "original_nifti", "rovir_optimal_wavelet")
    vcc = write(root, "FMP_1", "MPRAGE_preGad", "vcc24_original", "wavelet_selected_vcc24")
    # Model and phase files in the same directory must not enter the deck.
    vcc.with_name(vcc.name.replace(".nii.gz", "_uncond.nii.gz")).write_bytes(vcc.read_bytes())
    vcc.with_name(vcc.name.replace("part-mag", "part-phase")).write_bytes(vcc.read_bytes())
    write(root, "FMP_1", "SWI_preGad", "vcc24_original", "wavelet_selected_vcc24", echo="01")
    write(root, "FMP_1", "SWI_preGad", "original_nifti", "selected_wavelet", echo="02")
    write(root, "FMP_2", "MPRAGE_preGad", "original_nifti", "optimal_wavelet", nested=True)
    write(root, "FMP_10", "MPRAGE_postGad", "original_nifti", "rovir_optimal_wavelet")
    return original, vcc, rovir


def test_selection_preserves_all_subjects_echoes_and_fallbacks_without_duplicates(tmp_path):
    original, vcc, rovir = collection(tmp_path)
    selected, inventory = discover(tmp_path)
    assert len(selected) == 5 and len(inventory) == 7
    assert selected[0].path == rovir
    assert original not in {item.path for item in selected}
    assert vcc not in {item.path for item in selected}
    assert selected[1].container == "vcc24_original"
    assert [item.subject for item in selected] == ["FMP_1", "FMP_1", "FMP_1", "FMP_2", "FMP_10"]
    assert [item.echo for item in selected[1:3]] == ["1", "2"]
    assert selected[2].branch == "selected_wavelet"
    assert "ROVir" in selected[-1].title


def test_build_has_only_identified_image_pages_and_records_selection(tmp_path):
    root = tmp_path / "input"
    collection(root)
    output = tmp_path / "output/wavelet.pptx"
    assert build(root, output, template=TOOL_ROOT / "assets/slide_template.pptx") == 5
    prs = Presentation(output)
    assert len(prs.slides) == 5
    for slide in prs.slides:
        assert slide.slide_layout.name == "Title and Content"
        assert len(pictures(slide)) == 1
        assert "R3x1, wavelet" in slide.shapes.title.text
        assert slide.shapes.title.text.startswith("FMP ")
    with (output.parent / "wavelet_assets/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 and len({row["source"] for row in rows}) == 5
    assert rows[0]["branch"] == "rovir_optimal_wavelet"
    assert all(float(row["intensity_scale"]) == 1 for row in rows)
    assert all(row["orientation"] == "RAS" for row in rows)


def test_selection_refuses_same_priority_duplicates(tmp_path):
    path = write(tmp_path, "FMP_1", "MPRAGE_preGad", "original_nifti", "optimal_wavelet")
    path.with_name(path.name.replace("BART", "BARTduplicate")).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Ambiguous"):
        discover(tmp_path)


def test_selection_reports_subjects_without_a_baseline(tmp_path):
    write(tmp_path, "FMP_1", "MPRAGE_preGad", "original_nifti", "optimal_wavelet")
    (tmp_path / "FMP_2").mkdir()
    with pytest.raises(ValueError, match="FMP_2"):
        discover(tmp_path)
