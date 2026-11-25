#!/usr/bin/env python3
"""Build an image-only deck of all subjects' R3x1 wavelet magnitude baselines.

Choose one reconstruction per subject/contrast/echo/acquisition. Prefer ROVir
whenever available, then VCC24, then original optimal/selected wavelet.
Every page includes its subject and contrast in the title; no divider pages.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from build_retrospective_slides import TOOL_ROOT, canonical_magnitude, natural_key, physical_strip
from insert_baseline_slides import echo_acquisition
from nifti_slides import deck, images, metrics
from nifti_slides.discovery import find_subjects


BRANCH_PRIORITY = (
    ("original_nifti", "rovir_optimal_wavelet"),
    ("vcc24_original", "wavelet_selected_vcc24"),
    ("original_nifti", "optimal_wavelet"),
    ("original_nifti", "selected_wavelet"),
)
CONTRAST_PRIORITY = ("MPRAGE_preGad", "MPRAGE_postGad", "SWI_preGad", "SWI_postGad")


@dataclass(frozen=True)
class Baseline:
    path: Path
    subject: str
    contrast: str
    container: str
    branch: str
    echo: str | None
    acquisition: str | None
    priority: int

    @property
    def key(self) -> tuple:
        return self.subject, self.contrast, self.echo, self.acquisition

    @property
    def title(self) -> str:
        title = f"{self.subject.replace('_', ' ')} {self.contrast.replace('_', ' ')} – Wave R3x1, wavelet"
        if "rovir" in self.branch:
            title += " (ROVir)"
        if self.echo is not None:
            title += f", echo {self.echo}"
        if self.acquisition is not None and self.acquisition != "wave":
            title += f", {self.acquisition}"
        return title


def discover(root: Path) -> tuple[list[Baseline], list[Baseline]]:
    """Select one baseline per scan, retaining an inventory of all alternatives."""
    groups = defaultdict(list)
    inventory = []
    subjects = find_subjects(root)
    for subject in subjects:
        directory = root / subject
        if (directory / subject).is_dir():
            directory /= subject
        for contrast_dir in sorted(directory.iterdir()):
            if not contrast_dir.is_dir():
                continue
            for priority, (container, branch) in enumerate(BRANCH_PRIORITY):
                normal = contrast_dir / container / branch / "normal"
                for path in sorted(normal.glob("*.nii*")):
                    if not path.name.endswith((".nii", ".nii.gz")) or "part-mag" not in path.name:
                        continue
                    # 'normal' is the MPRAGE R3x1 convention; SWI and ROVir
                    # explicitly encode R3x1 in their filenames.
                    if not (path.name.startswith("sub-normal_") or
                            "r3x1" in path.name.lower()):
                        continue
                    if not path.name.endswith(("Wavelet.nii.gz", "Wavelet.nii",
                                               "WaveletSelectedVCC24.nii.gz", "WaveletSelectedVCC24.nii")):
                        continue  # exclude model outputs in the same directory
                    echo, acquisition = echo_acquisition(path.name)
                    item = Baseline(path, subject, contrast_dir.name, container,
                                    branch, echo, acquisition, priority)
                    groups[item.key].append(item)
                    inventory.append(item)
    selected = []
    for candidates in groups.values():
        best_priority = min(item.priority for item in candidates)
        best = [item for item in candidates if item.priority == best_priority]
        if len(best) != 1:
            raise ValueError(f"Ambiguous wavelet baseline for {best[0].key}: {best}")
        selected.append(best[0])
    selected.sort(key=lambda item: (
        natural_key(item.subject),
        CONTRAST_PRIORITY.index(item.contrast) if item.contrast in CONTRAST_PRIORITY else len(CONTRAST_PRIORITY),
        natural_key(item.contrast), int(item.echo or 0), item.acquisition or ""))
    if not selected:
        raise ValueError(f"No R3x1 wavelet baselines found under {root}")
    missing = sorted(set(subjects) - {item.subject for item in selected}, key=natural_key)
    if missing:
        raise ValueError(f"Subjects have no supported R3x1 wavelet baseline: {missing}")
    return selected, inventory


def build(root: Path, output: Path, *, template: Path,
          percentile: float = 99.5) -> int:
    if not 0 < percentile <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    selected, inventory = discover(root)
    assets = output.parent / f"{output.stem}_assets"
    builder = deck.DeckBuilder(template)
    rows = []
    for index, item in enumerate(selected, 1):
        image = canonical_magnitude(item.path)
        data = image.get_fdata(dtype=np.float32)
        foreground = metrics.build_foreground(data, image, mode="intensity").mask
        display_max = float(np.percentile(data[foreground], percentile))
        strip = physical_strip(data, nib.affines.voxel_sizes(image.affine))
        relative = item.path.relative_to(root)
        tiff = assets / "pptx_img" / relative.parent / f"{item.path.name}.tiff"
        images.write_strip_tiff(strip, display_max, tiff)
        raster, width, height = deck.brightened_png_bytes(tiff)
        builder.add_picture_slide(deck.SlideSpec(item.title, raster, width, height,
                                                with_axes=False))
        rows.append(dict(subject=item.subject, contrast=item.contrast, echo=item.echo or "",
                         acquisition=item.acquisition or "", container=item.container,
                         branch=item.branch, source=str(relative), display_max=display_max,
                         display_percentile=percentile, intensity_scale=1.0,
                         display_mask="intensity", orientation="RAS",
                         image=str(tiff.relative_to(output.parent))))
        print(f"[{index}/{len(selected)}] {item.title}", flush=True)
    builder.save(output)
    with (assets / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_paths = {item.path for item in selected}
    with (assets / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "priority", "selected"))
        writer.writeheader()
        writer.writerows(dict(source=str(item.path.relative_to(root)), priority=item.priority,
                              selected=item.path in selected_paths) for item in inventory)
    print(f"Saved {output}: {len({item.subject for item in selected})} subjects, "
          f"{len(selected)} image pages.", flush=True)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--display-percentile", type=float, default=99.5)
    parser.add_argument("--template", type=Path,
                        default=TOOL_ROOT / "assets/slide_template.pptx")
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    output = (args.output or root / "all_subjects_r3x1_wavelet.pptx").expanduser().resolve()
    build(root, output, template=args.template, percentile=args.display_percentile)


if __name__ == "__main__":
    main()
