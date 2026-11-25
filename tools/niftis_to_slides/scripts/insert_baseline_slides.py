#!/usr/bin/env python3
"""Insert matched R3x1 wavelet pages into a retrospective comparison deck.

Existing slide XML and images are retained. Baselines use the same display
reference, window, intensity matching, physical aspect ratio and brightness as
the corresponding retrospective pages, with one baseline per comparison group.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from pptx.enum.shapes import MSO_SHAPE_TYPE

from build_retrospective_slides import TOOL_ROOT, canonical_magnitude, physical_strip
from nifti_slides import deck, images, metrics
from nifti_slides.discovery import subject_contrast_dir


def echo_acquisition(name: str) -> tuple[str | None, str | None]:
    echo = re.search(r"(?:^|_)echo-(\d+)(?:_|\.)", name)
    acquisition = re.search(r"(?:^|_)acq-([^_\.]+)", name)
    return (str(int(echo[1])) if echo else None,
            acquisition[1] if acquisition else None)


def find_baseline(reference_root: Path, row: dict[str, str]) -> Path:
    """Require a unique unmasked wavelet normal/R3x1 in the same container."""
    directory = subject_contrast_dir(reference_root, row["subject"], row["contrast"])
    if directory is None:
        raise FileNotFoundError(f"Missing reference subject/contrast: {row['subject']}/{row['contrast']}")
    relative = Path(row["source"])
    index = relative.parts.index("retro")
    container = directory.joinpath(*relative.parts[2:index - 1])
    wanted = echo_acquisition(relative.name)
    candidates = []
    for path in container.rglob("*.nii*"):
        if not path.name.endswith((".nii", ".nii.gz")) or "part-mag" not in path.name:
            continue
        if path.parent.name != "normal" or "wavelet" not in path.parent.parent.name.lower():
            continue
        if not (path.name.startswith("sub-normal_") or
                re.search(r"r3x1", path.name, re.IGNORECASE)):
            continue
        if echo_acquisition(path.name) == wanted:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one matched R3x1 wavelet baseline for {relative}, "
                         f"found {len(candidates)}: {candidates}")
    return candidates[0]


def insert(input_root: Path, reference_root: Path, source_deck: Path, output: Path,
           *, template: Path, manifest: Path | None = None) -> int:
    manifest = manifest or source_deck.parent / f"{source_deck.stem}_assets/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Retrospective manifest is empty.")
    builder = deck.DeckBuilder(template, existing_path=source_deck)
    picture_indices = [index for index, slide in enumerate(builder.presentation.slides)
                       if any(shape.shape_type == MSO_SHAPE_TYPE.PICTURE
                              for shape in slide.shapes)]
    if len(picture_indices) != len(rows):
        raise ValueError("Deck picture count does not match the retrospective manifest; "
                         "use the original comparison deck before baseline insertion.")
    plan = []
    seen = set()
    for index, row in zip(picture_indices, rows):
        key = row["display_reference"]
        if key not in seen:
            seen.add(key)
            plan.append((index, row, find_baseline(reference_root, row)))
    # Resolve all matches before writing any output.
    assets = output.parent / f"{output.stem}_assets"
    baseline_rows = []
    for offset, (index, row, baseline_path) in enumerate(plan):
        reference = canonical_magnitude(input_root / row["display_reference"])
        reference_data = reference.get_fdata(dtype=np.float32)
        foreground = metrics.build_foreground(reference_data, reference,
                                               mode="intensity").mask
        baseline = canonical_magnitude(baseline_path)
        data, resampled = images.resample_to_baseline(baseline, reference)
        shown, gain = metrics.scale_to_baseline(reference_data, data, foreground,
                                               row["scale_mode"])
        display_max = float(row["display_max"])
        strip = physical_strip(shown, nib.affines.voxel_sizes(reference.affine))
        relative = baseline_path.relative_to(reference_root)
        group_tag = hashlib.sha256(row["display_reference"].encode()).hexdigest()[:12]
        tiff = assets / "baselines" / relative.parent / group_tag / f"{relative.name}.tiff"
        images.write_strip_tiff(strip, display_max, tiff)
        raster, width, height = deck.brightened_png_bytes(tiff)
        echo, _ = echo_acquisition(baseline_path.name)
        title = deck.BASELINE_TITLE + (f", echo {echo}" if echo else "")
        builder.add_picture_slide(deck.SlideSpec(title, raster, width, height,
                                                with_axes=False))
        # Insert immediately before the first original page of this group.
        entries = builder.presentation.slides._sldIdLst
        added = entries[-1]
        entries.remove(added)
        entries.insert(index + offset, added)
        baseline_rows.append(dict(subject=row["subject"], contrast=row["contrast"],
                                  baseline_source=str(relative),
                                  display_reference=row["display_reference"],
                                  display_max=display_max, intensity_scale=gain,
                                  scale_mode=row["scale_mode"], resampled=resampled,
                                  image=str(tiff.relative_to(output.parent))))
        print(f"{row['subject']}/{row['contrast']}: {title}", flush=True)
    builder.save(output)
    with (assets / "baseline_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    print(f"Inserted {len(plan)} baselines; {len(builder.presentation.slides)} slides: {output}",
          flush=True)
    return len(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Retrospective NIfTI collection")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--template", type=Path,
                        default=TOOL_ROOT / "assets/slide_template.pptx")
    args = parser.parse_args()
    insert(args.input_root.expanduser().resolve(), args.reference_root.expanduser().resolve(),
           args.deck.expanduser().resolve(), args.output.expanduser().resolve(),
           template=args.template, manifest=args.manifest)


if __name__ == "__main__":
    main()
