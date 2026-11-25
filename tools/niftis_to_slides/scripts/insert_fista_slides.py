#!/usr/bin/env python3
"""Insert original no-regularization FISTA pages after each uncond pair.

Uses the retrospective deck's manifest to match subject, contrast, container,
variant, echo and acquisition. All existing slides and images are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
from pptx.enum.shapes import MSO_SHAPE_TYPE

from build_retrospective_slides import (
    TOOL_ROOT, Volume, canonical_magnitude, physical_strip, slide_title,
)
from insert_baseline_slides import echo_acquisition
from nifti_slides import deck, images, metrics
from nifti_slides.discovery import subject_contrast_dir


def find_fista(reference_root: Path, row: dict[str, str]) -> Path:
    """Require a unique original FISTAR0 in the corresponding input directory."""
    directory = subject_contrast_dir(reference_root, row["subject"], row["contrast"])
    if directory is None:
        raise FileNotFoundError(f"Missing reference scan: {row['subject']}/{row['contrast']}")
    relative = Path(row["source"])
    parts = relative.parts
    index = parts.index("retro")
    if parts[index - 1] != "fista_r0":
        raise ValueError(f"Expected a no-regularization fista_r0 input: {relative}")
    reconstruction_dir = directory.joinpath(*parts[2:-1])
    candidates = [path for path in reconstruction_dir.glob("*.nii*")
                  if "part-mag" in path.name
                  and re.search(r"FISTA_?R0\.nii(?:\.gz)?$", path.name, re.IGNORECASE)
                  and echo_acquisition(path.name) == echo_acquisition(relative.name)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one original FISTAR0 for {relative}, "
                         f"found {len(candidates)}: {candidates}")
    return candidates[0]


def insertion_plan(builder: deck.DeckBuilder, rows: list[dict[str, str]],
                   reference_root: Path) -> list:
    """Locate reviewed uncond groups and resolve all raw inputs before writing."""
    indexed = []
    section = ""
    for index, slide in enumerate(builder.presentation.slides):
        title = slide.shapes.title.text if slide.shapes.title else ""
        if slide.slide_layout.name == deck.SECTION_LAYOUT:
            section = title
        if any(s.shape_type == MSO_SHAPE_TYPE.PICTURE for s in slide.shapes):
            if not title.startswith(deck.BASELINE_TITLE):
                indexed.append((index, title, section))
    if not rows or len(indexed) != len(rows):
        raise ValueError("Deck picture count does not match the uncond manifest; "
                         "use the deck before FISTA insertion.")
    groups = OrderedDict()
    for (index, title, section), row in zip(indexed, rows):
        expected_section = f"{row['subject']} {row['contrast']}".replace("_", " ")
        acceleration = deck.acceleration_label(row["variant"])
        echo, _ = echo_acquisition(Path(row["source"]).name)
        if (section != expected_section or not title.startswith(f"Wave {acceleration} – ")
                or not title.endswith(", uncond")
                or (echo is not None and f", echo {echo}," not in title)):
            raise ValueError(f"Uncond page does not match manifest row: {section}: {title}")
        groups.setdefault(row["display_reference"], []).append(
            (index, row, find_fista(reference_root, row)))
    plan = []
    for group in groups.values():
        indices = [item[0] for item in group]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("Uncond comparison groups must be contiguous.")
        previous = builder.presentation.slides[indices[0] - 1]
        if not previous.shapes.title.text.startswith(deck.BASELINE_TITLE):
            raise ValueError("Each uncond group must be preceded by its R3x1 baseline.")
        plan.append((indices[-1] + 1, group))
    return plan


def insert(input_root: Path, reference_root: Path, source_deck: Path, output: Path,
           *, template: Path, manifest: Path | None = None) -> int:
    manifest = manifest or source_deck.parent / f"{source_deck.stem}_assets/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    builder = deck.DeckBuilder(template, existing_path=source_deck)
    plan = insertion_plan(builder, rows, reference_root)
    assets = output.parent / f"{output.stem}_assets"
    fista_rows = []
    inserted = 0
    for target_index, group in plan:
        first = group[0][1]
        reference = canonical_magnitude(input_root / first["display_reference"])
        reference_data = reference.get_fdata(dtype=np.float32)
        foreground = metrics.build_foreground(reference_data, reference,
                                               mode="intensity").mask
        for _, row, path in group:
            source = canonical_magnitude(path)
            data, resampled = images.resample_to_baseline(source, reference)
            shown, gain = metrics.scale_to_baseline(reference_data, data, foreground,
                                                   row["scale_mode"])
            display_max = float(row["display_max"])
            strip = physical_strip(shown, nib.affines.voxel_sizes(reference.affine))
            relative = path.relative_to(reference_root)
            group_tag = hashlib.sha256(row["display_reference"].encode()).hexdigest()[:12]
            tiff = assets / "original_fista" / relative.parent / group_tag / f"{path.name}.tiff"
            images.write_strip_tiff(strip, display_max, tiff)
            raster, width, height = deck.brightened_png_bytes(tiff)
            parts = Path(row["source"]).parts
            volume = Volume(path, row["subject"], row["contrast"],
                            parts[2:parts.index("retro")], row["variant"], "")
            title = slide_title(volume, tuple(nib.affines.voxel_sizes(source.affine))) + ", FISTA"
            builder.add_picture_slide(deck.SlideSpec(title, raster, width, height,
                                                    with_axes=False))
            entries = builder.presentation.slides._sldIdLst
            added = entries[-1]
            entries.remove(added)
            entries.insert(target_index + inserted, added)
            inserted += 1
            fista_rows.append(dict(subject=row["subject"], contrast=row["contrast"],
                                   variant=row["variant"], fista_source=str(relative),
                                   comparison_source=row["source"],
                                   display_reference=row["display_reference"],
                                   display_max=display_max, intensity_scale=gain,
                                   scale_mode=row["scale_mode"], resampled=resampled,
                                   image=str(tiff.relative_to(output.parent))))
            print(f"{row['subject']}/{row['contrast']}: {title}", flush=True)
    builder.save(output)
    with (assets / "fista_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fista_rows[0]))
        writer.writeheader()
        writer.writerows(fista_rows)
    print(f"Inserted {inserted} FISTA pages; {len(builder.presentation.slides)} slides: {output}",
          flush=True)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Uncond NIfTI collection")
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
