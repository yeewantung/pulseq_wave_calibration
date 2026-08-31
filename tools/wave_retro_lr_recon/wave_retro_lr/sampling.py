"""Sampling contracts for measured Wave-MPRAGE TWIX data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SamplingPattern:
    """One supported image-stream sampling pattern on a logical PE grid."""

    name: str
    acceleration_lin_par: tuple[int, int]
    lin_residue: int | None
    matrix_lin_par: tuple[int, int]
    acquired_lin: tuple[int, ...]
    acquired_par: tuple[int, ...]
    measurement_index: int | None = None
    skip_lin_par: tuple[int, int] = (0, 0)

    def mask(self) -> np.ndarray:
        """Materialize the image-stream sampling coordinates.

        Returns:
            Boolean logical LIN/PAR sampling mask without calibration data.
        """
        result = np.zeros(self.matrix_lin_par, dtype=bool)
        result[np.ix_(self.acquired_lin, self.acquired_par)] = True
        return result

    def to_json(self) -> dict[str, Any]:
        """Convert the sampling contract to JSON-native provenance.

        Returns:
            Sampling dimensions, coordinates, residue, and center status.
        """
        center = (self.matrix_lin_par[0] // 2, self.matrix_lin_par[1] // 2)
        return {
            "name": self.name,
            "acceleration_lin_par": list(self.acceleration_lin_par),
            "lin_residue": self.lin_residue,
            "matrix_lin_par": list(self.matrix_lin_par),
            "acquired_lin": list(self.acquired_lin),
            "acquired_par": list(self.acquired_par),
            "measurement_index": self.measurement_index,
            "skip_lin_par": list(self.skip_lin_par),
            "image_kspace_center_acquired": bool(self.mask()[center]),
        }


def _integer_counters(values: Sequence[Any], label: str) -> tuple[int, ...]:
    """Validate MDH counters and convert them to exact integers.

    Args:
        values: Numeric counter values read from mapVBVD.
        label: Human-readable field name used in validation errors.

    Returns:
        Validated integer counter tuple.
    """
    converted: list[int] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} contains a non-integral value: {value!r}.")
        converted.append(int(numeric))
    return tuple(converted)


def classify_mprage_sampling(
    lines: Sequence[Any],
    partitions: Sequence[Any],
    *,
    matrix_lin_par: tuple[int, int],
    measurement_index: int | None = None,
    skip_lin_par: tuple[int, int] = (0, 0),
) -> SamplingPattern:
    """Accept only duplicate-free R1 or MPRAGE-compatible R3x1 sampling.

    The supported R3x1 acquisition has a single regular factor-three lattice
    on logical LIN and every logical PAR partition. Integrated calibration is
    handled separately from the image stream and is not folded into this mask.

    Args:
        lines: Image-stream logical LIN counters.
        partitions: Image-stream logical PAR counters paired with ``lines``.
        matrix_lin_par: Declared logical PE matrix dimensions.
        measurement_index: Selected TWIX measurement index for provenance.
        skip_lin_par: mapVBVD compact-payload offset in logical PE coordinates.

    Returns:
        Validated R1 or regular R3x1 sampling contract.
    """

    lin = _integer_counters(lines, "TWIX image.Lin")
    par = _integer_counters(partitions, "TWIX image.Par")
    if not lin or len(lin) != len(par):
        raise ValueError("TWIX image LIN/PAR counters must be non-empty and equally sized.")

    nlin, npar = (int(value) for value in matrix_lin_par)
    coordinates = tuple(zip(lin, par, strict=True))
    unique_coordinates = set(coordinates)
    if len(unique_coordinates) != len(coordinates):
        raise ValueError(
            "Duplicate TWIX image LIN/PAR coordinates are not supported; "
            "averages, repetitions, or ambiguous sampling may be present."
        )
    out_of_range = sorted(
        (line, partition)
        for line, partition in unique_coordinates
        if not (0 <= line < nlin and 0 <= partition < npar)
    )
    if out_of_range:
        raise ValueError(
            f"TWIX image sampling contains out-of-range coordinates: {out_of_range[:8]}."
        )

    acquired_par = tuple(sorted(set(par)))
    if acquired_par != tuple(range(npar)):
        raise ValueError("Supported MPRAGE data must acquire every logical PAR partition.")

    lines_by_partition = {
        partition: tuple(
            sorted(
                line
                for line, current_par in unique_coordinates
                if current_par == partition
            )
        )
        for partition in acquired_par
    }
    first_lines = lines_by_partition[0]
    if any(current != first_lines for current in lines_by_partition.values()):
        raise ValueError(
            "The MPRAGE LIN sampling pattern must be identical for every PAR partition."
        )

    if first_lines == tuple(range(nlin)):
        return SamplingPattern(
            name="R1",
            acceleration_lin_par=(1, 1),
            lin_residue=None,
            matrix_lin_par=(nlin, npar),
            acquired_lin=first_lines,
            acquired_par=acquired_par,
            measurement_index=measurement_index,
            skip_lin_par=skip_lin_par,
        )

    residues = {line % 3 for line in first_lines}
    if len(residues) != 1:
        raise ValueError("Non-R1 input is not one regular factor-three LIN lattice.")
    residue = next(iter(residues))
    expected_lines = tuple(range(residue, nlin, 3))
    if first_lines != expected_lines:
        raise ValueError("The factor-three LIN lattice is incomplete or irregular.")
    return SamplingPattern(
        name="R3x1",
        acceleration_lin_par=(3, 1),
        lin_residue=residue,
        matrix_lin_par=(nlin, npar),
        acquired_lin=first_lines,
        acquired_par=acquired_par,
        measurement_index=measurement_index,
        skip_lin_par=skip_lin_par,
    )


def inspect_twix_sampling(
    twix_path: str | Path, *, matrix_lin_par: tuple[int, int]
) -> tuple[SamplingPattern, Any]:
    """Inspect TWIX MDH counters and select the image measurement.

    Args:
        twix_path: Siemens TWIX file read without modifying it.
        matrix_lin_par: Expected logical PE matrix from the sequence.

    Returns:
        Validated sampling contract and its selected mapVBVD measurement.
    """

    import mapvbvd

    root = mapvbvd.mapVBVD(str(Path(twix_path)), quiet=True)
    measurements = list(root) if isinstance(root, (list, tuple)) else [root]
    candidates: list[tuple[int, int, Any]] = []
    for index, measurement in enumerate(measurements):
        image = measurement.get("image") if hasattr(measurement, "get") else None
        acquisitions = int(getattr(image, "NAcq", 0)) if image is not None else 0
        if acquisitions:
            candidates.append((acquisitions, index, measurement))
    if not candidates:
        raise ValueError("No TWIX measurement contains a populated image stream.")
    _, index, measurement = max(candidates, key=lambda item: (item[0], item[1]))
    # The pinned Wave-MPRAGE twix_import helpers select measurement 1 for a
    # multi-measurement TWIX and measurement 0 otherwise. Refuse a file where
    # that upstream payload selection would disagree with the MDH inspection.
    upstream_index = 1 if len(measurements) > 1 else 0
    if index != upstream_index:
        raise ValueError(
            "The largest TWIX image measurement does not match the pinned "
            f"Wave-MPRAGE loader selection ({index} versus {upstream_index})."
        )
    image = measurement["image"]
    pattern = classify_mprage_sampling(
        image.Lin,
        image.Par,
        matrix_lin_par=matrix_lin_par,
        measurement_index=index,
        skip_lin_par=(int(image.skipLin), int(image.skipPar)),
    )
    return pattern, measurement
