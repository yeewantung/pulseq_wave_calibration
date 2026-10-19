"""Strict reuse of coil-independent accepted MPRAGE PSF artifacts during fresh PCA preparation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .bart_io import cfl_record, open_cfl
from .metal_workflow import file_record, read_json, validate_cfl


PSF_NAMES = ("psf", "wave_trajectory", "psf_coefficients", "psf_coefficients_raw",
             "psf_coefficients_processing_input", "psf_coefficient_c_branch_turns")


def accepted_psf(inputs: Path, twix: Path, sequence: Path, settings: dict,
                 geometry: dict | None = None) -> dict:
    """Return an immutable source record after source/settings/geometry/artifact checks.

    Only standard MPRAGE or a ROVir source anchor is eligible, never projected
    ROVir maps. Existing corrupt artifacts raise rather than trigger a new fit.
    Explicit requested fitting bounds must equal the historical requested bounds.
    No source files are written and no scientific fitting is performed.
    """
    from .mprage import _file_identity

    inputs = inputs.resolve()
    manifest = read_json(inputs/"manifest.json")
    if manifest.get("status") not in ("measured_wave_mprage_bart_inputs_ready", "mprage_rovir_source_anchor_ready"):
        raise ValueError("Unsupported accepted PSF source")
    source = manifest.get("source", {})
    if (source.get("twix") != _file_identity(twix)
            or source.get("sequence") != _file_identity(sequence, include_hash=True)):
        raise ValueError("Accepted PSF belongs to different raw/sequence sources")
    psf = manifest["psf_calibration"]
    if psf.get("coefficient_processing") not in ("smooth", "sine-line"):
        raise ValueError("Accepted PSF processing mode is not recorded")
    for key in ("requested_fit_kx_range", "requested_fit_y_range", "requested_fit_z_range"):
        if settings.get(key) is not None and psf.get(key) != settings[key]:
            raise ValueError("Explicit PSF bounds conflict with accepted source")
    if settings.get("coefficient_processing") == "smooth" and psf.get("coefficient_processing") != "smooth":
        raise ValueError("Explicit smooth PSF request conflicts with accepted source")
    if geometry is not None and manifest["geometry"] != geometry:
        raise ValueError("Accepted PSF geometry differs from requested sequence")
    records = manifest.get("output_artifacts", {})
    if not records:
        records = manifest.get("artifacts", {})
    checked = {}
    for name in PSF_NAMES:
        if name not in records:
            raise ValueError(f"Accepted PSF lacks hash-bound artifact: {name}")
        record = cfl_record(inputs/name)
        if any(record.get(k) != v for k, v in records[name].items() if k in record):
            raise ValueError(f"Accepted PSF artifact hash mismatch: {name}")
        if not all(k in records[name] for k in ("header_sha256", "payload_sha256")):
            raise ValueError("Accepted PSF requires explicit header/payload hashes")
        checked[name] = validate_cfl(record)
    matrix = tuple(manifest["geometry"]["logical_matrix_ro_lin_par"])
    ro = matrix[0]*manifest["geometry"]["readout_oversampling_factor"]
    if (tuple(checked["psf"]["shape"][:3]) != (ro, *matrix[1:])
            or any(n != 1 for n in checked["psf"]["shape"][3:])
            or tuple(checked["wave_trajectory"]["shape"]) != (ro, 2)):
        raise ValueError("Accepted PSF/trajectory shape mismatch")
    for name, width in (("psf_coefficients", 3), ("psf_coefficients_raw", 3),
                        ("psf_coefficients_processing_input", 3), ("psf_coefficient_c_branch_turns", 1)):
        shape = checked[name]["shape"]
        if shape[0] != ro or int(np.prod(shape[1:])) != width:
            raise ValueError("Accepted PSF coefficient shape mismatch")
    return {"manifest": file_record(inputs/"manifest.json"), "artifacts": checked,
            "geometry": manifest["geometry"], "historical_calibration": copy.deepcopy(psf)}


def psf_tuple(record: dict) -> tuple:
    """Return existing trajectory/coefficient arrays in the calibration API format without fitting."""
    arrays = {}
    for name, identity in record["artifacts"].items():
        validate_cfl(identity)
        arrays[name] = np.asarray(open_cfl(identity["base"]))
    trajectory = arrays["wave_trajectory"].real
    coeff = arrays["psf_coefficients"].real
    raw = arrays["psf_coefficients_raw"].real
    processed = arrays["psf_coefficients_processing_input"].real
    for name in PSF_NAMES[1:]:
        if np.any(arrays[name].imag):
            raise ValueError("Accepted trajectory/coefficients must be real-valued")
    diagnostic = copy.deepcopy(record["historical_calibration"].get("processing_diagnostics", {}))
    diagnostic["kx_range"] = record["historical_calibration"].get("fit_kx_range")
    return (trajectory[:, 0], trajectory[:, 1], *[coeff[:, i] for i in range(3)],
            tuple(raw[:, i] for i in range(3)), tuple(processed[:, i] for i in range(3)),
            arrays["psf_coefficient_c_branch_turns"].real.reshape(-1), diagnostic, {})
