from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recon" / "recon_wave_mprage_retro_lr_us_batch.py"
spec = importlib.util.spec_from_file_location("retro", SCRIPT)
assert spec is not None and spec.loader is not None
retro = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = retro
spec.loader.exec_module(retro)


def geometry() -> retro.SourceGeometry:
    return retro.SourceGeometry(
        physical_matrix_xyz=(256, 256, 192),
        physical_fov_mm_xyz=(256.0, 256.0, 192.0),
        logical_matrix_ro_lin_par=(192, 256, 256),
        logical_fov_mm_ro_lin_par=(192.0, 256.0, 256.0),
        readout_oversampling_factor=4,
        ncalib=72,
        nacs=32,
    )


def test_center_crop_preserves_python_center() -> None:
    for source in (255, 256):
        for target in (127, 128, 129, 130):
            left, right = retro.center_crop_bounds(source, target)
            assert right - left == target
            assert left + target // 2 == source // 2


def test_infer_acceleration_from_centered_mask() -> None:
    nlin, npar = 256, 192
    lin = ((np.arange(nlin) - nlin // 2) % 3) == 0
    par = np.ones(npar, dtype=bool)
    mask = lin[:, None] & par[None, :]
    assert retro.infer_source_acceleration(mask) == (3, 1)


def test_retrospective_mask_keeps_center() -> None:
    mask = retro.make_retrospective_mask(171, 256, (3, 1), (3, 2))
    assert mask[171 // 2, 256 // 2]
    assert np.all(mask[:, 256 // 2][((np.arange(171) - 171 // 2) % 3) == 0])


def test_cannot_accelerate_already_accelerated_axis() -> None:
    with pytest.raises(ValueError, match="already accelerated"):
        retro.make_retrospective_mask(64, 64, (3, 1), (6, 1))


def test_resolution_resolves_in_physical_xyz_order() -> None:
    requested = retro.RequestedCase((1.5, 1.0, 1.0), (3, 2))
    case = retro.resolve_case(requested, geometry(), (3, 1))
    assert case.target_physical_matrix_xyz == (171, 256, 192)
    assert case.target_logical_matrix_ro_lin_par == (192, 256, 171)
    assert case.case_name == "res1.5x1x1mm_R3x2"
    assert case.achieved_resolution_mm_xyz[0] == pytest.approx(256.0 / 171)


def test_readout_resolution_cannot_change() -> None:
    requested = retro.RequestedCase((1.5, 1.0, 1.5), (3, 1))
    with pytest.raises(ValueError, match="readout resolution"):
        retro.resolve_case(requested, geometry(), (3, 1))


def test_sampling_mask_is_not_saved_or_needed() -> None:
    kspace = np.zeros((8, 7, 5, 2), dtype=np.complex64)
    kspace[:, 3, 2, :] = 1
    mask = retro.infer_sampling_mask(kspace)
    assert mask.shape == (7, 5)
    assert mask.sum() == 1


def test_target_psf_shape_and_finiteness() -> None:
    nx_os = 8
    delta_y = np.linspace(-1.0, 1.0, nx_os, dtype=np.float32)
    delta_z = np.linspace(1.0, -1.0, nx_os, dtype=np.float32)
    a = np.zeros(nx_os, dtype=np.float32)
    b = np.zeros(nx_os, dtype=np.float32)
    c = np.zeros(nx_os, dtype=np.float32)
    psf = retro.build_target_psf(delta_y, delta_z, a, b, c, 7, 5, -1, -1)
    assert psf.shape == (8, 7, 5)
    assert psf.dtype == np.complex64
    assert np.all(np.isfinite(psf))
    assert np.allclose(np.abs(psf), 1.0)


def test_csm_interpolation_has_target_shape_and_unit_rss() -> None:
    csm = np.ones((2, 4, 3, 3), dtype=np.complex64)
    target = retro.interpolate_target_csm(csm, (4, 5, 7))
    assert target.shape == (2, 4, 5, 7)
    rss = np.sqrt(np.sum(np.abs(target) ** 2, axis=0))
    assert np.allclose(rss, 1.0, atol=1e-5)
