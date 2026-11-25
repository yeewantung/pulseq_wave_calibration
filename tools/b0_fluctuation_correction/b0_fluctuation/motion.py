"""Rigid motion estimation per acquisition segment by Wave data consistency.

With LIN as the outer loop in linear order, each time segment covers a band
of ky lines, so a segment cannot be reconstructed into an image on its own.
Instead, every segment is aligned to a reference image reconstructed from all
data (aligned-SENSE / TAMER style): the rigid transform ``T_theta`` of the
object that minimizes

    || M_s A T_theta x_ref - y_s ||^2 / || y_s ||^2

over the segment's lines ``M_s`` is found by L-BFGS with automatic
differentiation through the Wave operator. Coil sensitivities stay fixed in
the scanner frame, so the object moves relative to them.

Parameters are translations in millimetres (RO, LIN, PAR) and rotations in
degrees about the RO, LIN, and PAR axes through the image centre.

Identifiability caveats:
    * A LIN (A-P) translation is a constant phase for a single-ky segment; it
      is only visible through the changed coil weighting and is degenerate
      with a uniform B0 phase, which the B0 navigator may already absorb.
    * Segments at high |ky| carry little energy, so their estimates are noisy;
      the per-segment energy is reported for weighting.
    * With a sequential LIN order, consecutive segments form a contiguous ky
      band. The least-squares reference absorbs a block of segments acquired
      in another position almost completely, so only the segments at the
      block edges remain inconsistent. Per-segment alignment against that
      reference then returns nearly zero motion (verified on the real Wave-GRE
      geometry: a 1.5 mm / 1 degree block was not recovered). Use
      :func:`segment_residual_profile` to detect such edges; correcting
      motion reliably needs a distributed phase-encoding order or navigators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from .acquisition import LineTiming
from .wave_operator import WaveOperator, fftc, ifftc

PARAMETER_NAMES = ("dx_mm", "dy_mm", "dz_mm", "rx_deg", "ry_deg", "rz_deg")


def rotation_matrix(angles_deg: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``Rz @ Ry @ Rx`` from three angles in degrees.

    Args:
        angles_deg: ``(3,)`` rotations about the RO, LIN, and PAR axes.

    Returns:
        ``(3, 3)`` rotation matrix.
    """
    rx, ry, rz = (angles_deg * math.pi / 180).unbind()
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    rot_x = torch.stack([one, zero, zero, zero, torch.cos(rx), -torch.sin(rx), zero, torch.sin(rx), torch.cos(rx)]).reshape(3, 3)
    rot_y = torch.stack([torch.cos(ry), zero, torch.sin(ry), zero, one, zero, -torch.sin(ry), zero, torch.cos(ry)]).reshape(3, 3)
    rot_z = torch.stack([torch.cos(rz), -torch.sin(rz), zero, torch.sin(rz), torch.cos(rz), zero, zero, zero, one]).reshape(3, 3)
    return rot_z @ rot_y @ rot_x


SHEAR_PADDING_VOXELS = (16, 16, 8)


def _fourier_shear(
    image: torch.Tensor, axis: int, along: int, factor: torch.Tensor, voxel_mm: Sequence[float]
) -> torch.Tensor:
    """Shift every line along ``axis`` by ``factor * coordinate_along`` (mm) with sinc interpolation.

    Args:
        image: ``(X, Y, Z)`` complex image.
        axis: Axis that is shifted.
        along: Axis whose physical coordinate sets the shift.
        factor: Dimensionless shear factor (differentiable scalar).
        voxel_mm: Voxel size in millimetres.

    Returns:
        Sheared image with the original shape. The shifted axis is zero padded
        by :data:`SHEAR_PADDING_VOXELS` to limit circular wrap-around.
    """
    pad = SHEAR_PADDING_VOXELS[axis]
    padding = [0, 0, 0, 0, 0, 0]
    # torch pad order is (last dim lo, hi, middle lo, hi, first lo, hi).
    padding[2 * (2 - axis)] = pad
    padding[2 * (2 - axis) + 1] = pad
    padded = functional.pad(image, padding)
    n_axis = padded.shape[axis]
    n_along = image.shape[along]
    coordinate = (torch.arange(n_along, device=image.device, dtype=torch.float32) - n_along // 2) * float(voxel_mm[along])
    shift_voxels = factor * coordinate / float(voxel_mm[axis])
    frequency = (torch.arange(n_axis, device=image.device, dtype=torch.float32) - n_axis // 2) / n_axis
    view_frequency = [1, 1, 1]
    view_frequency[axis] = n_axis
    view_shift = [1, 1, 1]
    view_shift[along] = n_along
    phase = frequency.reshape(view_frequency) * shift_voxels.reshape(view_shift)
    sheared = ifftc(fftc(padded, (axis,)) * torch.exp(-2j * math.pi * phase), (axis,))
    return sheared.narrow(axis, pad, image.shape[axis])


def fourier_rotate(image: torch.Tensor, angles_deg: torch.Tensor, voxel_mm: Sequence[float]) -> torch.Tensor:
    """Rotate an image by ``R = Rz Ry Rx`` using three Fourier shears per axis.

    The result equals ``image(R^T r)`` (the object rotated by ``R`` about the
    grid centre) without interpolation blur, so the loss is smooth in the
    angles, including at zero. Each plane rotation uses the decomposition
    ``[[1,-t],[0,1]] [[1,0],[s,1]] [[1,-t],[0,1]]`` with ``t = tan(a/2)``
    and ``s = sin(a)``.

    Args:
        image: ``(X, Y, Z)`` complex image.
        angles_deg: ``(3,)`` rotations about the RO, LIN, and PAR axes.
        voxel_mm: Voxel size in millimetres.

    Returns:
        Rotated image.
    """
    radians = angles_deg * math.pi / 180
    # Plane (u, v) of each axis rotation, matching rotation_matrix().
    planes = ((1, 2), (2, 0), (0, 1))
    rotated = image
    for angle, (u, v) in zip(radians.unbind(), planes):
        if not angle.requires_grad and float(angle) == 0.0:
            continue
        tangent = torch.tan(angle / 2)
        sine = torch.sin(angle)
        # Image operators apply in reverse matrix order: shear u, shear v, shear u.
        rotated = _fourier_shear(rotated, u, v, -tangent, voxel_mm)
        rotated = _fourier_shear(rotated, v, u, sine, voxel_mm)
        rotated = _fourier_shear(rotated, u, v, -tangent, voxel_mm)
    return rotated


def rigid_transform(image: torch.Tensor, parameters: torch.Tensor, voxel_mm: Sequence[float]) -> torch.Tensor:
    """Move an image by a rigid transform (Fourier-shear rotation, Fourier shift).

    Bilinear resampling is avoided on purpose: it blurs the image at every
    nonzero angle, which makes zero rotation an artificial local minimum of
    the data-consistency loss and stalls the estimator there.

    Args:
        image: ``(X, Y, Z)`` complex image.
        parameters: ``(6,)`` translations [mm] and rotations [deg].
        voxel_mm: Voxel size in millimetres.

    Returns:
        Transformed complex image (differentiable in ``parameters``).
    """
    shape = image.shape
    device = image.device
    spacing = torch.as_tensor(voxel_mm, dtype=torch.float32, device=device)
    moved = fourier_rotate(image, parameters[3:], voxel_mm)
    shifts = parameters[:3] / spacing
    phase = torch.zeros(shape, dtype=torch.float32, device=device)
    for axis, n in enumerate(shape):
        frequency = (torch.arange(n, device=device, dtype=torch.float32) - n // 2) / n
        view = [1, 1, 1]
        view[axis] = n
        phase = phase + (frequency * shifts[axis]).reshape(view)
    return ifftc(fftc(moved, (0, 1, 2)) * torch.exp(-2j * math.pi * phase), (0, 1, 2))


@dataclass
class MotionResult:
    """Per-segment rigid motion estimates.

    Attributes:
        parameters: ``(S, 6)`` estimates (see :data:`PARAMETER_NAMES`).
        time_s: ``(S,)`` segment centre times.
        lin: ``(S,)`` mean LIN index of each segment.
        energy: ``(S,)`` measured k-space energy of each segment.
        loss_initial: ``(S,)`` normalized residual at zero motion.
        loss_final: ``(S,)`` normalized residual after alignment.
        history: Optional diagnostics.
    """

    parameters: np.ndarray
    time_s: np.ndarray
    lin: np.ndarray
    energy: np.ndarray
    loss_initial: np.ndarray
    loss_final: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)


def segment_indices(timing: LineTiming, lines_per_segment: int) -> list[np.ndarray]:
    """Split time-ordered lines into consecutive segments.

    Args:
        timing: Line timing.
        lines_per_segment: Lines per segment (72 = one LIN sweep here).

    Returns:
        Index arrays into the timing arrays.
    """
    return [np.arange(start, min(start + lines_per_segment, timing.count)) for start in range(0, timing.count, lines_per_segment)]


def segment_residual_profile(
    operator: WaveOperator,
    image: torch.Tensor,
    kspace: torch.Tensor,
    timing: LineTiming,
    lines_per_segment: int = 72,
) -> dict[str, np.ndarray]:
    """Per-segment data residual and a step-motion score.

    With a sequential phase-encoding order, a block of segments acquired in a
    different position is largely absorbed by the least-squares image; the
    inconsistency then concentrates in the segments at the block edges. The
    score compares each segment's normalized residual with the geometric mean
    of its two time neighbours, so smooth k-space trends cancel and edges
    stand out (score well above 1).

    Args:
        operator: Wave operator.
        image: Reconstruction from all data.
        kspace: Measured k-space.
        timing: Line timing.
        lines_per_segment: Lines per segment.

    Returns:
        ``time_s``, ``lin`` (mean per segment), ``residual`` (normalized), and
        ``edge_score`` arrays, one entry per segment (score 1 at both ends).
    """
    segments = segment_indices(timing, lines_per_segment)
    lin_all = torch.as_tensor(timing.lin, device=image.device)
    par_all = torch.as_tensor(timing.par, device=image.device)
    residuals = []
    for members in segments:
        lin = lin_all[members]
        par = par_all[members]
        norm = torch.sum(torch.abs(kspace[:, :, lin, par]) ** 2)
        value, _ = segment_loss(operator, image, kspace, lin, par, norm, gradient=False)
        residuals.append(float(value))
    residual = np.asarray(residuals)
    score = np.ones_like(residual)
    if residual.size >= 3:
        score[1:-1] = residual[1:-1] / np.sqrt(residual[:-2] * residual[2:])
    return {
        "time_s": np.asarray([float(np.mean(timing.time_s[m])) for m in segments]),
        "lin": np.asarray([float(np.mean(timing.lin[m])) for m in segments]),
        "residual": residual,
        "edge_score": score,
    }


def segment_loss(
    operator: WaveOperator,
    moved: torch.Tensor,
    kspace: torch.Tensor,
    lin: torch.Tensor,
    par: torch.Tensor,
    norm: torch.Tensor,
    *,
    gradient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalized residual of one segment and its gradient w.r.t. the moved image.

    The loss is accumulated over coil chunks; each chunk is back-propagated
    immediately so device memory stays bounded.

    Args:
        operator: Wave operator.
        moved: ``(X, Y, Z)`` transformed image (detached leaf).
        kspace: ``(C, Xe, Y, Z)`` measured k-space.
        lin: Segment LIN indices.
        par: Segment PAR indices.
        norm: Squared norm of the segment's measured data.
        gradient: Also return the gradient (False for plain evaluation).

    Returns:
        Loss value and gradient with respect to ``moved`` (``None`` when
        ``gradient`` is False).
    """
    if not gradient:
        with torch.no_grad():
            total = torch.zeros((), dtype=torch.float32, device=moved.device)
            for start in range(0, operator.coils, operator.coil_chunk):
                stop = min(start + operator.coil_chunk, operator.coils)
                residual = operator.coil_forward(operator.maps[start:stop] * moved)[:, :, lin, par] - kspace[start:stop][:, :, lin, par]
                total = total + torch.sum(residual.real**2 + residual.imag**2) / norm
        return total, None
    leaf = moved.detach().requires_grad_(True)
    total = torch.zeros((), dtype=torch.float32, device=moved.device)
    for start in range(0, operator.coils, operator.coil_chunk):
        stop = min(start + operator.coil_chunk, operator.coils)
        predicted = operator.coil_forward(operator.maps[start:stop] * leaf)[:, :, lin, par]
        residual = predicted - kspace[start:stop][:, :, lin, par]
        loss = torch.sum(residual.real**2 + residual.imag**2) / norm
        loss.backward()
        total = total + loss.detach()
    return total, leaf.grad


def estimate_motion(
    operator: WaveOperator,
    reference: torch.Tensor,
    kspace: torch.Tensor,
    timing: LineTiming,
    voxel_mm: Sequence[float],
    *,
    lines_per_segment: int = 72,
    iterations: int = 15,
    estimate_rotation: bool = True,
    initial_parameters: np.ndarray | None = None,
    log: Callable[[str], None] = print,
) -> MotionResult:
    """Align every segment to the reference image by data consistency.

    Args:
        operator: Wave operator (all acquired lines).
        reference: Reference image from all data.
        kspace: Measured (optionally B0-corrected) k-space.
        timing: Line timing.
        voxel_mm: Voxel size in millimetres.
        lines_per_segment: Lines per segment.
        iterations: L-BFGS iterations per segment.
        estimate_rotation: Also estimate the three rotations.
        initial_parameters: Optional ``(S, 6)`` starting values (otherwise
            each segment starts from the previous segment's estimate).
        log: Progress callback.

    Returns:
        Per-segment estimates and residuals.
    """
    device = reference.device
    segments = segment_indices(timing, lines_per_segment)
    lin_all = torch.as_tensor(timing.lin, device=device)
    par_all = torch.as_tensor(timing.par, device=device)
    estimates, times, lins, energies, initial, final = [], [], [], [], [], []
    previous = torch.zeros(6, dtype=torch.float32, device=device)
    for index, members in enumerate(segments):
        lin = lin_all[members]
        par = par_all[members]
        measured = kspace[:, :, lin, par]
        norm = torch.sum(measured.real**2 + measured.imag**2)
        del measured
        # Warm start from a given value or the previous segment (motion is smooth).
        if initial_parameters is not None:
            previous = torch.as_tensor(initial_parameters[index], dtype=torch.float32, device=device)
        free = previous.clone() if estimate_rotation else previous[:3].clone()
        free.requires_grad_(True)
        optimizer = torch.optim.LBFGS([free], lr=1.0, max_iter=iterations, line_search_fn="strong_wolfe")

        def full(values: torch.Tensor) -> torch.Tensor:
            """Pad translation-only parameters with zero rotation."""
            return values if estimate_rotation else torch.cat([values, torch.zeros(3, device=device)])

        def closure() -> torch.Tensor:
            """Loss and parameter gradient by two-stage back-propagation."""
            optimizer.zero_grad()
            moved = rigid_transform(reference, full(free), voxel_mm)
            loss, image_grad = segment_loss(operator, moved, kspace, lin, par, norm)
            moved.backward(image_grad)
            return loss

        zero_loss, _ = segment_loss(operator, reference, kspace, lin, par, norm, gradient=False)
        optimizer.step(closure)
        with torch.no_grad():
            final_moved = rigid_transform(reference, full(free.detach()), voxel_mm)
        final_loss, _ = segment_loss(operator, final_moved, kspace, lin, par, norm, gradient=False)
        values = full(free.detach())
        previous = values.clone()
        estimates.append(values.cpu().numpy())
        times.append(float(np.mean(timing.time_s[members])))
        lins.append(float(np.mean(timing.lin[members])))
        energies.append(float(norm))
        initial.append(float(zero_loss))
        final.append(float(final_loss))
        if index % 10 == 0 or index == len(segments) - 1:
            log(
                f"segment {index + 1}/{len(segments)} (t {times[-1]:.1f} s, LIN {lins[-1]:.0f}): "
                f"residual {initial[-1]:.4f} -> {final[-1]:.4f}, params "
                + ", ".join(f"{name} {value:+.2f}" for name, value in zip(PARAMETER_NAMES, estimates[-1]))
            )
    return MotionResult(
        parameters=np.asarray(estimates),
        time_s=np.asarray(times),
        lin=np.asarray(lins),
        energy=np.asarray(energies),
        loss_initial=np.asarray(initial),
        loss_final=np.asarray(final),
    )


def _transform_adjoint(residual_image: torch.Tensor, parameters: torch.Tensor, voxel_mm: Sequence[float]) -> torch.Tensor:
    """Adjoint of :func:`rigid_transform` for fixed parameters (via a vector-Jacobian product).

    Args:
        residual_image: ``(X, Y, Z)`` image-domain vector.
        parameters: ``(6,)`` fixed motion parameters.
        voxel_mm: Voxel size.

    Returns:
        ``T^H residual_image``.
    """
    with torch.enable_grad():
        probe = torch.zeros_like(residual_image, requires_grad=True)
        moved = rigid_transform(probe, parameters, voxel_mm)
        # For f = Re<T x, r>, PyTorch's complex gradient with respect to x is T^H r.
        value = torch.real(torch.vdot(moved.reshape(-1), residual_image.reshape(-1)))
        (gradient,) = torch.autograd.grad(value, probe)
    return gradient


def group_segments(
    parameters: np.ndarray, segments: Sequence[np.ndarray], quantum: Sequence[float] = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group segments whose quantized motion parameters are identical.

    Args:
        parameters: ``(S, 6)`` motion estimates.
        segments: Line indices per segment.
        quantum: Quantization step per parameter (mm and degrees).

    Returns:
        ``(parameters, line_indices)`` pairs, one per distinct quantized value.
    """
    keys = np.round(parameters / np.asarray(quantum)).astype(np.int64)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        groups.setdefault(key, []).append(index)
    result = []
    for key, members in groups.items():
        lines = np.concatenate([segments[m] for m in members])
        result.append((np.asarray(key, dtype=np.float64) * np.asarray(quantum), lines))
    return result


def motion_aware_cg(
    operator: WaveOperator,
    kspace: torch.Tensor,
    timing: LineTiming,
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    voxel_mm: Sequence[float],
    *,
    iterations: int = 15,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """CG-SENSE where every group of lines is encoded from its own moved object.

    Solves ``min_x sum_g || M_g A T_g x - y_g ||^2``.

    Args:
        operator: Wave operator.
        kspace: Measured k-space.
        timing: Line timing (indexes the groups' line arrays).
        groups: ``(parameters, line_indices)`` pairs.
        voxel_mm: Voxel size.
        iterations: CG iterations.
        initial: Warm start.

    Returns:
        Reconstructed image in the reference position.
    """
    device = kspace.device
    lin_all = torch.as_tensor(timing.lin, device=device)
    par_all = torch.as_tensor(timing.par, device=device)
    prepared = []
    for parameters, lines in groups:
        mask = torch.zeros(operator.line_mask.shape, dtype=torch.bool, device=device)
        mask[lin_all[lines], par_all[lines]] = True
        prepared.append((torch.as_tensor(parameters, dtype=torch.float32, device=device), mask))
    original_mask = operator.line_mask

    def apply_groups(function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], argument: torch.Tensor) -> torch.Tensor:
        """Sum a per-group image-domain operation with the operator masked to that group."""
        total = torch.zeros(operator.image_shape, dtype=torch.complex64, device=device)
        for parameters, mask in prepared:
            operator.line_mask = mask
            operator.mask_float = mask.to(torch.float32)
            total += function(parameters, argument)
        operator.line_mask = original_mask
        operator.mask_float = original_mask.to(torch.float32)
        return total

    def normal(image: torch.Tensor) -> torch.Tensor:
        """``sum_g T_g^H A_g^H A_g T_g image``."""
        return apply_groups(
            lambda p, x: _transform_adjoint(operator.normal(rigid_transform(x, p, voxel_mm)), p, voxel_mm), image
        )

    with torch.no_grad():
        rhs = apply_groups(lambda p, y: _transform_adjoint(operator.adjoint(y), p, voxel_mm), kspace)
        x = torch.zeros_like(rhs) if initial is None else initial.clone()
        residual = rhs - normal(x) if initial is not None else rhs.clone()
        direction = residual.clone()
        rr = torch.vdot(residual.reshape(-1), residual.reshape(-1)).real
        for _ in range(iterations):
            normal_direction = normal(direction)
            alpha = rr / torch.vdot(direction.reshape(-1), normal_direction.reshape(-1)).real
            x = x + alpha * direction
            residual = residual - alpha * normal_direction
            rr_new = torch.vdot(residual.reshape(-1), residual.reshape(-1)).real
            direction = residual + (rr_new / rr) * direction
            rr = rr_new
    return x


def joint_motion_estimation(
    operator: WaveOperator,
    kspace: torch.Tensor,
    timing: LineTiming,
    voxel_mm: Sequence[float],
    *,
    outer_iterations: int = 5,
    lines_per_segment: int = 72,
    motion_iterations: int = 15,
    cg_iterations: int = 15,
    initial_image: torch.Tensor | None = None,
    log: Callable[[str], None] = print,
) -> tuple[MotionResult, torch.Tensor, list[dict[str, float]]]:
    """Alternate per-segment motion estimation and motion-aware reconstruction.

    A single estimation against a least-squares reference underestimates
    motion because that reference partly fits the motion-corrupted data. The
    alternation (aligned-SENSE style) removes this bias progressively. The
    energy-weighted mean motion is removed after every estimation because a
    common rigid transform of all segments is degenerate with moving the image.

    Args:
        operator: Wave operator.
        kspace: Measured (optionally B0-corrected) k-space.
        timing: Line timing.
        voxel_mm: Voxel size.
        outer_iterations: Alternation cycles.
        lines_per_segment: Lines per segment.
        motion_iterations: L-BFGS iterations per segment and cycle.
        cg_iterations: Motion-aware CG iterations per cycle.
        initial_image: Optional starting image (else CG-SENSE without motion).
        log: Progress callback.

    Returns:
        Final motion estimates, motion-corrected image, and per-cycle history.
    """
    from .recon import cg_sense

    image = initial_image if initial_image is not None else cg_sense(operator, kspace, iterations=30)[0]
    segments = segment_indices(timing, lines_per_segment)
    parameters = np.zeros((len(segments), 6))
    history = []
    result: MotionResult | None = None
    for cycle in range(outer_iterations):
        result = estimate_motion(
            operator, image, kspace, timing, voxel_mm, lines_per_segment=lines_per_segment,
            iterations=motion_iterations, log=lambda _: None, initial_parameters=parameters,
        )
        weights = result.energy / result.energy.sum()
        parameters = result.parameters - np.sum(weights[:, None] * result.parameters, axis=0)
        groups = group_segments(parameters, segments)
        image = motion_aware_cg(operator, kspace, timing, groups, voxel_mm, iterations=cg_iterations, initial=image)
        record = {
            "cycle": cycle + 1,
            "groups": len(groups),
            "mean_loss_initial": float(np.sum(weights * result.loss_initial)),
            "mean_loss_final": float(np.sum(weights * result.loss_final)),
            **{f"max_abs_{name}": float(np.max(np.abs(parameters[:, i]))) for i, name in enumerate(PARAMETER_NAMES)},
        }
        history.append(record)
        log(
            f"cycle {cycle + 1}: weighted residual {record['mean_loss_initial']:.4f} -> {record['mean_loss_final']:.4f}, "
            f"{len(groups)} motion groups, max |params| "
            + ", ".join(f"{name} {record['max_abs_' + name]:.2f}" for name in PARAMETER_NAMES)
        )
    assert result is not None
    result.parameters = parameters
    return result, image, history
