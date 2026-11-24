"""Conjugate-gradient SENSE reconstruction for the Wave operator."""

from __future__ import annotations

import torch

from .wave_operator import WaveOperator


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex inner product ``<a, b> = sum(conj(a) * b)``.

    Args:
        a: First tensor.
        b: Second tensor with the same shape.

    Returns:
        Scalar complex tensor.
    """
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def cg_sense(
    operator: WaveOperator,
    kspace: torch.Tensor,
    *,
    iterations: int = 30,
    tikhonov: float = 0.0,
    initial: torch.Tensor | None = None,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Solve ``(A^H A + lambda I) x = A^H y`` by conjugate gradients.

    Args:
        operator: Wave encoding operator.
        kspace: Measured k-space ``(C, Xe, Y, Z)``.
        iterations: Maximum CG iterations.
        tikhonov: Nonnegative Tikhonov weight in the operator's units
            (the orthonormal FFTs make ``A^H A`` of order one where the
            coil maps are normalized).
        initial: Optional warm start image.
        tolerance: Stop when the residual norm falls below this fraction of
            the initial right-hand-side norm.

    Returns:
        Reconstructed image and the relative residual history.
    """
    rhs = operator.adjoint(kspace)
    x = torch.zeros_like(rhs) if initial is None else initial.clone()

    def apply(v: torch.Tensor) -> torch.Tensor:
        """Normal operator plus Tikhonov term."""
        result = operator.normal(v)
        if tikhonov > 0:
            result = result + tikhonov * v
        return result

    residual = rhs - apply(x) if initial is not None else rhs.clone()
    direction = residual.clone()
    rr = _dot(residual, residual).real
    reference = torch.sqrt(_dot(rhs, rhs).real)
    history = {"relative_residual": [float(torch.sqrt(rr) / reference)]}
    for _ in range(iterations):
        if float(torch.sqrt(rr) / reference) < tolerance:
            break
        normal_direction = apply(direction)
        alpha = rr / _dot(direction, normal_direction).real
        x = x + alpha * direction
        residual = residual - alpha * normal_direction
        rr_new = _dot(residual, residual).real
        direction = residual + (rr_new / rr) * direction
        rr = rr_new
        history["relative_residual"].append(float(torch.sqrt(rr) / reference))
    return x, history


def fit_complex_scale(measured: torch.Tensor, synthesized: torch.Tensor) -> tuple[complex, float]:
    """Fit ``measured ~ alpha * synthesized`` and report the relative residual.

    Args:
        measured: Measured k-space.
        synthesized: Model k-space on the same support.

    Returns:
        Least-squares complex scale and ``||measured - alpha s|| / ||measured||``.
    """
    alpha = _dot(synthesized, measured) / _dot(synthesized, synthesized)
    residual = torch.linalg.vector_norm(measured - alpha * synthesized) / torch.linalg.vector_norm(measured)
    return complex(alpha.item()), float(residual.item())
