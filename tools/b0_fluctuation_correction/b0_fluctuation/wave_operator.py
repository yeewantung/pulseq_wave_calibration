"""GPU Wave-CAIPI forward and adjoint operators matching the BART ``wave`` model.

BART ``wave`` reconstructs with ``A = M F_yz W F_x R E``: coil maps ``E``,
centred zero padding of the readout ``R``, a readout FFT ``F_x``, the
calibrated PSF ``W(kx, y, z)`` in the hybrid domain, a phase-encoding FFT
``F_yz``, and the sampling pattern ``M``. Data and maps are FFT-modulated, so
the model is the centred-FFT model below up to a global complex constant.
This module uses centred orthonormal FFTs throughout; any global scale or
phase relative to BART is irrelevant for data-consistency analysis because
it cancels in the measured/synthesized ratio's time variation and is fitted
explicitly where needed.

Array layout on the device:
    image        (X, Y, Z)
    maps         (C, X, Y, Z)
    psf          (Xe, Y, Z)
    k-space      (C, Xe, Y, Z)
    line mask    (Y, Z) boolean, one entry per acquired phase-encoding line
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def fftc(values: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Centred orthonormal forward FFT along the given dimensions.

    Args:
        values: Complex tensor.
        dims: Dimensions to transform.

    Returns:
        Transformed tensor with the zero frequency at index ``N // 2``.
    """
    shifted = torch.fft.ifftshift(values, dim=dims)
    return torch.fft.fftshift(torch.fft.fftn(shifted, dim=dims, norm="ortho"), dim=dims)


def ifftc(values: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Centred orthonormal inverse FFT along the given dimensions.

    Args:
        values: Complex tensor.
        dims: Dimensions to transform.

    Returns:
        Inverse-transformed tensor (exact adjoint of :func:`fftc`).
    """
    shifted = torch.fft.ifftshift(values, dim=dims)
    return torch.fft.fftshift(torch.fft.ifftn(shifted, dim=dims, norm="ortho"), dim=dims)


@dataclass
class WaveOperator:
    """Multi-coil Wave-CAIPI encoding operator on one device.

    Attributes:
        maps: Coil sensitivities ``(C, X, Y, Z)``.
        psf: Wave PSF ``(Xe, Y, Z)`` in the hybrid (kx, y, z) domain.
        line_mask: Boolean acquired-line pattern ``(Y, Z)``.
        coil_chunk: Coils transformed per batch to bound device memory.
    """

    maps: torch.Tensor
    psf: torch.Tensor
    line_mask: torch.Tensor
    coil_chunk: int = 8

    def __post_init__(self) -> None:
        """Validate shapes and cache derived quantities.

        Raises:
            ValueError: If the maps, PSF, and mask dimensions disagree.
        """
        if self.maps.ndim != 4 or self.psf.ndim != 3 or self.line_mask.ndim != 2:
            raise ValueError("Expected maps (C,X,Y,Z), psf (Xe,Y,Z), and mask (Y,Z).")
        _, x, y, z = self.maps.shape
        xe = self.psf.shape[0]
        if self.psf.shape[1:] != (y, z) or tuple(self.line_mask.shape) != (y, z):
            raise ValueError("PSF and mask phase-encoding dimensions must match the maps.")
        if xe < x:
            raise ValueError("The Wave readout grid must not be smaller than the image readout.")
        self.start = (xe - x) // 2
        self.mask_float = self.line_mask.to(self.psf.real.dtype)

    @property
    def coils(self) -> int:
        """Number of coils.

        Returns:
            Coil count.
        """
        return int(self.maps.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Image dimensions.

        Returns:
            ``(X, Y, Z)``.
        """
        return tuple(int(n) for n in self.maps.shape[1:])  # type: ignore[return-value]

    @property
    def kspace_shape(self) -> tuple[int, int, int, int]:
        """K-space dimensions.

        Returns:
            ``(C, Xe, Y, Z)``.
        """
        return (self.coils, int(self.psf.shape[0]), *self.image_shape[1:])

    def coil_forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Encode coil images without applying coil maps (unmasked).

        Args:
            coil_images: ``(c, X, Y, Z)`` images.

        Returns:
            Fully sampled Wave k-space ``(c, Xe, Y, Z)``.
        """
        c, x, y, z = coil_images.shape
        padded = torch.zeros((c, self.psf.shape[0], y, z), dtype=coil_images.dtype, device=coil_images.device)
        padded[:, self.start : self.start + x] = coil_images
        hybrid = fftc(padded, (1,)) * self.psf
        return fftc(hybrid, (2, 3))

    def coil_adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        """Adjoint of :meth:`coil_forward` (no mask, no coil combination).

        Args:
            kspace: ``(c, Xe, Y, Z)`` k-space.

        Returns:
            Coil images ``(c, X, Y, Z)``.
        """
        hybrid = ifftc(kspace, (2, 3)) * self.psf.conj()
        padded = ifftc(hybrid, (1,))
        return padded[:, self.start : self.start + self.image_shape[0]]

    def forward(self, image: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the full masked multi-coil Wave encoding.

        Args:
            image: ``(X, Y, Z)`` complex image.
            out: Optional preallocated ``(C, Xe, Y, Z)`` output.

        Returns:
            Masked k-space ``(C, Xe, Y, Z)``.
        """
        if out is None:
            out = torch.empty(self.kspace_shape, dtype=image.dtype, device=image.device)
        for start in range(0, self.coils, self.coil_chunk):
            stop = min(start + self.coil_chunk, self.coils)
            out[start:stop] = self.coil_forward(self.maps[start:stop] * image) * self.mask_float
        return out

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint of :meth:`forward`.

        Args:
            kspace: ``(C, Xe, Y, Z)`` k-space.

        Returns:
            Coil-combined image ``(X, Y, Z)``.
        """
        image = torch.zeros(self.image_shape, dtype=kspace.dtype, device=kspace.device)
        for start in range(0, self.coils, self.coil_chunk):
            stop = min(start + self.coil_chunk, self.coils)
            coil_images = self.coil_adjoint(kspace[start:stop] * self.mask_float)
            image += torch.sum(self.maps[start:stop].conj() * coil_images, dim=0)
        return image

    def normal(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A`` without storing all coils at once.

        Args:
            image: ``(X, Y, Z)`` complex image.

        Returns:
            ``A^H A image``.
        """
        result = torch.zeros_like(image)
        for start in range(0, self.coils, self.coil_chunk):
            stop = min(start + self.coil_chunk, self.coils)
            kspace = self.coil_forward(self.maps[start:stop] * image) * self.mask_float
            result += torch.sum(self.maps[start:stop].conj() * self.coil_adjoint(kspace), dim=0)
        return result
