"""Phase-free coil-sensitivity consistency metrics for ESPIRiT diagnostics.

These pure NumPy functions compare measured coil-image vectors with ESPIRiT
sensitivity-map subspaces and describe the local structure of both. They do
not access files, launch external programs, handle image orientation or
regions of interest, or plot; callers pass arrays that are already loaded.

Array conventions:
    * The coil axis is always last. Coil vectors ``d`` have shape
      ``(..., Nc)``. Map matrices have shape ``(..., Nc, M)``, which is the
      layout of an ESPIRiT ``(RO, LIN, PAR, Nc, M)`` CFL array opened as a
      NumPy array. Eigenvalue maps have shape ``(..., M)`` once the singleton
      CFL dimension 3 is removed. Masks are boolean arrays of shape ``(...)``.
    * Neighbourhood operations act on the spatial axes 0, 1 and 2, i.e.
      ``(RO, LIN, PAR)``, and never wrap around array edges.
    * Metric outputs are float64 and NaN wherever a metric is undefined.

Scientific scope:
    The intended calibration is the diagnostic ``ecalib -m 2 -c 0`` output:
    two uncropped, unit-norm eigenvector maps (map 1 and map 2) with their
    eigenvalue maps. Map components are inspected separately and are never
    reduced to a root-sum-of-squares combination as the only result. The
    primary diagnostic is the phase-free coil-space projection residual
    ``rho_M``. The residual-to-noise ratio is a conditional secondary
    diagnostic that requires an independently validated noise covariance.
    Per-coil phase maps are never interpreted.
"""

from __future__ import annotations

import operator
from typing import Any, Sequence

import numpy as np

# Complex128 working copies processed per block stay near 64 MB per operand.
_CHUNK_ELEMENTS = 1 << 22
# Byte budget for one batch of gathered unit vectors and local Gram matrices.
_GRAM_BATCH_BYTES = 128 * 1024 * 1024
# Absolute tolerance on Q^H Q for bases whose columns are unit-norm or zero.
# It accepts orthonormal maps stored in complex64 and rejects unnormalized maps.
_BASIS_ORTHONORMALITY_TOLERANCE = 1e-4
# Relative tolerance for Hermitian symmetry and positive semidefiniteness.
_COVARIANCE_TOLERANCE = 1e-6
_COMPLEX_NAN = complex(np.nan, np.nan)


def _positive_int(value: Any, name: str) -> int:
    """Validate one strictly positive integer argument.

    Args:
        value: Candidate integer value.
        name: Human-readable argument name used in validation errors.

    Returns:
        The value as a Python ``int``.

    Raises:
        ValueError: If the value is boolean, non-integral, or not positive.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}.") from exc
    if resolved < 1:
        raise ValueError(f"{name} must be positive; got {resolved}.")
    return int(resolved)


def _finite_scalar(value: Any, name: str) -> float:
    """Validate one finite real scalar argument.

    Args:
        value: Candidate real number.
        name: Human-readable argument name used in validation errors.

    Returns:
        The value as a Python ``float``.

    Raises:
        ValueError: If the value is boolean, not a real number, or not finite.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number; got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return result


def _dimensions(shape: Sequence[int], name: str) -> tuple[int, ...]:
    """Convert a dimension sequence to positive Python integers.

    Args:
        shape: Candidate array dimensions.
        name: Human-readable shape name used in validation errors.

    Returns:
        Tuple of positive integers in the given order.

    Raises:
        ValueError: If the sequence is empty or not iterable, or holds a
            non-integral or non-positive entry.
    """
    try:
        entries = list(shape)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers; got {shape!r}.") from exc
    if not entries:
        raise ValueError(f"{name} must not be empty.")
    return tuple(_positive_int(entry, f"{name} entry") for entry in entries)


def _numeric_array(values: Any, name: str, min_ndim: int) -> np.ndarray:
    """Return an array view after validating its numeric dtype and rank.

    Args:
        values: Array-like input; NumPy arrays and memory maps are not copied.
        name: Human-readable array name used in validation errors.
        min_ndim: Minimum number of dimensions.

    Returns:
        ``np.asarray(values)``.

    Raises:
        ValueError: If the dtype is not integer, floating, or complex, or the
            array has fewer than ``min_ndim`` dimensions.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {array.dtype}.")
    if array.ndim < min_ndim:
        raise ValueError(
            f"{name} must have at least {min_ndim} dimensions; got shape {array.shape}."
        )
    return array


def _real_array(values: Any, name: str, min_ndim: int) -> np.ndarray:
    """Return a real-valued numeric array after validating dtype and rank.

    Args:
        values: Array-like input.
        name: Human-readable array name used in validation errors.
        min_ndim: Minimum number of dimensions.

    Returns:
        ``np.asarray(values)``.

    Raises:
        ValueError: If the array is complex, non-numeric, or too small.
    """
    array = _numeric_array(values, name, min_ndim)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued; got dtype {array.dtype}.")
    return array


def _bool_mask(mask: Any, shape: Sequence[int], name: str) -> np.ndarray:
    """Validate one boolean voxel mask.

    Args:
        mask: Candidate mask.
        shape: Exact required mask shape.
        name: Human-readable mask name used in validation errors.

    Returns:
        ``np.asarray(mask)`` with dtype ``bool``.

    Raises:
        ValueError: If the mask is not boolean or its shape differs.
    """
    values = np.asarray(mask)
    if values.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array; got dtype {values.dtype}.")
    if values.shape != tuple(shape):
        raise ValueError(f"{name} shape {values.shape} does not match {tuple(shape)}.")
    return values


def _axis0_blocks(shape: Sequence[int]) -> list[slice]:
    """Split axis 0 into blocks whose complex128 working copies stay bounded.

    Args:
        shape: Shape of the largest array processed per block.

    Returns:
        Contiguous axis-0 slices covering the full axis in order.
    """
    rows = int(shape[0])
    row_elements = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, _CHUNK_ELEMENTS // row_elements)
    return [slice(start, min(start + step, rows)) for start in range(0, rows, step)]


def _squared_norm(values: np.ndarray) -> np.ndarray:
    """Return squared Euclidean norms along the last (coil) axis.

    Args:
        values: Complex128 or float64 array with the coil axis last.

    Returns:
        Float64 array with the coil axis removed.
    """
    return np.sum(values.real**2 + values.imag**2, axis=-1)


def _unit_column_count(basis: np.ndarray, name: str) -> np.ndarray:
    """Count unit columns of bases whose columns are orthonormal or zero.

    Args:
        basis: Complex128 per-voxel bases of shape ``(..., Nc, M)``.
        name: Human-readable basis name used in validation errors.

    Returns:
        Integer array of shape ``(...)`` with the number of unit columns.

    Raises:
        ValueError: If any per-voxel ``Q^H Q`` deviates from a diagonal 0/1
            pattern by more than the orthonormality tolerance or is not finite.
    """
    gram = np.einsum("...cm,...cn->...mn", basis.conj(), basis)
    unit = np.real(np.diagonal(gram, axis1=-2, axis2=-1)) > 0.5
    target = unit[..., np.newaxis] * np.eye(basis.shape[-1])
    deviation = np.max(np.abs(gram - target), axis=(-2, -1), initial=0.0)
    violations = int(np.count_nonzero(~(deviation <= _BASIS_ORTHONORMALITY_TOLERANCE)))
    if violations:
        raise ValueError(
            f"{name} columns must be orthonormal or zero inside the mask; {violations} "
            f"voxels exceed the Gram tolerance {_BASIS_ORTHONORMALITY_TOLERANCE:g}. "
            "Build the basis with orthonormal_map_basis."
        )
    return np.count_nonzero(unit, axis=-1)


def _validated_covariance(matrix: Any, coils: int | None, name: str) -> np.ndarray:
    """Validate a Hermitian positive-semidefinite coil covariance matrix.

    Args:
        matrix: Candidate ``(Nc, Nc)`` covariance.
        coils: Required coil count, or ``None`` to accept any square size.
        name: Human-readable matrix name used in validation errors.

    Returns:
        Exactly Hermitian complex128 copy ``(C + C^H) / 2``.

    Raises:
        ValueError: If the matrix is not square, has the wrong coil count, is
            non-finite, or is not Hermitian positive semidefinite within a
            relative tolerance.
    """
    values = _numeric_array(matrix, name, 2)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square (Nc, Nc) matrix; got {values.shape}.")
    if coils is not None and values.shape[0] != coils:
        raise ValueError(f"{name} has {values.shape[0]} coils; expected {coils}.")
    values = np.asarray(values, dtype=np.complex128)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    scale = float(np.max(np.abs(values), initial=0.0))
    asymmetry = float(np.max(np.abs(values - values.conj().T), initial=0.0))
    if asymmetry > _COVARIANCE_TOLERANCE * scale:
        raise ValueError(f"{name} is not Hermitian (maximum asymmetry {asymmetry:g}).")
    hermitian = 0.5 * (values + values.conj().T)
    spectrum = np.linalg.eigvalsh(hermitian)
    if spectrum[0] < -_COVARIANCE_TOLERANCE * float(np.max(np.abs(spectrum))):
        raise ValueError(
            f"{name} is not positive semidefinite (smallest eigenvalue {spectrum[0]:g})."
        )
    return hermitian


def _validated_window(window: Any, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Validate real k-space weights that broadcast to the spatial grid.

    Args:
        window: Candidate weights, for example of shape ``(1, LIN, PAR)``.
        spatial_shape: ``(RO, LIN, PAR)`` grid shape.

    Returns:
        Float64 weights with their original broadcastable shape.

    Raises:
        ValueError: If the weights are complex, non-numeric, non-finite, or do
            not broadcast to the grid without enlarging it.
    """
    weights = _real_array(window, "Apodization window", 0)
    try:
        broadcast = np.broadcast_shapes(weights.shape, spatial_shape)
    except ValueError as exc:
        raise ValueError(
            f"Apodization window shape {weights.shape} does not broadcast to {spatial_shape}."
        ) from exc
    if broadcast != spatial_shape:
        raise ValueError(
            f"Apodization window shape {weights.shape} enlarges the grid {spatial_shape}."
        )
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("Apodization window contains non-finite values.")
    return weights


def _fft_axes(ndim: int, axes: Sequence[int]) -> tuple[int, ...]:
    """Normalize a sequence of distinct FFT axes.

    Args:
        ndim: Number of dimensions of the transformed array.
        axes: Requested axes; negative values count from the end.

    Returns:
        Nonnegative axis indices in the requested order.

    Raises:
        ValueError: If the axes are empty, non-integral, repeated, or out of
            range.
    """
    try:
        requested = [operator.index(axis) for axis in axes]
    except TypeError as exc:
        raise ValueError(f"FFT axes must be a sequence of integers; got {axes!r}.") from exc
    resolved = []
    for axis in requested:
        normalized = axis + ndim if axis < 0 else axis
        if not 0 <= normalized < ndim:
            raise ValueError(f"FFT axis {axis} is invalid for an array with {ndim} dimensions.")
        resolved.append(normalized)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"FFT axes must be non-empty and distinct; got {tuple(axes)}.")
    return tuple(resolved)


def _axis_pair(axis: int, length: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Build the index pair that couples each voxel with its forward neighbour.

    Args:
        axis: Spatial axis 0, 1, or 2.
        length: Array length along that axis.

    Returns:
        ``(lower, upper)`` basic-slicing indices selecting positions
        ``0 .. length - 2`` and ``1 .. length - 1`` along ``axis``; no index
        wraps around the array edge.
    """
    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(0, length - 1)
    upper[axis] = slice(1, length)
    return tuple(lower), tuple(upper)


def _update_minimum(
    target: np.ndarray,
    lower: tuple[slice, ...],
    upper: tuple[slice, ...],
    values: np.ndarray,
) -> None:
    """Fold one neighbour-pair score into the running minimum of both voxels.

    Args:
        target: Running per-voxel minimum, modified in place.
        lower: Index of the voxel at the lower end of each pair.
        upper: Index of the voxel at the upper end of each pair.
        values: Pair scores, with ``inf`` for pairs that must be ignored.
    """
    np.minimum(target[lower], values, out=target[lower])
    np.minimum(target[upper], values, out=target[upper])


def dc_centered_hann(length: int) -> np.ndarray:
    """Return a Hann window centered on the k-space DC sample of a block.

    For a block of length ``n`` whose DC sample sits at local index
    ``n // 2``, the window is ``w(k) = cos(pi * k / n) ** 2`` with
    ``k = j - n // 2`` for ``j = 0, ..., n - 1``. It equals 1 at DC, is
    symmetric about DC wherever both ``k`` and ``-k`` exist, and is exactly 0
    at ``k = -n / 2`` for even ``n``.

    Args:
        length: Positive block length ``n``.

    Returns:
        Float64 window of shape ``(n,)``.

    Raises:
        ValueError: If ``length`` is not a positive integer.
    """
    size = _positive_int(length, "Hann window length")
    offsets = np.abs(np.arange(size, dtype=np.float64) - size // 2)
    # cos(pi*k/n)**2 == (1 + cos(2*pi*|k|/n)) / 2; evaluating on |k| makes the
    # window exactly symmetric about DC and exactly zero at k = -n/2.
    return 0.5 * (1.0 + np.cos(np.pi * (2.0 * offsets / size)))


def acs_block_slices(lin_size: int, par_size: int, acs_size: int) -> tuple[slice, slice]:
    """Locate the measured square ACS block on the centered LIN/PAR grid.

    The block starts at ``N // 2 - acs_size // 2`` along each phase-encoding
    axis, the embedding used by the MPRAGE calibration export, so the k-space
    DC sample ``N // 2`` sits at local offset ``acs_size // 2``.

    Args:
        lin_size: Positive LIN grid size.
        par_size: Positive PAR grid size.
        acs_size: Positive ACS edge length that fits both axes.

    Returns:
        ``(lin_slice, par_slice)`` selecting the ACS block.

    Raises:
        ValueError: If a size is not a positive integer or the ACS block does
            not fit the grid.
    """
    lin = _positive_int(lin_size, "LIN size")
    par = _positive_int(par_size, "PAR size")
    acs = _positive_int(acs_size, "ACS size")
    if acs > lin or acs > par:
        raise ValueError(f"ACS size {acs} does not fit the LIN/PAR grid {(lin, par)}.")
    lin_start = lin // 2 - acs // 2
    par_start = par // 2 - acs // 2
    return slice(lin_start, lin_start + acs), slice(par_start, par_start + acs)


def acs_apodization_window(grid_shape: Sequence[int], acs_size: int) -> np.ndarray:
    """Build the separable DC-centered Hann apodization of the ACS block.

    Args:
        grid_shape: Reconstruction grid ``(RO, LIN, PAR)``.
        acs_size: Edge length of the measured square ACS block.

    Returns:
        Float64 array of shape ``(1, LIN, PAR)`` equal to
        ``hann(k_lin) * hann(k_par)`` from :func:`dc_centered_hann` inside the
        block placed by :func:`acs_block_slices` and 0 outside. The singleton
        RO axis broadcasts, so the window is 1 along RO.

    Raises:
        ValueError: If the grid shape or ACS size is invalid.
    """
    shape = _dimensions(grid_shape, "Grid shape")
    if len(shape) != 3:
        raise ValueError(f"Grid shape must be (RO, LIN, PAR); got {shape}.")
    _, lin, par = shape
    lin_block, par_block = acs_block_slices(lin, par, acs_size)
    profile = dc_centered_hann(acs_size)
    window = np.zeros((1, lin, par), dtype=np.float64)
    window[0, lin_block, par_block] = np.outer(profile, profile)
    return window


def centered_fft(image: np.ndarray, axes: Sequence[int] = (0, 1, 2)) -> np.ndarray:
    """Apply the centered orthonormal forward FFT, the inverse of centered_ifft.

    ``centered_fft(x) = fftshift(fftn(ifftshift(x, axes), axes, norm="ortho"), axes)``.
    For even sizes this is the grid convention of BART's centered FFT (image
    center and k-space DC both at index ``N // 2``) and of
    ``core.centered_fftn``. A global constant phase between conventions is
    irrelevant here because every diagnostic in this module is phase-free or
    quadratic.

    Args:
        image: Numeric array with the image center at index ``N // 2`` of
            each transformed axis.
        axes: Distinct axes to transform; negative values count from the end.

    Returns:
        Complex k-space array of the same shape; precision follows NumPy's
        FFT promotion (complex64 input stays complex64).

    Raises:
        ValueError: If the input is not numeric or the axes are invalid.
    """
    values = _numeric_array(image, "FFT input", 1)
    resolved = _fft_axes(values.ndim, axes)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(values, axes=resolved), axes=resolved, norm="ortho"),
        axes=resolved,
    )


def centered_ifft(kspace: np.ndarray, axes: Sequence[int] = (0, 1, 2)) -> np.ndarray:
    """Apply the centered orthonormal inverse FFT used to form coil images.

    ``centered_ifft(k) = fftshift(ifftn(ifftshift(k, axes), axes, norm="ortho"), axes)``.
    For even sizes this is the grid convention of BART's centered FFT
    (k-space DC and image center both at index ``N // 2``) and of
    ``core.centered_fftn``. A global constant phase between conventions is
    irrelevant here because every diagnostic in this module is phase-free or
    quadratic.

    Args:
        kspace: Numeric array with the k-space DC sample at index ``N // 2``
            of each transformed axis.
        axes: Distinct axes to transform; negative values count from the end.

    Returns:
        Complex image-domain array of the same shape; precision follows
        NumPy's FFT promotion (complex64 input stays complex64).

    Raises:
        ValueError: If the input is not numeric or the axes are invalid.
    """
    values = _numeric_array(kspace, "Inverse FFT input", 1)
    resolved = _fft_axes(values.ndim, axes)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(values, axes=resolved), axes=resolved, norm="ortho"),
        axes=resolved,
    )


def coil_images(
    kspace: np.ndarray,
    window: np.ndarray | None = None,
    *,
    dtype: Any = np.complex64,
) -> np.ndarray:
    """Reconstruct zero-filled coil images one coil at a time.

    Each coil volume is optionally multiplied by the real k-space window,
    transformed with :func:`centered_ifft` over ``(RO, LIN, PAR)`` in
    complex128, and stored in ``dtype``. Looping over coils bounds the working
    memory to a few single-coil volumes. The DC-centered Hann window from
    :func:`acs_apodization_window` is the primary variant and ``window=None``
    is the unapodized secondary variant.

    Args:
        kspace: Finite k-space of shape ``(RO, LIN, PAR, Nc)`` with the DC
            sample at index ``N // 2`` of every spatial axis.
        window: Optional real finite weights broadcastable to
            ``(RO, LIN, PAR)``, for example of shape ``(1, LIN, PAR)``.
            ``None`` applies no window.
        dtype: Complex output dtype.

    Returns:
        Coil images of shape ``(RO, LIN, PAR, Nc)`` and dtype ``dtype``.

    Raises:
        ValueError: If the k-space shape or values, the window, or the output
            dtype are invalid.
    """
    values = _numeric_array(kspace, "Coil k-space", 4)
    if values.ndim != 4:
        raise ValueError(f"Coil k-space must be (RO, LIN, PAR, Nc); got {values.shape}.")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.complexfloating):
        raise ValueError(f"Coil-image dtype must be complex; got {output_dtype}.")
    spatial_shape = tuple(int(size) for size in values.shape[:3])
    weights = None if window is None else _validated_window(window, spatial_shape)
    images = np.empty(values.shape, dtype=output_dtype)
    for coil in range(values.shape[3]):
        # np.array always copies, so the in-place window never alters the input.
        channel = np.array(values[..., coil], dtype=np.complex128)
        if not np.isfinite(channel).all():
            raise ValueError(f"Coil k-space channel {coil} contains non-finite values.")
        if weights is not None:
            channel *= weights
        images[..., coil] = centered_ifft(channel, axes=(0, 1, 2))
    return images


def coil_vector_norm(coil_vectors: np.ndarray) -> np.ndarray:
    """Compute the Euclidean norm of every coil vector.

    Args:
        coil_vectors: Numeric coil vectors of shape ``(..., Nc)`` with at least
            one voxel axis.

    Returns:
        Float64 norms ``||d||_2`` of shape ``(...)``.

    Raises:
        ValueError: If the input is not numeric or has no voxel axis.
    """
    values = _numeric_array(coil_vectors, "Coil vectors", 2)
    norms = np.empty(values.shape[:-1], dtype=np.float64)
    for block in _axis0_blocks(values.shape):
        norms[block] = np.sqrt(_squared_norm(np.asarray(values[block], dtype=np.complex128)))
    return norms


def normalize_coil_vectors(coil_vectors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale masked coil vectors to unit Euclidean norm.

    Args:
        coil_vectors: Numeric coil vectors of shape ``(..., Nc)``.
        mask: Boolean voxel mask of shape ``(...)``.

    Returns:
        Complex128 array of shape ``(..., Nc)`` holding ``d / ||d||`` inside
        the mask and complex NaN outside it or where ``||d||`` is zero or
        non-finite.

    Raises:
        ValueError: If the input is not numeric, has no voxel axis, or the
            mask is invalid.
    """
    values = _numeric_array(coil_vectors, "Coil vectors", 2)
    selection = _bool_mask(mask, values.shape[:-1], "Coil-vector mask")
    normalized = np.full(values.shape, _COMPLEX_NAN, dtype=np.complex128)
    for block in _axis0_blocks(values.shape):
        chunk = np.asarray(values[block], dtype=np.complex128)
        norms = np.sqrt(_squared_norm(chunk))
        valid = selection[block] & np.isfinite(norms) & (norms > 0)
        normalized[block][valid] = chunk[valid] / norms[valid, np.newaxis]
    return normalized


def empirical_coil_covariance(
    coil_vectors: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Estimate the coil covariance as the masked mean of ``d d^H``.

    The estimate is uncentered because image-domain noise has zero mean; it is
    intended for noise-only (for example background-air) masks.

    Args:
        coil_vectors: Numeric coil vectors of shape ``(..., Nc)``.
        mask: Boolean voxel mask of shape ``(...)`` with more than ``Nc``
            voxels.

    Returns:
        ``(covariance, count)``: the exactly Hermitian complex128 ``(Nc, Nc)``
        matrix ``mean(d d^H)`` and the number of masked voxels.

    Raises:
        ValueError: If inputs are invalid, the mask holds ``Nc`` or fewer
            voxels, or a masked coil vector is non-finite.
    """
    values = _numeric_array(coil_vectors, "Coil vectors", 2)
    selection = _bool_mask(mask, values.shape[:-1], "Covariance mask")
    coils = int(values.shape[-1])
    count = int(np.count_nonzero(selection))
    if count <= coils:
        raise ValueError(
            f"Empirical coil covariance needs more than {coils} masked voxels; found {count}."
        )
    accumulator = np.zeros((coils, coils), dtype=np.complex128)
    for block in _axis0_blocks(values.shape):
        local = selection[block]
        if not local.any():
            continue
        samples = np.asarray(values[block][local], dtype=np.complex128)
        if not np.isfinite(samples).all():
            raise ValueError("Masked coil vectors contain non-finite values.")
        # Row v of ``samples`` is d_v^T, so samples^T conj(samples) = sum_v d_v d_v^H.
        accumulator += samples.T @ samples.conj()
    covariance = accumulator / count
    return 0.5 * (covariance + covariance.conj().T), count


def modelled_image_noise_covariance(
    kspace_covariance: np.ndarray,
    window: np.ndarray | None,
    grid_voxels: int,
    acquired_samples: int,
) -> np.ndarray:
    """Model the image-domain coil noise covariance of windowed coil images.

    Under a white k-space noise model (independent samples sharing the
    per-sample coil covariance ``Psi_k``), a windowed zero-filled orthonormal
    inverse FFT gives every image voxel the covariance
    ``Psi_d = Psi_k * sum_acquired |w(k)|^2 / grid_voxels``; without a window
    the sum equals ``acquired_samples``. This is a model only: it ignores
    sample-dependent noise, filtering, and interpolation in the acquisition
    chain, so it must be validated against an independent estimate before any
    residual-to-noise ratio is interpreted.

    Args:
        kspace_covariance: Hermitian positive-semidefinite ``(Nc, Nc)``
            per-sample noise covariance in the units of the k-space passed to
            :func:`coil_images`.
        window: ``None`` or real finite weights on the reconstruction grid
            that vanish outside the acquired samples, such as the output of
            :func:`acs_apodization_window`. Singleton axes broadcast, so the
            grid sum equals ``(grid_voxels / window.size) * sum(|w|^2)``.
        grid_voxels: Number of reconstruction-grid voxels ``RO * LIN * PAR``.
        acquired_samples: Number of acquired (non-zero-filled) k-space samples.

    Returns:
        Complex128 Hermitian ``(Nc, Nc)`` modelled image-domain covariance.

    Raises:
        ValueError: If the covariance or counts are invalid, the window size
            does not divide ``grid_voxels``, or the window weights more samples
            than were acquired.
    """
    covariance = _validated_covariance(kspace_covariance, None, "k-space noise covariance")
    voxels = _positive_int(grid_voxels, "Grid voxel count")
    samples = _positive_int(acquired_samples, "Acquired sample count")
    if samples > voxels:
        raise ValueError(f"Acquired samples {samples} exceed the grid voxel count {voxels}.")
    if window is None:
        energy = float(samples)
    else:
        weights = np.asarray(_real_array(window, "Apodization window", 1), dtype=np.float64)
        if weights.size == 0 or voxels % weights.size:
            raise ValueError(
                f"Apodization window size {weights.size} does not divide the grid voxel "
                f"count {voxels}."
            )
        if not np.isfinite(weights).all():
            raise ValueError("Apodization window contains non-finite values.")
        repeats = voxels // weights.size
        weighted_samples = repeats * int(np.count_nonzero(weights))
        if weighted_samples > samples:
            raise ValueError(
                f"Apodization window weights {weighted_samples} samples but only {samples} "
                "were acquired; it must vanish outside the acquired k-space samples."
            )
        energy = repeats * float(np.sum(weights**2))
    return covariance * (energy / voxels)


def signal_to_noise_energy(coil_vectors: np.ndarray, noise_trace: float) -> np.ndarray:
    """Compute the per-voxel signal-to-noise energy ratio ``||d||^2 / tr(Psi_d)``.

    Args:
        coil_vectors: Numeric coil vectors of shape ``(..., Nc)``.
        noise_trace: Positive expected noise energy per voxel, normally the
            real trace of the image-domain noise covariance.

    Returns:
        Float64 energy ratio of shape ``(...)``.

    Raises:
        ValueError: If the vectors are invalid or ``noise_trace`` is not a
            positive finite number.
    """
    trace = _finite_scalar(noise_trace, "Noise trace")
    if trace <= 0:
        raise ValueError(f"Noise trace must be positive; got {trace:g}.")
    return coil_vector_norm(coil_vectors) ** 2 / trace


def support_mask(snr: np.ndarray, kappa: float) -> np.ndarray:
    """Select voxels whose signal-to-noise energy reaches a threshold.

    Args:
        snr: Real signal-to-noise energy ratios of any shape.
        kappa: Finite inclusive threshold.

    Returns:
        Boolean mask ``isfinite(snr) & (snr >= kappa)`` with the shape of
        ``snr``.

    Raises:
        ValueError: If ``snr`` is not real numeric or ``kappa`` is not finite.
    """
    values = _real_array(snr, "SNR energy", 0)
    threshold = _finite_scalar(kappa, "SNR threshold")
    return np.isfinite(values) & (values >= threshold)


def orthonormal_map_basis(
    maps: np.ndarray, rank_tolerance: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis of each voxel's map span by truncated SVD.

    For each voxel the ``(Nc, M)`` map matrix is decomposed as
    ``U diag(s) V^H``; left singular vectors with ``s > 0`` and
    ``s >= rank_tolerance * max(s)`` are kept. Basis columns follow singular
    value order, not map order, so pass ``maps[..., :M]`` to obtain the basis
    for ``span{S_1, ..., S_M}``.

    Args:
        maps: Finite map matrices of shape ``(..., Nc, M)``.
        rank_tolerance: Relative singular-value threshold in ``(0, 1]``.

    Returns:
        ``(basis, rank)``: complex128 basis of shape ``(..., Nc, M)`` whose
        dropped columns are exactly 0, and int32 rank of shape ``(...)``,
        which is 0 where all maps vanish. The output contains no NaN.

    Raises:
        ValueError: If the maps are invalid or non-finite, or the tolerance is
            outside ``(0, 1]``.
    """
    values = _numeric_array(maps, "Sensitivity maps", 3)
    tolerance = _finite_scalar(rank_tolerance, "Rank tolerance")
    if not 0 < tolerance <= 1:
        raise ValueError(f"Rank tolerance must lie in (0, 1]; got {tolerance:g}.")
    coils, count = (int(size) for size in values.shape[-2:])
    kept = min(coils, count)
    basis = np.zeros(values.shape, dtype=np.complex128)
    rank = np.zeros(values.shape[:-2], dtype=np.int32)
    for block in _axis0_blocks(values.shape):
        chunk = np.asarray(values[block], dtype=np.complex128)
        if not np.isfinite(chunk).all():
            raise ValueError("Sensitivity maps contain non-finite values.")
        left, singular, _ = np.linalg.svd(chunk, full_matrices=False)
        keep = (singular > 0) & (singular >= tolerance * singular[..., :1])
        basis[block][..., :kept] = left * keep[..., np.newaxis, :]
        rank[block] = np.count_nonzero(keep, axis=-1)
    return basis, rank


def map_orthonormality_error(maps: np.ndarray) -> np.ndarray:
    """Measure how far each voxel's maps are from an orthonormal set.

    Args:
        maps: Map matrices of shape ``(..., Nc, M)``.

    Returns:
        Float64 Frobenius error ``||S^H S - I_M||_F`` of shape ``(...)``;
        non-finite maps give NaN.

    Raises:
        ValueError: If the maps are not numeric or have fewer than three
            dimensions.
    """
    values = _numeric_array(maps, "Sensitivity maps", 3)
    identity = np.eye(values.shape[-1])
    errors = np.empty(values.shape[:-2], dtype=np.float64)
    for block in _axis0_blocks(values.shape):
        chunk = np.asarray(values[block], dtype=np.complex128)
        gram = np.einsum("...cm,...cn->...mn", chunk.conj(), chunk)
        errors[block] = np.sqrt(np.sum(np.abs(gram - identity) ** 2, axis=(-2, -1)))
    return errors


def _masked_projection_inputs(
    coil_vectors: np.ndarray, basis: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the shared inputs of the projection diagnostics.

    Args:
        coil_vectors: Candidate coil vectors of shape ``(..., Nc)``.
        basis: Candidate basis of shape ``(..., Nc, M)``.
        mask: Candidate boolean mask of shape ``(...)``.

    Returns:
        ``(vectors, subspace, selection)`` array views.

    Raises:
        ValueError: If dtypes or shapes are inconsistent.
    """
    vectors = _numeric_array(coil_vectors, "Coil vectors", 2)
    subspace = _numeric_array(basis, "Projection basis", 3)
    if subspace.shape[:-1] != vectors.shape:
        raise ValueError(
            f"Projection basis shape {subspace.shape} must be coil-vector shape "
            f"{vectors.shape} plus one trailing map axis."
        )
    selection = _bool_mask(mask, vectors.shape[:-1], "Projection mask")
    return vectors, subspace, selection


def _residual_energy(vectors: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    """Return ``||d - Q Q^H d||^2`` for gathered vectors and bases.

    Args:
        vectors: Complex128 coil vectors of shape ``(V, Nc)``.
        subspace: Complex128 orthonormal-or-zero bases of shape ``(V, Nc, M)``.

    Returns:
        Float64 residual energies of shape ``(V,)``.
    """
    coefficients = np.einsum("vcm,vc->vm", subspace.conj(), vectors)
    # Subtracting the explicit projection keeps near-zero residuals accurate.
    return _squared_norm(vectors - np.einsum("vcm,vm->vc", subspace, coefficients))


def projection_residual(
    coil_vectors: np.ndarray, basis: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute the phase-free coil-space projection residual ``rho_M``.

    ``rho_M = ||d - Q_M Q_M^H d||_2 / ||d||_2``, where ``Q_M`` is an
    orthonormal basis of ``span{S_1, ..., S_M}`` from
    ``orthonormal_map_basis(maps[..., :M])``. It depends on the maps only
    through the projector ``Q_M Q_M^H``, so it is invariant to per-voxel map
    phases and to positive column scaling that keeps the retained rank. This
    is the primary map-consistency diagnostic.

    Args:
        coil_vectors: Numeric coil vectors ``d`` of shape ``(..., Nc)``.
        basis: Basis of shape ``(..., Nc, M)`` whose columns are orthonormal
            or zero inside the mask.
        mask: Boolean voxel mask of shape ``(...)``.

    Returns:
        Float64 residual in ``[0, 1]`` of shape ``(...)``; NaN outside the
        mask, where ``||d||`` is zero or non-finite, or where every basis
        column is zero.

    Raises:
        ValueError: If shapes or dtypes are inconsistent, or the basis is not
            orthonormal-or-zero inside the mask.
    """
    vectors, subspace, selection = _masked_projection_inputs(coil_vectors, basis, mask)
    residual = np.full(vectors.shape[:-1], np.nan, dtype=np.float64)
    for block in _axis0_blocks(subspace.shape):
        local = selection[block]
        if not local.any():
            continue
        d = np.asarray(vectors[block][local], dtype=np.complex128)
        q = np.asarray(subspace[block][local], dtype=np.complex128)
        _unit_column_count(q, "Projection basis")
        energy = _squared_norm(d)
        valid = np.any(q != 0, axis=(-2, -1)) & np.isfinite(energy) & (energy > 0)
        values = np.full(d.shape[0], np.nan, dtype=np.float64)
        values[valid] = np.minimum(
            np.sqrt(_residual_energy(d[valid], q[valid]) / energy[valid]), 1.0
        )
        residual[block][local] = values
    return residual


def residual_to_noise_ratio(
    coil_vectors: np.ndarray,
    basis: np.ndarray,
    noise_covariance: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the conditional residual-to-noise ratio ``RNR_M``.

    ``RNR_M = ||(I - P_M) d||^2 / tr[(I - P_M) Psi_d]`` with
    ``P_M = Q_M Q_M^H`` and
    ``tr[(I - P_M) Psi_d] = tr(Psi_d) - sum_m Re(q_m^H Psi_d q_m)``. Its
    expectation is 1 when the noiseless coil vector lies in the map subspace
    and ``Psi_d`` is the true image-domain noise covariance.

    This is a conditional, secondary diagnostic. It is interpretable only
    after ``Psi_d`` has been validated against an independent estimate of the
    image-domain noise covariance (a separate module performs that check); a
    modelled covariance alone does not qualify. A value near 1 does not by
    itself establish consistency between data and maps: a mis-scaled
    covariance, residual signal that mimics the noise structure, or averaging
    over heterogeneous voxels can also produce values near 1. Always report
    it next to the primary projection residual, never in place of it.

    Args:
        coil_vectors: Numeric coil vectors ``d`` of shape ``(..., Nc)``.
        basis: Basis of shape ``(..., Nc, M)`` whose columns are orthonormal
            or zero inside the mask.
        noise_covariance: Hermitian positive-semidefinite ``(Nc, Nc)``
            image-domain noise covariance ``Psi_d``.
        mask: Boolean voxel mask of shape ``(...)``.

    Returns:
        Float64 ratio of shape ``(...)``; NaN outside the mask, where every
        basis column is zero, where the residual is non-finite, or where the
        denominator is not positive.

    Raises:
        ValueError: If shapes or dtypes are inconsistent, the covariance is
            invalid, or the basis is not orthonormal-or-zero inside the mask.
    """
    vectors, subspace, selection = _masked_projection_inputs(coil_vectors, basis, mask)
    covariance = _validated_covariance(
        noise_covariance, int(vectors.shape[-1]), "Image noise covariance"
    )
    noise_trace = float(np.real(np.trace(covariance)))
    ratio = np.full(vectors.shape[:-1], np.nan, dtype=np.float64)
    for block in _axis0_blocks(subspace.shape):
        local = selection[block]
        if not local.any():
            continue
        d = np.asarray(vectors[block][local], dtype=np.complex128)
        q = np.asarray(subspace[block][local], dtype=np.complex128)
        _unit_column_count(q, "Projection basis")
        numerator = _residual_energy(d, q)
        # sum_m Re(q_m^H Psi q_m) is the noise energy captured by the subspace.
        captured = np.real(np.sum(q.conj() * np.matmul(covariance, q), axis=(-2, -1)))
        denominator = noise_trace - captured
        valid = np.any(q != 0, axis=(-2, -1)) & np.isfinite(numerator) & (denominator > 0)
        values = np.full(d.shape[0], np.nan, dtype=np.float64)
        values[valid] = numerator[valid] / denominator[valid]
        ratio[block][local] = values
    return ratio


def _check_calibration_layout(
    shape: tuple[int, ...],
    expected: tuple[int, ...],
    labels: tuple[str, ...],
    name: str,
) -> None:
    """Compare one ESPIRiT output shape with its required leading dimensions.

    Args:
        shape: Observed dimensions.
        expected: Required leading dimensions.
        labels: Human-readable name of each leading dimension.
        name: Output name used in validation errors.

    Raises:
        ValueError: If the rank is too small, a leading dimension differs, or
            a trailing dimension is not singleton.
    """
    if len(shape) < len(expected):
        raise ValueError(
            f"{name} must have at least {len(expected)} dimensions {expected}; got {shape}."
        )
    for label, observed, required in zip(labels, shape, expected):
        if observed != required:
            raise ValueError(
                f"{name} {label} is {observed} but must be {required}; got {shape}, expected "
                f"{expected} plus trailing singleton dimensions."
            )
    trailing = shape[len(expected) :]
    if any(size != 1 for size in trailing):
        raise ValueError(f"{name} has non-singleton trailing dimensions {trailing}; got {shape}.")


def validate_two_map_outputs(
    maps_shape: Sequence[int],
    eigenvalue_shape: Sequence[int],
    spatial_shape: Sequence[int],
    coils: int,
) -> None:
    """Validate the array layout of a two-map ESPIRiT calibration.

    ``ecalib -m 2 -c 0`` writes maps of shape ``(RO, LIN, PAR, Nc, 2)`` and
    eigenvalue maps of shape ``(RO, LIN, PAR, 1, 2)``; CFL headers may append
    further singleton dimensions.

    Args:
        maps_shape: Observed map dimensions.
        eigenvalue_shape: Observed eigenvalue-map dimensions.
        spatial_shape: Expected ``(RO, LIN, PAR)``.
        coils: Expected coil count ``Nc``.

    Raises:
        ValueError: If any shape is malformed, the spatial or coil dimensions
            differ, the MAPS dimension is not 2, the eigenvalue dimension 3 is
            not singleton, or any trailing dimension is not singleton.
    """
    spatial = _dimensions(spatial_shape, "Spatial shape")
    if len(spatial) != 3:
        raise ValueError(f"Spatial shape must be (RO, LIN, PAR); got {spatial}.")
    coil_count = _positive_int(coils, "Coil count")
    _check_calibration_layout(
        _dimensions(maps_shape, "Map shape"),
        spatial + (coil_count, 2),
        ("RO", "LIN", "PAR", "coil dimension 3", "MAPS dimension 4"),
        "Two-map sensitivity output",
    )
    _check_calibration_layout(
        _dimensions(eigenvalue_shape, "Eigenvalue shape"),
        spatial + (1, 2),
        ("RO", "LIN", "PAR", "singleton dimension 3", "MAPS dimension 4"),
        "Two-map eigenvalue output",
    )


def eigenvalue_arrays(
    eigenvalue_values: np.ndarray, *, imaginary_tolerance: float = 1e-5
) -> np.ndarray:
    """Convert an opened ESPIRiT eigenvalue CFL array to real eigenvalue maps.

    Args:
        eigenvalue_values: Array of shape ``(RO, LIN, PAR, 1, M)`` optionally
            followed by singleton dimensions.
        imaginary_tolerance: Nonnegative relative tolerance; the largest
            finite ``|imag|`` must not exceed
            ``imaginary_tolerance * max(1, max finite |real|)``.

    Returns:
        C-ordered float64 array of shape ``(RO, LIN, PAR, M)`` holding the
        real parts.

    Raises:
        ValueError: If the array has fewer than five dimensions, dimension 3
            or a trailing dimension is not singleton, the tolerance is
            invalid, or the imaginary parts are too large.
    """
    values = _numeric_array(eigenvalue_values, "Eigenvalue array", 5)
    if values.shape[3] != 1:
        raise ValueError(f"Eigenvalue array dimension 3 must be singleton; got {values.shape}.")
    if any(size != 1 for size in values.shape[5:]):
        raise ValueError(
            f"Eigenvalue array has non-singleton trailing dimensions; got {values.shape}."
        )
    tolerance = _finite_scalar(imaginary_tolerance, "Imaginary tolerance")
    if tolerance < 0:
        raise ValueError(f"Imaginary tolerance must be nonnegative; got {tolerance:g}.")
    core = values[(slice(None),) * 3 + (0, slice(None)) + (0,) * (values.ndim - 5)]
    real = np.array(core.real, dtype=np.float64, order="C")
    if np.iscomplexobj(core):
        imaginary = np.abs(np.asarray(core.imag, dtype=np.float64))
        finite_real = np.abs(real[np.isfinite(real)])
        scale = max(1.0, float(np.max(finite_real, initial=0.0)))
        largest = float(np.max(imaginary[np.isfinite(imaginary)], initial=0.0))
        if largest > tolerance * scale:
            raise ValueError(
                f"Eigenvalue imaginary parts reach {largest:g}, above the allowed "
                f"{tolerance * scale:g}."
            )
    return real


def _real_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """Validate real eigenvalue maps with the map axis last.

    Args:
        eigenvalues: Candidate real array of shape ``(..., M)``.

    Returns:
        ``np.asarray(eigenvalues)``.

    Raises:
        ValueError: If the array is complex, non-numeric, or has no map axis.
    """
    values = _real_array(eigenvalues, "Eigenvalues", 1)
    if values.shape[-1] < 1:
        raise ValueError(f"Eigenvalues need at least one map; got shape {values.shape}.")
    return values


def eigenvalue_qc(eigenvalues: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Summarize ESPIRiT eigenvalue maps for quality control.

    Args:
        eigenvalues: Real eigenvalues of shape ``(..., M)``, map 1 at index 0.
        mask: Optional boolean voxel mask of shape ``(...)``; ``None`` uses
            every voxel.

    Returns:
        JSON-native dictionary with ``map_count`` (M), ``voxels`` (evaluated
        voxels), ``finite_fraction`` (finite share of the ``voxels * M``
        entries, ``None`` without entries), ``min`` and ``max`` (over finite
        entries, ``None`` without any), ``count_ge_one`` (finite entries with
        ``|lambda| >= 1``), ``count_negative`` (finite entries below 0), and
        ``descending_order_violations`` (voxels where some
        ``lambda_{m+1} > lambda_m``; comparisons involving NaN do not count).

    Raises:
        ValueError: If the eigenvalues are not real or the mask is invalid.
    """
    values = _real_eigenvalues(eigenvalues)
    count = int(values.shape[-1])
    if mask is None:
        selected = np.asarray(values, dtype=np.float64).reshape(-1, count)
    else:
        selection = _bool_mask(mask, values.shape[:-1], "Eigenvalue QC mask")
        selected = np.asarray(values[selection], dtype=np.float64)
    finite = np.isfinite(selected)
    finite_values = selected[finite]
    return {
        "map_count": count,
        "voxels": int(selected.shape[0]),
        "finite_fraction": (
            float(np.count_nonzero(finite) / selected.size) if selected.size else None
        ),
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
        "count_ge_one": int(np.count_nonzero(np.abs(finite_values) >= 1.0)),
        "count_negative": int(np.count_nonzero(finite_values < 0.0)),
        "descending_order_violations": int(
            np.count_nonzero(np.any(selected[:, 1:] > selected[:, :-1], axis=-1))
        ),
    }


def eigenvalue_gap(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute the signed gap between the map-1 and map-2 eigenvalues.

    Args:
        eigenvalues: Real eigenvalues of shape ``(..., M)`` with ``M >= 2``.

    Returns:
        Float64 ``lambda_1 - lambda_2`` of shape ``(...)``.

    Raises:
        ValueError: If the eigenvalues are not real or fewer than two maps
            are present.
    """
    values = _real_eigenvalues(eigenvalues)
    if values.shape[-1] < 2:
        raise ValueError(f"Eigenvalue gap needs at least two maps; got shape {values.shape}.")
    return np.asarray(values[..., 0], dtype=np.float64) - np.asarray(
        values[..., 1], dtype=np.float64
    )


def map_reproduction(
    map_values: np.ndarray, reference_map: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Measure phase-invariant agreement between one map and a reference map.

    ``alpha = |s^H s_ref| / (||s|| ||s_ref||)`` is 1 when the two coil
    vectors agree up to a per-voxel complex factor and 0 when they are
    orthogonal.

    Args:
        map_values: Map coil vectors ``s`` of shape ``(..., Nc)``.
        reference_map: Reference coil vectors ``s_ref`` of the same shape.
        mask: Boolean voxel mask of shape ``(...)``.

    Returns:
        Float64 ``alpha`` in ``[0, 1]`` of shape ``(...)``; NaN outside the
        mask or where either norm is zero or non-finite.

    Raises:
        ValueError: If dtypes or shapes are invalid.
    """
    current = _numeric_array(map_values, "Map values", 2)
    reference = _numeric_array(reference_map, "Reference map", 2)
    if current.shape != reference.shape:
        raise ValueError(f"Map shape {current.shape} does not match reference {reference.shape}.")
    selection = _bool_mask(mask, current.shape[:-1], "Map reproduction mask")
    alpha = np.full(current.shape[:-1], np.nan, dtype=np.float64)
    for block in _axis0_blocks(current.shape):
        local = selection[block]
        if not local.any():
            continue
        s = np.asarray(current[block][local], dtype=np.complex128)
        t = np.asarray(reference[block][local], dtype=np.complex128)
        norms = np.sqrt(_squared_norm(s) * _squared_norm(t))
        valid = np.isfinite(norms) & (norms > 0)
        values = np.full(s.shape[0], np.nan, dtype=np.float64)
        overlap = np.abs(np.sum(s[valid].conj() * t[valid], axis=-1))
        values[valid] = np.minimum(overlap / norms[valid], 1.0)
        alpha[block][local] = values
    return alpha


def _neighborhood_shape(neighborhood: Sequence[int]) -> tuple[int, int, int]:
    """Validate a centered box neighbourhood with odd edge lengths.

    Args:
        neighborhood: Candidate ``(RO, LIN, PAR)`` box edge lengths.

    Returns:
        Three odd positive integers.

    Raises:
        ValueError: If the box does not have three odd positive edge lengths.
    """
    box = _dimensions(neighborhood, "Neighbourhood")
    if len(box) != 3 or any(size % 2 == 0 for size in box):
        raise ValueError(f"Neighbourhood must hold three odd sizes (RO, LIN, PAR); got {box}.")
    return box[0], box[1], box[2]


def local_coil_rank(
    coil_vectors: np.ndarray,
    support: np.ndarray,
    *,
    neighborhood: Sequence[int] = (5, 1, 1),
    min_members: int = 4,
    par_chunk: int = 8,
) -> dict[str, np.ndarray]:
    """Describe the model-free local rank of measured coil vectors.

    For every support voxel, the members are the support voxels with a
    nonzero coil vector inside the centered box ``neighborhood``, clipped at
    the array edges without wrap-around. The default ``(5, 1, 1)`` spans five
    samples along RO, the metal-displacement axis. Each member vector is
    scaled to unit norm, which removes intensity and phase differences, and
    the eigenvalues ``mu_1 >= mu_2 >= ...`` of the Gram matrix ``D^H D`` of
    the unit members (or of ``D D^H`` when the box holds more voxels than
    there are coils; both share their nonzero eigenvalues) give

    * ``e1 = mu_1 / sum_i max(mu_i, 0)``: 1 for a locally rank-1 field and
      smaller when several coil-vector directions coexist, and
    * ``kappa2 = max(mu_2, 0) / mu_1``: the relative weight of a second
      direction.

    No noise debiasing is applied, so noise lowers ``e1`` at low SNR. No
    ESPIRiT model is involved. PAR planes are processed in chunks with a halo
    and Gram matrices in bounded batches, so a ``(256, 32, 32, 52)`` input
    with a ``(5, 3, 3)`` box needs a few hundred MB of working memory.

    Args:
        coil_vectors: Coil vectors of shape ``(RO, LIN, PAR, Nc)`` with
            ``Nc >= 2``.
        support: Boolean support mask of shape ``(RO, LIN, PAR)``.
        neighborhood: Odd box edge lengths along ``(RO, LIN, PAR)``.
        min_members: Minimum member count, between 2 and the box size.
        par_chunk: Maximum number of PAR planes processed per chunk.

    Returns:
        Dictionary with ``"e1"`` and ``"kappa2"`` (float64, NaN outside the
        support, at zero-norm voxels, or with fewer than ``min_members``
        members) and ``"members"`` (int32 member count at support voxels, 0
        elsewhere), each of shape ``(RO, LIN, PAR)``.

    Raises:
        ValueError: If the inputs or parameters are invalid or a coil vector
            inside the support is non-finite.
    """
    values = _numeric_array(coil_vectors, "Coil vectors", 4)
    if values.ndim != 4:
        raise ValueError(f"Coil vectors must be (RO, LIN, PAR, Nc); got {values.shape}.")
    ro, lin, par, coils = (int(size) for size in values.shape)
    if coils < 2:
        raise ValueError(f"Local coil rank needs at least two coils; got {coils}.")
    selection = _bool_mask(support, (ro, lin, par), "Local-rank support")
    box = _neighborhood_shape(neighborhood)
    box_size = box[0] * box[1] * box[2]
    required = _positive_int(min_members, "Minimum member count")
    if not 2 <= required <= box_size:
        raise ValueError(
            f"Minimum member count must lie in [2, {box_size}] for box {box}; got {required}."
        )
    planes = _positive_int(par_chunk, "PAR chunk")
    half = tuple(size // 2 for size in box)
    # Offsets are expressed in padded coordinates: offset half[a] is the center.
    offset_grid = np.meshgrid(*(np.arange(size) for size in box), indexing="ij")
    offsets = [grid.ravel() for grid in offset_grid]
    use_gram = box_size <= coils
    matrix_size = box_size if use_gram else coils
    voxel_bytes = 16 * (2 * box_size * coils + 2 * matrix_size * matrix_size)
    batch = max(1, _GRAM_BATCH_BYTES // voxel_bytes)

    e1 = np.full((ro, lin, par), np.nan, dtype=np.float64)
    kappa2 = np.full((ro, lin, par), np.nan, dtype=np.float64)
    members = np.zeros((ro, lin, par), dtype=np.int32)
    for start in range(0, par, planes):
        stop = min(start + planes, par)
        width = stop - start
        low = max(start - half[2], 0)
        high = min(stop + half[2], par)
        block = np.ascontiguousarray(values[:, :, low:high, :], dtype=np.complex128)
        block_support = selection[:, :, low:high]
        norms = np.sqrt(_squared_norm(block))
        if not np.isfinite(norms[block_support]).all():
            raise ValueError("Coil vectors inside the local-rank support must be finite.")
        valid = block_support & (norms > 0)
        # Zero-padding marks out-of-array and non-member neighbours with zero
        # vectors; they add zero Gram rows and columns, which leave the nonzero
        # spectrum unchanged, so edges are clipped instead of wrapped.
        before = half[2] - (start - low)
        padded_shape = (ro + 2 * half[0], lin + 2 * half[1], width + 2 * half[2])
        unit = np.zeros(padded_shape + (coils,), dtype=np.complex128)
        occupied = np.zeros(padded_shape, dtype=bool)
        region = (
            slice(half[0], half[0] + ro),
            slice(half[1], half[1] + lin),
            slice(before, before + high - low),
        )
        unit[region][valid] = block[valid] / norms[valid, np.newaxis]
        occupied[region] = valid
        del block, norms

        counts = np.zeros((ro, lin, width), dtype=np.int32)
        for a, b, c in zip(*offsets):
            counts += occupied[a : a + ro, b : b + lin, c : c + width]
        center = slice(start - low, start - low + width)
        members[:, :, start:stop] = np.where(block_support[:, :, center], counts, 0)
        index_ro, index_lin, index_par = np.nonzero(valid[:, :, center] & (counts >= required))
        for batch_start in range(0, index_ro.size, batch):
            part = slice(batch_start, batch_start + batch)
            gathered = unit[
                index_ro[part, np.newaxis] + offsets[0],
                index_lin[part, np.newaxis] + offsets[1],
                index_par[part, np.newaxis] + offsets[2],
            ]
            if use_gram:
                matrices = np.matmul(gathered.conj(), gathered.transpose(0, 2, 1))
            else:
                matrices = np.matmul(gathered.transpose(0, 2, 1), gathered.conj())
            spectrum = np.linalg.eigvalsh(matrices)
            leading = spectrum[:, -1]
            target = (index_ro[part], index_lin[part], index_par[part] + start)
            e1[target] = leading / np.sum(np.maximum(spectrum, 0.0), axis=-1)
            kappa2[target] = np.maximum(spectrum[:, -2], 0.0) / leading
    return {"e1": e1, "kappa2": kappa2, "members": members}


def csm_coherence(
    map_values: np.ndarray,
    basis: np.ndarray,
    rank: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure phase-free smoothness of map 1 and of the rank-2 map subspace.

    With the six face neighbours ``delta`` of voxel ``r`` that lie inside the
    array (no wrap-around) and inside the mask:

    * ``C1(r) = min_delta |S1(r)^H S1(r + delta)| / (||S1(r)|| ||S1(r + delta)||)``
      compares map-1 directions and ignores per-voxel phase;
    * ``C2(r) = min_delta ||Q2(r)^H Q2(r + delta)||_F^2 / 2`` compares the
      rank-2 subspaces spanned by maps 1 and 2 (1 for identical subspaces),
      considering only neighbours whose rank is 2 as well.

    A low ``C1`` with ``C2`` near 1 indicates that map 1 changes direction
    inside a stable two-dimensional subspace, for example by switching
    eigenvectors. Per-coil phase maps are never interpreted.

    Args:
        map_values: Map 1 ``S1`` of shape ``(RO, LIN, PAR, Nc)``.
        basis: Rank-2 basis ``Q2`` of shape ``(RO, LIN, PAR, Nc, 2)`` from
            :func:`orthonormal_map_basis` applied to maps 1 and 2.
        rank: Integer rank of shape ``(RO, LIN, PAR)`` returned with ``basis``.
        mask: Boolean voxel mask of shape ``(RO, LIN, PAR)``.

    Returns:
        ``(C1, C2)`` float64 arrays of shape ``(RO, LIN, PAR)`` in ``[0, 1]``.
        ``C1`` is NaN outside the mask, where ``||S1||`` is zero or
        non-finite, or without a valid neighbour. ``C2`` is NaN outside the
        mask, where the rank is below 2, or without a valid rank-2 neighbour.

    Raises:
        ValueError: If shapes or dtypes are invalid, the basis is not
            orthonormal-or-zero inside the mask, or ``rank`` does not match its
            number of unit columns.
    """
    maps_1 = _numeric_array(map_values, "Map 1 values", 4)
    if maps_1.ndim != 4:
        raise ValueError(f"Map 1 values must be (RO, LIN, PAR, Nc); got {maps_1.shape}.")
    ro, lin, par, coils = (int(size) for size in maps_1.shape)
    subspace = _numeric_array(basis, "Rank-2 basis", 5)
    if subspace.shape != (ro, lin, par, coils, 2):
        raise ValueError(
            f"Rank-2 basis must have shape {(ro, lin, par, coils, 2)}; got {subspace.shape}."
        )
    ranks = np.asarray(rank)
    if ranks.shape != (ro, lin, par) or not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(
            f"Rank must be an integer array of shape {(ro, lin, par)}; got {ranks.dtype} "
            f"{ranks.shape}."
        )
    selection = _bool_mask(mask, (ro, lin, par), "Coherence mask")
    c1 = np.full((ro, lin, par), np.nan, dtype=np.float64)
    c2 = np.full((ro, lin, par), np.nan, dtype=np.float64)
    planes = max(1, _CHUNK_ELEMENTS // max(1, ro * lin * coils * 2))
    for start in range(0, par, planes):
        stop = min(start + planes, par)
        # A one-plane PAR halo supplies the face neighbours across chunk borders.
        low = max(start - 1, 0)
        high = min(stop + 1, par)
        keep = slice(start - low, start - low + stop - start)
        local_mask = selection[:, :, low:high]
        local_rank = ranks[:, :, low:high]
        local_basis = np.asarray(subspace[:, :, low:high], dtype=np.complex128)
        # Each plane is validated once, as a kept plane; halo planes are
        # validated in the chunk that keeps them.
        kept_mask = local_mask[:, :, keep]
        unit_columns = _unit_column_count(local_basis[:, :, keep][kept_mask], "Rank-2 basis")
        if np.any(unit_columns != local_rank[:, :, keep][kept_mask]):
            raise ValueError("Rank does not match the number of unit basis columns in the mask.")
        local_map = np.asarray(maps_1[:, :, low:high], dtype=np.complex128)
        norms = np.sqrt(_squared_norm(local_map))
        map_valid = local_mask & np.isfinite(norms) & (norms > 0)
        unit = np.zeros_like(local_map)
        unit[map_valid] = local_map[map_valid] / norms[map_valid, np.newaxis]
        del local_map, norms
        subspace_valid = local_mask & (local_rank >= 2)
        best_map = np.full(local_mask.shape, np.inf)
        best_subspace = np.full(local_mask.shape, np.inf)
        for axis in range(3):
            # RO and LIN pairs are needed only inside the kept planes, whereas
            # PAR pairs also couple the kept planes with the halo planes.
            planes_used = slice(None) if axis == 2 else keep
            unit_view = unit[:, :, planes_used]
            basis_view = local_basis[:, :, planes_used]
            map_view = map_valid[:, :, planes_used]
            subspace_view = subspace_valid[:, :, planes_used]
            length = unit_view.shape[axis]
            if length < 2:
                continue
            lower, upper = _axis_pair(axis, length)
            overlap = np.abs(np.sum(unit_view[lower].conj() * unit_view[upper], axis=-1))
            pair = map_view[lower] & map_view[upper]
            _update_minimum(
                best_map[:, :, planes_used], lower, upper, np.where(pair, overlap, np.inf)
            )
            cross = np.einsum(
                "...cm,...cn->...mn", basis_view[lower].conj(), basis_view[upper]
            )
            energy = 0.5 * np.sum(np.abs(cross) ** 2, axis=(-2, -1))
            pair = subspace_view[lower] & subspace_view[upper]
            _update_minimum(
                best_subspace[:, :, planes_used], lower, upper, np.where(pair, energy, np.inf)
            )
        score = best_map[:, :, keep]
        c1[:, :, start:stop] = np.where(
            map_valid[:, :, keep] & np.isfinite(score), np.minimum(score, 1.0), np.nan
        )
        score = best_subspace[:, :, keep]
        c2[:, :, start:stop] = np.where(
            subspace_valid[:, :, keep] & np.isfinite(score), np.minimum(score, 1.0), np.nan
        )
    return c1, c2


def map_switching_mask(
    c1: np.ndarray,
    c2: np.ndarray,
    lambda1: np.ndarray,
    lambda2: np.ndarray,
    *,
    c1_max: float = 0.9,
    c2_min: float = 0.99,
    gap_max: float = 0.05,
) -> np.ndarray:
    """Label voxels whose map-1 discontinuity looks like eigenvector switching.

    This mask is descriptive only. It marks voxels where map 1 is locally
    incoherent (``C1 <= c1_max``) while the rank-2 subspace stays coherent
    (``C2 >= c2_min``) and the two eigenvalues are nearly degenerate
    (``|lambda1 - lambda2| <= gap_max``). It is a label for review, not a
    decision rule, a correction, or evidence of a particular physical cause.

    Args:
        c1: Map-1 coherence from :func:`csm_coherence`.
        c2: Rank-2 subspace coherence from :func:`csm_coherence`.
        lambda1: Map-1 eigenvalues.
        lambda2: Map-2 eigenvalues.
        c1_max: Inclusive upper bound on ``C1`` in ``[0, 1]``.
        c2_min: Inclusive lower bound on ``C2`` in ``[0, 1]``.
        gap_max: Nonnegative inclusive bound on ``|lambda1 - lambda2|``.

    Returns:
        Boolean mask with the common input shape; NaN inputs give ``False``.

    Raises:
        ValueError: If the inputs are not real arrays of one shape or a
            threshold is invalid.
    """
    arrays = [
        _real_array(values, name, 0)
        for values, name in ((c1, "C1"), (c2, "C2"), (lambda1, "lambda1"), (lambda2, "lambda2"))
    ]
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError(
            "Map-switching inputs must share one shape; got "
            f"{[values.shape for values in arrays]}."
        )
    upper = _finite_scalar(c1_max, "C1 maximum")
    lower = _finite_scalar(c2_min, "C2 minimum")
    gap = _finite_scalar(gap_max, "Eigenvalue gap maximum")
    if not (0 <= upper <= 1 and 0 <= lower <= 1 and gap >= 0):
        raise ValueError(
            "Map-switching thresholds need c1_max and c2_min in [0, 1] and gap_max >= 0; "
            f"got {(upper, lower, gap)}."
        )
    c1_values, c2_values, lambda1_values, lambda2_values = (
        np.asarray(values, dtype=np.float64) for values in arrays
    )
    return (
        (c1_values <= upper)
        & (c2_values >= lower)
        & (np.abs(lambda1_values - lambda2_values) <= gap)
    )
