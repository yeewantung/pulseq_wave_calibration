"""Estimate B0 fluctuations on a standard GRE tree and write a corrected tree.

Output layout below one user-approved root::

    estimation/
        manifest.json                  settings, timing, history, summary
        b0_fluctuation_estimate.npz    per-line times, indices, phases/traces
        figures/                       traces, spectra, k-space maps, images
    normal/bart_inputs/                corrected k-space, PSF copies, manifest
    normal/bart_output/                coil maps + ecalib record (copies)
    retro/<case>/bart_inputs/          corrected retrospective cases
    logs/
"""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import plots
from .acquisition import LineTiming, timing_from_loop_order, timing_from_twix
from .field_model import FieldModelSettings, coordinate_images, modelled_gradient_kspace, run_linear_field_navigator
from .navigator import NavigatorSettings, apply_line_phase, run_consistency_navigator
from .recon import cg_sense
from .standard_tree import (
    CaseInputs,
    install_shared_outputs,
    load_case,
    load_kspace,
    load_maps,
    load_psf,
    maps_base,
    write_case_inputs,
    write_json,
)
from .wave_operator import WaveOperator

METHODS = ("linear_field", "global", "coil_affine", "per_coil")
RETRO_STRATEGIES = ("from-normal", "self")


def _line_mask(kspace: torch.Tensor) -> torch.Tensor:
    """Acquired ``(LIN, PAR)`` lines of a k-space tensor.

    Args:
        kspace: ``(C, Xe, Y, Z)`` k-space.

    Returns:
        Boolean device tensor.
    """
    return torch.sum(torch.abs(kspace), dim=(0, 1)) > 0


class Estimator:
    """Run one estimation method on one case and apply it to every echo.

    Attributes:
        method: One of :data:`METHODS`.
        settings: ``FieldModelSettings`` or ``NavigatorSettings``.
        log: Progress callback.
    """

    def __init__(self, method: str, settings: FieldModelSettings | NavigatorSettings, log: Callable[[str], None]):
        """Store the method and settings.

        Args:
            method: One of :data:`METHODS`.
            settings: Matching settings object.
            log: Progress callback.

        Raises:
            ValueError: If the method is unknown.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")
        self.method = method
        self.settings = settings
        self.log = log

    def estimate(self, operator: WaveOperator, kspace: torch.Tensor, timing: LineTiming) -> dict[str, Any]:
        """Estimate per-line corrections on the reference echo.

        Args:
            operator: Reference-echo Wave operator.
            kspace: Reference-echo k-space.
            timing: Acquisition order.

        Returns:
            Dictionary with the result object, corrected k-space, final image,
            and NumPy arrays for saving.
        """
        if self.method == "linear_field":
            result = run_linear_field_navigator(operator, kspace, timing, self.settings, log=self.log)  # type: ignore[arg-type]
            arrays = {
                "parameters_rad": result.parameters,
                "parameter_names": np.asarray(["uniform"] + [f"gradient_{a}" for a in result.axes]),
                "line_energy": result.line_energy,
                "reference_mm": result.reference_mm,
            }
            traces = result.traces(self.settings.echo_time_s)
        else:
            result = run_consistency_navigator(operator, kspace, timing, self.settings, log=self.log)  # type: ignore[arg-type]
            arrays = {
                "phase_rad_by_coil": result.phase,
                "line_energy_by_coil": result.weights,
                "coil_positions_mm": result.coil_positions_mm,
            }
            traces = {"uniform_hz": result.global_hz(self.settings.echo_time_s)}
            affine = result.affine_traces(self.settings.echo_time_s, ("z",))
            traces["coil_affine_gradient_z_hz_per_mm"] = affine["gradient_z_hz_per_mm"]
        return {"result": result, "arrays": arrays, "traces": traces}

    def apply(
        self,
        estimate: dict[str, Any],
        operator: WaveOperator,
        kspace: torch.Tensor,
        timing: LineTiming,
        echo_time_ratio: float,
        *,
        is_reference: bool,
    ) -> torch.Tensor:
        """Correct one echo with the reference-echo estimate scaled by TE.

        Args:
            estimate: Output of :meth:`estimate`.
            operator: This echo's Wave operator.
            kspace: This echo's measured k-space.
            timing: Acquisition order (same lines as the estimate).
            echo_time_ratio: ``TE_echo / TE_reference``.
            is_reference: True for the estimated echo itself.

        Returns:
            Corrected ``(C, Xe, Y, Z)`` k-space.
        """
        result = estimate["result"]
        if is_reference and echo_time_ratio == 1.0:
            return result.corrected
        device = kspace.device
        lin = torch.as_tensor(timing.lin, device=device)
        par = torch.as_tensor(timing.par, device=device)
        if self.method != "linear_field":
            phase = torch.as_tensor(result.phase * echo_time_ratio, device=device)
            return apply_line_phase(kspace.clone(), phase, lin, par)
        parameters = torch.as_tensor(result.parameters * echo_time_ratio, device=device)
        grid = torch.zeros(operator.line_mask.shape, dtype=torch.float64, device=device)
        grid[lin, par] = parameters[0]
        corrected = kspace * torch.exp(-1j * grid).to(kspace.dtype)[None, None]
        if result.axes:
            # The gradient term needs this echo's own image.
            image, _ = cg_sense(operator, corrected, iterations=self.settings.cg_iterations, tikhonov=self.settings.tikhonov)
            coordinates = coordinate_images(image.shape, self.settings.voxel_mm, result.reference_mm, result.axes, device)
            corrected -= modelled_gradient_kspace(operator, image, coordinates, parameters[1:], lin, par)
        return corrected


def build_settings(method: str, *, echo_time_s: float, voxel_mm: Sequence[float], **options: Any) -> FieldModelSettings | NavigatorSettings:
    """Create the settings object for a method.

    Args:
        method: One of :data:`METHODS`.
        echo_time_s: Reference echo time.
        voxel_mm: Voxel size.
        **options: Keyword overrides accepted by the settings class.

    Returns:
        Settings instance.
    """
    clean = {key: value for key, value in options.items() if value is not None}
    if method == "linear_field":
        return FieldModelSettings(echo_time_s=echo_time_s, voxel_mm=tuple(voxel_mm), **clean)
    axes = clean.pop("axes", None)
    clean.pop("gradient_fwhm_s", None)
    if method == "coil_affine" and axes:
        clean["affine_axes"] = tuple(axes)
    return NavigatorSettings(mode=method, echo_time_s=echo_time_s, voxel_mm=tuple(voxel_mm), **clean)


def run_case(
    case: CaseInputs,
    maps: torch.Tensor,
    timing_all: LineTiming,
    estimator: Estimator,
    reference_echo: int,
    device: torch.device,
    figures: Path,
    log: Callable[[str], None],
) -> tuple[dict[int, torch.Tensor], dict[str, Any], LineTiming]:
    """Estimate on one case's reference echo and correct all its echoes.

    Args:
        case: Case to correct.
        maps: ``(C, X, Y, Z)`` coil maps.
        timing_all: Timing of all lines of the scan.
        estimator: Estimation method.
        reference_echo: One-based estimated echo.
        device: Device.
        figures: Figure directory.
        log: Progress callback.

    Returns:
        Corrected k-space per echo, the estimate, and the case's timing.
    """
    reference_index = case.echo_index(reference_echo)
    reference_record = case.echoes[reference_index]
    kspace = load_kspace(case.inputs / str(reference_record["wave_kspace"]), device)
    mask = _line_mask(kspace)
    timing = timing_all.subset(mask.cpu().numpy())
    operator = WaveOperator(maps, load_psf(case.inputs / str(reference_record["psf"]), device), mask)
    uncorrected, _ = cg_sense(operator, kspace, iterations=estimator.settings.cg_iterations)
    estimate = estimator.estimate(operator, kspace, timing)
    corrected_image = estimate["result"].image
    plots.plot_image_comparison(
        {"uncorrected CG": uncorrected.cpu().numpy(), "corrected CG": corrected_image.cpu().numpy()},
        figures / f"{case.case_id}_cg_images.png",
        title=f"{case.case_id}: unregularized CG-SENSE before/after correction ({estimator.method})",
        voxel_mm=case.voxel_mm,
    )
    estimate["uncorrected_image"] = uncorrected.cpu().numpy()
    estimate["corrected_image"] = corrected_image.cpu().numpy()
    del uncorrected
    reference_te = float(reference_record["te_s"])
    corrected: dict[int, torch.Tensor] = {}
    for record in case.echoes:
        echo = int(record["echo"])
        if echo == reference_echo:
            corrected[echo] = estimate["result"].corrected.cpu()
            continue
        echo_kspace = load_kspace(case.inputs / str(record["wave_kspace"]), device)
        echo_operator = WaveOperator(maps, load_psf(case.inputs / str(record["psf"]), device), mask)
        ratio = float(record["te_s"]) / reference_te
        log(f"{case.case_id}: applying the echo-{reference_echo} estimate to echo {echo} (TE ratio {ratio:.3f})")
        corrected[echo] = estimator.apply(estimate, echo_operator, echo_kspace, timing, ratio, is_reference=False).cpu()
        del echo_kspace, echo_operator
    estimate["result"].corrected = None  # release device memory
    return corrected, estimate, timing


def estimate_and_correct(
    standard_root: str | Path,
    output_root: str | Path,
    *,
    method: str = "linear_field",
    twix: str | Path | None = None,
    repetition_time_s: float | None = None,
    reference_echo: int | None = None,
    retro_cases: Sequence[str] = ("native_r3x3",),
    retro_strategy: str = "from-normal",
    device: str = "cuda",
    log: Callable[[str], None] = print,
    **options: Any,
) -> dict[str, Any]:
    """Estimate B0 fluctuations on the normal case and write a corrected tree.

    Args:
        standard_root: Standard GRE ``vccN`` root.
        output_root: User-approved output root.
        method: One of :data:`METHODS`.
        twix: TWIX file for per-line MDH timestamps (preferred).
        repetition_time_s: TR for loop-order timing when ``twix`` is absent.
        reference_echo: One-based estimated echo; defaults to the last echo.
        retro_cases: Retrospective cases of the standard tree to correct.
        retro_strategy: ``from-normal`` masks the corrected normal data;
            ``self`` re-estimates on each retrospective case's own lines.
        device: Torch device.
        log: Progress callback.
        **options: Settings overrides (see :func:`build_settings`).

    Returns:
        Estimation manifest.

    Raises:
        ValueError: If neither timing source is given or options are invalid.

    Side Effects:
        Writes the corrected tree, estimation arrays, and figures.
    """
    if retro_strategy not in RETRO_STRATEGIES:
        raise ValueError(f"retro_strategy must be one of {RETRO_STRATEGIES}.")
    if twix is None and repetition_time_s is None:
        raise ValueError("Provide --twix (preferred) or --repetition-time-s.")
    root = Path(output_root).expanduser().resolve()
    estimation_dir = root / "estimation"
    figures = estimation_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    normal = load_case(standard_root, "normal")
    reference_echo = int(reference_echo or normal.echoes[-1]["echo"])
    reference_record = normal.echoes[normal.echo_index(reference_echo)]
    settings = build_settings(
        method, echo_time_s=float(reference_record["te_s"]), voxel_mm=normal.voxel_mm, **options
    )
    estimator = Estimator(method, settings, log)
    maps = load_maps(maps_base(standard_root), torch_device)
    if twix is not None:
        timing_all = timing_from_twix(twix, echo=reference_echo - 1)
    else:
        mask = np.load(normal.inputs / "sampling_mask.npy") if (normal.inputs / "sampling_mask.npy").is_file() else None
        if mask is None:
            raise ValueError("Loop-order timing needs normal/bart_inputs/sampling_mask.npy.")
        timing_all = timing_from_loop_order(mask, float(repetition_time_s))
    log(f"timing: {timing_all.to_json()}")

    record: dict[str, Any] = {
        "method": method,
        "settings": {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(settings).items()},
        "reference_echo": reference_echo,
        "timing": timing_all.to_json(),
        "standard_root": str(Path(standard_root).expanduser().resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "Meineke J, Nielsen T. Magn Reson Med 2019;81:3046-3055 (consistency navigator), Wave forward model",
    }
    cases_summary: dict[str, Any] = {}

    def save_case(case: CaseInputs, corrected: dict[int, torch.Tensor], estimate: dict[str, Any] | None,
                  timing: LineTiming, mask: np.ndarray | None) -> None:
        """Write one case's corrected inputs, arrays, and figures."""
        provenance = {**record, "estimated_on_case": case.case_id if estimate is not None else "normal"}
        write_case_inputs(case, root, corrected, provenance, sampling_mask=mask)
        summary: dict[str, Any] = {"line_count": timing.count}
        if estimate is not None:
            traces = estimate["traces"]
            np.savez(
                estimation_dir / f"{case.case_id}_b0_fluctuation_estimate.npz",
                lin=timing.lin, par=timing.par, time_s=timing.time_s,
                **estimate["arrays"], **{f"trace_{k}": v for k, v in traces.items() if isinstance(v, np.ndarray) and v.shape == timing.time_s.shape},
            )
            plots.plot_traces(
                timing.time_s,
                {k: v for k, v in traces.items() if isinstance(v, np.ndarray) and v.shape == timing.time_s.shape},
                figures / f"{case.case_id}_traces.png",
                title=f"{case.case_id}: B0 fluctuation estimate ({method})",
                lin=timing.lin,
            )
            plots.plot_kspace_phase(
                timing.lin, timing.par, traces["uniform_hz"], (maps.shape[2], maps.shape[3]),
                figures / f"{case.case_id}_uniform_hz_on_kspace.png", f"{case.case_id}: uniform B0 fluctuation [Hz]",
            )
            np.save(estimation_dir / f"{case.case_id}_cg_uncorrected.npy", estimate["uncorrected_image"])
            np.save(estimation_dir / f"{case.case_id}_cg_corrected.npy", estimate["corrected_image"])
            uniform = traces["uniform_hz"]
            summary.update(
                {
                    "uniform_hz_std": float(np.std(uniform)),
                    "uniform_hz_peak_to_peak": float(np.ptp(uniform)),
                    "history": estimate["result"].history,
                }
            )
            for key, value in traces.items():
                if key.startswith("gradient") or key.startswith("coil_affine"):
                    summary[f"{key}_std"] = float(np.std(value))
        cases_summary[case.case_id] = summary

    corrected, estimate, timing = run_case(normal, maps, timing_all, estimator, reference_echo, torch_device, figures, log)
    save_case(normal, corrected, estimate, timing, None)
    for case_id in retro_cases:
        case = load_case(standard_root, case_id)
        mask = np.load(case.inputs / "sampling_mask.npy")
        if retro_strategy == "from-normal":
            mask_tensor = torch.as_tensor(mask, dtype=torch.complex64)
            masked = {echo: value * mask_tensor[None, None] for echo, value in corrected.items()}
            save_case(case, masked, None, timing.subset(mask), mask)
        else:
            case_corrected, case_estimate, case_timing = run_case(
                case, maps, timing_all, estimator, reference_echo, torch_device, figures, log
            )
            save_case(case, case_corrected, case_estimate, case_timing, mask)
            del case_corrected
        log(f"wrote corrected {case_id} ({retro_strategy})")
    install_shared_outputs(standard_root, root)
    manifest = {
        **record,
        "status": "b0_fluctuation_corrected_inputs_ready",
        "retro_cases": list(retro_cases),
        "retro_strategy": retro_strategy,
        "cases": cases_summary,
        "units": {"uniform_hz": "Hz at the signal-weighted reference position", "gradient": "Hz/mm"},
    }
    write_json(estimation_dir / "manifest.json", manifest)
    return manifest
