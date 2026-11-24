"""Real-geometry validation with an injected, known B0 fluctuation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import plots
from .acquisition import LineTiming
from .pipeline import Estimator, build_settings
from .recon import cg_sense
from .simulation import FluctuationModel, add_noise, inject_fluctuation
from .standard_tree import load_case, load_kspace, load_maps, load_psf, maps_base, write_json
from .wave_operator import WaveOperator


def run_simulation(
    standard_root: str | Path,
    output_dir: str | Path,
    timing_all: LineTiming,
    *,
    methods: Sequence[str] = ("linear_field", "global"),
    model: FluctuationModel = FluctuationModel(),
    noise_scale: float = 1.0,
    device: str = "cuda",
    log: Callable[[str], None] = print,
    **options: Any,
) -> dict[str, Any]:
    """Inject a known fluctuation into data synthesized from a real scan and correct it.

    The ground-truth image is the CG-SENSE reconstruction of the measured
    normal case (reference echo 1). Noise is added with the per-sample
    standard deviation of the measured data residual times ``noise_scale``,
    which is an upper bound because that residual also contains real
    inconsistencies.

    Args:
        standard_root: Standard GRE ``vccN`` root (geometry, PSF, maps, data).
        output_dir: Directory for the summary JSON and figures.
        timing_all: Line timing of the scan.
        methods: Estimation methods to evaluate.
        model: Injected fluctuation.
        noise_scale: Multiplier of the measured-residual noise level.
        device: Torch device.
        log: Progress callback.
        **options: Settings overrides (see ``pipeline.build_settings``).

    Returns:
        Summary with image NRMSE and trace errors per method.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    case = load_case(standard_root, "normal")
    record = case.echoes[0]
    echo_time = float(record["te_s"])
    measured = load_kspace(case.inputs / str(record["wave_kspace"]), torch_device)
    mask = torch.sum(torch.abs(measured), dim=(0, 1)) > 0
    timing = timing_all.subset(mask.cpu().numpy())
    operator = WaveOperator(load_maps(maps_base(standard_root), torch_device), load_psf(case.inputs / str(record["psf"]), torch_device), mask)
    truth, _ = cg_sense(operator, measured, iterations=40)
    clean = operator.forward(truth)
    samples = int(mask.sum()) * operator.coils * operator.kspace_shape[1]
    sigma = noise_scale * float(torch.linalg.vector_norm(measured - clean)) / math.sqrt(2 * samples)
    del measured
    corrupted = inject_fluctuation(operator, truth, timing, model, echo_time_s=echo_time, voxel_mm_par=case.voxel_mm[2])
    add_noise(corrupted, sigma, mask, seed=1)
    head = truth.abs() > 0.1 * truth.abs().max()

    def nrmse(image: torch.Tensor) -> dict[str, float]:
        """Head NRMSE of complex values (complex scale) and magnitudes (real scale).

        A static linear field cannot be identified from data consistency and
        only adds an image phase ramp, which the complex NRMSE penalizes and
        the magnitude NRMSE ignores.
        """
        scale = torch.vdot(image[head], truth[head]) / torch.vdot(image[head], image[head])
        complex_error = torch.linalg.vector_norm(scale * image[head] - truth[head]) / torch.linalg.vector_norm(truth[head])
        a, b = image.abs()[head], truth.abs()[head]
        magnitude_error = torch.linalg.vector_norm(torch.sum(a * b) / torch.sum(a * a) * a - b) / torch.linalg.vector_norm(b)
        return {"complex": float(complex_error), "magnitude": float(magnitude_error)}

    noise_only = add_noise(clean.clone(), sigma, mask, seed=1)
    floor_image, _ = cg_sense(operator, noise_only, iterations=40)
    del noise_only, clean
    uncorrected, _ = cg_sense(operator, corrupted, iterations=40)
    weights_z = torch.sum(torch.abs(truth) ** 2, dim=(0, 1))
    z_mm = (torch.arange(truth.shape[2], device=torch_device) - truth.shape[2] // 2) * case.voxel_mm[2]
    z_ref = float(torch.sum(weights_z * z_mm) / torch.sum(weights_z))
    true_uniform = model.uniform_trace_hz(timing.time_s) + model.gradient_trace_hz_per_mm(timing.time_s) * z_ref
    true_gradient = model.gradient_trace_hz_per_mm(timing.time_s)
    summary: dict[str, Any] = {
        "model": asdict(model),
        "noise_sigma": sigma,
        "noise_scale": noise_scale,
        "nrmse_noise_only": nrmse(floor_image),
        "nrmse_uncorrected": nrmse(uncorrected),
        "methods": {},
    }
    traces_for_plot = {"true uniform [Hz]": true_uniform}
    log(f"NRMSE noise only {summary['nrmse_noise_only']}, uncorrected {summary['nrmse_uncorrected']}")
    for method in methods:
        settings = build_settings(method, echo_time_s=echo_time, voxel_mm=case.voxel_mm, **options)
        estimate = Estimator(method, settings, lambda _: None).estimate(operator, corrupted, timing)
        traces = estimate["traces"]
        uniform = traces["uniform_hz"]
        entry = {
            "nrmse_corrected": nrmse(estimate["result"].image),
            "uniform_correlation": float(np.corrcoef(uniform - uniform.mean(), true_uniform - true_uniform.mean())[0, 1]),
            "uniform_rms_error_hz": float(np.sqrt(np.mean(((uniform - uniform.mean()) - (true_uniform - true_uniform.mean())) ** 2))),
            "cycles": len(estimate["result"].history) - 1,
        }
        gradient = traces.get("gradient_z_hz_per_mm", traces.get("coil_affine_gradient_z_hz_per_mm"))
        if gradient is not None and model.gradient_hz_per_mm:
            centred = true_gradient - true_gradient.mean()
            entry["gradient_z_correlation"] = float(np.corrcoef(gradient - gradient.mean(), centred)[0, 1])
            entry["gradient_z_slope"] = float(np.polyfit(centred, gradient - gradient.mean(), 1)[0])
        summary["methods"][method] = entry
        traces_for_plot[f"{method} uniform [Hz]"] = uniform
        log(f"{method}: {entry}")
        del estimate
    plots.plot_traces(timing.time_s, traces_for_plot, output / "simulation_traces.png",
                      title="Injected (true) vs recovered uniform B0 fluctuation", lin=timing.lin)
    write_json(output / "simulation_summary.json", summary)
    return summary
