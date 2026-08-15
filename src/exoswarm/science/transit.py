"""Deterministic BLS transit search, lightweight refinement, and phase folding."""

from __future__ import annotations

import matplotlib
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.optimize import least_squares

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from exoswarm.domain.models import (
    Candidate,
    ExperimentType,
    InterpretationCode,
    MeasurementUncertainty,
    QualityFlag,
    QualitySeverity,
    ScientificFailure,
    ScientificResult,
    ScientificStatus,
)

from .common import (
    TOOL_VERSION,
    ArtifactStore,
    ScienceManifest,
    load_npz,
    robust_sigma,
    setup_science_axes,
    tess_provenance,
)


def search_transits(
    manifest: ScienceManifest,
    store: ArtifactStore,
    *,
    min_period_days: float = 0.5,
    max_period_days: float = 10.0,
    min_duration_hours: float = 1.0,
    max_duration_hours: float = 6.0,
    frequency_samples: int = 12000,
    harmonic_tie_fraction: float = 0.04,
    alternate_detrending: bool = False,
) -> ScientificResult | ScientificFailure:
    """Run an Astropy Box Least Squares search over a declared bounded grid.

    A near-degenerate P/P/2 family is handled conservatively: when P/2 retains at
    least ``1-harmonic_tie_fraction`` of the best BLS S/N, the shorter event
    cadence is recorded as the initial ephemeris and explicitly flagged for
    odd/even plus harmonic resolution.  This is target-independent and prevents
    an eclipsing binary's alternating events from being silently collapsed into a
    confident orbital-period claim.
    """

    parameters: dict[str, int | float | bool | str] = {
        "min_period_days": min_period_days,
        "max_period_days": max_period_days,
        "min_duration_hours": min_duration_hours,
        "max_duration_hours": max_duration_hours,
        "frequency_samples": frequency_samples,
        "harmonic_tie_fraction": harmonic_tie_fraction,
        "objective": "snr",
        "algorithm": "astropy.timeseries.BoxLeastSquares",
        "alternate_detrending": alternate_detrending,
    }
    if (
        min_period_days <= 0
        or max_period_days <= min_period_days
        or min_duration_hours <= 0
        or max_duration_hours <= min_duration_hours
        or frequency_samples < 2000
        or frequency_samples > 50000
        or not 0 <= harmonic_tie_fraction <= 0.1
    ):
        return _failure(
            ExperimentType.TRANSIT_SEARCH,
            "invalid bounded BLS search parameters",
            "INVALID_BLS_GRID",
            parameters=parameters,
        )
    stem = "alternate_cleaned_light_curve" if alternate_detrending else "cleaned_light_curve"
    cleaned_path = store.path(stem)
    if not cleaned_path.is_file():
        return _failure(
            ExperimentType.TRANSIT_SEARCH,
            "detrended light-curve artifact is required before transit search",
            "CLEANED_CURVE_MISSING",
            alternatives=[ExperimentType.DETRENDING],
            parameters=parameters,
        )
    arrays = load_npz(cleaned_path)
    time = arrays["time_btjd"].astype(float)
    flux = arrays["flux"].astype(float)
    flux_err = arrays["flux_err"].astype(float)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    if time.size < 1000:
        return _failure(
            ExperimentType.TRANSIT_SEARCH,
            "BLS requires at least 1000 usable cadences",
            "INSUFFICIENT_CADENCES",
            alternatives=[ExperimentType.QUALITY_INSPECTION],
            parameters=parameters,
        )
    baseline = float(time[-1] - time[0])
    effective_max_period = min(max_period_days, baseline / 2.0)
    if effective_max_period <= min_period_days:
        return _failure(
            ExperimentType.TRANSIT_SEARCH,
            "observing baseline cannot support two events in the requested period range",
            "BASELINE_TOO_SHORT",
            alternatives=[ExperimentType.QUALITY_INSPECTION],
            parameters=parameters,
        )
    duration_min = min_duration_hours / 24.0
    duration_max = min(max_duration_hours / 24.0, 0.45 * min_period_days)
    durations = np.linspace(duration_min, duration_max, 9)
    frequencies = np.linspace(1.0 / effective_max_period, 1.0 / min_period_days, frequency_samples)
    periods = 1.0 / frequencies
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    coarse = bls.power(periods, durations, objective="snr", method="fast", oversample=10)
    global_index = int(np.nanargmax(coarse.power))
    coarse_frequency_step = float(frequencies[1] - frequencies[0])
    global_peak = _refine_peak(
        bls,
        center_period=float(coarse.period[global_index]),
        frequency_half_width=4.0 * coarse_frequency_step,
        duration_min=duration_min,
        duration_max=duration_max,
    )

    selected = global_peak
    initial_alias_selected = False
    half_ratio = float("nan")
    half_period = float(global_peak["period"] / 2.0)
    if half_period >= min_period_days:
        half_peak = _refine_peak(
            bls,
            center_period=half_period,
            frequency_half_width=4.0 * coarse_frequency_step,
            duration_min=duration_min,
            duration_max=duration_max,
        )
        half_ratio = float(half_peak["power"] / global_peak["power"])
        if half_ratio >= 1.0 - harmonic_tie_fraction:
            selected = half_peak
            initial_alias_selected = True

    cadence_days = float(np.median(np.diff(time)[np.diff(time) < 0.1]))
    period = float(selected["period"])
    epoch = float(selected["transit_time"])
    depth = float(selected["depth"])
    duration = float(selected["duration"])
    depth_error = float(selected["depth_err"])
    bls_snr = float(selected["depth_snr"])
    refinement_used = False
    if not initial_alias_selected:
        refined = _fit_trapezoid(
            time,
            flux,
            flux_err,
            period=period,
            epoch=epoch,
            depth=max(depth, 1e-6),
            duration=duration,
            cadence_days=cadence_days,
            period_half_width=max(
                5.0 * coarse_frequency_step * period**2,
                0.002 * period,
            ),
        )
        if refined is not None:
            period = refined["period"]
            epoch = refined["epoch"]
            depth = refined["depth"]
            duration = refined["duration"]
            depth_error = refined["depth_error"]
            refinement_used = True

    event_count = _observed_event_count(time, period, epoch, duration)
    period_grid_resolution = float((8.0 * coarse_frequency_step / 2999.0) * period**2)
    period_tolerance = max(period_grid_resolution, cadence_days / max(event_count - 1, 1))
    epoch_tolerance = cadence_days / 2.0
    duration_resolution_hours = max(cadence_days * 24.0, (duration_max - duration_min) * 24 / 64)
    radius_ratio = float(np.sqrt(max(depth, 0.0)))
    radius_ratio_error = float(depth_error / (2.0 * radius_ratio)) if radius_ratio > 0 else 0.0
    detected = bool(bls_snr >= 7.0 and event_count >= 2 and depth > 0)
    quality_flags: list[QualityFlag] = [
        QualityFlag(
            code="FORMAL_WHITE_NOISE_SNR",
            severity=QualitySeverity.INFO,
            detail="BLS depth S/N uses SPOC per-cadence errors and is not a false-alarm probability.",
        )
    ]
    if initial_alias_selected:
        quality_flags.append(
            QualityFlag(
                code="NEAR_DEGENERATE_HALF_PERIOD",
                severity=QualitySeverity.WARNING,
                detail=(
                    "P/2 retained at least the declared fraction of the strongest BLS peak; "
                    "the shorter event cadence is provisional pending odd/even and harmonic tests."
                ),
            )
        )

    order = np.argsort(np.asarray(coarse.period))
    period_data = store.save_npz(
        "bls_periodogram",
        "bls_periodogram_data",
        period_days=np.asarray(coarse.period)[order],
        power_snr=np.asarray(coarse.power)[order],
        duration_days_at_peak=np.asarray(coarse.duration)[order],
    )
    figure, axis = plt.subplots(figsize=(9.5, 3.8))
    axis.plot(
        np.asarray(coarse.period)[order],
        np.asarray(coarse.power)[order],
        color="#64d8ff",
        linewidth=0.8,
    )
    axis.axvline(period, color="#ffcc66", linewidth=1.2, label=f"initial P = {period:.6f} d")
    if initial_alias_selected:
        axis.axvline(
            float(global_peak["period"]),
            color="#ff758f",
            linewidth=0.9,
            linestyle="--",
            label="slightly stronger 2P family peak",
        )
    axis.set_title("Deterministic Box Least Squares search")
    axis.legend(frameon=False, labelcolor="#d7e4f5")
    setup_science_axes(axis, xlabel="Trial period (d)", ylabel="BLS depth S/N")
    period_plot = store.save_figure("bls_periodogram", "bls_periodogram_plot", figure)
    plt.close(figure)

    numerical_results = {
        "period_days": period,
        "epoch_btjd": epoch,
        "transit_depth_ppm": depth * 1e6,
        "duration_hours": duration * 24.0,
        "signal_to_noise": bls_snr,
        "observed_events": int(event_count),
        "bls_global_period_days": float(global_peak["period"]),
        "bls_global_peak_snr": float(global_peak["depth_snr"]),
        "half_period_snr_ratio": float(half_ratio if np.isfinite(half_ratio) else 0.0),
        "initial_alias_selected": int(initial_alias_selected),
        "approx_radius_ratio": radius_ratio,
        "search_baseline_days": baseline,
    }
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.TRANSIT_SEARCH,
        tool_name="exoswarm.science.search_transits",
        tool_version=TOOL_VERSION,
        parameters=parameters | {"effective_max_period_days": effective_max_period},
        numerical_results=numerical_results,
        result_units={
            "period_days": "d",
            "epoch_btjd": "BTJD",
            "transit_depth_ppm": "ppm",
            "duration_hours": "h",
            "signal_to_noise": "dimensionless",
            "observed_events": "count",
            "bls_global_period_days": "d",
            "bls_global_peak_snr": "dimensionless",
            "half_period_snr_ratio": "ratio",
            "initial_alias_selected": "boolean integer",
            "approx_radius_ratio": "ratio",
            "search_baseline_days": "d",
        },
        uncertainties={
            "period_days": MeasurementUncertainty(
                value=period_tolerance,
                unit="d",
                method="maximum of BLS frequency resolution and cadence/event-baseline resolution",
                kind="tolerance",
            ),
            "epoch_btjd": MeasurementUncertainty(
                value=epoch_tolerance,
                unit="d",
                method="half-cadence timing resolution",
                kind="resolution",
            ),
            "transit_depth_ppm": MeasurementUncertainty(
                value=depth_error * 1e6,
                unit="ppm",
                method=(
                    "robust trapezoid residual standard error"
                    if refinement_used
                    else "Astropy BLS formal depth uncertainty"
                ),
                kind="standard_uncertainty",
            ),
            "duration_hours": MeasurementUncertainty(
                value=duration_resolution_hours,
                unit="h",
                method="maximum of cadence and local duration-grid resolution",
                kind="resolution",
            ),
            "approx_radius_ratio": MeasurementUncertainty(
                value=radius_ratio_error,
                unit="ratio",
                method="first-order propagation from transit-depth uncertainty",
                kind="standard_uncertainty",
            ),
        },
        quality_flags=quality_flags,
        interpretation_code=(
            InterpretationCode.DETECTED if detected else InterpretationCode.NOT_DETECTED
        ),
        limitations=[
            "BLS assumes a periodic box and is a search statistic, not a physical transit model.",
            "The reported period tolerance is an explicit numerical tolerance, not a calibrated confidence interval.",
            "The approximate radius ratio sqrt(depth) neglects dilution, limb darkening and grazing geometry.",
            "Formal depth S/N does not account fully for time-correlated stellar or instrumental noise.",
            *(
                [
                    "The initial period is a deliberately provisional P/2 event cadence requiring harmonic resolution."
                ]
                if initial_alias_selected
                else []
            ),
        ],
        input_artifacts=[
            store.existing(
                stem,
                "alternate_cleaned_light_curve_data"
                if alternate_detrending
                else "cleaned_light_curve_data",
            )
        ],
        output_artifacts=[period_data, period_plot],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def compare_detrending_sensitivity(
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
    alternate_preprocessing: ScientificResult,
    *,
    max_period_fractional_shift: float = 0.005,
    max_depth_fractional_change: float = 0.25,
    max_depth_difference_sigma: float = 5.0,
    minimum_snr_ratio: float = 0.70,
    minimum_signal_to_noise: float = 7.0,
) -> ScientificResult | ScientificFailure:
    """Re-measure a candidate under nominal and alternate detrending.

    The comparison deliberately performs the same local BLS re-search on both
    light curves.  This avoids comparing a nominal trapezoid fit to an alternate
    box fit, and lets the adaptive experiment test whether period, depth and
    signal quality persist under a declared preprocessing change.
    """

    experiment_type = ExperimentType.ALTERNATE_DETRENDING
    if alternate_preprocessing.experiment_type is not experiment_type:
        return _failure(
            experiment_type,
            "alternate preprocessing result is required for sensitivity comparison",
            "ALTERNATE_PREPROCESSING_REQUIRED",
            alternatives=[ExperimentType.ALTERNATE_DETRENDING],
        )
    nominal_path = store.path("cleaned_light_curve")
    alternate_path = store.path("alternate_cleaned_light_curve")
    if not nominal_path.is_file() or not alternate_path.is_file():
        return _failure(
            experiment_type,
            "both nominal and alternate detrended light curves are required",
            "DETRENDING_PAIR_MISSING",
            alternatives=[ExperimentType.DETRENDING],
        )
    nominal_arrays = load_npz(nominal_path)
    alternate_arrays = load_npz(alternate_path)
    nominal = _local_candidate_measurement(nominal_arrays, candidate)
    alternate = _local_candidate_measurement(alternate_arrays, candidate)
    if nominal is None or alternate is None:
        return _failure(
            experiment_type,
            "candidate could not be re-measured on both detrending products",
            "LOCAL_CANDIDATE_RESEARCH_FAILED",
            alternatives=[ExperimentType.HARMONIC_TEST, ExperimentType.SIGNAL_QUALITY],
        )

    period_fractional_shift = abs(alternate["period"] - nominal["period"]) / nominal["period"]
    depth_difference = abs(alternate["depth"] - nominal["depth"])
    depth_fractional_change = depth_difference / max(nominal["depth"], np.finfo(float).eps)
    combined_depth_error = float(np.hypot(nominal["depth_error"], alternate["depth_error"]))
    depth_difference_sigma = (
        depth_difference / combined_depth_error if combined_depth_error > 0 else 0.0
    )
    snr_ratio = alternate["snr"] / max(nominal["snr"], np.finfo(float).eps)
    depth_consistent = bool(
        depth_fractional_change <= max_depth_fractional_change
        or depth_difference_sigma <= max_depth_difference_sigma
    )
    robust = bool(
        period_fractional_shift <= max_period_fractional_shift
        and depth_consistent
        and alternate["snr"] >= minimum_signal_to_noise
        and snr_ratio >= minimum_snr_ratio
        and alternate["observed_events"] >= 2
    )

    data_artifact = store.save_npz(
        "detrending_sensitivity",
        "detrending_sensitivity_data",
        configuration_code=np.asarray([0, 1], dtype=np.int8),
        period_days=np.asarray([nominal["period"], alternate["period"]]),
        epoch_btjd=np.asarray([nominal["epoch"], alternate["epoch"]]),
        depth_ppm=np.asarray([nominal["depth"], alternate["depth"]]) * 1e6,
        depth_uncertainty_ppm=np.asarray([nominal["depth_error"], alternate["depth_error"]]) * 1e6,
        duration_hours=np.asarray([nominal["duration"], alternate["duration"]]) * 24.0,
        signal_to_noise=np.asarray([nominal["snr"], alternate["snr"]]),
        observed_events=np.asarray(
            [nominal["observed_events"], alternate["observed_events"]], dtype=np.int64
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    labels = ["Nominal", "Alternate"]
    colors = ["#64d8ff", "#ffca62"]
    plotted = (
        ("Period (d)", [nominal["period"], alternate["period"]]),
        ("Depth (ppm)", [nominal["depth"] * 1e6, alternate["depth"] * 1e6]),
        ("BLS depth S/N", [nominal["snr"], alternate["snr"]]),
    )
    for axis, (ylabel, values) in zip(axes, plotted, strict=True):
        axis.bar(labels, values, color=colors, width=0.62)
        setup_science_axes(axis, xlabel="", ylabel=ylabel)
    figure.suptitle(
        "Candidate re-measurement under alternate detrending"
        f" — {'robust' if robust else 'sensitive'}",
        color="#e8f2ff",
    )
    plot_artifact = store.save_figure(
        "detrending_sensitivity", "detrending_sensitivity_plot", figure
    )
    plt.close(figure)

    numerical_results = {
        "nominal_period_days": float(nominal["period"]),
        "alternate_period_days": float(alternate["period"]),
        "period_fractional_shift": float(period_fractional_shift),
        "nominal_epoch_btjd": float(nominal["epoch"]),
        "alternate_epoch_btjd": float(alternate["epoch"]),
        "nominal_depth_ppm": float(nominal["depth"] * 1e6),
        "alternate_depth_ppm": float(alternate["depth"] * 1e6),
        "depth_fractional_change": float(depth_fractional_change),
        "depth_difference_sigma": float(depth_difference_sigma),
        "nominal_duration_hours": float(nominal["duration"] * 24.0),
        "alternate_duration_hours": float(alternate["duration"] * 24.0),
        "nominal_signal_to_noise": float(nominal["snr"]),
        "alternate_signal_to_noise": float(alternate["snr"]),
        "alternate_to_nominal_snr_ratio": float(snr_ratio),
        "nominal_observed_events": int(nominal["observed_events"]),
        "alternate_observed_events": int(alternate["observed_events"]),
        "passes_preprocessing_robustness": int(robust),
    }
    result_units = {
        "nominal_period_days": "d",
        "alternate_period_days": "d",
        "period_fractional_shift": "fraction",
        "nominal_epoch_btjd": "BTJD",
        "alternate_epoch_btjd": "BTJD",
        "nominal_depth_ppm": "ppm",
        "alternate_depth_ppm": "ppm",
        "depth_fractional_change": "fraction",
        "depth_difference_sigma": "sigma",
        "nominal_duration_hours": "h",
        "alternate_duration_hours": "h",
        "nominal_signal_to_noise": "dimensionless",
        "alternate_signal_to_noise": "dimensionless",
        "alternate_to_nominal_snr_ratio": "ratio",
        "nominal_observed_events": "count",
        "alternate_observed_events": "count",
        "passes_preprocessing_robustness": "boolean integer",
    }
    uncertainties = {
        f"{prefix}_period_days": MeasurementUncertainty(
            value=float(measurement["period_tolerance"]),
            unit="d",
            method="local BLS frequency-grid tolerance",
            kind="tolerance",
        )
        for prefix, measurement in (("nominal", nominal), ("alternate", alternate))
    }
    uncertainties.update(
        {
            f"{prefix}_epoch_btjd": MeasurementUncertainty(
                value=float(measurement["cadence_days"] / 2.0),
                unit="d",
                method="half-cadence timing resolution",
                kind="resolution",
            )
            for prefix, measurement in (("nominal", nominal), ("alternate", alternate))
        }
    )
    uncertainties.update(
        {
            f"{prefix}_depth_ppm": MeasurementUncertainty(
                value=float(measurement["depth_error"] * 1e6),
                unit="ppm",
                method="Astropy BLS formal depth uncertainty",
            )
            for prefix, measurement in (("nominal", nominal), ("alternate", alternate))
        }
    )
    uncertainties.update(
        {
            f"{prefix}_duration_hours": MeasurementUncertainty(
                value=float(measurement["duration_resolution"] * 24.0),
                unit="h",
                method="local BLS duration-grid resolution",
                kind="resolution",
            )
            for prefix, measurement in (("nominal", nominal), ("alternate", alternate))
        }
    )
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=experiment_type,
        tool_name="exoswarm.science.compare_detrending_sensitivity",
        tool_version=TOOL_VERSION,
        parameters={
            **alternate_preprocessing.parameters,
            "comparison_algorithm": "matched local Astropy BoxLeastSquares re-search",
            "max_period_fractional_shift": max_period_fractional_shift,
            "max_depth_fractional_change": max_depth_fractional_change,
            "max_depth_difference_sigma": max_depth_difference_sigma,
            "minimum_snr_ratio": minimum_snr_ratio,
            "minimum_signal_to_noise": minimum_signal_to_noise,
        },
        numerical_results=numerical_results,
        result_units=result_units,
        uncertainties=uncertainties,
        quality_flags=[
            QualityFlag(
                code="DETRENDING_ROBUST" if robust else "DETRENDING_SENSITIVE",
                severity=QualitySeverity.INFO if robust else QualitySeverity.WARNING,
                detail=(
                    "The candidate passed all declared preprocessing-persistence thresholds."
                    if robust
                    else "At least one declared preprocessing-persistence threshold failed."
                ),
            )
        ],
        interpretation_code=(
            InterpretationCode.ROBUST if robust else InterpretationCode.PREPROCESSING_SENSITIVE
        ),
        limitations=[
            "This is a local candidate re-search, not an unrestricted search for a different dominant period.",
            "The thresholds are declared heuristic robustness criteria, not calibrated probabilities.",
            "Both products inherit SPOC PDC corrections and therefore are not fully independent reductions.",
        ],
        input_artifacts=[
            store.existing("normalized_light_curve", "normalized_light_curve_data"),
            store.existing("cleaned_light_curve", "cleaned_light_curve_data"),
        ],
        output_artifacts=[
            *alternate_preprocessing.output_artifacts,
            data_artifact,
            plot_artifact,
        ],
        provenance=alternate_preprocessing.provenance,
    )


def phase_fold_candidate(
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
    *,
    bins: int = 180,
    alternate_detrending: bool = False,
) -> ScientificResult | ScientificFailure:
    if bins < 50 or bins > 500:
        return _failure(
            ExperimentType.PHASE_FOLD,
            "phase-fold bins must be between 50 and 500",
            "INVALID_PHASE_BINS",
            parameters={"bins": bins},
        )
    stem = "alternate_cleaned_light_curve" if alternate_detrending else "cleaned_light_curve"
    path = store.path(stem)
    if not path.is_file():
        return _failure(
            ExperimentType.PHASE_FOLD,
            "detrended light curve is missing",
            "CLEANED_CURVE_MISSING",
            alternatives=[ExperimentType.DETRENDING],
        )
    arrays = load_npz(path)
    time = arrays["time_btjd"].astype(float)
    flux = arrays["flux"].astype(float)
    flux_err = arrays["flux_err"].astype(float)
    phase = _phase(time, candidate.period_days, candidate.epoch_btjd)
    order = np.argsort(phase)
    phase, flux, flux_err = phase[order], flux[order], flux_err[order]
    bin_phase, bin_flux, bin_error, bin_count = _bin_series(phase, flux, bins=bins)
    duration_days = candidate.duration_hours / 24.0
    model_flux = _trapezoid_model(
        time,
        period=candidate.period_days,
        epoch=candidate.epoch_btjd,
        depth=candidate.transit_depth_ppm / 1e6,
        duration=duration_days,
        ingress_fraction=0.25,
        baseline=1.0,
    )[order]
    data_artifact = store.save_npz(
        "phase_folded",
        "phase_folded_data",
        phase=phase,
        phase_time_days=phase * candidate.period_days,
        flux=flux,
        flux_err=flux_err,
        bin_phase=bin_phase,
        bin_flux=bin_flux,
        bin_flux_err=bin_error,
        bin_count=bin_count,
        model_flux=model_flux,
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    axis.scatter(phase, flux, s=1.0, color="#476b8e", alpha=0.16, rasterized=True)
    good_bins = np.isfinite(bin_flux)
    axis.errorbar(
        bin_phase[good_bins],
        bin_flux[good_bins],
        yerr=bin_error[good_bins],
        fmt="o",
        markersize=2.6,
        linewidth=0.7,
        color="#69e5ff",
        ecolor="#69e5ff",
        alpha=0.95,
        label="phase bins",
    )
    axis.plot(
        phase,
        model_flux,
        color="#ffca62",
        linewidth=1.1,
        alpha=0.9,
        label="measured box/trapezoid summary",
    )
    half_window = max(0.08, 4.0 * duration_days / candidate.period_days)
    axis.set_xlim(-half_window, half_window)
    axis.set_title("Phase-folded candidate")
    axis.legend(frameon=False, labelcolor="#d7e4f5")
    setup_science_axes(axis, xlabel="Orbital phase", ylabel="Detrended flux")
    plot_artifact = store.save_figure("phase_folded", "phase_folded_plot", figure)
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.PHASE_FOLD,
        tool_name="exoswarm.science.phase_fold_candidate",
        tool_version=TOOL_VERSION,
        parameters={"bins": bins, "phase_convention": "[-0.5, 0.5), transit at 0"},
        numerical_results={
            "period_days": float(candidate.period_days),
            "epoch_btjd": float(candidate.epoch_btjd),
            "folded_points": int(phase.size),
            "nonempty_bins": int(np.count_nonzero(good_bins)),
        },
        result_units={
            "period_days": "d",
            "epoch_btjd": "BTJD",
            "folded_points": "count",
            "nonempty_bins": "count",
        },
        uncertainties={
            key: value
            for key, value in candidate.uncertainties.items()
            if key in {"period_days", "epoch_btjd"}
        },
        quality_flags=[],
        interpretation_code=InterpretationCode.PROCESSED,
        limitations=[
            "Phase folding is a visualization and does not create an independent measurement.",
            "The plotted shape is a lightweight summary, not a limb-darkened physical fit.",
        ],
        input_artifacts=[
            store.existing(
                stem,
                "alternate_cleaned_light_curve_data"
                if alternate_detrending
                else "cleaned_light_curve_data",
            )
        ],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def _local_candidate_measurement(
    arrays: dict[str, np.ndarray], candidate: Candidate
) -> dict[str, float] | None:
    time = np.asarray(arrays.get("time_btjd", []), dtype=float)
    flux = np.asarray(arrays.get("flux", []), dtype=float)
    flux_err = np.asarray(arrays.get("flux_err", []), dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    if time.size < 1000:
        return None
    differences = np.diff(time)
    short_differences = differences[(differences > 0) & (differences < 0.1)]
    if short_differences.size == 0:
        return None
    cadence_days = float(np.median(short_differences))
    baseline = float(time[-1] - time[0])
    center_frequency = 1.0 / candidate.period_days
    frequency_half_width = min(0.02 * center_frequency, 3.0 / baseline)
    frequencies = np.linspace(
        max(np.finfo(float).eps, center_frequency - frequency_half_width),
        center_frequency + frequency_half_width,
        2400,
    )
    periods = 1.0 / frequencies
    candidate_duration = candidate.duration_hours / 24.0
    minimum_duration = max(3.0 * cadence_days, 0.65 * candidate_duration)
    maximum_duration = min(0.30, 1.45 * candidate_duration, 0.40 * candidate.period_days)
    if maximum_duration <= minimum_duration:
        return None
    durations = np.linspace(minimum_duration, maximum_duration, 17)
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    try:
        result = bls.power(periods, durations, objective="snr", method="fast", oversample=20)
        index = int(np.nanargmax(result.power))
    except (ValueError, FloatingPointError):
        return None
    period = float(result.period[index])
    epoch = float(result.transit_time[index])
    duration = float(result.duration[index])
    return {
        "period": period,
        "epoch": epoch,
        "depth": float(result.depth[index]),
        "depth_error": float(result.depth_err[index]),
        "duration": duration,
        "snr": float(result.depth_snr[index]),
        "observed_events": float(_observed_event_count(time, period, epoch, duration)),
        "period_tolerance": float(abs(periods[1] - periods[0])),
        "duration_resolution": float(durations[1] - durations[0]),
        "cadence_days": cadence_days,
    }


def _refine_peak(
    bls: BoxLeastSquares,
    *,
    center_period: float,
    frequency_half_width: float,
    duration_min: float,
    duration_max: float,
) -> dict[str, float]:
    center_frequency = 1.0 / center_period
    frequencies = np.linspace(
        max(np.finfo(float).eps, center_frequency - frequency_half_width),
        center_frequency + frequency_half_width,
        3000,
    )
    periods = 1.0 / frequencies
    durations = np.linspace(duration_min, duration_max, 33)
    result = bls.power(periods, durations, objective="snr", method="fast", oversample=20)
    index = int(np.nanargmax(result.power))
    return {
        "period": float(result.period[index]),
        "duration": float(result.duration[index]),
        "transit_time": float(result.transit_time[index]),
        "depth": float(result.depth[index]),
        "depth_err": float(result.depth_err[index]),
        "depth_snr": float(result.depth_snr[index]),
        "power": float(result.power[index]),
    }


def _fit_trapezoid(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period: float,
    epoch: float,
    depth: float,
    duration: float,
    cadence_days: float,
    period_half_width: float,
) -> dict[str, float] | None:
    initial = np.array([period, epoch, depth, duration, 0.25, 1.0])
    lower = np.array(
        [
            period - period_half_width,
            epoch - max(duration, 2 * cadence_days),
            0.0,
            max(3 * cadence_days, 0.02),
            0.03,
            0.98,
        ]
    )
    upper = np.array(
        [
            period + period_half_width,
            epoch + max(duration, 2 * cadence_days),
            min(0.5, max(0.05, 4 * depth)),
            min(0.35 * period, max(0.3, 2.5 * duration)),
            0.49,
            1.02,
        ]
    )
    if np.any(initial <= lower) or np.any(initial >= upper):
        return None

    def residuals(values: np.ndarray) -> np.ndarray:
        model = _trapezoid_model(
            time,
            period=float(values[0]),
            epoch=float(values[1]),
            depth=float(values[2]),
            duration=float(values[3]),
            ingress_fraction=float(values[4]),
            baseline=float(values[5]),
        )
        return (flux - model) / flux_err

    try:
        fit = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=350,
        )
    except (ValueError, FloatingPointError):
        return None
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return None
    values = fit.x
    model = _trapezoid_model(
        time,
        period=float(values[0]),
        epoch=float(values[1]),
        depth=float(values[2]),
        duration=float(values[3]),
        ingress_fraction=float(values[4]),
        baseline=float(values[5]),
    )
    in_transit = np.abs(_phase_time(time, float(values[0]), float(values[1]))) <= values[3] / 2
    residual_scatter = robust_sigma(flux - model)
    depth_error = residual_scatter / np.sqrt(max(int(np.count_nonzero(in_transit)), 1))
    return {
        "period": float(values[0]),
        "epoch": float(values[1]),
        "depth": float(values[2]),
        "duration": float(values[3]),
        "depth_error": float(max(depth_error, np.finfo(float).eps)),
    }


def _trapezoid_model(
    time: np.ndarray,
    *,
    period: float,
    epoch: float,
    depth: float,
    duration: float,
    ingress_fraction: float,
    baseline: float,
) -> np.ndarray:
    distance = np.abs(_phase_time(time, period, epoch))
    half_duration = duration / 2.0
    ingress = max(np.finfo(float).eps, ingress_fraction * duration)
    flat_half_width = max(0.0, half_duration - ingress)
    shape = np.where(
        distance <= flat_half_width,
        1.0,
        np.where(distance < half_duration, (half_duration - distance) / ingress, 0.0),
    )
    return baseline - depth * shape


def _observed_event_count(time: np.ndarray, period: float, epoch: float, duration: float) -> int:
    cycles = np.floor((time - epoch) / period + 0.5).astype(np.int64)
    in_transit = np.abs(_phase_time(time, period, epoch)) <= duration / 2.0
    return int(
        sum(np.count_nonzero(in_transit & (cycles == cycle)) >= 3 for cycle in np.unique(cycles))
    )


def _phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return ((time - epoch + 0.5 * period) % period) / period - 0.5


def _phase_time(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return _phase(time, period, epoch) * period


def _bin_series(
    phase: np.ndarray, flux: np.ndarray, *, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)
    values = np.full(bins, np.nan)
    errors = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=np.int64)
    for index in range(bins):
        selected = flux[indices == index]
        selected = selected[np.isfinite(selected)]
        counts[index] = selected.size
        if selected.size:
            values[index] = np.median(selected)
            errors[index] = robust_sigma(selected) / np.sqrt(selected.size)
    return centers, values, errors, counts


def _failure(
    experiment_type: ExperimentType,
    reason: str,
    reason_code: str,
    *,
    alternatives: list[ExperimentType] | None = None,
    parameters: dict[str, int | float | bool | str] | None = None,
) -> ScientificFailure:
    return ScientificFailure(
        status=ScientificStatus.PRECONDITION_FAILED,
        experiment_type=experiment_type,
        tool_name=f"exoswarm.science.{experiment_type.value}",
        tool_version=TOOL_VERSION,
        parameters=parameters or {},
        reason=reason,
        reason_code=reason_code,
        suggested_alternatives=alternatives or [],
        interpretation_code=InterpretationCode.PRECONDITION_NOT_MET,
    )
