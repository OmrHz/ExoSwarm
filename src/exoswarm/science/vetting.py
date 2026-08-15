"""Deterministic mandatory and adaptive photometric vetting diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from astropy.timeseries import BoxLeastSquares

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
    product_path,
    robust_sigma,
    setup_science_axes,
    tess_provenance,
)
from .data import load_spoc_light_curve
from .transit import _fit_trapezoid


def assess_signal_quality(
    manifest: ScienceManifest,
    candidate: Candidate,
    *,
    minimum_snr: float = 7.0,
    minimum_events: int = 2,
    maximum_fractional_duration: float = 0.2,
) -> ScientificResult:
    duration_ratio = candidate.duration_hours / 24.0 / candidate.period_days
    passes = (
        candidate.signal_to_noise >= minimum_snr
        and candidate.observed_events >= minimum_events
        and 0 < candidate.transit_depth_ppm < 500_000
        and 0 < duration_ratio <= maximum_fractional_duration
    )
    failed_checks = int(candidate.signal_to_noise < minimum_snr)
    failed_checks += int(candidate.observed_events < minimum_events)
    failed_checks += int(not 0 < candidate.transit_depth_ppm < 500_000)
    failed_checks += int(not 0 < duration_ratio <= maximum_fractional_duration)
    flags: list[QualityFlag] = []
    if not passes:
        flags.append(
            QualityFlag(
                code="MINIMUM_SIGNAL_QUALITY_FAILED",
                severity=QualitySeverity.WARNING,
                detail="One or more declared signal-quality thresholds failed.",
            )
        )
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.SIGNAL_QUALITY,
        tool_name="exoswarm.science.assess_signal_quality",
        tool_version=TOOL_VERSION,
        parameters={
            "minimum_snr": minimum_snr,
            "minimum_events": minimum_events,
            "maximum_fractional_duration": maximum_fractional_duration,
            "maximum_depth_ppm": 500_000,
        },
        numerical_results={
            "signal_to_noise": float(candidate.signal_to_noise),
            "observed_events": int(candidate.observed_events),
            "transit_depth_ppm": float(candidate.transit_depth_ppm),
            "fractional_duration": float(duration_ratio),
            "failed_checks": failed_checks,
            "passes_minimum_quality": int(passes),
        },
        result_units={
            "signal_to_noise": "dimensionless",
            "observed_events": "count",
            "transit_depth_ppm": "ppm",
            "fractional_duration": "fraction",
            "failed_checks": "count",
            "passes_minimum_quality": "boolean integer",
        },
        uncertainties={},
        quality_flags=flags,
        interpretation_code=InterpretationCode.PASS if passes else InterpretationCode.FAIL,
        limitations=[
            "Thresholds are declared MVP screening rules, not a calibrated planet probability.",
            "Passing minimum signal quality does not distinguish planets from eclipsing binaries.",
        ],
        input_artifacts=[],
        output_artifacts=[],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def odd_even_test(
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
) -> ScientificResult | ScientificFailure:
    loaded = _load_cleaned(store, ExperimentType.ODD_EVEN)
    if isinstance(loaded, ScientificFailure):
        return loaded
    time, flux, flux_err, input_artifact = loaded
    cycles = np.floor((time - candidate.epoch_btjd) / candidate.period_days + 0.5).astype(int)
    duration_days = candidate.duration_hours / 24.0
    phase_time = _phase_time(time, candidate.period_days, candidate.epoch_btjd)
    in_transit = np.abs(phase_time) <= duration_days / 2.0
    usable_cycles = [
        int(cycle)
        for cycle in np.unique(cycles)
        if np.count_nonzero(in_transit & (cycles == cycle)) >= 3
    ]
    odd_cycles = [cycle for cycle in usable_cycles if cycle % 2]
    even_cycles = [cycle for cycle in usable_cycles if not cycle % 2]
    if len(usable_cycles) < 4 or len(odd_cycles) < 2 or len(even_cycles) < 2:
        return _failure(
            ExperimentType.ODD_EVEN,
            (
                "odd/even requires >=4 usable transits with >=2 of each parity; "
                f"found {len(usable_cycles)} ({len(odd_cycles)} odd, {len(even_cycles)} even)"
            ),
            "TOO_FEW_TRANSITS_FOR_ODD_EVEN",
            alternatives=[ExperimentType.SECONDARY_ECLIPSE, ExperimentType.HARMONIC_TEST],
        )

    bls = BoxLeastSquares(time, flux, dy=flux_err)
    stats = bls.compute_stats(
        candidate.period_days,
        duration_days,
        candidate.epoch_btjd,
    )
    odd_depth, odd_error = (float(value) for value in stats["depth_odd"])
    even_depth, even_error = (float(value) for value in stats["depth_even"])
    difference = odd_depth - even_depth
    difference_error = float(np.hypot(odd_error, even_error))
    difference_sigma = float(abs(difference) / difference_error) if difference_error > 0 else 0.0
    inconsistent = bool(difference_sigma >= 3.0)

    event_numbers: list[int] = []
    event_depths: list[float] = []
    event_errors: list[float] = []
    parities: list[int] = []
    for cycle in usable_cycles:
        depth, uncertainty, count = _event_depth(
            time,
            flux,
            flux_err,
            center=candidate.epoch_btjd + cycle * candidate.period_days,
            duration_days=duration_days,
        )
        if count >= 3 and np.isfinite(depth) and np.isfinite(uncertainty):
            event_numbers.append(cycle)
            event_depths.append(depth * 1e6)
            event_errors.append(uncertainty * 1e6)
            parities.append(abs(cycle) % 2)
    data_artifact = store.save_npz(
        "odd_even",
        "odd_even_data",
        event_number=np.asarray(event_numbers, dtype=np.int64),
        event_depth_ppm=np.asarray(event_depths, dtype=float),
        event_depth_uncertainty_ppm=np.asarray(event_errors, dtype=float),
        event_parity=np.asarray(parities, dtype=np.int8),
    )
    figure, axis = plt.subplots(figsize=(8.6, 4.0))
    events = np.asarray(event_numbers)
    depths = np.asarray(event_depths)
    errors = np.asarray(event_errors)
    parity = np.asarray(parities)
    for value, label, color in [(1, "odd", "#ff8ca1"), (0, "even", "#6de4ff")]:
        selected = parity == value
        axis.errorbar(
            events[selected],
            depths[selected],
            yerr=errors[selected],
            fmt="o",
            markersize=4,
            linewidth=0.7,
            color=color,
            label=label,
        )
    axis.axhline(odd_depth * 1e6, color="#ff8ca1", linestyle="--", linewidth=0.8)
    axis.axhline(even_depth * 1e6, color="#6de4ff", linestyle="--", linewidth=0.8)
    axis.set_title(f"Odd/even event depths — difference {difference_sigma:.2f} sigma")
    axis.legend(frameon=False, labelcolor="#d7e4f5")
    setup_science_axes(axis, xlabel="Event number", ylabel="Depth (ppm)")
    plot_artifact = store.save_figure("odd_even", "odd_even_plot", figure)
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.ODD_EVEN,
        tool_name="exoswarm.science.odd_even_test",
        tool_version=TOOL_VERSION,
        parameters={"consistency_threshold_sigma": 3.0, "minimum_usable_transits": 4},
        numerical_results={
            "odd_depth_ppm": odd_depth * 1e6,
            "even_depth_ppm": even_depth * 1e6,
            "odd_depth_uncertainty_ppm": odd_error * 1e6,
            "even_depth_uncertainty_ppm": even_error * 1e6,
            "depth_difference_ppm": difference * 1e6,
            "depth_difference_sigma": difference_sigma,
            "usable_transits": len(usable_cycles),
            "odd_transits": len(odd_cycles),
            "even_transits": len(even_cycles),
        },
        result_units={
            "odd_depth_ppm": "ppm",
            "even_depth_ppm": "ppm",
            "odd_depth_uncertainty_ppm": "ppm",
            "even_depth_uncertainty_ppm": "ppm",
            "depth_difference_ppm": "ppm",
            "depth_difference_sigma": "sigma",
            "usable_transits": "count",
            "odd_transits": "count",
            "even_transits": "count",
        },
        uncertainties={
            "odd_depth_ppm": MeasurementUncertainty(
                value=odd_error * 1e6,
                unit="ppm",
                method="Astropy BLS inverse-variance odd-event depth uncertainty",
            ),
            "even_depth_ppm": MeasurementUncertainty(
                value=even_error * 1e6,
                unit="ppm",
                method="Astropy BLS inverse-variance even-event depth uncertainty",
            ),
            "depth_difference_ppm": MeasurementUncertainty(
                value=difference_error * 1e6,
                unit="ppm",
                method="quadrature propagation of odd/even depth uncertainties",
            ),
        },
        quality_flags=[],
        interpretation_code=(
            InterpretationCode.INCONSISTENT if inconsistent else InterpretationCode.CONSISTENT
        ),
        limitations=[
            "Odd/even consistency cannot exclude an equal-depth eclipsing binary.",
            "Formal uncertainties use per-cadence SPOC errors and may underrepresent red noise.",
            "A significant mismatch can indicate a half-period alias and requires harmonic testing.",
        ],
        input_artifacts=[input_artifact],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def secondary_eclipse_test(
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
    *,
    phase_half_threshold_sigma: float = 5.0,
    scan_threshold_sigma: float = 10.0,
) -> ScientificResult | ScientificFailure:
    loaded = _load_cleaned(store, ExperimentType.SECONDARY_ECLIPSE)
    if isinstance(loaded, ScientificFailure):
        return loaded
    time, flux, flux_err, input_artifact = loaded
    duration_days = candidate.duration_hours / 24.0
    phase = _phase(time, candidate.period_days, candidate.epoch_btjd)
    half_depth, half_error, half_count = _phase_box_depth(
        phase,
        flux,
        flux_err,
        center_phase=0.5,
        half_width=duration_days / (2 * candidate.period_days),
    )
    if half_count < 3 or not np.isfinite(half_depth) or not np.isfinite(half_error):
        return _failure(
            ExperimentType.SECONDARY_ECLIPSE,
            "insufficient phase-0.5 coverage for a secondary-eclipse measurement",
            "SECONDARY_PHASE_UNCOVERED",
            alternatives=[ExperimentType.HARMONIC_TEST],
        )
    phase_centers = np.linspace(0.2, 0.8, 121)
    scan_depths = np.full_like(phase_centers, np.nan)
    scan_errors = np.full_like(phase_centers, np.nan)
    scan_significance = np.full_like(phase_centers, np.nan)
    half_width = duration_days / (2 * candidate.period_days)
    for index, center in enumerate(phase_centers):
        depth, error, count = _phase_box_depth(
            phase,
            flux,
            flux_err,
            center_phase=float(center),
            half_width=half_width,
        )
        if count >= 3 and error > 0:
            scan_depths[index] = depth
            scan_errors[index] = error
            scan_significance[index] = depth / error
    best_index = int(np.nanargmax(scan_significance))
    best_depth = float(scan_depths[best_index])
    best_error = float(scan_errors[best_index])
    best_sigma = float(scan_significance[best_index])
    best_phase = float(phase_centers[best_index] % 1.0)
    half_sigma = float(half_depth / half_error) if half_error > 0 else 0.0
    significant = bool(
        (half_depth > 0 and half_sigma >= phase_half_threshold_sigma)
        or (best_depth > 0 and best_sigma >= scan_threshold_sigma)
    )
    primary_depth = candidate.transit_depth_ppm / 1e6
    ratio = float(max(best_depth, 0) / primary_depth) if primary_depth > 0 else 0.0
    bin_phase, bin_flux, bin_error, _ = _bin_phase(phase, flux, bins=200)
    data_artifact = store.save_npz(
        "secondary_eclipse",
        "secondary_eclipse_data",
        phase=phase,
        flux=flux,
        bin_phase=bin_phase,
        bin_flux=bin_flux,
        bin_flux_err=bin_error,
        scan_phase=phase_centers,
        scan_depth_ppm=scan_depths * 1e6,
        scan_significance=scan_significance,
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=False)
    axes[0].scatter(phase, flux, s=1.0, color="#426384", alpha=0.15, rasterized=True)
    good = np.isfinite(bin_flux)
    axes[0].plot(bin_phase[good], bin_flux[good], "o", ms=2.4, color="#6de4ff")
    axes[0].axvline(-0.5, color="#ffca62", linewidth=0.9, linestyle="--")
    axes[0].axvline(
        best_phase if best_phase <= 0.5 else best_phase - 1, color="#ff7c95", linewidth=0.9
    )
    axes[0].set_title("Full-orbit phase fold and secondary-event search")
    setup_science_axes(axes[0], xlabel="Orbital phase", ylabel="Detrended flux")
    axes[1].plot(phase_centers, scan_significance, color="#ff8ca1", linewidth=1.0)
    axes[1].axhline(scan_threshold_sigma, color="#ffca62", linestyle="--", linewidth=0.8)
    setup_science_axes(axes[1], xlabel="Trial secondary phase", ylabel="Formal depth significance")
    plot_artifact = store.save_figure("secondary_eclipse", "secondary_eclipse_plot", figure)
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.SECONDARY_ECLIPSE,
        tool_name="exoswarm.science.secondary_eclipse_test",
        tool_version=TOOL_VERSION,
        parameters={
            "phase_half_threshold_sigma": phase_half_threshold_sigma,
            "scan_threshold_sigma": scan_threshold_sigma,
            "scan_phase_min": 0.2,
            "scan_phase_max": 0.8,
            "scan_trials": int(phase_centers.size),
            "box_duration_hours": float(candidate.duration_hours),
        },
        numerical_results={
            "phase_half_depth_ppm": half_depth * 1e6,
            "phase_half_depth_uncertainty_ppm": half_error * 1e6,
            "phase_half_significance_sigma": half_sigma,
            "secondary_depth_ppm": best_depth * 1e6,
            "secondary_depth_uncertainty_ppm": best_error * 1e6,
            "secondary_significance_sigma": best_sigma,
            "secondary_phase": best_phase,
            "secondary_to_primary_depth_ratio": ratio,
        },
        result_units={
            "phase_half_depth_ppm": "ppm",
            "phase_half_depth_uncertainty_ppm": "ppm",
            "phase_half_significance_sigma": "sigma",
            "secondary_depth_ppm": "ppm",
            "secondary_depth_uncertainty_ppm": "ppm",
            "secondary_significance_sigma": "sigma",
            "secondary_phase": "phase",
            "secondary_to_primary_depth_ratio": "ratio",
        },
        uncertainties={
            "phase_half_depth_ppm": MeasurementUncertainty(
                value=half_error * 1e6,
                unit="ppm",
                method="inverse-variance box depth at phase 0.5",
            ),
            "secondary_depth_ppm": MeasurementUncertainty(
                value=best_error * 1e6,
                unit="ppm",
                method="inverse-variance box depth at maximum scanned phase",
            ),
        },
        quality_flags=[
            QualityFlag(
                code="SECONDARY_PHASE_SCAN",
                severity=QualitySeverity.INFO,
                detail="The scan threshold is conservative but is not a calibrated trials-corrected false-alarm probability.",
            )
        ],
        interpretation_code=(
            InterpretationCode.SIGNIFICANT if significant else InterpretationCode.NOT_SIGNIFICANT
        ),
        limitations=[
            "A non-detection does not exclude a cool, grazing, eccentric or diluted eclipsing binary.",
            "A planetary thermal/reflected-light eclipse can also produce a real secondary event.",
            "If the input ephemeris is P/2, primary and secondary events overlap at phase zero; odd/even and 2P testing must resolve that alias.",
        ],
        input_artifacts=[input_artifact],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def contamination_screen(
    data_root: str | Path,
    manifest: ScienceManifest,
    *,
    aperture_context_radius_arcsec: float = 42.0,
) -> ScientificResult | ScientificFailure:
    try:
        curve = load_spoc_light_curve(product_path(data_root, manifest, "light_curve"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _failure(
            ExperimentType.CONTAMINATION_SCREEN,
            str(exc),
            "LIGHT_CURVE_UNREADABLE",
            alternatives=[ExperimentType.LOAD_CACHED_DATA],
        )
    neighbor = manifest.neighbor_context
    separation = neighbor.nearest_neighbor_separation_arcsec if neighbor else float("nan")
    delta_tmag = neighbor.nearest_neighbor_delta_tmag if neighbor else float("nan")
    crowdsap = curve.crowdsap if curve.crowdsap is not None else float("nan")
    flfrcsap = curve.flfrcsap if curve.flfrcsap is not None else float("nan")
    contamination = float(max(0.0, 1.0 - crowdsap)) if np.isfinite(crowdsap) else float("nan")
    neighbor_near = bool(np.isfinite(separation) and separation <= aperture_context_radius_arcsec)
    material_crowding = bool(np.isfinite(contamination) and contamination >= 0.05)
    concern = neighbor_near or material_crowding
    flags: list[QualityFlag] = []
    if concern:
        flags.append(
            QualityFlag(
                code="SPATIAL_LOCALIZATION_RECOMMENDED",
                severity=QualitySeverity.WARNING,
                detail="Catalog geometry or SPOC CROWDSAP warrants a target-pixel localization test.",
            )
        )
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.CONTAMINATION_SCREEN,
        tool_name="exoswarm.science.contamination_screen",
        tool_version=TOOL_VERSION,
        parameters={
            "aperture_context_radius_arcsec": aperture_context_radius_arcsec,
            "material_contamination_fraction": 0.05,
        },
        numerical_results={
            "crowdsap": crowdsap if np.isfinite(crowdsap) else 0.0,
            "estimated_contaminating_flux_fraction": (
                contamination if np.isfinite(contamination) else 0.0
            ),
            "flfrcsap": flfrcsap if np.isfinite(flfrcsap) else 0.0,
            "nearest_neighbor_separation_arcsec": separation if np.isfinite(separation) else 0.0,
            "nearest_neighbor_delta_tmag": delta_tmag if np.isfinite(delta_tmag) else 0.0,
            "spatial_localization_recommended": int(concern),
        },
        result_units={
            "crowdsap": "fraction",
            "estimated_contaminating_flux_fraction": "fraction",
            "flfrcsap": "fraction",
            "nearest_neighbor_separation_arcsec": "arcsec",
            "nearest_neighbor_delta_tmag": "mag",
            "spatial_localization_recommended": "boolean integer",
        },
        uncertainties={},
        quality_flags=flags,
        interpretation_code=(
            InterpretationCode.NEIGHBOR_DETECTED if concern else InterpretationCode.NO_NEARBY_SOURCE
        ),
        limitations=[
            "CROWDSAP estimates aperture crowding; it does not identify which source dims.",
            "The cached TIC neighbor query is contextual evidence, not a ground-truth planet lookup.",
            "One TESS pixel is approximately 21 arcsec, so sub-pixel blends require pixel-level testing.",
        ],
        input_artifacts=[],
        output_artifacts=[],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def harmonic_alias_test(
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
    *,
    double_preference_snr_ratio: float = 1.01,
    odd_even_threshold_sigma: float = 3.0,
) -> ScientificResult | ScientificFailure:
    loaded = _load_cleaned(store, ExperimentType.HARMONIC_TEST)
    if isinstance(loaded, ScientificFailure):
        return loaded
    time, flux, flux_err, input_artifact = loaded
    duration_days = candidate.duration_hours / 24.0
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    baseline = float(time[-1] - time[0])
    factors = np.array([0.5, 1.0, 2.0])
    peaks = [
        _local_harmonic_peak(
            bls,
            center_period=candidate.period_days * float(factor),
            baseline=baseline,
            duration_days=duration_days,
        )
        for factor in factors
    ]
    scores = np.asarray([peak["snr"] for peak in peaks])
    log_likelihoods = np.asarray([peak["log_likelihood"] for peak in peaks])

    stats = bls.compute_stats(candidate.period_days, duration_days, candidate.epoch_btjd)
    odd_depth, odd_error = (float(value) for value in stats["depth_odd"])
    even_depth, even_error = (float(value) for value in stats["depth_even"])
    difference_error = float(np.hypot(odd_error, even_error))
    odd_even_sigma = (
        float(abs(odd_depth - even_depth) / difference_error) if difference_error > 0 else 0.0
    )
    preferred_index = 1
    interpretation = InterpretationCode.PREFERRED_NOMINAL_PERIOD
    if (
        scores[2] >= double_preference_snr_ratio * scores[1]
        and odd_even_sigma >= odd_even_threshold_sigma
    ):
        preferred_index = 2
        interpretation = InterpretationCode.PREFERRED_DOUBLE_PERIOD
    elif scores[0] > 1.05 * scores[1]:
        preferred_index = 0
        interpretation = InterpretationCode.PREFERRED_HALF_PERIOD
    preferred = peaks[preferred_index]
    preferred_factor = float(factors[preferred_index])
    short_differences = np.diff(time)
    short_differences = short_differences[(short_differences > 0) & (short_differences < 0.1)]
    cadence_days = float(np.median(short_differences))
    if preferred_index != 1:
        trapezoid = _fit_trapezoid(
            time,
            flux,
            flux_err,
            period=preferred["period"],
            epoch=preferred["epoch"],
            depth=max(preferred["depth"], 1e-6),
            duration=preferred["duration"],
            cadence_days=cadence_days,
            period_half_width=max(0.002 * preferred["period"], 5e-4),
        )
        if trapezoid is not None:
            preferred = preferred | {
                "period": trapezoid["period"],
                "epoch": trapezoid["epoch"],
                "duration": trapezoid["duration"],
                "depth": trapezoid["depth"],
                "depth_error": trapezoid["depth_error"],
            }
    preferred_phase_time = _phase_time(time, preferred["period"], preferred["epoch"])
    preferred_cycles = np.floor((time - preferred["epoch"]) / preferred["period"] + 0.5).astype(int)
    preferred_in_transit = np.abs(preferred_phase_time) <= preferred["duration"] / 2.0
    preferred_observed_events = int(
        sum(
            np.count_nonzero(preferred_in_transit & (preferred_cycles == cycle)) >= 3
            for cycle in np.unique(preferred_cycles)
        )
    )
    local_frequency_half_width = min(0.02 / preferred["period"], 3.0 / baseline)
    frequency_resolution = 2.0 * local_frequency_half_width / 2399.0
    period_tolerance = max(
        frequency_resolution * preferred["period"] ** 2,
        cadence_days / max(preferred_observed_events - 1, 1),
    )
    epoch_tolerance = cadence_days / 2.0
    duration_tolerance_hours = max(
        cadence_days * 24.0,
        0.05 * preferred["duration"] * 24.0,
    )

    primary_depth, _, _ = _phase_box_depth(
        _phase(time, preferred["period"], preferred["epoch"]),
        flux,
        flux_err,
        center_phase=0.0,
        half_width=preferred["duration"] / (2 * preferred["period"]),
    )
    secondary_depth, secondary_error, _ = _phase_box_depth(
        _phase(time, preferred["period"], preferred["epoch"]),
        flux,
        flux_err,
        center_phase=0.5,
        half_width=preferred["duration"] / (2 * preferred["period"]),
    )
    data_artifact = store.save_npz(
        "harmonic_test",
        "harmonic_test_data",
        tested_period_days=np.asarray([peak["period"] for peak in peaks]),
        bls_snr=scores,
        delta_log_likelihood=log_likelihoods - log_likelihoods[1],
        factor=factors,
        refined_epoch_btjd=np.asarray([peak["epoch"] for peak in peaks]),
        refined_duration_hours=np.asarray([peak["duration"] * 24 for peak in peaks]),
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#68a8d8", "#6de4ff", "#ffca62"]
    axis.bar(["P/2", "P", "2P"], scores, color=colors, width=0.6)
    axis.set_title(f"Harmonic family — preferred factor {preferred_factor:g}")
    setup_science_axes(axis, xlabel="Tested ephemeris", ylabel="BLS depth S/N")
    plot_artifact = store.save_figure("harmonic_test", "harmonic_test_plot", figure)
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.HARMONIC_TEST,
        tool_name="exoswarm.science.harmonic_alias_test",
        tool_version=TOOL_VERSION,
        parameters={
            "tested_factors": "0.5,1,2",
            "double_preference_snr_ratio": double_preference_snr_ratio,
            "odd_even_threshold_sigma": odd_even_threshold_sigma,
            "selection_rule": "2P requires both SNR gain and odd/even inconsistency",
        },
        numerical_results={
            "tested_half_period_days": float(peaks[0]["period"]),
            "tested_nominal_period_days": float(peaks[1]["period"]),
            "tested_double_period_days": float(peaks[2]["period"]),
            "score_half_period_snr": float(scores[0]),
            "score_nominal_period_snr": float(scores[1]),
            "score_double_period_snr": float(scores[2]),
            "odd_even_difference_sigma_at_nominal": odd_even_sigma,
            "preferred_factor": preferred_factor,
            "preferred_period_days": float(preferred["period"]),
            "preferred_epoch_btjd": float(preferred["epoch"]),
            "preferred_duration_hours": float(preferred["duration"] * 24.0),
            "preferred_signal_to_noise": float(preferred["snr"]),
            "preferred_observed_events": preferred_observed_events,
            "preferred_primary_depth_ppm": float(preferred["depth"] * 1e6),
            "preferred_primary_box_depth_ppm": float(primary_depth * 1e6),
            "preferred_secondary_depth_ppm": float(secondary_depth * 1e6),
            "preferred_secondary_significance_sigma": float(
                secondary_depth / secondary_error if secondary_error > 0 else 0.0
            ),
        },
        result_units={
            "tested_half_period_days": "d",
            "tested_nominal_period_days": "d",
            "tested_double_period_days": "d",
            "score_half_period_snr": "dimensionless",
            "score_nominal_period_snr": "dimensionless",
            "score_double_period_snr": "dimensionless",
            "odd_even_difference_sigma_at_nominal": "sigma",
            "preferred_factor": "factor",
            "preferred_period_days": "d",
            "preferred_epoch_btjd": "BTJD",
            "preferred_duration_hours": "h",
            "preferred_signal_to_noise": "dimensionless",
            "preferred_observed_events": "count",
            "preferred_primary_depth_ppm": "ppm",
            "preferred_primary_box_depth_ppm": "ppm",
            "preferred_secondary_depth_ppm": "ppm",
            "preferred_secondary_significance_sigma": "sigma",
        },
        uncertainties={
            "preferred_period_days": MeasurementUncertainty(
                value=float(period_tolerance),
                unit="d",
                method="local harmonic BLS grid plus cadence/event-baseline resolution",
                kind="tolerance",
            ),
            "preferred_epoch_btjd": MeasurementUncertainty(
                value=float(epoch_tolerance),
                unit="d",
                method="half-cadence timing resolution at the revised ephemeris",
                kind="resolution",
            ),
            "preferred_duration_hours": MeasurementUncertainty(
                value=float(duration_tolerance_hours),
                unit="h",
                method="maximum of one cadence and five percent of the fitted duration",
                kind="tolerance",
            ),
            "preferred_primary_depth_ppm": MeasurementUncertainty(
                value=float(preferred["depth_error"] * 1e6),
                unit="ppm",
                method=(
                    "robust trapezoid residual standard error"
                    if preferred_index != 1
                    else "Astropy BLS formal depth uncertainty"
                ),
            ),
            "preferred_secondary_depth_ppm": MeasurementUncertainty(
                value=float(secondary_error * 1e6),
                unit="ppm",
                method="inverse-variance phase-box depth",
            ),
        },
        quality_flags=(
            [
                QualityFlag(
                    code="ORBITAL_PERIOD_REVISED_FROM_ALIAS",
                    severity=QualitySeverity.WARNING,
                    detail="The preferred orbital ephemeris differs from the initial event cadence.",
                )
            ]
            if preferred_index != 1
            else []
        ),
        interpretation_code=interpretation,
        limitations=[
            "Harmonic scores compare simple box models and do not constitute a physical binary fit.",
            "The declared preference rule is heuristic and auditable, not a calibrated model probability.",
            "A 2P preference is required to show both improved BLS score and odd/even inconsistency.",
        ],
        input_artifacts=[input_artifact],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def _local_harmonic_peak(
    bls: BoxLeastSquares,
    *,
    center_period: float,
    baseline: float,
    duration_days: float,
) -> dict[str, float]:
    center_frequency = 1.0 / center_period
    half_width = min(0.02 * center_frequency, 3.0 / baseline)
    frequencies = np.linspace(center_frequency - half_width, center_frequency + half_width, 2400)
    periods = 1.0 / frequencies
    durations = np.linspace(
        max(0.025, 0.65 * duration_days),
        min(0.3, 1.45 * duration_days),
        17,
    )
    snr_result = bls.power(periods, durations, objective="snr", method="fast", oversample=20)
    index = int(np.nanargmax(snr_result.power))
    period = float(snr_result.period[index])
    duration = float(snr_result.duration[index])
    epoch = float(snr_result.transit_time[index])
    likelihood_result = bls.power(
        np.asarray([period]),
        np.asarray([duration]),
        objective="likelihood",
        method="fast",
        oversample=20,
    )
    return {
        "period": period,
        "duration": duration,
        "epoch": epoch,
        "depth": float(snr_result.depth[index]),
        "depth_error": float(snr_result.depth_err[index]),
        "snr": float(snr_result.depth_snr[index]),
        "log_likelihood": float(likelihood_result.power[0]),
    }


def _load_cleaned(
    store: ArtifactStore, experiment_type: ExperimentType
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object] | ScientificFailure:
    path = store.path("cleaned_light_curve")
    if not path.is_file():
        return _failure(
            experiment_type,
            "detrended light curve is required",
            "CLEANED_CURVE_MISSING",
            alternatives=[ExperimentType.DETRENDING],
        )
    arrays = load_npz(path)
    artifact = store.existing("cleaned_light_curve", "cleaned_light_curve_data")
    return (
        arrays["time_btjd"].astype(float),
        arrays["flux"].astype(float),
        arrays["flux_err"].astype(float),
        artifact,
    )


def _event_depth(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    center: float,
    duration_days: float,
) -> tuple[float, float, int]:
    distance = np.abs(time - center)
    inside = distance <= duration_days / 2.0
    outside = (distance >= duration_days) & (distance <= 2.5 * duration_days)
    if np.count_nonzero(inside) < 3 or np.count_nonzero(outside) < 6:
        return float("nan"), float("nan"), int(np.count_nonzero(inside))
    in_mean, in_error = _weighted_mean(flux[inside], flux_err[inside])
    out_mean, out_error = _weighted_mean(flux[outside], flux_err[outside])
    return out_mean - in_mean, float(np.hypot(in_error, out_error)), int(np.count_nonzero(inside))


def _phase_box_depth(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    center_phase: float,
    half_width: float,
) -> tuple[float, float, int]:
    center = ((center_phase + 0.5) % 1.0) - 0.5
    distance = np.abs(((phase - center + 0.5) % 1.0) - 0.5)
    primary_distance = np.abs(phase)
    inside = distance <= half_width
    outside = (distance >= 2.0 * half_width) & (distance <= 5.0 * half_width)
    if abs(center) > 2 * half_width:
        outside &= primary_distance > 2.0 * half_width
    if np.count_nonzero(inside) < 3 or np.count_nonzero(outside) < 6:
        return float("nan"), float("nan"), int(np.count_nonzero(inside))
    in_mean, in_error = _weighted_mean(flux[inside], flux_err[inside])
    out_mean, out_error = _weighted_mean(flux[outside], flux_err[outside])
    return out_mean - in_mean, float(np.hypot(in_error, out_error)), int(np.count_nonzero(inside))


def _weighted_mean(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        return float("nan"), float("nan")
    weights = 1.0 / np.square(errors[valid])
    mean = float(np.sum(weights * values[valid]) / np.sum(weights))
    formal = float(np.sqrt(1.0 / np.sum(weights)))
    empirical = robust_sigma(values[valid]) / np.sqrt(np.count_nonzero(valid))
    return mean, max(formal, empirical if np.isfinite(empirical) else 0.0)


def _phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return ((time - epoch + 0.5 * period) % period) / period - 0.5


def _phase_time(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return _phase(time, period, epoch) * period


def _bin_phase(
    phase: np.ndarray, flux: np.ndarray, *, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    index = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)
    values = np.full(bins, np.nan)
    errors = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for item in range(bins):
        selected = flux[index == item]
        selected = selected[np.isfinite(selected)]
        counts[item] = selected.size
        if selected.size:
            values[item] = np.median(selected)
            errors[item] = robust_sigma(selected) / np.sqrt(selected.size)
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
