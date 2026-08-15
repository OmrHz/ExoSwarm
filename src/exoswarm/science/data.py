"""Cached SPOC light-curve loading, quality inspection, cleaning and detrending."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from exoswarm.domain.models import (
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
    LightCurveData,
    ScienceManifest,
    load_npz,
    product_path,
    robust_sigma,
    setup_science_axes,
    source_artifact,
    tess_provenance,
    verify_cached_product,
)


def _failure(
    experiment_type: ExperimentType,
    reason: str,
    reason_code: str,
    *,
    alternatives: list[ExperimentType] | None = None,
    parameters: dict[str, int | float | str | bool] | None = None,
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


def load_spoc_light_curve(path: str | Path) -> LightCurveData:
    """Load arrays from one locally cached SPOC light-curve FITS product.

    Only calibrated SPOC columns needed by deterministic tools are returned.  FITS
    identity headers never enter the returned object.
    """

    with fits.open(Path(path), memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError("SPOC light-curve FITS has no LIGHTCURVE extension")
        table = hdul[1].data
        required = {"TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"}
        missing = required - set(table.names or ())
        if missing:
            raise ValueError(f"SPOC light curve missing columns: {sorted(missing)}")
        primary = hdul[0].header
        extension = hdul[1].header
        time = np.asarray(table["TIME"], dtype=np.float64).copy()
        flux = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64).copy()
        flux_err = np.asarray(table["PDCSAP_FLUX_ERR"], dtype=np.float64).copy()
        quality = np.asarray(table["QUALITY"], dtype=np.int64).copy()
        finite_times = np.sort(time[np.isfinite(time)])
        differences = np.diff(finite_times)
        positive_differences = differences[differences > 0]
        cadence_seconds = (
            float(np.median(positive_differences) * 86400.0)
            if positive_differences.size
            else float("nan")
        )
        return LightCurveData(
            time_btjd=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            sector=int(primary.get("SECTOR", 0)),
            cadence_seconds=cadence_seconds,
            crowdsap=_optional_float(extension.get("CROWDSAP")),
            flfrcsap=_optional_float(extension.get("FLFRCSAP")),
        )


def load_cached_observation(
    data_root: str | Path, manifest: ScienceManifest, store: ArtifactStore
) -> ScientificResult | ScientificFailure:
    """Verify both source products and emit an identity-sanitized raw artifact."""

    try:
        light_curve_path = product_path(data_root, manifest, "light_curve")
        target_pixel_path = product_path(data_root, manifest, "target_pixel")
        verify_cached_product(light_curve_path, manifest.products["light_curve"])
        verify_cached_product(target_pixel_path, manifest.products["target_pixel"])
        curve = load_spoc_light_curve(light_curve_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _failure(
            ExperimentType.LOAD_CACHED_DATA,
            str(exc),
            "CACHED_PRODUCT_INVALID",
        )

    raw_npz = store.save_npz(
        "raw_light_curve",
        "raw_light_curve_data",
        time_btjd=curve.time_btjd,
        flux=curve.flux,
        flux_err=curve.flux_err,
        quality=curve.quality,
    )
    figure, axis = plt.subplots(figsize=(10, 3.2))
    finite = np.isfinite(curve.time_btjd) & np.isfinite(curve.flux)
    if finite.any():
        scale = np.nanmedian(curve.flux[finite])
        display_flux = curve.flux[finite] / scale if scale > 0 else curve.flux[finite]
        axis.scatter(
            curve.time_btjd[finite],
            display_flux,
            s=1.4,
            color="#6bdcff",
            alpha=0.65,
            rasterized=True,
        )
    axis.set_title("Cached SPOC observation (unfiltered PDCSAP flux)")
    setup_science_axes(axis, xlabel="Time (BTJD)", ylabel="Relative flux for display")
    raw_plot = store.save_figure("raw_light_curve", "raw_light_curve_plot", figure)
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.LOAD_CACHED_DATA,
        tool_name="exoswarm.science.load_cached_observation",
        tool_version=TOOL_VERSION,
        parameters={"verify_sha256": True, "flux_column": "PDCSAP_FLUX"},
        numerical_results={
            "total_cadences": int(curve.time_btjd.size),
            "sector": int(manifest.sector),
            "measured_cadence_seconds": float(curve.cadence_seconds),
            "source_bytes": int(
                manifest.products["light_curve"].size_bytes
                + manifest.products["target_pixel"].size_bytes
            ),
        },
        result_units={
            "total_cadences": "count",
            "sector": "sector",
            "measured_cadence_seconds": "s",
            "source_bytes": "byte",
        },
        uncertainties={
            "measured_cadence_seconds": MeasurementUncertainty(
                value=0.001,
                unit="s",
                method="reported numerical resolution",
                kind="resolution",
            )
        },
        quality_flags=[],
        interpretation_code=InterpretationCode.LOADED,
        limitations=[
            "PDCSAP_FLUX is a calibrated SPOC product, not untouched detector counts.",
            "Loading verifies file integrity but does not by itself establish scientific usability.",
        ],
        input_artifacts=[
            source_artifact(manifest, "light_curve"),
            source_artifact(manifest, "target_pixel"),
        ],
        output_artifacts=[raw_npz, raw_plot],
        provenance=[
            tess_provenance(manifest, "light_curve"),
            tess_provenance(manifest, "target_pixel"),
        ],
    )


def inspect_light_curve_quality(
    data_root: str | Path, manifest: ScienceManifest
) -> ScientificResult | ScientificFailure:
    try:
        path = product_path(data_root, manifest, "light_curve")
        curve = load_spoc_light_curve(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _failure(
            ExperimentType.QUALITY_INSPECTION,
            str(exc),
            "LIGHT_CURVE_UNREADABLE",
            alternatives=[ExperimentType.LOAD_CACHED_DATA],
        )

    finite = (
        np.isfinite(curve.time_btjd)
        & np.isfinite(curve.flux)
        & np.isfinite(curve.flux_err)
        & (curve.flux_err > 0)
    )
    quality_zero = curve.quality == 0
    usable = finite & quality_zero
    usable_times = np.sort(curve.time_btjd[usable])
    gaps = np.diff(usable_times) if usable_times.size > 1 else np.array([], dtype=float)
    baseline = float(usable_times[-1] - usable_times[0]) if usable_times.size > 1 else 0.0
    median_cadence_days = float(np.median(gaps[gaps < 0.1])) if np.any(gaps < 0.1) else float("nan")
    expected = baseline / median_cadence_days if median_cadence_days > 0 else 0.0
    duty_cycle = float(min(1.0, usable.sum() / expected)) if expected > 0 else 0.0
    usable_fraction = float(usable.mean()) if usable.size else 0.0
    large_gap_count = int(np.count_nonzero(gaps > 0.25))
    flags: list[QualityFlag] = []
    if np.count_nonzero(~quality_zero):
        flags.append(
            QualityFlag(
                code="TESS_QUALITY_CADENCES_PRESENT",
                severity=QualitySeverity.INFO,
                detail="Non-zero SPOC QUALITY cadences are excluded from analysis.",
            )
        )
    if large_gap_count:
        flags.append(
            QualityFlag(
                code="OBSERVATION_GAPS_PRESENT",
                severity=QualitySeverity.INFO,
                detail="Detrending is performed independently across gaps larger than 0.25 d.",
            )
        )
    acceptable = usable.sum() >= 1000 and usable_fraction >= 0.5 and baseline >= 10.0
    if not acceptable:
        flags.append(
            QualityFlag(
                code="INSUFFICIENT_USABLE_BASELINE",
                severity=QualitySeverity.ERROR,
                detail="Usable cadence count/fraction or observing baseline is below the MVP threshold.",
            )
        )

    results = {
        "total_cadences": int(curve.time_btjd.size),
        "usable_cadences": int(usable.sum()),
        "quality_rejected_cadences": int(np.count_nonzero(finite & ~quality_zero)),
        "nonfinite_or_invalid_cadences": int(np.count_nonzero(~finite)),
        "usable_fraction": usable_fraction,
        "baseline_days": baseline,
        "median_cadence_minutes": float(median_cadence_days * 1440.0),
        "large_gap_count": large_gap_count,
        "approximate_duty_cycle": duty_cycle,
    }
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.QUALITY_INSPECTION,
        tool_name="exoswarm.science.inspect_light_curve_quality",
        tool_version=TOOL_VERSION,
        parameters={
            "quality_policy": "QUALITY == 0",
            "large_gap_days": 0.25,
            "minimum_usable_cadences": 1000,
        },
        numerical_results=results,
        result_units={
            "total_cadences": "count",
            "usable_cadences": "count",
            "quality_rejected_cadences": "count",
            "nonfinite_or_invalid_cadences": "count",
            "usable_fraction": "fraction",
            "baseline_days": "d",
            "median_cadence_minutes": "min",
            "large_gap_count": "count",
            "approximate_duty_cycle": "fraction",
        },
        uncertainties={},
        quality_flags=flags,
        interpretation_code=(
            InterpretationCode.ACCEPTABLE if acceptable else InterpretationCode.POOR_QUALITY
        ),
        limitations=[
            "SPOC quality bit zero is conservative and removes all cadences with known flags.",
            "Duty cycle is approximate because it is inferred from the median cadence and baseline.",
        ],
        input_artifacts=[source_artifact(manifest, "light_curve")],
        output_artifacts=[],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def normalize_and_clean_light_curve(
    data_root: str | Path,
    manifest: ScienceManifest,
    store: ArtifactStore,
    *,
    upper_outlier_sigma: float = 8.0,
) -> ScientificResult | ScientificFailure:
    if not 4.0 <= upper_outlier_sigma <= 20.0:
        return _failure(
            ExperimentType.NORMALIZATION,
            "upper_outlier_sigma must be between 4 and 20",
            "INVALID_OUTLIER_THRESHOLD",
            parameters={"upper_outlier_sigma": upper_outlier_sigma},
        )
    try:
        curve = load_spoc_light_curve(product_path(data_root, manifest, "light_curve"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _failure(
            ExperimentType.NORMALIZATION,
            str(exc),
            "LIGHT_CURVE_UNREADABLE",
            alternatives=[ExperimentType.LOAD_CACHED_DATA],
        )

    finite = (
        np.isfinite(curve.time_btjd)
        & np.isfinite(curve.flux)
        & np.isfinite(curve.flux_err)
        & (curve.flux_err > 0)
        & (curve.flux > 0)
    )
    base_mask = finite & (curve.quality == 0)
    if np.count_nonzero(base_mask) < 1000:
        return _failure(
            ExperimentType.NORMALIZATION,
            "fewer than 1000 finite QUALITY==0 cadences",
            "INSUFFICIENT_CADENCES",
            alternatives=[ExperimentType.QUALITY_INSPECTION],
        )
    median_flux = float(np.median(curve.flux[base_mask]))
    normalized = curve.flux[base_mask] / median_flux
    normalized_err = curve.flux_err[base_mask] / median_flux
    times = curve.time_btjd[base_mask]
    scatter = robust_sigma(normalized)
    upper_limit = float(np.median(normalized) + upper_outlier_sigma * scatter)
    # Critical preservation rule: only clip the bright tail.  Negative excursions
    # may be genuine transits/eclipses and are never sigma-clipped here.
    keep = normalized <= upper_limit
    order = np.argsort(times[keep])
    times = times[keep][order]
    normalized = normalized[keep][order]
    normalized_err = normalized_err[keep][order]
    outliers = int(np.count_nonzero(~keep))

    data_artifact = store.save_npz(
        "normalized_light_curve",
        "normalized_light_curve_data",
        time_btjd=times,
        flux=normalized,
        flux_err=normalized_err,
    )
    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.scatter(times, normalized, s=1.5, color="#77ddff", alpha=0.7, rasterized=True)
    axis.axhline(1.0, color="#ffcf6b", linewidth=0.8, alpha=0.7)
    axis.set_title("Quality-filtered, normalized SPOC light curve")
    setup_science_axes(axis, xlabel="Time (BTJD)", ylabel="Normalized flux")
    plot_artifact = store.save_figure(
        "normalized_light_curve", "normalized_light_curve_plot", figure
    )
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.NORMALIZATION,
        tool_name="exoswarm.science.normalize_and_clean_light_curve",
        tool_version=TOOL_VERSION,
        parameters={
            "flux_column": "PDCSAP_FLUX",
            "quality_policy": "QUALITY == 0",
            "upper_outlier_sigma": upper_outlier_sigma,
            "negative_sigma_clipping": False,
        },
        numerical_results={
            "input_cadences": int(curve.time_btjd.size),
            "normalized_cadences": int(times.size),
            "positive_outliers_removed": outliers,
            "normalization_flux": median_flux,
            "pre_detrend_robust_scatter_ppm": float(scatter * 1e6),
        },
        result_units={
            "input_cadences": "count",
            "normalized_cadences": "count",
            "positive_outliers_removed": "count",
            "normalization_flux": "electron / s",
            "pre_detrend_robust_scatter_ppm": "ppm",
        },
        uncertainties={},
        quality_flags=[
            QualityFlag(
                code="NEGATIVE_EXCURSIONS_PRESERVED",
                severity=QualitySeverity.INFO,
                detail="Transit-like dimmings are not removed by symmetric sigma clipping.",
            )
        ],
        interpretation_code=InterpretationCode.PROCESSED,
        limitations=[
            "PDCSAP_FLUX already includes deterministic SPOC cotrending corrections.",
            "Bright-tail rejection may remove genuine stellar flares; it is intended only for transit search preparation.",
        ],
        input_artifacts=[source_artifact(manifest, "light_curve")],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def detrend_light_curve(
    manifest: ScienceManifest,
    store: ArtifactStore,
    *,
    window_days: float = 1.0,
    gap_days: float = 0.25,
    method: str = "median_filter",
    sigma_clip: float = 5.0,
    alternate: bool = False,
) -> ScientificResult | ScientificFailure:
    experiment_type = (
        ExperimentType.ALTERNATE_DETRENDING if alternate else ExperimentType.DETRENDING
    )
    if not 0.5 <= window_days <= 3.0:
        return _failure(
            experiment_type,
            "running-median window_days must be between 0.5 and 3.0",
            "INVALID_DETREND_WINDOW",
            parameters={"window_days": window_days, "gap_days": gap_days},
        )
    normalized_method = method.casefold().replace("-", "_")
    if normalized_method in {"running_median", "median", "median_filter"}:
        normalized_method = "median_filter"
    if normalized_method not in {"median_filter", "savgol"}:
        return _failure(
            experiment_type,
            "method must be median_filter or savgol",
            "UNSUPPORTED_DETREND_METHOD",
            parameters={"method": method, "window_days": window_days},
        )
    if not 2.0 <= sigma_clip <= 10.0:
        return _failure(
            experiment_type,
            "sigma_clip must be between 2 and 10",
            "INVALID_SIGMA_CLIP",
            parameters={"sigma_clip": sigma_clip},
        )
    normalized_path = store.path("normalized_light_curve")
    if not normalized_path.is_file():
        return _failure(
            experiment_type,
            "normalization artifact is required before detrending",
            "NORMALIZED_CURVE_MISSING",
            alternatives=[ExperimentType.NORMALIZATION],
        )
    arrays = load_npz(normalized_path)
    time = arrays["time_btjd"].astype(float)
    flux = arrays["flux"].astype(float)
    flux_err = arrays["flux_err"].astype(float)
    if time.size < 1000:
        return _failure(
            experiment_type,
            "fewer than 1000 normalized cadences",
            "INSUFFICIENT_CADENCES",
            alternatives=[ExperimentType.QUALITY_INSPECTION],
        )
    cadence_days = float(np.median(np.diff(time)[np.diff(time) < 0.1]))
    requested_window_points = max(3, int(round(window_days / cadence_days)))
    if requested_window_points % 2 == 0:
        requested_window_points += 1
    split_points = np.flatnonzero(np.diff(time) > gap_days) + 1
    starts = np.r_[0, split_points]
    ends = np.r_[split_points, time.size]
    trend = np.full_like(flux, np.nan)
    for start, end in zip(starts, ends, strict=True):
        count = int(end - start)
        if count < 3:
            trend[start:end] = np.median(flux[start:end])
            continue
        local_window = min(requested_window_points, count if count % 2 else count - 1)
        local_window = max(3, local_window)
        segment = flux[start:end]
        preliminary = median_filter(segment, size=local_window, mode="nearest")
        if normalized_method == "median_filter":
            trend[start:end] = preliminary
        else:
            residual = segment / preliminary - 1.0
            scale = robust_sigma(residual)
            training = np.isfinite(segment)
            if np.isfinite(scale) and scale > 0:
                # Exclude transit-like dips and bright excursions only from the
                # trend-training series.  The original cadences remain in output.
                training &= residual > -sigma_clip * scale
                training &= residual < sigma_clip * scale
            positions = np.arange(count)
            if np.count_nonzero(training) >= max(5, local_window // 4):
                interpolated = np.interp(positions, positions[training], segment[training])
            else:
                interpolated = preliminary
            trend[start:end] = savgol_filter(
                interpolated,
                window_length=local_window,
                polyorder=2,
                mode="interp",
            )
    valid = np.isfinite(trend) & (trend > 0)
    time = time[valid]
    cleaned_flux = flux[valid] / trend[valid]
    cleaned_err = flux_err[valid] / trend[valid]
    trend = trend[valid]
    center = float(np.median(cleaned_flux))
    cleaned_flux /= center
    cleaned_err /= center
    point_to_point_sigma = robust_sigma(np.diff(cleaned_flux)) / np.sqrt(2.0)
    robust_scatter = robust_sigma(cleaned_flux - 1.0)

    stem = "alternate_cleaned_light_curve" if alternate else "cleaned_light_curve"
    data_artifact = store.save_npz(
        stem,
        "alternate_cleaned_light_curve_data" if alternate else "cleaned_light_curve_data",
        time_btjd=time,
        flux=cleaned_flux,
        flux_err=cleaned_err,
        trend=trend,
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    axes[0].scatter(time, flux[valid], s=1.2, color="#6bdcff", alpha=0.55, rasterized=True)
    axes[0].plot(time, trend, color="#ffca62", linewidth=1.0, label=normalized_method)
    axes[0].legend(frameon=False, labelcolor="#c5d5e8")
    axes[0].set_title(f"Limited detrending ({window_days:g} d window)")
    setup_science_axes(axes[0], xlabel="", ylabel="Normalized flux")
    axes[1].scatter(time, cleaned_flux, s=1.2, color="#80f0c0", alpha=0.62, rasterized=True)
    axes[1].axhline(1.0, color="#ffca62", linewidth=0.8, alpha=0.7)
    setup_science_axes(axes[1], xlabel="Time (BTJD)", ylabel="Detrended flux")
    plot_artifact = store.save_figure(
        stem,
        "alternate_cleaned_light_curve_plot" if alternate else "cleaned_light_curve_plot",
        figure,
    )
    plt.close(figure)

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=experiment_type,
        tool_name="exoswarm.science.detrend_light_curve",
        tool_version=TOOL_VERSION,
        parameters={
            "method": normalized_method,
            "window_days": window_days,
            "gap_days": gap_days,
            "window_points": requested_window_points,
            "trend_training_sigma_clip": sigma_clip,
            "negative_sigma_clipping": False,
        },
        numerical_results={
            "cleaned_cadences": int(time.size),
            "segments": int(len(starts)),
            "window_days": float(window_days),
            "point_to_point_noise_ppm": float(point_to_point_sigma * 1e6),
            "robust_scatter_ppm": float(robust_scatter * 1e6),
        },
        result_units={
            "cleaned_cadences": "count",
            "segments": "count",
            "window_days": "d",
            "point_to_point_noise_ppm": "ppm",
            "robust_scatter_ppm": "ppm",
        },
        uncertainties={},
        quality_flags=[
            QualityFlag(
                code="LIMITED_DETRENDING",
                severity=QualitySeverity.INFO,
                detail="Only bounded median-filter and transit-protected Savitzky-Golay methods are permitted.",
            )
        ],
        interpretation_code=InterpretationCode.PROCESSED,
        limitations=[
            "The selected smoother can attenuate signals with durations approaching half the selected window.",
            "The cleaned series inherits SPOC PDC corrections and is not an independent systematics model.",
            "Point-to-point MAD is a robust noise proxy, not a complete red-noise model.",
        ],
        input_artifacts=[store.existing("normalized_light_curve", "normalized_light_curve_data")],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "light_curve")],
    )


def _optional_float(value: object) -> float | None:
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
