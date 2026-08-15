"""Target-pixel difference imaging and transit-correlated centroid localization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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
    product_path,
    robust_sigma,
    source_artifact,
    tess_provenance,
)


def centroid_localization(
    data_root: str | Path,
    manifest: ScienceManifest,
    store: ArtifactStore,
    candidate: Candidate,
    *,
    bootstrap_samples: int = 96,
    random_seed: int = 17042,
    aperture_id: str | None = None,
    transit_window_scale: float = 1.0,
) -> ScientificResult | ScientificFailure:
    """Localize event-associated flux using the genuine cached SPOC pixel cube.

    The diagnostic combines two reproducible signals:

    * the centroid of the out-minus-in difference image; and
    * the direction of the transit-correlated aperture photocenter shift relative
      to the nearest catalog neighbor.

    The second signal is important for crowded apertures: when the target dims the
    photocenter should move *toward* a steady neighbor, whereas a dimming neighbor
    moves the photocenter in the opposite direction.
    """

    if bootstrap_samples < 32 or bootstrap_samples > 512:
        return _failure(
            "bootstrap_samples must be between 32 and 512",
            "INVALID_BOOTSTRAP_SAMPLES",
            parameters={"bootstrap_samples": bootstrap_samples},
        )
    if aperture_id not in {None, "pipeline", "spoc"}:
        return _failure(
            "centroid localization only permits the cached SPOC pipeline aperture",
            "UNKNOWN_APERTURE",
            parameters={"aperture_id": aperture_id},
        )
    if not 0.25 < transit_window_scale <= 3.0:
        return _failure(
            "transit_window_scale must be in (0.25, 3]",
            "INVALID_TRANSIT_WINDOW_SCALE",
            parameters={"transit_window_scale": transit_window_scale},
        )
    path = product_path(data_root, manifest, "target_pixel")
    try:
        with fits.open(path, memmap=True) as hdul:
            if len(hdul) < 3 or hdul[1].data is None or hdul[2].data is None:
                raise ValueError("target-pixel FITS lacks PIXELS or APERTURE extension")
            table = hdul[1].data
            required = {"TIME", "FLUX", "QUALITY"}
            missing = required - set(table.names or ())
            if missing:
                raise ValueError(f"target-pixel FITS missing columns: {sorted(missing)}")
            time = np.asarray(table["TIME"], dtype=float).copy()
            cube = np.asarray(table["FLUX"], dtype=float).copy()
            quality = np.asarray(table["QUALITY"], dtype=np.int64).copy()
            aperture = (np.asarray(hdul[2].data, dtype=np.int64) & 2) > 0
            wcs = WCS(hdul[2].header)
            # The target coordinate is used only inside deterministic code and is
            # immediately converted to anonymous stamp pixels.
            target_x, target_y = wcs.world_to_pixel_values(
                float(hdul[0].header["RA_OBJ"]), float(hdul[0].header["DEC_OBJ"])
            )
            pixel_scale_arcsec = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        return _failure(
            str(exc),
            "TARGET_PIXEL_UNREADABLE",
            alternatives=[ExperimentType.CONTAMINATION_SCREEN],
        )

    duration_days = candidate.duration_hours / 24.0 * transit_window_scale
    distance = np.abs(_phase_time(time, candidate.period_days, candidate.epoch_btjd))
    finite_frames = np.isfinite(time) & np.any(np.isfinite(cube), axis=(1, 2))
    good = finite_frames & (quality == 0)
    in_transit = good & (distance <= duration_days / 2.0)
    out_of_transit = good & (distance >= 1.5 * duration_days) & (distance <= 3.0 * duration_days)
    in_indices = np.flatnonzero(in_transit)
    out_indices = np.flatnonzero(out_of_transit)
    if in_indices.size < 20 or out_indices.size < 40:
        return _failure(
            (
                "centroid localization requires >=20 in-transit and >=40 local "
                f"out-of-transit cadences; found {in_indices.size} and {out_indices.size}"
            ),
            "INSUFFICIENT_PIXEL_CADENCES",
            alternatives=[ExperimentType.CONTAMINATION_SCREEN],
        )

    out_image = np.nanmean(cube[out_indices], axis=0)
    in_image = np.nanmean(cube[in_indices], axis=0)
    difference_image = out_image - in_image
    difference_x, difference_y = _difference_centroid(difference_image, aperture)
    difference_offset = float(
        np.hypot(difference_x - float(target_x), difference_y - float(target_y))
    )

    in_centroids = _aperture_photocenters(cube[in_indices], aperture)
    out_centroids = _aperture_photocenters(cube[out_indices], aperture)
    in_centroid = np.nanmedian(in_centroids, axis=0)
    out_centroid = np.nanmedian(out_centroids, axis=0)
    shift = in_centroid - out_centroid
    shift_pixels = float(np.hypot(shift[0], shift[1]))
    shift_x_error = float(
        np.hypot(
            _median_standard_error(in_centroids[:, 0]),
            _median_standard_error(out_centroids[:, 0]),
        )
    )
    shift_y_error = float(
        np.hypot(
            _median_standard_error(in_centroids[:, 1]),
            _median_standard_error(out_centroids[:, 1]),
        )
    )
    shift_error = float(np.hypot(shift_x_error, shift_y_error))
    shift_significance = shift_pixels / shift_error if shift_error > 0 else 0.0

    rng = np.random.default_rng(random_seed)
    bootstrap_centroids = np.empty((bootstrap_samples, 2), dtype=float)
    for index in range(bootstrap_samples):
        selected_in = rng.choice(in_indices, size=in_indices.size, replace=True)
        selected_out = rng.choice(out_indices, size=out_indices.size, replace=True)
        sample_difference = np.nanmean(cube[selected_out], axis=0) - np.nanmean(
            cube[selected_in], axis=0
        )
        bootstrap_centroids[index] = _difference_centroid(sample_difference, aperture)
    difference_centroid_error = float(
        np.hypot(
            robust_sigma(bootstrap_centroids[:, 0]),
            robust_sigma(bootstrap_centroids[:, 1]),
        )
    )

    direction_cosine = 0.0
    neighbor_x = float(target_x)
    neighbor_y = float(target_y)
    neighbor = manifest.neighbor_context
    if neighbor is not None:
        neighbor_vector = np.array(
            [
                neighbor.nearest_neighbor_pixel_offset_x,
                neighbor.nearest_neighbor_pixel_offset_y,
            ],
            dtype=float,
        )
        neighbor_x += neighbor_vector[0]
        neighbor_y += neighbor_vector[1]
        if shift_pixels > 0 and np.linalg.norm(neighbor_vector) > 0:
            direction_cosine = float(
                np.dot(shift, neighbor_vector) / (shift_pixels * np.linalg.norm(neighbor_vector))
            )

    target_consistent = bool(
        difference_offset <= 0.5 and (shift_significance < 3.0 or direction_cosine >= 0.0)
    )
    offset_detected = bool(shift_significance >= 3.0 and direction_cosine <= -0.25)
    if offset_detected:
        interpretation = InterpretationCode.OFFSET_DETECTED
    elif target_consistent:
        interpretation = InterpretationCode.TARGET_CONSISTENT
    else:
        interpretation = InterpretationCode.INCONCLUSIVE

    data_artifact = store.save_npz(
        "centroid_localization",
        "centroid_localization_data",
        out_of_transit_image=out_image,
        in_transit_image=in_image,
        difference_image=difference_image,
        aperture_mask=aperture.astype(np.uint8),
        target_xy=np.asarray([target_x, target_y], dtype=float),
        neighbor_xy=np.asarray([neighbor_x, neighbor_y], dtype=float),
        difference_centroid_xy=np.asarray([difference_x, difference_y], dtype=float),
        in_transit_photocenter_xy=in_centroids,
        out_of_transit_photocenter_xy=out_centroids,
        centroid_bootstrap_xy=bootstrap_centroids,
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    images = [out_image, in_image, difference_image]
    titles = ["Out of transit", "In transit", "Difference (out − in)"]
    for axis, image, title in zip(axes, images, titles, strict=True):
        shown = axis.imshow(image, origin="lower", cmap="magma")
        axis.plot(target_x, target_y, "+", color="#68f5c1", ms=10, mew=1.4, label="target")
        axis.plot(
            neighbor_x, neighbor_y, "x", color="#6de4ff", ms=7, mew=1.0, label="nearest neighbor"
        )
        axis.set_title(title, color="#e8f2ff")
        axis.tick_params(colors="#a9bdd6")
        figure.colorbar(shown, ax=axis, fraction=0.045, pad=0.03)
    axes[2].plot(
        difference_x,
        difference_y,
        "o",
        mfc="none",
        mec="#ffca62",
        ms=9,
        mew=1.2,
        label="difference centroid",
    )
    axes[2].legend(frameon=False, fontsize=7, labelcolor="#d7e4f5", loc="upper right")
    figure.suptitle("Transit-associated target-pixel localization", color="#e8f2ff")
    plot_artifact = store.save_figure("centroid_localization", "centroid_localization_plot", figure)
    plt.close(figure)

    flags: list[QualityFlag] = []
    if shift_significance < 3.0:
        flags.append(
            QualityFlag(
                code="NO_SIGNIFICANT_PHOTOCENTER_SHIFT",
                severity=QualitySeverity.INFO,
                detail="The in/out aperture photocenter motion is below 3 sigma.",
            )
        )
    elif direction_cosine >= 0:
        flags.append(
            QualityFlag(
                code="SHIFT_DIRECTION_TARGET_DIMMING",
                severity=QualitySeverity.INFO,
                detail="The photocenter moves toward the steady neighbor, as expected when the target dims.",
            )
        )
    elif offset_detected:
        flags.append(
            QualityFlag(
                code="SHIFT_DIRECTION_NEIGHBOR_DIMMING",
                severity=QualitySeverity.WARNING,
                detail="The photocenter shift direction is more compatible with the neighbor dimming.",
            )
        )

    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.CENTROID_LOCALIZATION,
        tool_name="exoswarm.science.centroid_localization",
        tool_version=TOOL_VERSION,
        parameters={
            "in_transit_half_width_hours": candidate.duration_hours / 2.0,
            "out_of_transit_annulus_duration_multiple_min": 1.5,
            "out_of_transit_annulus_duration_multiple_max": 3.0,
            "quality_policy": "QUALITY == 0",
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "aperture_id": aperture_id or "pipeline",
            "transit_window_scale": transit_window_scale,
            "target_consistency_radius_pixels": 0.5,
        },
        numerical_results={
            "in_transit_cadences": int(in_indices.size),
            "out_of_transit_cadences": int(out_indices.size),
            "aperture_pixels": int(np.count_nonzero(aperture)),
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "difference_centroid_x": difference_x,
            "difference_centroid_y": difference_y,
            "target_pixel_x": float(target_x),
            "target_pixel_y": float(target_y),
            "difference_source_offset_pixels": difference_offset,
            "difference_source_offset_arcsec": difference_offset * pixel_scale_arcsec,
            "photocenter_shift_pixels": shift_pixels,
            "photocenter_shift_arcsec": shift_pixels * pixel_scale_arcsec,
            "photocenter_shift_significance_sigma": shift_significance,
            "target_dimming_direction_cosine": direction_cosine,
            "nearest_neighbor_separation_arcsec": (
                neighbor.nearest_neighbor_separation_arcsec if neighbor else 0.0
            ),
        },
        result_units={
            "in_transit_cadences": "count",
            "out_of_transit_cadences": "count",
            "aperture_pixels": "count",
            "pixel_scale_arcsec": "arcsec / pixel",
            "difference_centroid_x": "pixel",
            "difference_centroid_y": "pixel",
            "target_pixel_x": "pixel",
            "target_pixel_y": "pixel",
            "difference_source_offset_pixels": "pixel",
            "difference_source_offset_arcsec": "arcsec",
            "photocenter_shift_pixels": "pixel",
            "photocenter_shift_arcsec": "arcsec",
            "photocenter_shift_significance_sigma": "sigma",
            "target_dimming_direction_cosine": "cosine",
            "nearest_neighbor_separation_arcsec": "arcsec",
        },
        uncertainties={
            "difference_source_offset_pixels": MeasurementUncertainty(
                value=difference_centroid_error,
                unit="pixel",
                method="deterministic bootstrap of in/out difference-image centroids",
            ),
            "difference_source_offset_arcsec": MeasurementUncertainty(
                value=difference_centroid_error * pixel_scale_arcsec,
                unit="arcsec",
                method="difference-centroid bootstrap propagated through local WCS pixel scale",
            ),
            "photocenter_shift_pixels": MeasurementUncertainty(
                value=shift_error,
                unit="pixel",
                method="robust standard errors of median in/out aperture photocenters",
            ),
            "photocenter_shift_arcsec": MeasurementUncertainty(
                value=shift_error * pixel_scale_arcsec,
                unit="arcsec",
                method="photocenter uncertainty propagated through local WCS pixel scale",
            ),
        },
        quality_flags=flags,
        interpretation_code=interpretation,
        limitations=[
            "This is a limited single-sector difference-image/centroid diagnostic, not a calibrated SPOC PRF-centroid or probabilistic validation pipeline.",
            "Moment centroids can be biased by the asymmetric undersampled TESS pixel response and aperture truncation.",
            "TARGET_CONSISTENT cannot exclude unresolved companions at substantially sub-pixel separation.",
            "The neighbor-direction rule is only applied to the nearest cached TIC source.",
        ],
        input_artifacts=[source_artifact(manifest, "target_pixel")],
        output_artifacts=[data_artifact, plot_artifact],
        provenance=[tess_provenance(manifest, "target_pixel")],
    )


def _difference_centroid(image: np.ndarray, aperture: np.ndarray) -> tuple[float, float]:
    weights = np.where(aperture, np.clip(image, 0.0, None), 0.0)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    y, x = np.indices(image.shape)
    return float(np.nansum(weights * x) / total), float(np.nansum(weights * y) / total)


def _aperture_photocenters(cube: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    if np.any(~aperture):
        backgrounds = np.nanmedian(cube[:, ~aperture], axis=1)
    else:
        backgrounds = np.zeros(cube.shape[0])
    weights = np.where(
        aperture[None, :, :],
        np.clip(cube - backgrounds[:, None, None], 0.0, None),
        0.0,
    )
    totals = np.nansum(weights, axis=(1, 2))
    y, x = np.indices(aperture.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        centers_x = np.nansum(weights * x[None, :, :], axis=(1, 2)) / totals
        centers_y = np.nansum(weights * y[None, :, :], axis=(1, 2)) / totals
    return np.column_stack([centers_x, centers_y])


def _median_standard_error(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("inf")
    # 1.253 converts the standard error of a mean to that of a median under a
    # Gaussian approximation; MAD makes the scale robust to pointing outliers.
    return float(1.253 * robust_sigma(finite) / np.sqrt(finite.size))


def _phase_time(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return (((time - epoch + 0.5 * period) % period) / period - 0.5) * period


def _failure(
    reason: str,
    reason_code: str,
    *,
    alternatives: list[ExperimentType] | None = None,
    parameters: dict[str, int | float | bool | str] | None = None,
) -> ScientificFailure:
    return ScientificFailure(
        status=ScientificStatus.PRECONDITION_FAILED,
        experiment_type=ExperimentType.CENTROID_LOCALIZATION,
        tool_name="exoswarm.science.centroid_localization",
        tool_version=TOOL_VERSION,
        parameters=parameters or {},
        reason=reason,
        reason_code=reason_code,
        suggested_alternatives=alternatives or [],
        interpretation_code=InterpretationCode.PRECONDITION_NOT_MET,
    )
