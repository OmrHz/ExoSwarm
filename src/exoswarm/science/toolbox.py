"""Narrow runtime adapter for the bounded deterministic experiment registry."""

from __future__ import annotations

from pathlib import Path

from exoswarm.domain.models import (
    Candidate,
    ExperimentType,
    InterpretationCode,
    InvestigationState,
    ScientificFailure,
    ScientificResult,
    ScientificStatus,
    ScientificToolResponse,
    ToolRequest,
)

from .common import TOOL_VERSION, ArtifactStore, ScienceManifest, load_science_manifest
from .data import (
    detrend_light_curve,
    inspect_light_curve_quality,
    load_cached_observation,
    normalize_and_clean_light_curve,
)
from .pixels import centroid_localization
from .transit import (
    compare_detrending_sensitivity,
    phase_fold_candidate,
    search_transits,
)
from .vetting import (
    assess_signal_quality,
    contamination_screen,
    harmonic_alias_test,
    odd_even_test,
    secondary_eclipse_test,
)


class ScienceToolbox:
    """Execute one validated scientific operation against opaque cached products.

    ``runs_root`` is a directory containing one run directory per opaque target.
    Generated artifacts therefore follow
    ``runs/<TARGET>/artifacts/science/<role>.<format>`` and are indexed by a
    UI-safe ``artifacts/artifacts.json``.
    """

    def __init__(
        self,
        data_root: str | Path = "data/tess",
        runs_root: str | Path = "runs",
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.runs_root = Path(runs_root).resolve()

    def load_manifest(self, opaque_target_id: str) -> ScienceManifest:
        return load_science_manifest(self.data_root, opaque_target_id)

    def run_directory(self, opaque_target_id: str) -> Path:
        return self.runs_root / opaque_target_id

    def execute(
        self,
        request: ToolRequest,
        state: InvestigationState,
    ) -> ScientificToolResponse:
        requested_target = request.parameters.get("opaque_target_id")
        if requested_target is not None and requested_target != state.opaque_target_id:
            return _invalid_request(
                request.experiment_type,
                "request opaque target does not match investigation state",
                "TARGET_MISMATCH",
                request.parameters,
            )
        try:
            manifest = self.load_manifest(state.opaque_target_id)
        except (FileNotFoundError, ValueError) as exc:
            return _invalid_request(
                request.experiment_type,
                str(exc),
                "UNKNOWN_OPAQUE_TARGET",
                request.parameters,
            )
        store = ArtifactStore(self.run_directory(state.opaque_target_id), state.opaque_target_id)
        parameters = {
            key: value for key, value in request.parameters.items() if key != "opaque_target_id"
        }

        experiment = request.experiment_type
        if experiment is ExperimentType.LOAD_CACHED_DATA:
            invalid = _reject_unknown(experiment, parameters, {"product_id"})
            return invalid or load_cached_observation(self.data_root, manifest, store)
        if experiment is ExperimentType.QUALITY_INSPECTION:
            invalid = _reject_unknown(experiment, parameters, set())
            return invalid or inspect_light_curve_quality(self.data_root, manifest)
        if experiment is ExperimentType.NORMALIZATION:
            invalid = _reject_unknown(experiment, parameters, {"upper_outlier_sigma"})
            return invalid or normalize_and_clean_light_curve(
                self.data_root,
                manifest,
                store,
                upper_outlier_sigma=float(parameters.get("upper_outlier_sigma", 8.0)),
            )
        if experiment in {ExperimentType.DETRENDING, ExperimentType.ALTERNATE_DETRENDING}:
            candidate_id = parameters.pop("candidate_id", None)
            candidate: Candidate | None = None
            if experiment is ExperimentType.ALTERNATE_DETRENDING:
                candidate_check = _candidate_or_failure(state, experiment, candidate_id)
                if isinstance(candidate_check, ScientificFailure):
                    return candidate_check
                candidate = candidate_check
            allowed = {
                "method",
                "window_hours",
                "window_days",
                "sigma_clip",
                "gap_days",
            }
            invalid = _reject_unknown(experiment, parameters, allowed)
            if invalid is not None:
                return invalid
            window_days = float(
                parameters.get(
                    "window_days",
                    float(parameters.get("window_hours", 24.0)) / 24.0,
                )
            )
            detrending_result = detrend_light_curve(
                manifest,
                store,
                window_days=window_days,
                gap_days=float(parameters.get("gap_days", 0.25)),
                method=str(parameters.get("method", "median_filter")),
                sigma_clip=float(parameters.get("sigma_clip", 5.0)),
                alternate=experiment is ExperimentType.ALTERNATE_DETRENDING,
            )
            if experiment is ExperimentType.ALTERNATE_DETRENDING and isinstance(
                detrending_result, ScientificResult
            ):
                assert candidate is not None
                return compare_detrending_sensitivity(
                    manifest,
                    store,
                    candidate,
                    detrending_result,
                )
            return detrending_result
        if experiment is ExperimentType.TRANSIT_SEARCH:
            allowed = {
                "min_period_days",
                "max_period_days",
                "min_duration_hours",
                "max_duration_hours",
                "durations_hours",
                "frequency_samples",
                "harmonic_tie_fraction",
                "alternate_detrending",
            }
            invalid = _reject_unknown(experiment, parameters, allowed)
            durations = parameters.get("durations_hours", [1.0, 2.0, 4.0, 6.0])
            if not isinstance(durations, list) or not durations:
                return _invalid_request(
                    experiment,
                    "durations_hours must be a non-empty list",
                    "INVALID_DURATION_GRID",
                    parameters,
                )
            return invalid or search_transits(
                manifest,
                store,
                min_period_days=float(parameters.get("min_period_days", 0.5)),
                max_period_days=float(parameters.get("max_period_days", 10.0)),
                min_duration_hours=float(parameters.get("min_duration_hours", min(durations))),
                max_duration_hours=float(parameters.get("max_duration_hours", max(durations))),
                frequency_samples=int(parameters.get("frequency_samples", 12000)),
                harmonic_tie_fraction=float(parameters.get("harmonic_tie_fraction", 0.04)),
                alternate_detrending=bool(parameters.get("alternate_detrending", False)),
            )

        candidate_id = parameters.pop("candidate_id", None)
        candidate_or_failure = _candidate_or_failure(state, experiment, candidate_id)
        if isinstance(candidate_or_failure, ScientificFailure):
            return candidate_or_failure
        candidate = candidate_or_failure
        if experiment is ExperimentType.PHASE_FOLD:
            invalid = _reject_unknown(experiment, parameters, {"bins", "alternate_detrending"})
            return invalid or phase_fold_candidate(
                manifest,
                store,
                candidate,
                bins=int(parameters.get("bins", 180)),
                alternate_detrending=bool(parameters.get("alternate_detrending", False)),
            )
        if experiment is ExperimentType.SIGNAL_QUALITY:
            invalid = _reject_unknown(
                experiment,
                parameters,
                {"minimum_snr", "minimum_events", "maximum_fractional_duration"},
            )
            return invalid or assess_signal_quality(
                manifest,
                candidate,
                minimum_snr=float(parameters.get("minimum_snr", 7.0)),
                minimum_events=int(parameters.get("minimum_events", 2)),
                maximum_fractional_duration=float(
                    parameters.get("maximum_fractional_duration", 0.2)
                ),
            )
        if experiment is ExperimentType.ODD_EVEN:
            invalid = _reject_unknown(experiment, parameters, set())
            return invalid or odd_even_test(manifest, store, candidate)
        if experiment is ExperimentType.SECONDARY_ECLIPSE:
            invalid = _reject_unknown(
                experiment,
                parameters,
                {"phase_half_threshold_sigma", "scan_threshold_sigma"},
            )
            return invalid or secondary_eclipse_test(
                manifest,
                store,
                candidate,
                phase_half_threshold_sigma=float(parameters.get("phase_half_threshold_sigma", 5.0)),
                scan_threshold_sigma=float(parameters.get("scan_threshold_sigma", 10.0)),
            )
        if experiment is ExperimentType.CONTAMINATION_SCREEN:
            invalid = _reject_unknown(experiment, parameters, {"aperture_context_radius_arcsec"})
            return invalid or contamination_screen(
                self.data_root,
                manifest,
                aperture_context_radius_arcsec=float(
                    parameters.get("aperture_context_radius_arcsec", 42.0)
                ),
            )
        if experiment is ExperimentType.HARMONIC_TEST:
            base_period = parameters.pop("base_period_days", candidate.period_days)
            factors = parameters.pop("factors", [0.5, 1.0, 2.0])
            if factors != [0.5, 1.0, 2.0]:
                return _invalid_request(
                    experiment,
                    "harmonic factors must be exactly [0.5, 1.0, 2.0]",
                    "INVALID_HARMONIC_FACTORS",
                    parameters,
                )
            if abs(float(base_period) - candidate.period_days) > max(
                1e-9, 1e-6 * candidate.period_days
            ):
                return _invalid_request(
                    experiment,
                    "base_period_days does not match the selected candidate",
                    "CANDIDATE_PERIOD_MISMATCH",
                    parameters,
                )
            invalid = _reject_unknown(
                experiment,
                parameters,
                {"double_preference_snr_ratio", "odd_even_threshold_sigma"},
            )
            return invalid or harmonic_alias_test(
                manifest,
                store,
                candidate,
                double_preference_snr_ratio=float(
                    parameters.get("double_preference_snr_ratio", 1.01)
                ),
                odd_even_threshold_sigma=float(parameters.get("odd_even_threshold_sigma", 3.0)),
            )
        if experiment is ExperimentType.CENTROID_LOCALIZATION:
            invalid = _reject_unknown(
                experiment,
                parameters,
                {
                    "aperture_id",
                    "transit_window_scale",
                    "bootstrap_samples",
                    "random_seed",
                },
            )
            return invalid or centroid_localization(
                self.data_root,
                manifest,
                store,
                candidate,
                bootstrap_samples=int(parameters.get("bootstrap_samples", 96)),
                random_seed=int(parameters.get("random_seed", 17042)),
                aperture_id=(
                    str(parameters["aperture_id"])
                    if parameters.get("aperture_id") is not None
                    else None
                ),
                transit_window_scale=float(parameters.get("transit_window_scale", 1.0)),
            )
        if experiment is ExperimentType.ALTERNATE_APERTURE:
            return ScientificFailure(
                status=ScientificStatus.PRECONDITION_FAILED,
                experiment_type=experiment,
                tool_name="exoswarm.science.alternate_aperture",
                tool_version=TOOL_VERSION,
                parameters=parameters,
                reason="alternate-aperture extraction is outside the MVP experiment registry",
                reason_code="NOT_IMPLEMENTED_IN_MVP",
                suggested_alternatives=[ExperimentType.CENTROID_LOCALIZATION],
                interpretation_code=InterpretationCode.PRECONDITION_NOT_MET,
                limitations=[
                    "The MVP implements one genuine target-pixel centroid diagnostic rather than multiple aperture families."
                ],
            )
        return _invalid_request(
            experiment,
            f"unsupported scientific experiment {experiment.value}",
            "UNKNOWN_EXPERIMENT",
            parameters,
        )


def candidate_from_search_result(result: ScientificResult) -> Candidate:
    """Create the shared domain Candidate from a validated transit-search result."""

    if result.experiment_type is not ExperimentType.TRANSIT_SEARCH:
        raise ValueError("candidate can only be constructed from a transit-search result")
    if result.interpretation_code is not InterpretationCode.DETECTED:
        raise ValueError("transit search did not detect a viable candidate")
    values = result.numerical_results
    required = {
        "period_days",
        "epoch_btjd",
        "transit_depth_ppm",
        "duration_hours",
        "signal_to_noise",
        "observed_events",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"transit search missing canonical candidate keys: {sorted(missing)}")
    return Candidate(
        period_days=float(values["period_days"]),
        epoch_btjd=float(values["epoch_btjd"]),
        transit_depth_ppm=float(values["transit_depth_ppm"]),
        duration_hours=float(values["duration_hours"]),
        signal_to_noise=float(values["signal_to_noise"]),
        observed_events=int(values["observed_events"]),
        uncertainties={
            key: value for key, value in result.uncertainties.items() if key in required
        },
        search_statistic=float(values["bls_global_peak_snr"]),
        artifact_ids=[artifact.artifact_id for artifact in result.output_artifacts],
    )


def _candidate_or_failure(
    state: InvestigationState,
    experiment_type: ExperimentType,
    candidate_id: object | None = None,
) -> Candidate | ScientificFailure:
    if state.candidates:
        if candidate_id is None:
            return state.candidates[-1]
        selected = next(
            (candidate for candidate in state.candidates if candidate.candidate_id == candidate_id),
            None,
        )
        if selected is not None:
            return selected
    return ScientificFailure(
        status=ScientificStatus.PRECONDITION_FAILED,
        experiment_type=experiment_type,
        tool_name=f"exoswarm.science.{experiment_type.value}",
        tool_version=TOOL_VERSION,
        parameters={},
        reason=(
            f"{experiment_type.value} requires a validated transit candidate"
            if candidate_id is None
            else f"no candidate matches candidate_id={candidate_id!r}"
        ),
        reason_code="CANDIDATE_REQUIRED",
        suggested_alternatives=[ExperimentType.TRANSIT_SEARCH],
        interpretation_code=InterpretationCode.PRECONDITION_NOT_MET,
    )


def _reject_unknown(
    experiment_type: ExperimentType,
    parameters: dict[str, object],
    allowed: set[str],
) -> ScientificFailure | None:
    unknown = set(parameters) - allowed
    if not unknown:
        return None
    return _invalid_request(
        experiment_type,
        f"unknown parameters for {experiment_type.value}: {sorted(unknown)}",
        "UNKNOWN_PARAMETER",
        parameters,
    )


def _invalid_request(
    experiment_type: ExperimentType,
    reason: str,
    reason_code: str,
    parameters: dict[str, object],
) -> ScientificFailure:
    return ScientificFailure(
        status=ScientificStatus.INVALID_REQUEST,
        experiment_type=experiment_type,
        tool_name="exoswarm.science.ScienceToolbox.execute",
        tool_version=TOOL_VERSION,
        parameters=parameters,  # type: ignore[arg-type]
        reason=reason,
        reason_code=reason_code,
        suggested_alternatives=[],
        interpretation_code=InterpretationCode.TOOL_ERROR,
    )
