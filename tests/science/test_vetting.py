from __future__ import annotations

from exoswarm.domain.models import (
    Candidate,
    ExperimentType,
    InterpretationCode,
    ScientificFailure,
    ScientificResult,
    ScientificStatus,
    ToolRequest,
)


def _execute(case, experiment: ExperimentType, **parameters) -> ScientificResult:
    result = case.toolbox.execute(
        ToolRequest(
            experiment_type=experiment,
            requested_by="pytest",
            parameters=parameters,
        ),
        case.state,
    )
    assert isinstance(result, ScientificResult), result
    return result


def test_planet_mandatory_vetting_and_centroid_are_conservative(prepared_cases) -> None:
    case = prepared_cases["TARGET-X17"]
    quality = _execute(case, ExperimentType.SIGNAL_QUALITY)
    odd_even = _execute(case, ExperimentType.ODD_EVEN)
    secondary = _execute(case, ExperimentType.SECONDARY_ECLIPSE)
    contamination = _execute(case, ExperimentType.CONTAMINATION_SCREEN)
    centroid = _execute(case, ExperimentType.CENTROID_LOCALIZATION, bootstrap_samples=32)
    harmonic = _execute(case, ExperimentType.HARMONIC_TEST)
    assert quality.interpretation_code is InterpretationCode.PASS
    assert odd_even.interpretation_code is InterpretationCode.CONSISTENT
    assert odd_even.numerical_results["depth_difference_sigma"] < 3
    assert secondary.interpretation_code is InterpretationCode.NOT_SIGNIFICANT
    assert contamination.interpretation_code is InterpretationCode.NEIGHBOR_DETECTED
    assert contamination.numerical_results["spatial_localization_recommended"] == 1
    assert centroid.interpretation_code is InterpretationCode.TARGET_CONSISTENT
    assert centroid.numerical_results["difference_source_offset_pixels"] < 0.5
    assert harmonic.interpretation_code is InterpretationCode.PREFERRED_NOMINAL_PERIOD


def test_alternate_detrending_researches_candidate_and_reports_robustness(
    prepared_cases,
) -> None:
    case = prepared_cases["TARGET-X17"]
    result = _execute(
        case,
        ExperimentType.ALTERNATE_DETRENDING,
        method="savgol",
        window_hours=36.0,
    )
    assert result.interpretation_code in {
        InterpretationCode.ROBUST,
        InterpretationCode.PREPROCESSING_SENSITIVE,
    }
    assert result.interpretation_code is not InterpretationCode.PROCESSED
    assert "alternate_period_days" in result.numerical_results
    assert "alternate_depth_ppm" in result.numerical_results
    assert "alternate_to_nominal_snr_ratio" in result.numerical_results
    assert "passes_preprocessing_robustness" in result.numerical_results
    assert {item.role for item in result.output_artifacts} >= {
        "alternate_cleaned_light_curve_data",
        "detrending_sensitivity_data",
        "detrending_sensitivity_plot",
    }


def test_eb_vetting_resolves_two_p_and_produces_different_path(prepared_cases) -> None:
    case = prepared_cases["TARGET-X42"]
    odd_even = _execute(case, ExperimentType.ODD_EVEN)
    harmonic = _execute(case, ExperimentType.HARMONIC_TEST)
    centroid = _execute(case, ExperimentType.CENTROID_LOCALIZATION, bootstrap_samples=32)
    assert odd_even.interpretation_code is InterpretationCode.INCONSISTENT
    assert odd_even.numerical_results["depth_difference_sigma"] > 100
    assert harmonic.interpretation_code is InterpretationCode.PREFERRED_DOUBLE_PERIOD
    assert harmonic.numerical_results["preferred_factor"] == 2.0
    assert abs(harmonic.numerical_results["preferred_period_days"] - 3.0319175718) < 0.001
    assert harmonic.numerical_results["preferred_secondary_depth_ppm"] > 20_000
    assert harmonic.numerical_results["preferred_secondary_significance_sigma"] > 20
    assert harmonic.numerical_results["preferred_observed_events"] >= 6
    assert harmonic.numerical_results["preferred_signal_to_noise"] > 100
    assert {
        "preferred_period_days",
        "preferred_epoch_btjd",
        "preferred_duration_hours",
        "preferred_primary_depth_ppm",
    } <= harmonic.uncertainties.keys()
    assert centroid.interpretation_code is InterpretationCode.TARGET_CONSISTENT
    assert centroid.numerical_results["target_dimming_direction_cosine"] > 0


def test_odd_even_precondition_failure_is_structured(prepared_cases) -> None:
    case = prepared_cases["TARGET-X17"]
    original = case.state.candidates[-1]
    too_long = Candidate(
        period_days=12.0,
        epoch_btjd=original.epoch_btjd,
        transit_depth_ppm=original.transit_depth_ppm,
        duration_hours=original.duration_hours,
        signal_to_noise=original.signal_to_noise,
        observed_events=2,
    )
    case.state.candidates.append(too_long)
    try:
        result = case.toolbox.execute(
            ToolRequest(experiment_type=ExperimentType.ODD_EVEN, requested_by="pytest"),
            case.state,
        )
    finally:
        case.state.candidates.pop()
    assert isinstance(result, ScientificFailure)
    assert result.status is ScientificStatus.PRECONDITION_FAILED
    assert result.reason_code == "TOO_FEW_TRANSITS_FOR_ODD_EVEN"
    assert ExperimentType.SECONDARY_ECLIPSE in result.suggested_alternatives


def test_every_successful_vetting_output_round_trips_shared_schema(prepared_cases) -> None:
    case = prepared_cases["TARGET-X17"]
    for experiment in (
        ExperimentType.PHASE_FOLD,
        ExperimentType.SIGNAL_QUALITY,
        ExperimentType.ODD_EVEN,
        ExperimentType.SECONDARY_ECLIPSE,
        ExperimentType.CONTAMINATION_SCREEN,
        ExperimentType.HARMONIC_TEST,
        ExperimentType.CENTROID_LOCALIZATION,
    ):
        params = (
            {"bootstrap_samples": 32} if experiment is ExperimentType.CENTROID_LOCALIZATION else {}
        )
        result = _execute(case, experiment, **params)
        assert ScientificResult.model_validate(result.model_dump()) == result


def test_phase_fold_propagates_canonical_ephemeris_uncertainties(prepared_cases) -> None:
    case = prepared_cases["TARGET-X17"]
    candidate = case.state.candidates[-1]
    result = _execute(case, ExperimentType.PHASE_FOLD)
    assert result.numerical_results["period_days"] == candidate.period_days
    assert result.numerical_results["epoch_btjd"] == candidate.epoch_btjd
    assert result.uncertainties["period_days"] == candidate.uncertainties["period_days"]
    assert result.uncertainties["epoch_btjd"] == candidate.uncertainties["epoch_btjd"]
