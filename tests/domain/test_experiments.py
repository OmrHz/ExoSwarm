from __future__ import annotations

from exoswarm.domain import (
    ArtifactRef,
    BackendTargetMappingRef,
    Budget,
    Candidate,
    ExperimentRegistry,
    ExperimentType,
    InvestigationState,
    ReviewStatus,
    ToolRequest,
)


def state_with_candidate(*, events: int = 5, budget: int = 8) -> InvestigationState:
    return InvestigationState(
        opaque_target_id="TARGET-X17",
        backend_target_mapping=BackendTargetMappingRef(mapping_key="c" * 24),
        candidates=[
            Candidate(
                candidate_id="CAND-1",
                period_days=6.2,
                epoch_btjd=1900,
                transit_depth_ppm=1200,
                duration_hours=2.4,
                signal_to_noise=12,
                observed_events=events,
            )
        ],
        available_data_products=[
            ArtifactRef(artifact_id="TPF-1", role="target_pixel", path="cached/tpf.fits")
        ],
        completed_tests=[ExperimentType.TRANSIT_SEARCH],
        available_tests=[
            ExperimentType.ODD_EVEN,
            ExperimentType.HARMONIC_TEST,
            ExperimentType.CENTROID_LOCALIZATION,
        ],
        experiment_budget=Budget(limit=budget),
    )


def test_odd_even_structured_precondition_failure() -> None:
    registry = ExperimentRegistry()
    state = state_with_candidate(events=3)
    request = ToolRequest(
        experiment_type=ExperimentType.ODD_EVEN,
        parameters={"candidate_id": "CAND-1"},
        requested_by="runtime",
    )
    review = registry.validate(request, state)
    assert review.status is ReviewStatus.PRECONDITION_FAILED
    assert review.reason_code == "INSUFFICIENT_OBSERVED_EVENTS"
    assert ExperimentType.HARMONIC_TEST in review.suggested_alternatives
    failure = registry.as_scientific_failure(review)
    assert failure.reason_code == "INSUFFICIENT_OBSERVED_EVENTS"


def test_registry_rejects_nonsensical_parameters_and_unknown_raw_tool() -> None:
    registry = ExperimentRegistry()
    state = state_with_candidate()
    state.available_tests.append(ExperimentType.TRANSIT_SEARCH)
    bad_range = ToolRequest(
        experiment_type=ExperimentType.TRANSIT_SEARCH,
        parameters={"min_period_days": 10, "max_period_days": 2},
        requested_by="agent",
    )
    assert registry.validate(bad_range, state).reason_code == "INVALID_EXPERIMENT_PARAMETERS"

    unknown = registry.validate_raw(
        {
            "request_id": "REQ-X",
            "experiment_type": "query_nasa_catalog",
            "parameters": {},
            "requested_by": "agent",
        },
        state,
    )
    assert unknown.status is ReviewStatus.REJECTED
    assert unknown.reason_code == "INVALID_TOOL_REQUEST_SCHEMA"

    invalid_detrender = ToolRequest(
        experiment_type=ExperimentType.ALTERNATE_DETRENDING,
        parameters={
            "candidate_id": "CAND-1",
            "method": "llm_invented_smoother",
            "window_hours": 36,
        },
        adaptive=True,
        requested_by="agent",
    )
    state.available_tests.append(ExperimentType.ALTERNATE_DETRENDING)
    assert (
        registry.validate(invalid_detrender, state).reason_code == "INVALID_EXPERIMENT_PARAMETERS"
    )
    outside_scientific_window = invalid_detrender.model_copy(
        update={
            "parameters": {
                "candidate_id": "CAND-1",
                "method": "savgol",
                "window_hours": 6,
            }
        }
    )
    assert (
        registry.validate(outside_scientific_window, state).reason_code
        == "INVALID_EXPERIMENT_PARAMETERS"
    )


def test_registry_enforces_budget_availability_duplicates_and_lock() -> None:
    registry = ExperimentRegistry()
    state = state_with_candidate(budget=0)
    request = ToolRequest(
        experiment_type=ExperimentType.HARMONIC_TEST,
        parameters={
            "candidate_id": "CAND-1",
            "base_period_days": 6.2,
            "factors": [0.5, 1, 2],
        },
        adaptive=True,
        requested_by="skeptic",
    )
    assert registry.validate(request, state).status is ReviewStatus.BUDGET_EXHAUSTED

    state.experiment_budget = Budget(limit=3)
    assert registry.validate(request, state).status is ReviewStatus.ALLOWED
    registry.record_attempt(state, request, successful=True)
    state.available_tests.append(ExperimentType.HARMONIC_TEST)
    repeat = request.model_copy(update={"request_id": "REQ-REPEAT", "justification": "recheck"})
    assert registry.validate(repeat, state).reason_code == "MAX_EXECUTIONS_REACHED"


def test_registry_exposes_only_current_precondition_valid_tools() -> None:
    registry = ExperimentRegistry()
    state = InvestigationState(
        opaque_target_id="TARGET-X17",
        backend_target_mapping=BackendTargetMappingRef(mapping_key="d" * 24),
    )
    registry.initialize_available_tests(state)
    assert state.available_tests == [ExperimentType.LOAD_CACHED_DATA]
    request = ToolRequest(
        experiment_type=ExperimentType.LOAD_CACHED_DATA,
        requested_by="runtime",
    )
    registry.record_attempt(state, request, successful=True)
    assert state.available_tests == [ExperimentType.QUALITY_INSPECTION]


def test_registry_exposes_adaptive_tools_when_candidate_preconditions_hold() -> None:
    registry = ExperimentRegistry()
    state = state_with_candidate()
    registry.initialize_available_tests(state)
    assert ExperimentType.HARMONIC_TEST in state.available_tests
    assert ExperimentType.CENTROID_LOCALIZATION in state.available_tests
    assert ExperimentType.ALTERNATE_DETRENDING in state.available_tests
    assert ExperimentType.ALTERNATE_APERTURE not in state.available_tests


def test_harmonic_period_must_match_deterministic_candidate() -> None:
    registry = ExperimentRegistry()
    state = state_with_candidate()
    request = ToolRequest(
        experiment_type=ExperimentType.HARMONIC_TEST,
        parameters={
            "candidate_id": "CAND-1",
            "base_period_days": 9.9,
            "factors": [0.5, 1.0, 2.0],
        },
        adaptive=True,
        requested_by="agent",
    )
    review = registry.validate(request, state)
    assert review.status is ReviewStatus.REJECTED
    assert review.reason_code == "CANDIDATE_EPHEMERIS_MISMATCH"
