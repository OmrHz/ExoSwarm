from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from exoswarm.domain import (
    BackendTargetMappingRef,
    Budget,
    CriticDecision,
    CriticVerdict,
    ExperimentType,
    Hypothesis,
    InterpretationCode,
    InvestigationState,
    LockState,
    ScientificDisposition,
    ScientificFailure,
    ScientificResult,
    ScientificStatus,
    ScientificToolResponse,
    SkepticAction,
    SkepticDecision,
)


def test_scientific_tool_response_discriminated_union() -> None:
    adapter = TypeAdapter(ScientificToolResponse)
    result = adapter.validate_python(
        {
            "status": "SUCCESS",
            "experiment_type": "transit_search",
            "tool_name": "bls",
            "tool_version": "1.0",
            "numerical_results": {"period_days": 6.2},
            "result_units": {"period_days": "day"},
            "interpretation_code": "DETECTED",
        }
    )
    assert isinstance(result, ScientificResult)

    failure = adapter.validate_python(
        {
            "status": "PRECONDITION_FAILED",
            "experiment_type": "odd_even",
            "tool_name": "odd-even",
            "tool_version": "1.0",
            "reason": "requires >=4 events; found 3",
            "reason_code": "INSUFFICIENT_EVENTS",
            "suggested_alternatives": ["secondary_eclipse", "harmonic_test"],
            "interpretation_code": "PRECONDITION_NOT_MET",
        }
    )
    assert isinstance(failure, ScientificFailure)


def test_scientific_results_reject_unlinked_uncertainty() -> None:
    with pytest.raises(ValidationError, match="unknown numerical results"):
        ScientificResult(
            status=ScientificStatus.SUCCESS,
            experiment_type=ExperimentType.TRANSIT_SEARCH,
            tool_name="bls",
            tool_version="1",
            numerical_results={"period_days": 2.0},
            uncertainties={"depth_ppm": {"value": 10, "unit": "ppm", "method": "bootstrap"}},
            interpretation_code=InterpretationCode.DETECTED,
        )


def test_agent_view_excludes_backend_mapping_and_artifact_paths() -> None:
    state = InvestigationState(
        opaque_target_id="TARGET-X17",
        backend_target_mapping=BackendTargetMappingRef(mapping_key="a" * 24),
        available_data_products=[
            {
                "artifact_id": "LC-1",
                "path": "private/target.fits",
                "source_uri": "https://private.example/record",
            }
        ],
    )
    serialized = state.to_agent_view().model_dump_json()
    assert "mapping" not in serialized.casefold()
    assert "private" not in serialized.casefold()
    assert "source_uri" not in serialized
    assert json.loads(serialized)["opaque_target_id"] == "TARGET-X17"


def test_budget_and_lock_transitions_are_bounded() -> None:
    assert Budget(limit=2).consume().remaining == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        Budget(limit=1, used=1).consume()

    state = InvestigationState(
        opaque_target_id="TARGET-X17",
        backend_target_mapping=BackendTargetMappingRef(mapping_key="b" * 24),
        final_disposition=ScientificDisposition.INCONCLUSIVE,
    )
    state.transition_lock(LockState.READY_TO_LOCK)
    state.transition_lock(LockState.RESULT_LOCKED)
    with pytest.raises(ValueError, match="invalid lock transition"):
        state.transition_lock(LockState.UNLOCKED)


def test_skeptic_and_critic_structured_contracts() -> None:
    decision = SkepticDecision(
        hypothesis_under_test=Hypothesis.BACKGROUND_CONTAMINANT,
        requested_experiment=ExperimentType.CENTROID_LOCALIZATION,
        parameters={"candidate_id": "CAND-1"},
        reason_code="NEARBY_SOURCE_IN_APERTURE",
        explanation="Spatial evidence can distinguish the remaining contaminant hypothesis.",
        expected_discriminating_result="Test whether transit-associated motion is offset.",
        predicted_outcomes={"TARGET_CONSISTENT": "weakens contamination"},
        expected_information_value=0.8,
    )
    assert decision.action is SkepticAction.REQUEST_EXPERIMENT

    with pytest.raises(ValidationError, match="STOP decisions"):
        SkepticDecision(
            action=SkepticAction.STOP,
            requested_experiment=ExperimentType.HARMONIC_TEST,
            reason_code="ENOUGH_EVIDENCE",
            explanation="Stop.",
            expected_discriminating_result="No further result needed.",
            expected_information_value=0,
        )

    with pytest.raises(ValidationError, match="REVISE requires"):
        CriticDecision(
            reviewed_request_id="REQ-1",
            verdict=CriticVerdict.REVISE,
            reason_code="BETTER_TEST_AVAILABLE",
            reason="Use the spatial diagnostic.",
        )
