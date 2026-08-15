from __future__ import annotations

from exoswarm.agents.guardrails import sanitize_critic_decision, sanitize_skeptic_decision
from exoswarm.domain.ledger import EvidenceLedger
from exoswarm.domain.models import (
    CriticDecision,
    CriticVerdict,
    DecisionPriority,
    ExperimentType,
    Hypothesis,
    InterpretationCode,
    ScientificResult,
    ScientificStatus,
    SkepticDecision,
)


def _ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.append_result(
        ScientificResult(
            status=ScientificStatus.SUCCESS,
            experiment_type=ExperimentType.TRANSIT_SEARCH,
            tool_name="deterministic_bls",
            tool_version="test",
            parameters={"durations_hours": [1.0, 2.0, 4.0, 6.0]},
            numerical_results={"duration_hours": 2.0},
            result_units={"duration_hours": "h"},
            interpretation_code=InterpretationCode.DETECTED,
        )
    )
    return ledger


def test_skeptic_prose_repairs_unsupported_number_and_identity() -> None:
    decision = SkepticDecision(
        hypothesis_under_test=Hypothesis.ECLIPSING_BINARY,
        requested_experiment=ExperimentType.HARMONIC_TEST,
        parameters={},
        reason_code="TIC123",
        explanation="TIC 123 has a 4.2 hour event.",
        expected_discriminating_result="Compare the recorded 2.0 hour morphology.",
        predicted_outcomes={"TOI905": "No unsupported values."},
        expected_information_value=0.5,
        priority=DecisionPriority.MEDIUM,
    )
    repaired, report = sanitize_skeptic_decision(decision, _ledger())
    assert "4.2" not in repaired.explanation
    assert "TIC 123" not in repaired.explanation
    assert repaired.reason_code == "IDENTITY_WITHHELD"
    assert "TOI905" not in repaired.predicted_outcomes
    assert "2.0 hour" in repaired.expected_discriminating_result
    assert report["changed"]


def test_critic_prose_allows_ledger_number() -> None:
    decision = CriticDecision(
        reviewed_request_id="REQ-1",
        verdict=CriticVerdict.APPROVE,
        reason_code="SUPPORTED",
        reason="The recorded duration is 2.0 hours.",
    )
    repaired, report = sanitize_critic_decision(decision, _ledger())
    assert repaired.reason == decision.reason
    assert not report["changed"]


def test_critic_reason_code_cannot_guess_target_identity() -> None:
    decision = CriticDecision(
        reviewed_request_id="REQ-1",
        verdict=CriticVerdict.APPROVE,
        reason_code="TOI905",
        reason="The recorded duration is 2.0 hours.",
    )
    repaired, report = sanitize_critic_decision(decision, _ledger())
    assert repaired.reason_code == "IDENTITY_WITHHELD"
    assert report["changed"]


def test_search_grid_parameter_cannot_masquerade_as_measurement() -> None:
    decision = CriticDecision(
        reviewed_request_id="REQ-1",
        verdict=CriticVerdict.APPROVE,
        reason_code="UNSUPPORTED_PARAMETER_AS_RESULT",
        reason="The measured duration is 4.0 hours.",
    )
    repaired, report = sanitize_critic_decision(decision, _ledger())
    assert "4.0" not in repaired.reason
    assert report["changed"]
