from __future__ import annotations

import pytest

from exoswarm.domain import (
    DeterministicHypothesisUpdater,
    EvidenceLedger,
    EvidenceState,
    ExperimentType,
    Hypothesis,
    InterpretationCode,
    NumericProvenanceGuard,
    NumericProvenanceViolation,
    ScientificResult,
    ScientificStatus,
    initial_hypotheses,
)


def evidence_result(
    experiment: ExperimentType, interpretation: InterpretationCode
) -> ScientificResult:
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=experiment,
        tool_name="deterministic_test",
        tool_version="1.0",
        numerical_results={
            "duration_hours": 4.2,
            "period_days": 6.26814,
            "snr": 12.8,
        },
        result_units={"duration_hours": "hour", "period_days": "day"},
        interpretation_code=interpretation,
    )


def test_declared_hypothesis_rules_strengthen_eclipsing_binary() -> None:
    ledger = EvidenceLedger()
    item = ledger.append_result(
        evidence_result(ExperimentType.ODD_EVEN, InterpretationCode.INCONSISTENT),
        evidence_id="EV-ODDEVEN",
    )
    updater = DeterministicHypothesisUpdater()
    states, report = updater.apply(initial_hypotheses(), item)
    assert report.applied
    assert states[Hypothesis.ECLIPSING_BINARY].evidence_state is EvidenceState.STRONGLY_SUPPORTED
    assert states[Hypothesis.PLANETARY_TRANSIT].evidence_state is EvidenceState.WEAKENED
    # Applying one ledger item twice is idempotent.
    states_again, second_report = updater.apply(states, item)
    assert not second_report.applied
    assert states_again == states


def test_numeric_provenance_accepts_recorded_values_and_rounding() -> None:
    ledger = EvidenceLedger()
    ledger.append_result(
        evidence_result(ExperimentType.TRANSIT_SEARCH, InterpretationCode.DETECTED),
        evidence_id="EV-BLS",
    )
    guard = NumericProvenanceGuard(ledger)
    report = guard.validate(
        "TARGET-X17: duration 4.2 hours, period 6.268 days, SNR 12.8; test P/2."
    )
    assert report.valid
    assert len(report.claims) == 3
    # Unit conversion is also traced to the deterministic duration measurement.
    assert guard.validate("Duration is 0.175 days.").valid


def test_numeric_provenance_rejects_and_repairs_invented_measurement() -> None:
    ledger = EvidenceLedger()
    ledger.append_result(
        evidence_result(ExperimentType.TRANSIT_SEARCH, InterpretationCode.DETECTED),
        evidence_id="EV-BLS",
    )
    guard = NumericProvenanceGuard(ledger)
    with pytest.raises(NumericProvenanceViolation, match="9.9 hours"):
        guard.enforce("The duration is 9.9 hours.")
    assert guard.repair("The duration is 9.9 hours.") == (
        "The duration is [measurement unavailable]."
    )
