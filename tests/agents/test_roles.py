from __future__ import annotations

import json

from exoswarm.agents.provider import UnavailableProvider
from exoswarm.agents.roles import CriticAgent, SkepticAgent, build_evidence_packet
from exoswarm.agents.structured import StructuredAgentRunner
from exoswarm.domain.ledger import EvidenceLedger
from exoswarm.domain.models import (
    BackendTargetMappingRef,
    Candidate,
    CriticVerdict,
    ExperimentType,
    InterpretationCode,
    InvestigationState,
    ScientificResult,
    ScientificStatus,
    ToolRequest,
)


def _state(*available: ExperimentType) -> InvestigationState:
    return InvestigationState(
        opaque_target_id="TARGET-X17",
        backend_target_mapping=BackendTargetMappingRef(mapping_key="opaque-handle-0001"),
        candidates=[
            Candidate(
                candidate_id="CAND-TEST",
                period_days=3.25,
                epoch_btjd=1450.5,
                transit_depth_ppm=1_200,
                duration_hours=2.2,
                signal_to_noise=11.0,
                observed_events=6,
            )
        ],
        available_tests=list(available),
    )


def _append(
    ledger: EvidenceLedger,
    experiment: ExperimentType,
    interpretation: InterpretationCode,
) -> None:
    ledger.append_result(
        ScientificResult(
            status=ScientificStatus.SUCCESS,
            experiment_type=experiment,
            tool_name=f"test_{experiment.value}",
            tool_version="test",
            interpretation_code=interpretation,
            limitations=["Synthetic unit-test evidence; not a scientific result."],
        )
    )


def _runner() -> StructuredAgentRunner:
    return StructuredAgentRunner(UnavailableProvider("unit test"))


def test_skeptic_path_changes_for_eb_evidence() -> None:
    state = _state(ExperimentType.HARMONIC_TEST, ExperimentType.CENTROID_LOCALIZATION)
    ledger = EvidenceLedger()
    _append(ledger, ExperimentType.ODD_EVEN, InterpretationCode.INCONSISTENT)
    packet = build_evidence_packet(state, ledger)

    decision = SkepticAgent(_runner()).decide(packet).value
    assert decision.requested_experiment is ExperimentType.HARMONIC_TEST
    assert decision.reason_code == "EB_EVIDENCE_TEST_PERIOD_ALIAS"


def test_skeptic_path_changes_for_contamination_evidence() -> None:
    state = _state(ExperimentType.HARMONIC_TEST, ExperimentType.CENTROID_LOCALIZATION)
    ledger = EvidenceLedger()
    _append(
        ledger,
        ExperimentType.CONTAMINATION_SCREEN,
        InterpretationCode.NEIGHBOR_DETECTED,
    )
    packet = build_evidence_packet(state, ledger)

    decision = SkepticAgent(_runner()).decide(packet).value
    assert decision.requested_experiment is ExperimentType.CENTROID_LOCALIZATION
    assert decision.reason_code == "LOCALIZE_BACKGROUND_ALTERNATIVE"


def test_critic_approves_a_nonredundant_discriminating_request() -> None:
    state = _state(ExperimentType.HARMONIC_TEST)
    ledger = EvidenceLedger()
    _append(ledger, ExperimentType.ODD_EVEN, InterpretationCode.INCONSISTENT)
    packet = build_evidence_packet(state, ledger)
    request = ToolRequest(
        experiment_type=ExperimentType.HARMONIC_TEST,
        parameters={
            "candidate_id": "CAND-TEST",
            "base_period_days": 3.25,
            "factors": [0.5, 1.0, 2.0],
        },
        adaptive=True,
        requested_by="skeptic",
    )

    verdict = CriticAgent(_runner()).review(packet, request).value
    assert verdict.verdict is CriticVerdict.APPROVE
    assert verdict.reviewed_request_id == request.request_id


def test_critic_vetoes_a_redundant_request() -> None:
    state = _state(ExperimentType.HARMONIC_TEST)
    state.completed_tests = [ExperimentType.HARMONIC_TEST]
    packet = build_evidence_packet(state, EvidenceLedger())
    request = ToolRequest(
        experiment_type=ExperimentType.HARMONIC_TEST,
        parameters={
            "candidate_id": "CAND-TEST",
            "base_period_days": 3.25,
            "factors": [0.5, 1.0, 2.0],
        },
        adaptive=True,
        requested_by="skeptic",
    )
    verdict = CriticAgent(_runner()).review(packet, request).value
    assert verdict.verdict is CriticVerdict.VETO
    assert verdict.reason_code == "REDUNDANT_WITH_LEDGER"


def test_agent_packet_contains_no_backend_mapping_identity_or_raw_arrays() -> None:
    packet = build_evidence_packet(_state(ExperimentType.HARMONIC_TEST), EvidenceLedger())
    serialized = json.dumps(packet).lower()
    assert "backend_target_mapping" not in serialized
    assert "opaque-handle" not in serialized
    assert "tic_id" not in serialized
    assert "real_target" not in serialized
    assert "raw_flux" not in serialized
    assert packet["opaque_target_id"] == "TARGET-X17"


def test_agent_packet_exposes_bounded_parameter_contracts() -> None:
    packet = build_evidence_packet(
        _state(
            ExperimentType.HARMONIC_TEST,
            ExperimentType.CENTROID_LOCALIZATION,
            ExperimentType.ALTERNATE_DETRENDING,
        ),
        EvidenceLedger(),
    )
    harmonic = packet["experiment_contracts"][ExperimentType.HARMONIC_TEST.value]
    assert harmonic["copy_parameters_exactly"] == {
        "candidate_id": "CAND-TEST",
        "base_period_days": 3.25,
        "factors": [0.5, 1.0, 2.0],
    }
    centroid = packet["experiment_contracts"][ExperimentType.CENTROID_LOCALIZATION.value]
    assert centroid["copy_parameters_exactly"]["candidate_id"] == "CAND-TEST"
    assert "backend_target_mapping" not in json.dumps(packet)
