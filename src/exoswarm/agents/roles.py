"""Bounded Skeptic and Critic roles over compact deterministic evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from exoswarm.domain.ledger import EvidenceLedger
from exoswarm.domain.models import (
    CriticDecision,
    CriticVerdict,
    DecisionPriority,
    ExperimentType,
    Hypothesis,
    InvestigationState,
    SkepticAction,
    SkepticDecision,
    ToolRequest,
)

from .structured import StructuredAgentRunner, StructuredCall

SKEPTIC_OBJECTIVE = (
    "Identify the strongest plausible non-planetary explanation still compatible with the "
    "recorded evidence and select exactly one available experiment expected to best discriminate "
    "between it and the planetary interpretation. Prefer an experiment capable of changing the "
    "disposition. Return STOP when no useful unused experiment remains. Expected information "
    "value is an uncalibrated decision-utility estimate, never a planet probability. Copy the "
    "selected experiment's supplied parameter contract; never create candidate identifiers or "
    "ephemerides."
)

CRITIC_OBJECTIVE = (
    "Evaluate whether the proposed adaptive experiment is permitted, unused, and genuinely "
    "discriminating given the supplied Evidence Ledger. Return APPROVE, REVISE, or VETO. A "
    "revision may contain exactly one permitted alternative; do not begin a debate and do not "
    "invent measurements. The reviewed_request_id must exactly copy proposal.request_id."
)


def build_evidence_packet(
    state: InvestigationState,
    ledger: EvidenceLedger,
    *,
    max_evidence_items: int = 24,
) -> dict[str, Any]:
    """Build the only state view sent to a scientific decision model.

    Artifact paths, backend mappings, source identities, raw flux arrays, pixel cubes, and
    external catalog values are deliberately absent.
    """

    view = state.to_agent_view()
    recent = ledger.items[-max_evidence_items:]
    candidate = view.candidates[0].model_dump(mode="json") if view.candidates else None
    experiment_contracts = _available_experiment_contracts(view.available_tests, candidate)
    return {
        "opaque_target_id": view.opaque_target_id,
        "investigation_status": view.status.value,
        "lock_state": view.lock_state.value,
        "current_candidate": candidate,
        "hypotheses": {
            hypothesis.value: {
                "evidence_state": item.evidence_state.value,
                "heuristic_evidence_weight": item.heuristic_evidence_weight,
                "supporting_evidence_ids": item.supporting_evidence_ids,
                "opposing_evidence_ids": item.opposing_evidence_ids,
            }
            for hypothesis, item in view.hypotheses.items()
        },
        "evidence": [
            {
                "id": item.id,
                "experiment_type": item.experiment_type.value,
                "interpretation_code": item.interpretation_code.value,
                "numerical_results": item.numerical_results,
                "result_units": item.result_units,
                "uncertainties": {
                    name: uncertainty.model_dump(mode="json")
                    for name, uncertainty in item.uncertainties.items()
                },
                "quality_flags": [flag.model_dump(mode="json") for flag in item.quality_flags],
                "limitations": item.limitations,
            }
            for item in recent
        ],
        "completed_tests": [item.value for item in view.completed_tests],
        "available_experiments": [item.value for item in view.available_tests],
        "experiment_contracts": experiment_contracts,
        "unresolved_questions": view.unresolved_questions,
        "budgets": {
            "experiments_remaining": view.experiment_budget_remaining,
            "agent_turns_remaining": view.agent_turn_budget_remaining,
        },
    }


def _available_experiment_contracts(
    available: Iterable[ExperimentType],
    candidate: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Expose only safe, bounded parameter contracts for currently permitted actions."""

    candidate_id = str(candidate.get("candidate_id")) if candidate else None
    period_days = candidate.get("period_days") if candidate else None
    contracts: dict[str, dict[str, Any]] = {}
    for experiment in available:
        if experiment is ExperimentType.HARMONIC_TEST and candidate_id and period_days is not None:
            contracts[experiment.value] = {
                "copy_parameters_exactly": {
                    "candidate_id": candidate_id,
                    "base_period_days": period_days,
                    "factors": [0.5, 1.0, 2.0],
                },
                "purpose": "compare the recorded candidate at P/2, P, and 2P",
            }
        elif experiment is ExperimentType.CENTROID_LOCALIZATION and candidate_id:
            contracts[experiment.value] = {
                "copy_parameters_exactly": {
                    "candidate_id": candidate_id,
                    "aperture_id": None,
                    "transit_window_scale": 1.0,
                },
                "purpose": "localize the recorded dimming in target-pixel data",
            }
        elif experiment is ExperimentType.ALTERNATE_DETRENDING and candidate_id:
            contracts[experiment.value] = {
                "required_parameters": {
                    "candidate_id": candidate_id,
                    "method": "savgol",
                    "window_hours": 36.0,
                },
                "allowed_method": ["median_filter", "savgol"],
                "window_hours_bounds": [12.0, 72.0],
                "purpose": "rerun a matched search under one permitted preprocessing choice",
            }
        else:
            contracts[experiment.value] = {"copy_parameters_exactly": {}}
    return contracts


class SkepticAgent:
    def __init__(self, runner: StructuredAgentRunner) -> None:
        self.runner = runner

    def decide(self, packet: dict[str, Any]) -> StructuredCall[SkepticDecision]:
        return self.runner.request(
            role="SKEPTIC",
            objective=SKEPTIC_OBJECTIVE,
            packet=packet,
            response_model=SkepticDecision,
            fallback=_skeptic_fallback,
        )


class CriticAgent:
    def __init__(self, runner: StructuredAgentRunner) -> None:
        self.runner = runner

    def review(
        self,
        packet: dict[str, Any],
        request: ToolRequest,
    ) -> StructuredCall[CriticDecision]:
        critic_packet = {
            **packet,
            "proposal": request.model_dump(mode="json"),
        }
        return self.runner.request(
            role="CRITIC",
            objective=CRITIC_OBJECTIVE,
            packet=critic_packet,
            response_model=CriticDecision,
            fallback=_critic_fallback,
        )


def _skeptic_fallback(packet: dict[str, Any], reason: str) -> SkepticDecision:
    """Safe, declared policy used only when structured model inference is unavailable."""

    available = _available(packet)
    candidate = packet.get("current_candidate") or {}
    candidate_id = str(candidate.get("candidate_id") or "")
    period_days = candidate.get("period_days")

    if ExperimentType.HARMONIC_TEST in available and (
        _has_interpretation(packet, ExperimentType.ODD_EVEN, "INCONSISTENT")
        or _has_interpretation(packet, ExperimentType.SECONDARY_ECLIPSE, "SIGNIFICANT")
    ):
        return SkepticDecision(
            hypothesis_under_test=Hypothesis.ECLIPSING_BINARY,
            requested_experiment=ExperimentType.HARMONIC_TEST,
            parameters={
                "candidate_id": candidate_id,
                "base_period_days": float(period_days),
                "factors": [0.5, 1.0, 2.0],
            },
            reason_code="EB_EVIDENCE_TEST_PERIOD_ALIAS",
            explanation=(
                "Odd/even or secondary-event evidence leaves an eclipsing binary as the "
                "strongest alternative; the declared harmonic comparison is discriminating."
            ),
            expected_discriminating_result=(
                "Determine whether P, P/2, or 2P better separates the repeating dimmings."
            ),
            predicted_outcomes={
                "PREFERRED_DOUBLE_PERIOD": "strengthens the eclipsing-binary/alias explanation",
                "PREFERRED_NOMINAL_PERIOD": "weakens the period-alias explanation",
            },
            expected_information_value=0.90,
            stop_if="the harmonic result resolves the dominant remaining alternative",
            priority=DecisionPriority.HIGH,
        )

    if ExperimentType.CENTROID_LOCALIZATION in available and (
        _has_interpretation(packet, ExperimentType.CONTAMINATION_SCREEN, "NEIGHBOR_DETECTED")
        or _hypothesis_unresolved(packet, Hypothesis.BACKGROUND_CONTAMINANT)
    ):
        return SkepticDecision(
            hypothesis_under_test=Hypothesis.BACKGROUND_CONTAMINANT,
            requested_experiment=ExperimentType.CENTROID_LOCALIZATION,
            parameters={"candidate_id": candidate_id, "transit_window_scale": 1.0},
            reason_code="LOCALIZE_BACKGROUND_ALTERNATIVE",
            explanation=(
                "The background-source explanation remains unresolved; transit-associated "
                "pixel motion is more discriminating than repeating a photometric check."
            ),
            expected_discriminating_result=(
                "Test whether the dimming-associated centroid is spatially consistent with "
                "the target aperture."
            ),
            predicted_outcomes={
                "TARGET_CONSISTENT": "weakens the background-contaminant explanation",
                "OFFSET_DETECTED": "strengthens the background-contaminant explanation",
            },
            expected_information_value=0.85,
            stop_if="spatial evidence resolves the strongest remaining alternative",
            priority=DecisionPriority.HIGH,
        )

    if ExperimentType.ALTERNATE_DETRENDING in available and (
        _has_interpretation(packet, ExperimentType.SIGNAL_QUALITY, "FAIL")
        or _has_quality_flag(packet, {"HIGH_VARIABILITY", "DETRENDING_SENSITIVE"})
    ):
        return SkepticDecision(
            hypothesis_under_test=Hypothesis.INSTRUMENTAL_SYSTEMATIC,
            requested_experiment=ExperimentType.ALTERNATE_DETRENDING,
            parameters={
                "candidate_id": candidate_id,
                "method": "savgol",
                "window_hours": 36.0,
            },
            reason_code="TEST_PREPROCESSING_ROBUSTNESS",
            explanation=(
                "Signal quality leaves variability or preprocessing sensitivity unresolved; "
                "an allowed alternate detrending run can test robustness."
            ),
            expected_discriminating_result=(
                "Determine whether the candidate ephemeris and depth persist under an allowed "
                "alternative preprocessing configuration."
            ),
            predicted_outcomes={
                "ROBUST": "weakens preprocessing/systematic explanations",
                "PREPROCESSING_SENSITIVE": "strengthens variability/systematic explanations",
            },
            expected_information_value=0.70,
            stop_if="the candidate is shown to be preprocessing-sensitive",
            priority=DecisionPriority.MEDIUM,
        )

    return SkepticDecision(
        action=SkepticAction.STOP,
        reason_code="NO_UNUSED_DISCRIMINATING_EXPERIMENT",
        explanation=f"No permitted unused experiment adds decisive evidence. Fallback reason: {reason}",
        expected_discriminating_result="No additional deterministic result is requested.",
        predicted_outcomes={},
        expected_information_value=0.0,
        stop_if="mandatory diagnostics are complete",
        priority=DecisionPriority.LOW,
    )


def _critic_fallback(packet: dict[str, Any], reason: str) -> CriticDecision:
    proposal = ToolRequest.model_validate(packet["proposal"])
    completed = {ExperimentType(value) for value in packet.get("completed_tests", [])}
    available = _available(packet)
    experiment = proposal.experiment_type

    if experiment in completed:
        return CriticDecision(
            reviewed_request_id=proposal.request_id,
            verdict=CriticVerdict.VETO,
            reason_code="REDUNDANT_WITH_LEDGER",
            reason="The proposed experiment already has deterministic evidence in the ledger.",
        )
    if experiment not in available:
        return CriticDecision(
            reviewed_request_id=proposal.request_id,
            verdict=CriticVerdict.VETO,
            reason_code="EXPERIMENT_NOT_AVAILABLE",
            reason="The proposal is outside the currently permitted experiment surface.",
        )

    discriminating = {
        ExperimentType.HARMONIC_TEST: (
            _has_interpretation(packet, ExperimentType.ODD_EVEN, "INCONSISTENT")
            or _has_interpretation(packet, ExperimentType.SECONDARY_ECLIPSE, "SIGNIFICANT")
        ),
        ExperimentType.CENTROID_LOCALIZATION: (
            _has_interpretation(packet, ExperimentType.CONTAMINATION_SCREEN, "NEIGHBOR_DETECTED")
            or _hypothesis_unresolved(packet, Hypothesis.BACKGROUND_CONTAMINANT)
        ),
        ExperimentType.ALTERNATE_DETRENDING: (
            _has_interpretation(packet, ExperimentType.SIGNAL_QUALITY, "FAIL")
            or _has_quality_flag(packet, {"HIGH_VARIABILITY", "DETRENDING_SENSITIVE"})
        ),
        ExperimentType.ALTERNATE_APERTURE: _has_interpretation(
            packet, ExperimentType.CONTAMINATION_SCREEN, "NEIGHBOR_DETECTED"
        ),
    }.get(experiment, False)
    if not discriminating:
        alternative = _best_alternative_request(packet, proposal)
        if alternative is not None:
            return CriticDecision(
                reviewed_request_id=proposal.request_id,
                verdict=CriticVerdict.REVISE,
                reason_code="MORE_DISCRIMINATING_ALTERNATIVE",
                reason=(
                    "The proposal is permitted but weakly connected to the current evidence; "
                    "one more discriminating alternative is supplied."
                ),
                revised_request=alternative,
            )
        return CriticDecision(
            reviewed_request_id=proposal.request_id,
            verdict=CriticVerdict.VETO,
            reason_code="LOW_DISCRIMINATING_VALUE",
            reason=f"No current evidence makes this proposal discriminating. Fallback: {reason}",
        )

    return CriticDecision(
        reviewed_request_id=proposal.request_id,
        verdict=CriticVerdict.APPROVE,
        reason_code="DISCRIMINATING_AND_NONREDUNDANT",
        reason=(
            "The experiment targets a live alternative, is available, and has no existing "
            "deterministic result in the ledger."
        ),
    )


def _best_alternative_request(packet: dict[str, Any], original: ToolRequest) -> ToolRequest | None:
    available = _available(packet)
    candidate = packet.get("current_candidate") or {}
    candidate_id = str(candidate.get("candidate_id") or "")
    period = candidate.get("period_days")
    if (
        ExperimentType.HARMONIC_TEST in available
        and ExperimentType.HARMONIC_TEST is not original.experiment_type
        and (
            _has_interpretation(packet, ExperimentType.ODD_EVEN, "INCONSISTENT")
            or _has_interpretation(packet, ExperimentType.SECONDARY_ECLIPSE, "SIGNIFICANT")
        )
    ):
        return ToolRequest(
            experiment_type=ExperimentType.HARMONIC_TEST,
            parameters={
                "candidate_id": candidate_id,
                "base_period_days": float(period),
                "factors": [0.5, 1.0, 2.0],
            },
            adaptive=True,
            requested_by="critic-revision",
            justification="Resolve EB/period-alias evidence instead of repeating a weak test.",
            agent_decision_id=original.agent_decision_id,
        )
    if (
        ExperimentType.CENTROID_LOCALIZATION in available
        and ExperimentType.CENTROID_LOCALIZATION is not original.experiment_type
        and _hypothesis_unresolved(packet, Hypothesis.BACKGROUND_CONTAMINANT)
    ):
        return ToolRequest(
            experiment_type=ExperimentType.CENTROID_LOCALIZATION,
            parameters={"candidate_id": candidate_id, "transit_window_scale": 1.0},
            adaptive=True,
            requested_by="critic-revision",
            justification="Spatial localization addresses the unresolved contaminant hypothesis.",
            agent_decision_id=original.agent_decision_id,
        )
    return None


def _available(packet: dict[str, Any]) -> set[ExperimentType]:
    result: set[ExperimentType] = set()
    for value in packet.get("available_experiments", []):
        try:
            result.add(ExperimentType(value))
        except ValueError:
            continue
    return result


def _evidence(packet: dict[str, Any]) -> Iterable[dict[str, Any]]:
    items = packet.get("evidence", [])
    return (item for item in items if isinstance(item, dict))


def _has_interpretation(
    packet: dict[str, Any], experiment: ExperimentType, interpretation: str
) -> bool:
    return any(
        item.get("experiment_type") == experiment.value
        and item.get("interpretation_code") == interpretation
        for item in _evidence(packet)
    )


def _has_quality_flag(packet: dict[str, Any], codes: set[str]) -> bool:
    return any(
        isinstance(flag, dict) and flag.get("code") in codes
        for item in _evidence(packet)
        for flag in item.get("quality_flags", [])
    )


def _hypothesis_unresolved(packet: dict[str, Any], hypothesis: Hypothesis) -> bool:
    value = packet.get("hypotheses", {}).get(hypothesis.value, {})
    return value.get("evidence_state") not in {"WEAKENED", "DISFAVORED"}


__all__ = [
    "CRITIC_OBJECTIVE",
    "SKEPTIC_OBJECTIVE",
    "CriticAgent",
    "SkepticAgent",
    "build_evidence_packet",
]
