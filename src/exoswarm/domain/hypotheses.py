"""Declared deterministic evidence-to-hypothesis rules.

The numeric values here are auditable heuristic weights.  They are deliberately
not normalized and must never be presented as planet probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from .ledger import EvidenceItem
from .models import (
    EvidenceState,
    ExperimentType,
    FrozenDomainModel,
    Hypothesis,
    HypothesisState,
    InterpretationCode,
    InvestigationState,
    ScientificDisposition,
)


class HypothesisUpdate(FrozenDomainModel):
    hypothesis: Hypothesis
    delta: float
    previous_weight: float
    updated_weight: float
    previous_state: EvidenceState
    updated_state: EvidenceState
    evidence_id: str


class HypothesisUpdateReport(FrozenDomainModel):
    evidence_id: str
    rule_key: str
    applied: bool
    updates: list[HypothesisUpdate] = Field(default_factory=list)
    note: str | None = None


RuleKey = tuple[ExperimentType, InterpretationCode]


# A compact, declared table.  Values express only direction/relative strength of
# implemented evidence. They have no calibrated probabilistic interpretation.
DEFAULT_EVIDENCE_RULES: Mapping[RuleKey, Mapping[Hypothesis, float]] = {
    (ExperimentType.TRANSIT_SEARCH, InterpretationCode.DETECTED): {
        Hypothesis.PLANETARY_TRANSIT: 0.75,
        Hypothesis.ECLIPSING_BINARY: 0.50,
        Hypothesis.BACKGROUND_CONTAMINANT: 0.25,
    },
    (ExperimentType.TRANSIT_SEARCH, InterpretationCode.NOT_DETECTED): {
        Hypothesis.PLANETARY_TRANSIT: -2.0,
        Hypothesis.ECLIPSING_BINARY: -1.0,
        Hypothesis.PERIOD_ALIAS_HARMONIC: -0.5,
    },
    (ExperimentType.SIGNAL_QUALITY, InterpretationCode.PASS): {
        Hypothesis.PLANETARY_TRANSIT: 0.50,
        Hypothesis.STELLAR_VARIABILITY: -0.25,
        Hypothesis.INSTRUMENTAL_SYSTEMATIC: -0.50,
    },
    (ExperimentType.SIGNAL_QUALITY, InterpretationCode.FAIL): {
        Hypothesis.PLANETARY_TRANSIT: -1.0,
        Hypothesis.STELLAR_VARIABILITY: 0.50,
        Hypothesis.INSTRUMENTAL_SYSTEMATIC: 0.75,
    },
    (ExperimentType.ODD_EVEN, InterpretationCode.CONSISTENT): {
        Hypothesis.PLANETARY_TRANSIT: 1.0,
        Hypothesis.ECLIPSING_BINARY: -0.75,
        Hypothesis.PERIOD_ALIAS_HARMONIC: -0.25,
    },
    (ExperimentType.ODD_EVEN, InterpretationCode.INCONSISTENT): {
        Hypothesis.PLANETARY_TRANSIT: -1.5,
        Hypothesis.ECLIPSING_BINARY: 2.0,
        Hypothesis.PERIOD_ALIAS_HARMONIC: 1.0,
    },
    (ExperimentType.SECONDARY_ECLIPSE, InterpretationCode.NOT_SIGNIFICANT): {
        Hypothesis.PLANETARY_TRANSIT: 0.75,
        Hypothesis.ECLIPSING_BINARY: -0.75,
    },
    (ExperimentType.SECONDARY_ECLIPSE, InterpretationCode.SIGNIFICANT): {
        Hypothesis.PLANETARY_TRANSIT: -1.5,
        Hypothesis.ECLIPSING_BINARY: 2.0,
    },
    (ExperimentType.CONTAMINATION_SCREEN, InterpretationCode.NEIGHBOR_DETECTED): {
        Hypothesis.BACKGROUND_CONTAMINANT: 1.0,
    },
    (ExperimentType.CONTAMINATION_SCREEN, InterpretationCode.NO_NEARBY_SOURCE): {
        Hypothesis.BACKGROUND_CONTAMINANT: -0.5,
    },
    (ExperimentType.CENTROID_LOCALIZATION, InterpretationCode.TARGET_CONSISTENT): {
        Hypothesis.PLANETARY_TRANSIT: 0.75,
        Hypothesis.BACKGROUND_CONTAMINANT: -1.5,
    },
    (ExperimentType.CENTROID_LOCALIZATION, InterpretationCode.OFFSET_DETECTED): {
        Hypothesis.PLANETARY_TRANSIT: -1.5,
        Hypothesis.BACKGROUND_CONTAMINANT: 2.5,
    },
    (ExperimentType.HARMONIC_TEST, InterpretationCode.PREFERRED_NOMINAL_PERIOD): {
        Hypothesis.PLANETARY_TRANSIT: 0.5,
        Hypothesis.PERIOD_ALIAS_HARMONIC: -1.5,
    },
    (ExperimentType.HARMONIC_TEST, InterpretationCode.PREFERRED_HALF_PERIOD): {
        Hypothesis.PLANETARY_TRANSIT: -0.75,
        Hypothesis.ECLIPSING_BINARY: 0.75,
        Hypothesis.PERIOD_ALIAS_HARMONIC: 2.0,
    },
    (ExperimentType.HARMONIC_TEST, InterpretationCode.PREFERRED_DOUBLE_PERIOD): {
        Hypothesis.PLANETARY_TRANSIT: -0.75,
        Hypothesis.ECLIPSING_BINARY: 1.0,
        Hypothesis.PERIOD_ALIAS_HARMONIC: 2.0,
    },
    (ExperimentType.ALTERNATE_DETRENDING, InterpretationCode.ROBUST): {
        Hypothesis.PLANETARY_TRANSIT: 0.75,
        Hypothesis.STELLAR_VARIABILITY: -0.5,
        Hypothesis.INSTRUMENTAL_SYSTEMATIC: -0.5,
    },
    (
        ExperimentType.ALTERNATE_DETRENDING,
        InterpretationCode.PREPROCESSING_SENSITIVE,
    ): {
        Hypothesis.PLANETARY_TRANSIT: -1.0,
        Hypothesis.STELLAR_VARIABILITY: 0.75,
        Hypothesis.INSTRUMENTAL_SYSTEMATIC: 1.0,
    },
}


def evidence_state_for_weight(weight: float) -> EvidenceState:
    if weight >= 2.0:
        return EvidenceState.STRONGLY_SUPPORTED
    if weight >= 0.75:
        return EvidenceState.SUPPORTED
    if weight > -0.75:
        return EvidenceState.UNRESOLVED
    if weight > -2.0:
        return EvidenceState.WEAKENED
    return EvidenceState.DISFAVORED


class DeterministicHypothesisUpdater:
    def __init__(
        self,
        rules: Mapping[RuleKey, Mapping[Hypothesis, float]] | None = None,
    ) -> None:
        self._rules = dict(rules or DEFAULT_EVIDENCE_RULES)

    def apply(
        self,
        hypotheses: Mapping[Hypothesis, HypothesisState],
        evidence: EvidenceItem,
    ) -> tuple[dict[Hypothesis, HypothesisState], HypothesisUpdateReport]:
        key = (evidence.experiment_type, evidence.interpretation_code)
        rule_key = f"{key[0].value}:{key[1].value}"
        existing_ids = {
            evidence_id
            for state in hypotheses.values()
            for evidence_id in (
                *state.supporting_evidence_ids,
                *state.opposing_evidence_ids,
            )
        }
        if evidence.id in existing_ids:
            return dict(hypotheses), HypothesisUpdateReport(
                evidence_id=evidence.id,
                rule_key=rule_key,
                applied=False,
                note="evidence was already applied",
            )

        weights = self._rules.get(key)
        if weights is None:
            return dict(hypotheses), HypothesisUpdateReport(
                evidence_id=evidence.id,
                rule_key=rule_key,
                applied=False,
                note="no declared update rule for this result",
            )

        updated = dict(hypotheses)
        changes: list[HypothesisUpdate] = []
        for hypothesis, delta in weights.items():
            prior = updated.get(hypothesis, HypothesisState(hypothesis=hypothesis))
            next_weight = prior.heuristic_evidence_weight + delta
            next_state = evidence_state_for_weight(next_weight)
            supporting = list(prior.supporting_evidence_ids)
            opposing = list(prior.opposing_evidence_ids)
            (supporting if delta > 0 else opposing).append(evidence.id)
            updated[hypothesis] = prior.model_copy(
                update={
                    "heuristic_evidence_weight": next_weight,
                    "evidence_state": next_state,
                    "supporting_evidence_ids": supporting,
                    "opposing_evidence_ids": opposing,
                }
            )
            changes.append(
                HypothesisUpdate(
                    hypothesis=hypothesis,
                    delta=delta,
                    previous_weight=prior.heuristic_evidence_weight,
                    updated_weight=next_weight,
                    previous_state=prior.evidence_state,
                    updated_state=next_state,
                    evidence_id=evidence.id,
                )
            )
        return updated, HypothesisUpdateReport(
            evidence_id=evidence.id,
            rule_key=rule_key,
            applied=True,
            updates=changes,
        )

    def apply_to_investigation(
        self, state: InvestigationState, evidence: EvidenceItem
    ) -> HypothesisUpdateReport:
        hypotheses, report = self.apply(state.hypotheses, evidence)
        if report.applied:
            state.hypotheses = hypotheses
        if evidence.id not in state.evidence:
            state.evidence = [*state.evidence, evidence.id]
        return report


MANDATORY_VETTING = frozenset(
    {
        ExperimentType.SIGNAL_QUALITY,
        ExperimentType.ODD_EVEN,
        ExperimentType.SECONDARY_ECLIPSE,
        ExperimentType.CONTAMINATION_SCREEN,
    }
)


def derive_disposition(state: InvestigationState) -> ScientificDisposition:
    """Derive a conservative categorical disposition from recorded state."""

    if not state.candidates:
        if ExperimentType.TRANSIT_SEARCH in state.completed_tests:
            return ScientificDisposition.NO_CREDIBLE_PERIODIC_SIGNAL
        return ScientificDisposition.INCONCLUSIVE

    planet = state.hypotheses[Hypothesis.PLANETARY_TRANSIT]
    alternatives = [
        hypothesis_state
        for hypothesis, hypothesis_state in state.hypotheses.items()
        if hypothesis is not Hypothesis.PLANETARY_TRANSIT
    ]
    strongest_alternative = max(
        (item.heuristic_evidence_weight for item in alternatives), default=0
    )
    if (
        planet.evidence_state in {EvidenceState.WEAKENED, EvidenceState.DISFAVORED}
        or strongest_alternative >= 2.0
    ):
        return ScientificDisposition.PLANETARY_INTERPRETATION_WEAK

    missing_mandatory = MANDATORY_VETTING - set(state.completed_tests)
    if missing_mandatory:
        if state.experiment_budget.remaining == 0:
            return ScientificDisposition.INCONCLUSIVE
        return ScientificDisposition.TRANSIT_LIKE_SIGNAL

    if planet.heuristic_evidence_weight >= 2.0 and strongest_alternative < 0.75:
        return ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING
    if planet.heuristic_evidence_weight >= 0.75 and strongest_alternative < 2.0:
        return ScientificDisposition.PLANETARY_INTERPRETATION_PLAUSIBLE
    return ScientificDisposition.TRANSIT_LIKE_SIGNAL
