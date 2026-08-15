"""Evidence-backed view models for the mission-control interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from exoswarm.domain.ledger import EvidenceItem
from exoswarm.domain.models import Hypothesis
from exoswarm.domain.numeric_provenance import NumericProvenanceGuard
from exoswarm.domain.trace import TraceEvent, TraceEventType

from .artifacts import MissionControlRun

HYPOTHESIS_LABELS = {
    Hypothesis.PLANETARY_TRANSIT.value: "Planetary transit",
    Hypothesis.ECLIPSING_BINARY.value: "Eclipsing binary",
    Hypothesis.BACKGROUND_CONTAMINANT.value: "Background contaminant",
    Hypothesis.STELLAR_VARIABILITY.value: "Stellar variability",
    Hypothesis.INSTRUMENTAL_SYSTEMATIC.value: "Instrument / systematic",
    Hypothesis.PERIOD_ALIAS_HARMONIC.value: "Period alias / harmonic",
}

CANDIDATE_FIELDS = (
    ("period_days", "Period", "d"),
    ("epoch_btjd", "Transit epoch", "BTJD"),
    ("transit_depth_ppm", "Transit depth", "ppm"),
    ("duration_hours", "Duration", "h"),
    ("signal_to_noise", "Signal quality", "S/N"),
    ("observed_events", "Observed events", "count"),
)

HARMONIC_MEASUREMENT_ALIASES = {
    "period_days": "preferred_period_days",
    "epoch_btjd": "preferred_epoch_btjd",
    "transit_depth_ppm": "preferred_primary_depth_ppm",
    "duration_hours": "preferred_duration_hours",
    "signal_to_noise": "preferred_signal_to_noise",
    "observed_events": "preferred_observed_events",
}


@dataclass(frozen=True, slots=True)
class MeasurementVM:
    key: str
    label: str
    value: int | float
    unit: str
    uncertainty: float | None
    uncertainty_unit: str | None
    uncertainty_kind: str | None
    uncertainty_method: str | None
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HypothesisVM:
    code: str
    label: str
    state: str
    weight: float
    supporting_evidence_ids: tuple[str, ...]
    opposing_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkepticVM:
    decision_id: str
    action: str
    hypothesis_code: str | None
    hypothesis_label: str
    experiment: str | None
    reason_code: str
    explanation: str
    expected_result: str
    predicted_outcomes: tuple[tuple[str, str], ...]
    decision_utility: float
    priority: str
    decision_source: str
    provider: str
    model: str
    provider_request_ids: tuple[str, ...]
    attempts: int
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class CriticVM:
    decision_id: str
    verdict: str
    reason_code: str
    reason: str
    reviewed_request_id: str
    decision_source: str
    provider: str
    model: str
    provider_request_ids: tuple[str, ...]
    attempts: int
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class CatalogMeasurementVM:
    name: str
    value: float
    unit: str
    uncertainty: float | None
    source_field: str | None
    source_id: str = "reveal.json"


def candidate_measurements(run: MissionControlRun) -> tuple[MeasurementVM, ...]:
    if run.result is not None and run.result.candidate is None:
        return ()
    if run.result is None:
        search_evidence = next(
            (
                item
                for item in reversed(run.evidence)
                if item.experiment_type.value == "transit_search"
                and item.interpretation_code.value == "DETECTED"
            ),
            None,
        )
        if search_evidence is None:
            return ()
        return _measurements_from_evidence(search_evidence)

    candidate = run.result.candidate
    assert candidate is not None
    evidence_by_id = run.evidence_by_id()
    measurements: list[MeasurementVM] = []
    for key, label, fallback_unit in CANDIDATE_FIELDS:
        value = getattr(candidate, key)
        uncertainty = candidate.uncertainties.get(key)
        evidence_sources = tuple(
            evidence_id
            for evidence_id in candidate.source_evidence_ids
            if _evidence_matches(evidence_by_id.get(evidence_id), key, value)
        )
        if not evidence_sources:
            evidence_sources = tuple(
                item.id for item in run.evidence if _evidence_matches(item, key, value)
            )
        measurements.append(
            MeasurementVM(
                key=key,
                label=label,
                value=value,
                unit=fallback_unit,
                uncertainty=uncertainty.value if uncertainty else None,
                uncertainty_unit=uncertainty.unit if uncertainty else None,
                uncertainty_kind=uncertainty.kind if uncertainty else None,
                uncertainty_method=uncertainty.method if uncertainty else None,
                source_ids=("result.json", *evidence_sources),
            )
        )
    return tuple(measurements)


def _measurements_from_evidence(item: EvidenceItem) -> tuple[MeasurementVM, ...]:
    required = {key for key, _, _ in CANDIDATE_FIELDS}
    if not required <= item.numerical_results.keys():
        return ()
    measurements: list[MeasurementVM] = []
    for key, label, fallback_unit in CANDIDATE_FIELDS:
        value = item.numerical_results[key]
        uncertainty = item.uncertainties.get(key)
        measurements.append(
            MeasurementVM(
                key=key,
                label=label,
                value=value,
                unit=item.result_units.get(key, fallback_unit),
                uncertainty=uncertainty.value if uncertainty else None,
                uncertainty_unit=uncertainty.unit if uncertainty else None,
                uncertainty_kind=uncertainty.kind if uncertainty else None,
                uncertainty_method=uncertainty.method if uncertainty else None,
                source_ids=(item.id,),
            )
        )
    return tuple(measurements)


def hypothesis_views(trace: tuple[TraceEvent, ...]) -> tuple[HypothesisVM, ...]:
    values: dict[str, dict[str, Any]] = {
        code: {
            "state": "UNRESOLVED",
            "weight": 0.0,
            "supporting": [],
            "opposing": [],
        }
        for code in HYPOTHESIS_LABELS
    }
    for event in trace:
        if event.event_type is not TraceEventType.HYPOTHESIS_UPDATED:
            continue
        report = event.payload.get("report")
        if not isinstance(report, dict):
            continue
        updates = report.get("updates", [])
        if not isinstance(updates, list):
            continue
        for update in updates:
            if not isinstance(update, dict):
                continue
            code = str(update.get("hypothesis") or "")
            if code not in values:
                continue
            state = values[code]
            state["state"] = str(update.get("updated_state") or state["state"])
            try:
                state["weight"] = float(update.get("updated_weight", state["weight"]))
                delta = float(update.get("delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            evidence_id = str(update.get("evidence_id") or "")
            if evidence_id:
                bucket = "supporting" if delta > 0 else "opposing"
                if evidence_id not in state[bucket]:
                    state[bucket].append(evidence_id)
    return tuple(
        HypothesisVM(
            code=code,
            label=label,
            state=str(values[code]["state"]),
            weight=float(values[code]["weight"]),
            supporting_evidence_ids=tuple(values[code]["supporting"]),
            opposing_evidence_ids=tuple(values[code]["opposing"]),
        )
        for code, label in HYPOTHESIS_LABELS.items()
    )


def working_hypotheses(
    hypotheses: tuple[HypothesisVM, ...],
) -> tuple[HypothesisVM | None, HypothesisVM | None]:
    if not hypotheses:
        return None, None
    current = max(hypotheses, key=lambda item: item.weight)
    alternatives = [item for item in hypotheses if item.code != Hypothesis.PLANETARY_TRANSIT.value]
    alternative = max(alternatives, key=lambda item: item.weight) if alternatives else None
    return current, alternative


def latest_skeptic_decision(run: MissionControlRun) -> SkepticVM | None:
    event = _latest_event(run.trace, TraceEventType.AGENT_DECISION)
    if event is None:
        return None
    raw = event.payload.get("decision")
    if not isinstance(raw, dict):
        return None
    guard = NumericProvenanceGuard(run.evidence)
    explanation = guard.repair(str(raw.get("explanation") or "No explanation recorded."))
    expected = guard.repair(
        str(raw.get("expected_discriminating_result") or "No expected result recorded.")
    )
    outcomes_raw = raw.get("predicted_outcomes")
    outcomes: list[tuple[str, str]] = []
    if isinstance(outcomes_raw, dict):
        outcomes = [(str(key), guard.repair(str(value))) for key, value in outcomes_raw.items()]
    hypothesis_code = raw.get("hypothesis_under_test")
    code = str(hypothesis_code) if hypothesis_code else None
    try:
        utility = float(raw.get("expected_information_value", 0.0))
    except (TypeError, ValueError):
        utility = 0.0
    source, provider, model, request_ids, attempts = _agent_provenance(run.trace, event, "SKEPTIC")
    return SkepticVM(
        decision_id=str(raw.get("decision_id") or event.event_id),
        action=str(raw.get("action") or "UNKNOWN"),
        hypothesis_code=code,
        hypothesis_label=HYPOTHESIS_LABELS.get(code or "", "No live alternative"),
        experiment=(str(raw["requested_experiment"]) if raw.get("requested_experiment") else None),
        reason_code=str(raw.get("reason_code") or "UNRECORDED"),
        explanation=explanation,
        expected_result=expected,
        predicted_outcomes=tuple(outcomes),
        decision_utility=max(0.0, min(1.0, utility)),
        priority=str(raw.get("priority") or "UNRECORDED"),
        decision_source=source,
        provider=provider,
        model=model,
        provider_request_ids=request_ids,
        attempts=attempts,
        used_fallback=bool(event.payload.get("used_fallback", False)),
    )


def latest_critic_decision(run: MissionControlRun) -> CriticVM | None:
    event = _latest_event(run.trace, TraceEventType.CRITIC_DECISION)
    if event is None:
        return None
    raw = event.payload.get("decision")
    if not isinstance(raw, dict):
        return None
    guard = NumericProvenanceGuard(run.evidence)
    source, provider, model, request_ids, attempts = _agent_provenance(run.trace, event, "CRITIC")
    return CriticVM(
        decision_id=str(raw.get("critic_decision_id") or event.event_id),
        verdict=str(raw.get("verdict") or "UNKNOWN"),
        reason_code=str(raw.get("reason_code") or "UNRECORDED"),
        reason=guard.repair(str(raw.get("reason") or "No review reason recorded.")),
        reviewed_request_id=str(raw.get("reviewed_request_id") or "UNRECORDED"),
        decision_source=source,
        provider=provider,
        model=model,
        provider_request_ids=request_ids,
        attempts=attempts,
        used_fallback=bool(event.payload.get("used_fallback", False)),
    )


def _agent_provenance(
    trace: tuple[TraceEvent, ...],
    decision_event: TraceEvent,
    role: str,
) -> tuple[str, str, str, tuple[str, ...], int]:
    """Return trace-backed model provenance without guessing that a call was live.

    New traces persist this metadata on the decision event. For older traces, a
    role-matched provider response is required before the UI labels the decision
    as live; a bare ``used_fallback=False`` flag is deliberately insufficient.
    """

    payload = decision_event.payload
    used_fallback = bool(payload.get("used_fallback", False))
    repaired = bool(payload.get("repaired", False))
    recorded_source = str(payload.get("decision_source") or "")

    prior_role_events = [
        event
        for event in trace
        if event.sequence < decision_event.sequence
        and event.payload.get("role") == role
        and event.event_type in {TraceEventType.AGENT_REQUEST, TraceEventType.AGENT_RESPONSE}
    ]
    response = next(
        (
            event
            for event in reversed(prior_role_events)
            if event.event_type is TraceEventType.AGENT_RESPONSE
        ),
        None,
    )
    request = next(
        (
            event
            for event in reversed(prior_role_events)
            if event.event_type is TraceEventType.AGENT_REQUEST
        ),
        None,
    )
    provider_event = response or request

    valid_sources = {
        "LIVE_MODEL",
        "REPAIRED_LIVE_MODEL",
        "DETERMINISTIC_FALLBACK",
    }
    if used_fallback:
        source = "DETERMINISTIC_FALLBACK"
    elif recorded_source in valid_sources:
        source = recorded_source
    elif response is not None:
        source = "REPAIRED_LIVE_MODEL" if repaired else "LIVE_MODEL"
    else:
        source = "UNVERIFIED_MODEL_SOURCE"

    provider = str(
        payload.get("provider")
        or (provider_event.payload.get("provider") if provider_event is not None else None)
        or "unrecorded"
    )
    model = str(
        payload.get("model")
        or (provider_event.payload.get("model") if provider_event is not None else None)
        or "unrecorded"
    )
    raw_request_ids = payload.get("provider_request_ids")
    if isinstance(raw_request_ids, list):
        request_ids = tuple(str(value) for value in raw_request_ids if value)
    elif response is not None and response.payload.get("request_id"):
        request_ids = (str(response.payload["request_id"]),)
    else:
        request_ids = ()
    try:
        attempts = max(0, int(payload.get("attempts", 0)))
    except (TypeError, ValueError):
        attempts = 0
    return source, provider, model, request_ids, attempts


def catalog_measurements(run: MissionControlRun) -> tuple[CatalogMeasurementVM, ...]:
    if not run.ground_truth_visible or run.reveal is None:
        return ()
    return tuple(
        CatalogMeasurementVM(
            name=name,
            value=measurement.value,
            unit=measurement.unit,
            uncertainty=measurement.uncertainty,
            source_field=measurement.source_field,
        )
        for name, measurement in run.reveal.ground_truth.measurements.items()
    )


def evidence_tone(item: EvidenceItem) -> str:
    code = item.interpretation_code.value
    if code in {
        "DETECTED",
        "PASS",
        "CONSISTENT",
        "NOT_SIGNIFICANT",
        "NO_NEARBY_SOURCE",
        "TARGET_CONSISTENT",
        "PREFERRED_NOMINAL_PERIOD",
        "ROBUST",
        "ACCEPTABLE",
    }:
        return "positive"
    if code in {
        "NOT_DETECTED",
        "FAIL",
        "INCONSISTENT",
        "SIGNIFICANT",
        "OFFSET_DETECTED",
        "PREFERRED_HALF_PERIOD",
        "PREFERRED_DOUBLE_PERIOD",
        "PREPROCESSING_SENSITIVE",
        "POOR_QUALITY",
    }:
        return "negative"
    if code in {"NEIGHBOR_DETECTED", "INCONCLUSIVE"}:
        return "warning"
    return "neutral"


def evidence_headline(item: EvidenceItem) -> str:
    experiment = item.experiment_type.value.replace("_", " ").title()
    interpretation = item.interpretation_code.value.replace("_", " ").title()
    return f"{experiment} · {interpretation}"


def adaptive_experiments(run: MissionControlRun) -> tuple[str, ...]:
    return tuple(
        item.experiment_type.value for item in run.evidence if item.agent_request_id is not None
    )


def _evidence_matches(
    item: EvidenceItem | None,
    key: str,
    value: int | float,
) -> bool:
    if item is None:
        return False
    evidence_key = key
    if evidence_key not in item.numerical_results:
        alias = HARMONIC_MEASUREMENT_ALIASES.get(key)
        if alias is None or alias not in item.numerical_results:
            return False
        evidence_key = alias
    recorded = item.numerical_results[evidence_key]
    return math.isclose(float(recorded), float(value), rel_tol=1e-10, abs_tol=1e-12)


def _latest_event(trace: tuple[TraceEvent, ...], event_type: TraceEventType) -> TraceEvent | None:
    return next(
        (event for event in reversed(trace) if event.event_type is event_type),
        None,
    )


__all__ = [
    "CatalogMeasurementVM",
    "CriticVM",
    "HypothesisVM",
    "MeasurementVM",
    "SkepticVM",
    "adaptive_experiments",
    "candidate_measurements",
    "catalog_measurements",
    "evidence_headline",
    "evidence_tone",
    "hypothesis_views",
    "latest_critic_decision",
    "latest_skeptic_decision",
    "working_hypotheses",
]
