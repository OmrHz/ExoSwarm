"""Post-run audit for real-model Skeptic and Critic investigations.

The live validator never invokes a model and never reads the private target registry.  It
reconstructs what was knowable at each decision from the verified trace and Evidence Ledger,
then combines that audit with the existing result-lock and blindness graders.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from exoswarm.domain.experiments import ExperimentDefinition, ExperimentRegistry
from exoswarm.domain.hypotheses import MANDATORY_VETTING
from exoswarm.domain.ledger import EvidenceItem, EvidenceLedger
from exoswarm.domain.models import (
    CriticDecision,
    CriticVerdict,
    EvidenceState,
    ExperimentType,
    Hypothesis,
    LockedInvestigationResult,
    ScientificDisposition,
    SkepticAction,
    SkepticDecision,
    ToolRequest,
)
from exoswarm.domain.numeric_provenance import NumericProvenanceGuard
from exoswarm.domain.trace import TraceEvent, TraceEventType, TraceRecorder
from exoswarm.evaluation.graders import (
    EvalModel,
    EvaluationExpectation,
    EvaluationReport,
    evaluate_run,
)

_IDENTITY_TEXT = re.compile(
    r"\b(?:TIC\s*\d+|TOI[-\s]?\d+(?:\.\d+)?|TESS[-\s]?OBJECT[-\s]?OF[-\s]?INTEREST\s*\d+)\b",
    re.IGNORECASE,
)
_PROHIBITED_CONTEXT_KEYS = frozenset(
    {
        "actual_target_identity",
        "backend_target_mapping",
        "catalog",
        "catalog_measurements",
        "catalog_status",
        "confirmation_status",
        "ground_truth",
        "known_period",
        "real_target_id",
        "reveal",
        "target_name",
        "tic_id",
        "toi_id",
    }
)
_EXPECTED_CONTEXT_KEYS = frozenset(
    {
        "opaque_target_id",
        "investigation_status",
        "lock_state",
        "current_candidate",
        "hypotheses",
        "evidence",
        "experiment_contracts",
        "completed_tests",
        "available_experiments",
        "unresolved_questions",
        "budgets",
    }
)
_EXPERIMENT_TARGETS: dict[ExperimentType, frozenset[Hypothesis]] = {
    ExperimentType.HARMONIC_TEST: frozenset(
        {Hypothesis.ECLIPSING_BINARY, Hypothesis.PERIOD_ALIAS_HARMONIC}
    ),
    ExperimentType.CENTROID_LOCALIZATION: frozenset({Hypothesis.BACKGROUND_CONTAMINANT}),
    ExperimentType.ALTERNATE_DETRENDING: frozenset(
        {Hypothesis.STELLAR_VARIABILITY, Hypothesis.INSTRUMENTAL_SYSTEMATIC}
    ),
    ExperimentType.ALTERNATE_APERTURE: frozenset({Hypothesis.BACKGROUND_CONTAMINANT}),
}
_PLAUSIBLE_STATES = frozenset(
    {
        EvidenceState.UNRESOLVED,
        EvidenceState.SUPPORTED,
        EvidenceState.STRONGLY_SUPPORTED,
    }
)


class DecisionOrigin(StrEnum):
    """Mechanically derived origin of a validated structured decision."""

    LIVE_MODEL = "LIVE_MODEL"
    REPAIRED_LIVE_MODEL = "REPAIRED_LIVE_MODEL"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class TokenUsage(EvalModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class MeasurementComparison(EvalModel):
    name: str
    measured: float | None = None
    reference: float
    absolute_difference: float | None = None
    tolerance: float
    passed: bool


class DecisionQuality(EvalModel):
    evidence_grounded: bool
    hypothesis_relevant: bool | None = None
    scientifically_useful: bool | None = None
    experiment_available: bool | None = None
    preconditions_satisfied: bool | None = None
    non_redundant: bool | None = None
    no_remaining_unsupported_numerical_claims: bool
    stopping_justified: bool | None = None
    passed: bool
    details: list[str] = Field(default_factory=list)


class SkepticDecisionAudit(EvalModel):
    trace_sequence: int = Field(ge=1)
    decision_id: str
    origin: DecisionOrigin
    live_provider_verified: bool
    action: SkepticAction
    hypothesis_under_test: Hypothesis | None = None
    requested_experiment: ExperimentType | None = None
    reason_code: str
    critic_verdict: CriticVerdict | None = None
    experiment_executed: ExperimentType | None = None
    schema_attempts: int = Field(ge=0)
    unsupported_numeric_claims_repaired: int = Field(default=0, ge=0)
    quality: DecisionQuality


class CriticDecisionAudit(EvalModel):
    trace_sequence: int = Field(ge=1)
    critic_decision_id: str
    reviewed_request_id: str
    origin: DecisionOrigin
    live_provider_verified: bool
    verdict: CriticVerdict
    reason_code: str
    proposal_experiment: ExperimentType | None = None
    request_id_matches: bool
    proposal_scientifically_useful: bool | None = None
    proposal_available: bool | None = None
    proposal_preconditions_satisfied: bool | None = None
    proposal_non_redundant: bool | None = None
    verdict_consistent_with_checks: bool
    evidence_grounded: bool
    no_remaining_unsupported_numerical_claims: bool
    schema_attempts: int = Field(ge=0)
    unsupported_numeric_claims_repaired: int = Field(default=0, ge=0)
    passed: bool
    details: list[str] = Field(default_factory=list)


class BlindnessAudit(EvalModel):
    target_identity_hidden_before_lock: bool
    catalog_access_blocked_before_lock: bool
    known_parameters_hidden_before_lock: bool
    agent_context_keys_safe: bool
    details: list[str] = Field(default_factory=list)


class IntegrityAudit(EvalModel):
    result_locked_before_reveal: bool
    sha256_verified: bool
    separate_reveal_artifact: bool
    locked_result_immutable: bool
    trace_and_ledger_verified: bool
    details: list[str] = Field(default_factory=list)


class LiveRunExpectation(EvalModel):
    opaque_target_id: str
    expected_provider: str = "featherless"
    expected_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    accepted_dispositions: set[ScientificDisposition] = Field(default_factory=set)
    negative_control: bool = False
    require_reveal: bool = True
    reference_measurements: dict[str, float] = Field(default_factory=dict)
    measurement_tolerances: dict[str, float] = Field(default_factory=dict)


class LiveRunReport(EvalModel):
    opaque_target_id: str
    run_id: str
    run_directory: str
    configured_provider: str
    configured_model: str
    response_providers: list[str]
    response_models: list[str]
    authentication_passed: bool
    live_inference_passed: bool
    structured_outputs_passed: bool
    skeptic_decisions: list[SkepticDecisionAudit]
    critic_decisions: list[CriticDecisionAudit]
    adaptive_experiments: list[ExperimentType]
    mandatory_diagnostics_completed: list[ExperimentType]
    missing_mandatory_diagnostics: list[ExperimentType]
    total_experiments_requested: int = Field(ge=0)
    successful_scientific_results: int = Field(ge=0)
    structured_requests: int = Field(ge=0)
    live_llm_calls: int = Field(ge=0)
    successful_live_responses: int = Field(ge=0)
    first_pass_valid_responses: int = Field(ge=0)
    repair_attempts: int = Field(ge=0)
    successful_repairs: int = Field(ge=0)
    failed_repairs: int = Field(ge=0)
    deterministic_fallbacks: int = Field(ge=0)
    fallback_events: int = Field(ge=0)
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)
    invalid_experiment_requests: int = Field(ge=0)
    unsupported_numerical_claims: int = Field(ge=0)
    redundant_experiment_requests: int = Field(ge=0)
    critic_verdict_counts: dict[str, int] = Field(default_factory=dict)
    token_usage: TokenUsage
    investigation_latency_seconds: float = Field(ge=0)
    structured_decision_latency_seconds: float = Field(ge=0)
    final_disposition: ScientificDisposition
    result_locked: bool
    reveal_present: bool
    measurement_comparisons: list[MeasurementComparison]
    blindness: BlindnessAudit
    integrity: IntegrityAudit
    deterministic_evaluation: EvaluationReport
    errors: list[str] = Field(default_factory=list)
    scientifically_valid_trajectory: bool
    live_validation_passed: bool


class ReliabilitySummary(EvalModel):
    total_structured_requests: int = Field(ge=0)
    total_live_llm_calls: int = Field(ge=0)
    successful_live_responses: int = Field(ge=0)
    first_pass_valid_responses: int = Field(ge=0)
    first_pass_schema_valid_rate: float = Field(ge=0, le=1)
    repair_attempts: int = Field(ge=0)
    repair_rate: float = Field(ge=0, le=1)
    successful_repairs: int = Field(ge=0)
    repair_success_rate: float = Field(ge=0, le=1)
    failed_repairs: int = Field(ge=0)
    deterministic_fallbacks: int = Field(ge=0)
    fallback_rate: float = Field(ge=0, le=1)
    invalid_experiment_requests: int = Field(ge=0)
    unsupported_numerical_claims: int = Field(ge=0)
    redundant_experiment_requests: int = Field(ge=0)
    token_usage: TokenUsage


class TargetConsistency(EvalModel):
    opaque_target_id: str
    trials: int = Field(ge=0)
    scientifically_valid_trajectories: int = Field(ge=0)
    exact_adaptive_experiment_consistent: bool
    adaptive_trajectories: list[list[ExperimentType]]


class LiveValidationReport(EvalModel):
    generated_at: datetime
    runs: list[LiveRunReport]
    reliability: ReliabilitySummary
    target_consistency: list[TargetConsistency]
    independent_run_ids: bool
    required_trial_counts_met: bool
    different_evidence_changed_trajectory: bool
    overall_live_model_valid_trajectories: int = Field(ge=0)
    passed: bool


def demo_live_expectation(opaque_target_id: str) -> LiveRunExpectation:
    """Return post-lock regression references for the two curated demo targets."""

    if opaque_target_id == "TARGET-X17":
        return LiveRunExpectation(
            opaque_target_id=opaque_target_id,
            accepted_dispositions={
                ScientificDisposition.PLANETARY_INTERPRETATION_PLAUSIBLE,
                ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING,
            },
            reference_measurements={
                "period_days": 3.739466865,
                "transit_depth_ppm": 15_051.0,
                "signal_to_noise": 105.2,
            },
            measurement_tolerances={
                "period_days": 0.001,
                "transit_depth_ppm": 500.0,
                "signal_to_noise": 5.0,
            },
        )
    if opaque_target_id == "TARGET-X42":
        return LiveRunExpectation(
            opaque_target_id=opaque_target_id,
            accepted_dispositions={ScientificDisposition.PLANETARY_INTERPRETATION_WEAK},
            negative_control=True,
            reference_measurements={
                "initial_period_days": 1.515946219,
                "period_days": 3.031910460,
            },
            measurement_tolerances={
                "initial_period_days": 0.001,
                "period_days": 0.001,
            },
        )
    raise ValueError(f"no curated live-validation expectation for {opaque_target_id!r}")


def evaluate_live_run(
    run_directory: str | Path,
    expectation: LiveRunExpectation | None = None,
) -> LiveRunReport:
    """Audit one completed online run without invoking the provider."""

    directory = Path(run_directory).resolve()
    result = LockedInvestigationResult.model_validate_json((directory / "result.json").read_bytes())
    expectation = expectation or demo_live_expectation(result.opaque_target_id)
    if expectation.opaque_target_id != result.opaque_target_id:
        raise ValueError("live-validation expectation does not match the locked target")

    ledger = EvidenceLedger(directory / "evidence.jsonl")
    trace = TraceRecorder(
        trace_id=result.trace_id,
        opaque_target_id=result.opaque_target_id,
        path=directory / "trace.jsonl",
    )
    events = list(trace.events)
    base_report = evaluate_run(
        directory,
        EvaluationExpectation(
            opaque_target_id=result.opaque_target_id,
            expected_period_days=expectation.reference_measurements.get("period_days"),
            period_tolerance_days=expectation.measurement_tolerances.get("period_days"),
            accepted_dispositions=expectation.accepted_dispositions,
            negative_control=expectation.negative_control,
            require_reveal=expectation.require_reveal,
        ),
    )
    initial = _one_event(events, TraceEventType.INVESTIGATION_INITIALIZED)
    settings = initial.payload.get("settings", {}) if initial else {}
    configured_provider = str(settings.get("provider", ""))
    configured_model = str(settings.get("model", ""))
    response_events = [
        event for event in events if event.event_type is TraceEventType.AGENT_RESPONSE
    ]
    response_providers = sorted(
        {str(event.payload.get("provider", "")) for event in response_events}
    )
    response_models = sorted({str(event.payload.get("model", "")) for event in response_events})

    skeptic_audits = _audit_skeptic_decisions(
        events,
        ledger,
        result,
        expected_provider=expectation.expected_provider,
        expected_model=expectation.expected_model,
    )
    critic_audits = _audit_critic_decisions(
        events,
        ledger,
        result,
        skeptic_audits,
        expected_provider=expectation.expected_provider,
        expected_model=expectation.expected_model,
    )
    _attach_critic_verdicts(skeptic_audits, critic_audits)

    decision_events = [
        event
        for event in events
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}
    ]
    origins = [_decision_origin(event) for event in decision_events]
    structured_requests = len(decision_events)
    live_llm_calls = sum(int(event.payload.get("attempts", 0)) for event in decision_events)
    first_pass = sum(origin is DecisionOrigin.LIVE_MODEL for origin in origins)
    repairs = sum(
        bool(event.payload.get("repaired")) or int(event.payload.get("attempts", 0)) > 1
        for event in decision_events
    )
    repaired = sum(origin is DecisionOrigin.REPAIRED_LIVE_MODEL for origin in origins)
    deterministic_fallbacks = sum(
        origin is DecisionOrigin.DETERMINISTIC_FALLBACK for origin in origins
    )
    failed_repairs = sum(
        origin is DecisionOrigin.DETERMINISTIC_FALLBACK
        and int(event.payload.get("attempts", 0)) > 1
        for origin, event in zip(origins, decision_events, strict=True)
    )
    fallback_events = [event for event in events if event.event_type is TraceEventType.FALLBACK]
    fallback_reasons = Counter(
        str(event.payload.get("reason_code") or event.payload.get("source_event") or "UNKNOWN")
        for event in fallback_events
    )
    invalid_requests = (
        fallback_reasons["INVALID_AGENT_TOOL_REQUEST"] + fallback_reasons["CRITIC_REVISION_INVALID"]
    )
    redundant_requests = fallback_reasons["REPEATED_LOW_VALUE_ACTION"]
    unsupported_claims = sum(
        audit.unsupported_numeric_claims_repaired for audit in skeptic_audits
    ) + sum(audit.unsupported_numeric_claims_repaired for audit in critic_audits)

    token_usage = _sum_usage(response_events)
    tool_requests = [event for event in events if event.event_type is TraceEventType.TOOL_REQUESTED]
    adaptive_experiments = [
        ExperimentType(str(event.payload["request"]["experiment_type"]))
        for event in tool_requests
        if bool(event.payload.get("request", {}).get("adaptive"))
    ]
    completed = set(result.completed_tests)
    mandatory_completed = sorted(completed & MANDATORY_VETTING, key=lambda item: item.value)
    missing_mandatory = sorted(MANDATORY_VETTING - completed, key=lambda item: item.value)

    lock_event = _one_event(events, TraceEventType.RESULT_LOCKED)
    reveal_event = _one_event(events, TraceEventType.GROUND_TRUTH_REVEALED)
    started_at = events[0].timestamp if events else result.created_at
    stopped_at = lock_event.timestamp if lock_event else events[-1].timestamp
    decision_latency = _structured_decision_latency(events)
    comparisons = _measurement_comparisons(result, ledger, expectation)
    blindness = _blindness_audit(directory, events, ledger, base_report)
    integrity = _integrity_audit(
        directory,
        base_report,
        require_reveal=expectation.require_reveal,
    )
    errors = _collect_errors(events)

    expected_live_responses = bool(response_events) and all(
        str(event.payload.get("provider")) == expectation.expected_provider
        and str(event.payload.get("model")) == expectation.expected_model
        for event in response_events
    )
    configured_live = (
        configured_provider == expectation.expected_provider
        and configured_model == expectation.expected_model
        and bool(settings.get("provider_enabled"))
    )
    all_decisions_live = bool(decision_events) and all(
        origin is not DecisionOrigin.DETERMINISTIC_FALLBACK for origin in origins
    )
    all_provider_audits = all(
        audit.live_provider_verified for audit in [*skeptic_audits, *critic_audits]
    )
    authentication_passed = configured_live and expected_live_responses
    live_inference_passed = authentication_passed and all_decisions_live and all_provider_audits
    structured_outputs_passed = all_decisions_live
    quality_passed = (
        bool(skeptic_audits)
        and all(audit.quality.passed for audit in skeptic_audits)
        and all(audit.passed for audit in critic_audits)
    )
    measurements_passed = all(item.passed for item in comparisons)
    scientific_trajectory = (
        quality_passed
        and measurements_passed
        and not missing_mandatory
        and result.disposition in expectation.accepted_dispositions
        and (
            not expectation.negative_control
            or result.disposition is ScientificDisposition.PLANETARY_INTERPRETATION_WEAK
        )
    )
    live_passed = (
        live_inference_passed
        and structured_outputs_passed
        and deterministic_fallbacks == 0
        and invalid_requests == 0
        and scientific_trajectory
        and blindness.target_identity_hidden_before_lock
        and blindness.catalog_access_blocked_before_lock
        and blindness.known_parameters_hidden_before_lock
        and integrity.result_locked_before_reveal
        and integrity.sha256_verified
        and integrity.separate_reveal_artifact
        and integrity.locked_result_immutable
        and base_report.passed
    )
    return LiveRunReport(
        opaque_target_id=result.opaque_target_id,
        run_id=result.trace_id,
        run_directory=str(directory),
        configured_provider=configured_provider,
        configured_model=configured_model,
        response_providers=response_providers,
        response_models=response_models,
        authentication_passed=authentication_passed,
        live_inference_passed=live_inference_passed,
        structured_outputs_passed=structured_outputs_passed,
        skeptic_decisions=skeptic_audits,
        critic_decisions=critic_audits,
        adaptive_experiments=adaptive_experiments,
        mandatory_diagnostics_completed=mandatory_completed,
        missing_mandatory_diagnostics=missing_mandatory,
        total_experiments_requested=len(tool_requests),
        successful_scientific_results=len(ledger),
        structured_requests=structured_requests,
        live_llm_calls=live_llm_calls,
        successful_live_responses=len(response_events),
        first_pass_valid_responses=first_pass,
        repair_attempts=repairs,
        successful_repairs=repaired,
        failed_repairs=failed_repairs,
        deterministic_fallbacks=deterministic_fallbacks,
        fallback_events=len(fallback_events),
        fallback_reason_counts=dict(sorted(fallback_reasons.items())),
        invalid_experiment_requests=invalid_requests,
        unsupported_numerical_claims=unsupported_claims,
        redundant_experiment_requests=redundant_requests,
        critic_verdict_counts=dict(
            sorted(Counter(item.verdict.value for item in critic_audits).items())
        ),
        token_usage=token_usage,
        investigation_latency_seconds=max(0.0, (stopped_at - started_at).total_seconds()),
        structured_decision_latency_seconds=decision_latency,
        final_disposition=result.disposition,
        result_locked=lock_event is not None,
        reveal_present=reveal_event is not None and (directory / "reveal.json").is_file(),
        measurement_comparisons=comparisons,
        blindness=blindness,
        integrity=integrity,
        deterministic_evaluation=base_report,
        errors=errors,
        scientifically_valid_trajectory=scientific_trajectory,
        live_validation_passed=live_passed,
    )


def evaluate_live_trials(
    run_directories: Sequence[str | Path],
    *,
    expectations: Mapping[str, LiveRunExpectation] | None = None,
    required_trials: Mapping[str, int] | None = None,
) -> LiveValidationReport:
    """Aggregate independent run directories into the six-trial validation report."""

    from exoswarm.domain.models import utc_now

    expectations = dict(expectations or {})
    required_trials = dict(required_trials or {"TARGET-X17": 3, "TARGET-X42": 3})
    reports: list[LiveRunReport] = []
    for run_directory in run_directories:
        result = LockedInvestigationResult.model_validate_json(
            (Path(run_directory) / "result.json").read_bytes()
        )
        reports.append(
            evaluate_live_run(
                run_directory,
                expectations.get(result.opaque_target_id)
                or demo_live_expectation(result.opaque_target_id),
            )
        )

    consistency: list[TargetConsistency] = []
    for target in sorted({*required_trials, *(item.opaque_target_id for item in reports)}):
        target_runs = [item for item in reports if item.opaque_target_id == target]
        paths = [[*item.adaptive_experiments] for item in target_runs]
        signatures = {tuple(item.value for item in path) for path in paths}
        consistency.append(
            TargetConsistency(
                opaque_target_id=target,
                trials=len(target_runs),
                scientifically_valid_trajectories=sum(
                    item.live_validation_passed for item in target_runs
                ),
                exact_adaptive_experiment_consistent=(len(signatures) <= 1),
                adaptive_trajectories=paths,
            )
        )

    structured_requests = sum(item.structured_requests for item in reports)
    repair_attempts = sum(item.repair_attempts for item in reports)
    successful_repairs = sum(item.successful_repairs for item in reports)
    fallbacks = sum(item.deterministic_fallbacks for item in reports)
    usage = TokenUsage(
        prompt_tokens=sum(item.token_usage.prompt_tokens for item in reports),
        completion_tokens=sum(item.token_usage.completion_tokens for item in reports),
        total_tokens=sum(item.token_usage.total_tokens for item in reports),
    )
    reliability = ReliabilitySummary(
        total_structured_requests=structured_requests,
        total_live_llm_calls=sum(item.live_llm_calls for item in reports),
        successful_live_responses=sum(item.successful_live_responses for item in reports),
        first_pass_valid_responses=sum(item.first_pass_valid_responses for item in reports),
        first_pass_schema_valid_rate=_rate(
            sum(item.first_pass_valid_responses for item in reports), structured_requests
        ),
        repair_attempts=repair_attempts,
        repair_rate=_rate(repair_attempts, structured_requests),
        successful_repairs=successful_repairs,
        repair_success_rate=_rate(successful_repairs, repair_attempts),
        failed_repairs=sum(item.failed_repairs for item in reports),
        deterministic_fallbacks=fallbacks,
        fallback_rate=_rate(fallbacks, structured_requests),
        invalid_experiment_requests=sum(item.invalid_experiment_requests for item in reports),
        unsupported_numerical_claims=sum(item.unsupported_numerical_claims for item in reports),
        redundant_experiment_requests=sum(item.redundant_experiment_requests for item in reports),
        token_usage=usage,
    )
    trial_counts_met = all(
        sum(report.opaque_target_id == target for report in reports) == required
        for target, required in required_trials.items()
    )
    branches_by_target = {
        target: {
            tuple(item.value for item in report.adaptive_experiments)
            for report in reports
            if report.opaque_target_id == target
        }
        for target in required_trials
    }
    changed_trajectory = (
        len(branches_by_target) >= 2
        and all(branches_by_target.values())
        and len({branch for branches in branches_by_target.values() for branch in branches}) > 1
    )
    valid = sum(item.live_validation_passed for item in reports)
    independent = len({item.run_id for item in reports}) == len(reports) and len(
        {item.run_directory for item in reports}
    ) == len(reports)
    return LiveValidationReport(
        generated_at=utc_now(),
        runs=reports,
        reliability=reliability,
        target_consistency=consistency,
        independent_run_ids=independent,
        required_trial_counts_met=trial_counts_met,
        different_evidence_changed_trajectory=changed_trajectory,
        overall_live_model_valid_trajectories=valid,
        passed=(
            independent
            and trial_counts_met
            and changed_trajectory
            and bool(reports)
            and all(item.live_validation_passed for item in reports)
        ),
    )


def render_live_validation_markdown(report: LiveValidationReport) -> str:
    """Render a compact, credential-free report suitable for a demo artifact."""

    lines = ["# ExoSwarm live-model validation", ""]
    provider = sorted({item.configured_provider for item in report.runs})
    models = sorted({item.configured_model for item in report.runs})
    lines.extend(
        [
            "## Provider",
            "",
            f"- Provider: {', '.join(provider) or 'none'}",
            f"- Model: {', '.join(models) or 'none'}",
            f"- Authentication: {'PASS' if all(item.authentication_passed for item in report.runs) else 'FAIL'}",
            f"- Live inference: {'PASS' if all(item.live_inference_passed for item in report.runs) else 'FAIL'}",
            f"- Structured outputs: {'PASS' if all(item.structured_outputs_passed for item in report.runs) else 'FAIL'}",
            "",
        ]
    )
    for target in sorted({item.opaque_target_id for item in report.runs}):
        lines.extend([f"## {target}", ""])
        lines.append(
            "| Run ID | Skeptic choice | Critic | Executed | Repairs | Fallbacks | Disposition | Lock | Reveal |"
        )
        lines.append("|---|---|---|---|---:|---:|---|---:|---:|")
        for run in [item for item in report.runs if item.opaque_target_id == target]:
            choices = ", ".join(
                decision.requested_experiment.value if decision.requested_experiment else "STOP"
                for decision in run.skeptic_decisions
            )
            critics = ", ".join(item.verdict.value for item in run.critic_decisions) or "none"
            executed = ", ".join(item.value for item in run.adaptive_experiments) or "none"
            lines.append(
                f"| {run.run_id} | {choices} | {critics} | {executed} | "
                f"{run.repair_attempts} | {run.deterministic_fallbacks} | "
                f"{run.final_disposition.value} | {'PASS' if run.result_locked else 'FAIL'} | "
                f"{'PASS' if run.reveal_present else 'FAIL'} |"
            )
        lines.append("")

    reliability = report.reliability
    lines.extend(
        [
            "## Agent reliability",
            "",
            f"- Total structured requests: {reliability.total_structured_requests}",
            f"- Total live LLM calls: {reliability.total_live_llm_calls}",
            f"- First-pass schema-valid rate: {reliability.first_pass_schema_valid_rate:.1%}",
            f"- Repair attempts: {reliability.repair_attempts}",
            f"- Repair success rate: {reliability.repair_success_rate:.1%}",
            f"- Deterministic fallbacks: {reliability.deterministic_fallbacks}",
            f"- Fallback rate: {reliability.fallback_rate:.1%}",
            f"- Invalid experiment requests: {reliability.invalid_experiment_requests}",
            f"- Unsupported numerical claims: {reliability.unsupported_numerical_claims}",
            f"- Redundant experiment requests: {reliability.redundant_experiment_requests}",
            f"- Token usage: {reliability.token_usage.total_tokens}",
            "",
            "## Consistency",
            "",
        ]
    )
    for item in report.target_consistency:
        lines.append(
            f"- {item.opaque_target_id}: {item.scientifically_valid_trajectories}/{item.trials} "
            "scientifically valid; exact adaptive experiment "
            f"{'consistent' if item.exact_adaptive_experiment_consistent else 'varied'}"
        )
    lines.extend(
        [
            f"- Independent run IDs/directories: {'PASS' if report.independent_run_ids else 'FAIL'}",
            f"- Overall live-model valid trajectories: {report.overall_live_model_valid_trajectories}/{len(report.runs)}",
            f"- Different evidence changed trajectory: {'PASS' if report.different_evidence_changed_trajectory else 'FAIL'}",
            "",
            f"Overall: {'PASS' if report.passed else 'FAIL'}",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_skeptic_decisions(
    events: list[TraceEvent],
    ledger: EvidenceLedger,
    result: LockedInvestigationResult,
    *,
    expected_provider: str,
    expected_model: str,
) -> list[SkepticDecisionAudit]:
    audits: list[SkepticDecisionAudit] = []
    for event in events:
        if event.event_type is not TraceEventType.AGENT_DECISION:
            continue
        decision = SkepticDecision.model_validate(event.payload["decision"])
        prefix = _evidence_before(events, ledger, event.sequence)
        hypotheses = _hypotheses_before(events, event.sequence)
        completed = [item.experiment_type for item in prefix]
        request_event = _preceding_agent_request(events, event, role="SKEPTIC")
        response_ok = _live_response_between(
            events,
            request_event,
            event,
            expected_provider=expected_provider,
            expected_model=expected_model,
        )
        numeric_ok = _skeptic_numeric_grounding(decision, prefix)
        context_ok = _request_context_safe(request_event) and _request_context_matches_prefix(
            request_event,
            prefix,
            result,
        )
        parameter_grounded = _parameters_grounded(decision, prefix, result)
        evidence_grounded = numeric_ok and context_ok and parameter_grounded
        origin = _decision_origin(event)
        unsupported = _guardrail_repairs_before(events, event, role="SKEPTIC")
        details: list[str] = []
        if not context_ok:
            details.append("agent-request context keys were not the declared safe packet surface")
        if not numeric_ok:
            details.append("sanitized decision prose still contains an unsupported number")
        if not parameter_grounded:
            details.append("decision parameters are not grounded in the current candidate/evidence")

        if decision.action is SkepticAction.STOP:
            stop_ok = _stop_is_justified(completed, hypotheses, result)
            if not stop_ok:
                details.append("STOP left a supported alternative with a useful unused test")
            quality = DecisionQuality(
                evidence_grounded=evidence_grounded,
                no_remaining_unsupported_numerical_claims=numeric_ok,
                stopping_justified=stop_ok,
                passed=evidence_grounded and stop_ok,
                details=details,
            )
        else:
            assert decision.requested_experiment is not None
            assert decision.hypothesis_under_test is not None
            relevant = _hypothesis_is_relevant(
                decision.hypothesis_under_test,
                decision.requested_experiment,
                hypotheses,
            )
            useful = decision.hypothesis_under_test in _EXPERIMENT_TARGETS.get(
                decision.requested_experiment, frozenset()
            )
            available, preconditions = _experiment_available(
                decision.requested_experiment,
                decision.parameters,
                completed,
                result,
                _available_product_roles(events),
            )
            if request_event is not None and "available_experiments" in request_event.payload:
                available = available and decision.requested_experiment.value in {
                    str(item) for item in request_event.payload["available_experiments"]
                }
            redundant = _is_redundant_request(events, event, decision)
            if not relevant:
                details.append("chosen hypothesis was not the strongest actionable alternative")
            if not useful:
                details.append("experiment does not target the declared hypothesis")
            if not available:
                details.append("experiment was not available at the decision point")
            if not preconditions:
                details.append("experiment parameters or scientific preconditions were invalid")
            if redundant:
                details.append("an equivalent experiment request already existed")
            passed = all(
                (evidence_grounded, relevant, useful, available, preconditions, not redundant)
            )
            quality = DecisionQuality(
                evidence_grounded=evidence_grounded,
                hypothesis_relevant=relevant,
                scientifically_useful=useful,
                experiment_available=available,
                preconditions_satisfied=preconditions,
                non_redundant=not redundant,
                no_remaining_unsupported_numerical_claims=numeric_ok,
                passed=passed,
                details=details,
            )
        audits.append(
            SkepticDecisionAudit(
                trace_sequence=event.sequence,
                decision_id=decision.decision_id,
                origin=origin,
                live_provider_verified=response_ok,
                action=decision.action,
                hypothesis_under_test=decision.hypothesis_under_test,
                requested_experiment=decision.requested_experiment,
                reason_code=decision.reason_code,
                experiment_executed=_executed_experiment_for_decision(events, decision.decision_id),
                schema_attempts=int(event.payload.get("attempts", 0)),
                unsupported_numeric_claims_repaired=unsupported,
                quality=quality,
            )
        )
    return audits


def _audit_critic_decisions(
    events: list[TraceEvent],
    ledger: EvidenceLedger,
    result: LockedInvestigationResult,
    skeptic_audits: list[SkepticDecisionAudit],
    *,
    expected_provider: str,
    expected_model: str,
) -> list[CriticDecisionAudit]:
    audits: list[CriticDecisionAudit] = []
    registry = ExperimentRegistry()
    for event in events:
        if event.event_type is not TraceEventType.CRITIC_DECISION:
            continue
        decision = CriticDecision.model_validate(event.payload["decision"])
        skeptic = next(
            (item for item in reversed(skeptic_audits) if item.trace_sequence < event.sequence),
            None,
        )
        proposal = skeptic.requested_experiment if skeptic else None
        request_event = _preceding_agent_request(events, event, role="CRITIC")
        response_ok = _live_response_between(
            events,
            request_event,
            event,
            expected_provider=expected_provider,
            expected_model=expected_model,
        )
        origin = _decision_origin(event)
        prefix = _evidence_before(events, ledger, event.sequence)
        numeric_ok = NumericProvenanceGuard(prefix).validate(decision.reason).valid
        context_ok = _request_context_safe(
            request_event, critic=True
        ) and _request_context_matches_prefix(request_event, prefix, result)
        proposal_quality = skeptic.quality if skeptic else None
        useful = proposal_quality.scientifically_useful if proposal_quality else None
        available = proposal_quality.experiment_available if proposal_quality else None
        preconditions = proposal_quality.preconditions_satisfied if proposal_quality else None
        non_redundant = proposal_quality.non_redundant if proposal_quality else None

        traced_proposal_id = None
        if request_event is not None:
            traced_proposal_id = request_event.payload.get("proposal_request_id")
            audit = request_event.payload.get("context_audit", {})
            if traced_proposal_id is None and isinstance(audit, dict):
                traced_proposal_id = audit.get("proposal_request_id")
        executed = _tool_request_by_id(events, decision.reviewed_request_id)
        request_match = bool(
            (traced_proposal_id and traced_proposal_id == decision.reviewed_request_id)
            or (
                executed
                and executed.payload.get("request", {}).get("request_id")
                == decision.reviewed_request_id
            )
        )
        if decision.verdict is CriticVerdict.VETO and traced_proposal_id is None:
            # Old traces do not retain a vetoed proposal id. The runtime's explicit
            # mismatch guard still provides a weaker proof when no mismatch event follows.
            request_match = not _fallback_between(
                events, event.sequence, None, "CRITIC_REVIEW_TARGET_MISMATCH"
            )

        checks_pass = all(
            value is True for value in (useful, available, preconditions, non_redundant)
        )
        if decision.verdict is CriticVerdict.APPROVE:
            verdict_ok = checks_pass and executed is not None
        elif decision.verdict is CriticVerdict.VETO:
            verdict_ok = not checks_pass
        else:
            revised = decision.revised_request
            assert revised is not None
            revised_event = _tool_request_by_id(events, revised.request_id)
            normalized_ok = _parameters_validate(
                registry.definition(revised.experiment_type), revised
            )
            verdict_ok = normalized_ok and revised_event is not None

        details: list[str] = []
        if not context_ok:
            details.append("Critic context keys were outside the declared safe surface")
        if not request_match:
            details.append("Critic reviewed_request_id cannot be linked to its proposal")
        if not numeric_ok:
            details.append("sanitized Critic reason still contains an unsupported number")
        if not verdict_ok:
            details.append("Critic verdict is inconsistent with deterministic proposal checks")
        unsupported = _guardrail_repairs_before(events, event, role="CRITIC")
        passed = context_ok and request_match and numeric_ok and verdict_ok
        audits.append(
            CriticDecisionAudit(
                trace_sequence=event.sequence,
                critic_decision_id=decision.critic_decision_id,
                reviewed_request_id=decision.reviewed_request_id,
                origin=origin,
                live_provider_verified=response_ok,
                verdict=decision.verdict,
                reason_code=decision.reason_code,
                proposal_experiment=proposal,
                request_id_matches=request_match,
                proposal_scientifically_useful=useful,
                proposal_available=available,
                proposal_preconditions_satisfied=preconditions,
                proposal_non_redundant=non_redundant,
                verdict_consistent_with_checks=verdict_ok,
                evidence_grounded=context_ok and numeric_ok,
                no_remaining_unsupported_numerical_claims=numeric_ok,
                schema_attempts=int(event.payload.get("attempts", 0)),
                unsupported_numeric_claims_repaired=unsupported,
                passed=passed,
                details=details,
            )
        )
    return audits


def _attach_critic_verdicts(
    skeptic_audits: list[SkepticDecisionAudit],
    critic_audits: list[CriticDecisionAudit],
) -> None:
    for critic in critic_audits:
        skeptic = next(
            (
                item
                for item in reversed(skeptic_audits)
                if item.trace_sequence < critic.trace_sequence
            ),
            None,
        )
        if skeptic is not None:
            skeptic.critic_verdict = critic.verdict


def _decision_origin(event: TraceEvent) -> DecisionOrigin:
    explicit = event.payload.get("decision_source") or event.payload.get("decision_origin")
    if explicit is not None:
        return DecisionOrigin(str(explicit))
    if bool(event.payload.get("used_fallback")):
        return DecisionOrigin.DETERMINISTIC_FALLBACK
    if bool(event.payload.get("repaired")):
        return DecisionOrigin.REPAIRED_LIVE_MODEL
    return DecisionOrigin.LIVE_MODEL


def _evidence_before(
    events: list[TraceEvent], ledger: EvidenceLedger, sequence: int
) -> list[EvidenceItem]:
    ids = [
        str(event.payload["evidence_id"])
        for event in events
        if event.sequence < sequence
        and event.event_type is TraceEventType.EVIDENCE_APPENDED
        and event.payload.get("evidence_id")
    ]
    return [ledger.get(evidence_id) for evidence_id in ids]


def _hypotheses_before(
    events: list[TraceEvent], sequence: int
) -> dict[Hypothesis, tuple[EvidenceState, float]]:
    states = {hypothesis: (EvidenceState.UNRESOLVED, 0.0) for hypothesis in Hypothesis}
    for event in events:
        if event.sequence >= sequence or event.event_type is not TraceEventType.HYPOTHESIS_UPDATED:
            continue
        report = event.payload.get("report", {})
        for update in report.get("updates", []) if isinstance(report, dict) else []:
            hypothesis = Hypothesis(str(update["hypothesis"]))
            states[hypothesis] = (
                EvidenceState(str(update["updated_state"])),
                float(update["updated_weight"]),
            )
    return states


def _hypothesis_is_relevant(
    selected: Hypothesis,
    experiment: ExperimentType,
    states: dict[Hypothesis, tuple[EvidenceState, float]],
) -> bool:
    if selected is Hypothesis.PLANETARY_TRANSIT:
        return False
    state, _ = states[selected]
    if state not in _PLAUSIBLE_STATES:
        return False
    alternatives = {
        hypothesis: weight
        for hypothesis, (evidence_state, weight) in states.items()
        if hypothesis is not Hypothesis.PLANETARY_TRANSIT and evidence_state in _PLAUSIBLE_STATES
    }
    if not alternatives:
        return False
    strongest_weight = max(alternatives.values())
    strongest = {
        hypothesis
        for hypothesis, weight in alternatives.items()
        if math.isclose(weight, strongest_weight, abs_tol=1e-12)
    }
    targets = _EXPERIMENT_TARGETS.get(experiment, frozenset())
    return selected in strongest or bool(targets & strongest)


def _experiment_available(
    experiment: ExperimentType,
    parameters: dict[str, Any],
    completed: list[ExperimentType],
    result: LockedInvestigationResult,
    artifact_roles: set[str],
) -> tuple[bool, bool]:
    registry = ExperimentRegistry()
    if experiment not in registry.experiment_types:
        return False, False
    definition = registry.definition(experiment)
    request = ToolRequest(
        experiment_type=experiment,
        parameters=parameters,
        adaptive=True,
        requested_by="live-validation-replay",
        justification="post-run replay",
    )
    parameters_ok = _parameters_validate(definition, request)
    dependencies_ok = all(item in completed for item in definition.required_tests)
    candidate_ok = not definition.requires_candidate or result.candidate is not None
    candidate_id = parameters.get("candidate_id")
    if definition.requires_candidate and result.candidate is not None:
        candidate_ok = candidate_id == result.candidate.candidate_id
    events_ok = definition.minimum_events is None or (
        result.candidate is not None
        and result.candidate.observed_events >= definition.minimum_events
    )
    artifact_ok = (
        definition.required_artifact_role is None
        or definition.required_artifact_role in artifact_roles
    )
    execution_ok = completed.count(experiment) < definition.max_executions
    preconditions = parameters_ok and dependencies_ok and candidate_ok and events_ok and artifact_ok
    return preconditions and execution_ok, preconditions


def _parameters_validate(definition: ExperimentDefinition, request: ToolRequest) -> bool:
    try:
        definition.parameter_model.model_validate(request.parameters)
    except Exception:
        return False
    return True


def _is_redundant_request(
    events: list[TraceEvent], event: TraceEvent, decision: SkepticDecision
) -> bool:
    fingerprint = (
        decision.requested_experiment,
        json.dumps(decision.parameters, sort_keys=True, separators=(",", ":")),
    )
    for prior in events:
        if (
            prior.sequence >= event.sequence
            or prior.event_type is not TraceEventType.TOOL_REQUESTED
        ):
            continue
        request = prior.payload.get("request", {})
        if not bool(request.get("adaptive")):
            continue
        prior_fingerprint = (
            ExperimentType(str(request["experiment_type"])),
            json.dumps(request.get("parameters", {}), sort_keys=True, separators=(",", ":")),
        )
        if prior_fingerprint == fingerprint:
            return True
    return False


def _stop_is_justified(
    completed: list[ExperimentType],
    hypotheses: dict[Hypothesis, tuple[EvidenceState, float]],
    result: LockedInvestigationResult,
) -> bool:
    if result.candidate is None:
        return True
    roles = {"target_pixel"}
    for experiment, targets in _EXPERIMENT_TARGETS.items():
        for hypothesis in targets:
            state, weight = hypotheses[hypothesis]
            if state not in {EvidenceState.SUPPORTED, EvidenceState.STRONGLY_SUPPORTED}:
                continue
            available, _ = _experiment_available(
                experiment,
                _default_adaptive_parameters(experiment, result),
                completed,
                result,
                roles,
            )
            if available and weight >= 0.75:
                return False
    return True


def _default_adaptive_parameters(
    experiment: ExperimentType, result: LockedInvestigationResult
) -> dict[str, Any]:
    if result.candidate is None:
        return {}
    parameters: dict[str, Any] = {"candidate_id": result.candidate.candidate_id}
    if experiment is ExperimentType.HARMONIC_TEST:
        parameters.update(
            {"base_period_days": result.candidate.period_days, "factors": [0.5, 1.0, 2.0]}
        )
    elif experiment is ExperimentType.CENTROID_LOCALIZATION:
        parameters["transit_window_scale"] = 1.0
    elif experiment is ExperimentType.ALTERNATE_DETRENDING:
        parameters.update({"method": "savgol", "window_hours": 36.0})
    return parameters


def _skeptic_numeric_grounding(decision: SkepticDecision, evidence: Iterable[EvidenceItem]) -> bool:
    guard = NumericProvenanceGuard(evidence)
    texts = [
        decision.explanation,
        decision.expected_discriminating_result,
        *(decision.predicted_outcomes.values()),
    ]
    if decision.stop_if:
        texts.append(decision.stop_if)
    return all(guard.validate(text).valid for text in texts)


def _parameters_grounded(
    decision: SkepticDecision,
    evidence: list[EvidenceItem],
    result: LockedInvestigationResult,
) -> bool:
    if decision.action is SkepticAction.STOP:
        return True
    candidate_id = decision.parameters.get("candidate_id")
    if candidate_id is not None and (
        result.candidate is None or candidate_id != result.candidate.candidate_id
    ):
        return False
    base_period = decision.parameters.get("base_period_days")
    if base_period is not None:
        period_values = [
            float(value)
            for item in evidence
            for key, value in item.numerical_results.items()
            if key in {"period_days", "preferred_period_days"}
        ]
        if not any(
            math.isclose(float(base_period), value, rel_tol=0, abs_tol=1e-9)
            for value in period_values
        ):
            return False
    return True


def _request_context_safe(event: TraceEvent | None, *, critic: bool = False) -> bool:
    if event is None:
        return False
    keys = {str(item) for item in event.payload.get("context_keys", [])}
    expected = _EXPECTED_CONTEXT_KEYS | ({"proposal"} if critic else set())
    legacy_expected = expected - {"experiment_contracts"}
    if keys not in (expected, legacy_expected) or keys & _PROHIBITED_CONTEXT_KEYS:
        return False
    serialized = json.dumps(event.payload, sort_keys=True, default=str)
    return not _IDENTITY_TEXT.search(serialized) and not _contains_prohibited_key(event.payload)


def _request_context_matches_prefix(
    event: TraceEvent | None,
    evidence: list[EvidenceItem],
    result: LockedInvestigationResult,
) -> bool:
    """Cross-check the safe request audit fields against the exact ledger prefix."""

    if event is None:
        return False
    payload = event.payload
    if payload.get("opaque_target_id") not in {None, result.opaque_target_id}:
        return False
    if payload.get("lock_state") not in {None, "UNLOCKED"}:
        return False
    if payload.get("context_preflight") not in {None, "PASS"}:
        return False
    digest = payload.get("context_sha256")
    if digest is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", str(digest)):
        return False
    traced_ids = payload.get("evidence_ids")
    if traced_ids is not None and list(traced_ids) != [item.id for item in evidence]:
        return False
    traced_completed = payload.get("completed_tests")
    return traced_completed is None or list(traced_completed) == [
        item.experiment_type.value for item in evidence
    ]


def _contains_prohibited_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _PROHIBITED_CONTEXT_KEYS:
                return True
            if _contains_prohibited_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_prohibited_key(item) for item in value)
    return False


def _preceding_agent_request(
    events: list[TraceEvent], decision: TraceEvent, *, role: str
) -> TraceEvent | None:
    for event in reversed(events[: decision.sequence - 1]):
        if event.event_type is TraceEventType.AGENT_REQUEST and event.payload.get("role") == role:
            return event
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}:
            break
    return None


def _live_response_between(
    events: list[TraceEvent],
    request: TraceEvent | None,
    decision: TraceEvent,
    *,
    expected_provider: str,
    expected_model: str,
) -> bool:
    if request is None:
        return False
    responses = [
        event
        for event in events
        if request.sequence < event.sequence < decision.sequence
        and event.event_type is TraceEventType.AGENT_RESPONSE
        and event.payload.get("role") == request.payload.get("role")
    ]
    return bool(responses) and all(
        event.payload.get("provider") == expected_provider
        and event.payload.get("model") == expected_model
        for event in responses
    )


def _guardrail_repairs_before(events: list[TraceEvent], decision: TraceEvent, *, role: str) -> int:
    for event in reversed(events[: decision.sequence - 1]):
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}:
            break
        if (
            event.event_type is TraceEventType.FALLBACK
            and event.payload.get("reason_code") == "AGENT_PROSE_GUARDRAIL_REPAIR"
            and event.payload.get("role") == role
        ):
            summary = event.payload.get("repair_summary", {})
            repairs = (
                summary.get("numeric_claims_repaired", {}) if isinstance(summary, dict) else {}
            )
            return sum(int(value) for value in repairs.values()) if isinstance(repairs, dict) else 0
    return 0


def _tool_request_by_id(events: list[TraceEvent], request_id: str) -> TraceEvent | None:
    return next(
        (
            event
            for event in events
            if event.event_type is TraceEventType.TOOL_REQUESTED
            and event.payload.get("request", {}).get("request_id") == request_id
        ),
        None,
    )


def _executed_experiment_for_decision(
    events: list[TraceEvent], decision_id: str
) -> ExperimentType | None:
    event = next(
        (
            item
            for item in events
            if item.event_type is TraceEventType.TOOL_REQUESTED
            and item.payload.get("request", {}).get("agent_decision_id") == decision_id
            and item.payload.get("validation", {}).get("status") == "ALLOWED"
        ),
        None,
    )
    if event is None:
        return None
    return ExperimentType(str(event.payload["request"]["experiment_type"]))


def _fallback_between(
    events: list[TraceEvent], start: int, end: int | None, reason_code: str
) -> bool:
    return any(
        event.event_type is TraceEventType.FALLBACK
        and start < event.sequence < (end or len(events) + 1)
        and event.payload.get("reason_code") == reason_code
        for event in events
    )


def _available_product_roles(events: list[TraceEvent]) -> set[str]:
    initial = _one_event(events, TraceEventType.INVESTIGATION_INITIALIZED)
    if initial is None:
        return set()
    return {str(item) for item in initial.payload.get("available_product_roles", [])}


def _one_event(events: list[TraceEvent], kind: TraceEventType) -> TraceEvent | None:
    matches = [event for event in events if event.event_type is kind]
    return matches[0] if len(matches) == 1 else None


def _sum_usage(events: Iterable[TraceEvent]) -> TokenUsage:
    prompt = completion = total = 0
    for event in events:
        usage = event.payload.get("usage", {})
        if not isinstance(usage, dict):
            continue
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _structured_decision_latency(events: list[TraceEvent]) -> float:
    total = 0.0
    for event in events:
        if event.event_type not in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}:
            continue
        role = "SKEPTIC" if event.event_type is TraceEventType.AGENT_DECISION else "CRITIC"
        request = _preceding_agent_request(events, event, role=role)
        if request is not None:
            total += max(0.0, (event.timestamp - request.timestamp).total_seconds())
    return total


def _measurement_comparisons(
    result: LockedInvestigationResult,
    ledger: EvidenceLedger,
    expectation: LiveRunExpectation,
) -> list[MeasurementComparison]:
    measured: dict[str, float | None] = {
        "period_days": result.candidate.period_days if result.candidate else None,
        "transit_depth_ppm": result.candidate.transit_depth_ppm if result.candidate else None,
        "signal_to_noise": result.candidate.signal_to_noise if result.candidate else None,
    }
    search = next(
        (item for item in ledger.items if item.experiment_type is ExperimentType.TRANSIT_SEARCH),
        None,
    )
    measured["initial_period_days"] = (
        float(search.numerical_results["period_days"])
        if search and "period_days" in search.numerical_results
        else None
    )
    comparisons: list[MeasurementComparison] = []
    for name, reference in expectation.reference_measurements.items():
        value = measured.get(name)
        tolerance = expectation.measurement_tolerances.get(name, 0.0)
        difference = abs(value - reference) if value is not None else None
        comparisons.append(
            MeasurementComparison(
                name=name,
                measured=value,
                reference=reference,
                absolute_difference=difference,
                tolerance=tolerance,
                passed=difference is not None and difference <= tolerance,
            )
        )
    return comparisons


def _blindness_audit(
    directory: Path,
    events: list[TraceEvent],
    ledger: EvidenceLedger,
    base_report: EvaluationReport,
) -> BlindnessAudit:
    grades = {item.name: item for item in base_report.grades}
    access = _one_event(events, TraceEventType.CATALOG_ACCESS_ENABLED)
    cutoff = access.sequence if access else len(events) + 1
    pre_access = [event for event in events if event.sequence < cutoff]
    serialized = json.dumps(
        [event.model_dump(mode="json") for event in pre_access], sort_keys=True, default=str
    )
    ledger_text = "\n".join(item.model_dump_json() for item in ledger.items)
    result_text = (directory / "result.json").read_text(encoding="utf-8")
    identity_hidden = (
        not _IDENTITY_TEXT.search(serialized)
        and not _IDENTITY_TEXT.search(ledger_text)
        and not _IDENTITY_TEXT.search(result_text)
        and grades.get("target_identity_not_exposed") is not None
        and grades["target_identity_not_exposed"].passed
    )
    context_events = [event for event in events if event.event_type is TraceEventType.AGENT_REQUEST]
    context_safe = all(
        _request_context_safe(event, critic=event.payload.get("role") == "CRITIC")
        for event in context_events
    )
    numeric_grade = grades.get("no_unsupported_numerical_measurements")
    known_hidden = context_safe and bool(numeric_grade and numeric_grade.passed)
    catalog_grade = grades.get("catalog_unavailable_before_result_locked")
    catalog_blocked = bool(catalog_grade and catalog_grade.passed)
    details = []
    if not identity_hidden:
        details.append("identity-like text was found in a pre-access artifact")
    if not context_safe:
        details.append("one or more agent request context-key sets were unsafe")
    if not known_hidden:
        details.append("known-parameter isolation or numeric provenance could not be verified")
    return BlindnessAudit(
        target_identity_hidden_before_lock=identity_hidden,
        catalog_access_blocked_before_lock=catalog_blocked,
        known_parameters_hidden_before_lock=known_hidden,
        agent_context_keys_safe=context_safe,
        details=details,
    )


def _integrity_audit(
    directory: Path,
    base_report: EvaluationReport,
    *,
    require_reveal: bool,
) -> IntegrityAudit:
    grades = {item.name: item for item in base_report.grades}
    sha = bool(grades.get("result_hash_valid") and grades["result_hash_valid"].passed)
    reveal_grade = grades.get("post_lock_reveal_valid_and_ordered")
    if require_reveal:
        order = bool(reveal_grade and reveal_grade.passed)
    else:
        order = not (directory / "reveal.json").exists() and reveal_grade is None
    commitment = bool(
        grades.get("locked_result_commits_to_verified_trace")
        and grades["locked_result_commits_to_verified_trace"].passed
    )
    ledger = bool(
        grades.get("locked_result_matches_evidence_ledger")
        and grades["locked_result_matches_evidence_ledger"].passed
        and grades.get("tool_schemas_and_ledger_valid")
        and grades["tool_schemas_and_ledger_valid"].passed
    )
    result_text = (directory / "result.json").read_text(encoding="utf-8")
    reveal_exists = (directory / "reveal.json").is_file()
    separate = (
        (reveal_exists if require_reveal else not reveal_exists)
        and "ground_truth" not in result_text
        and "catalog_status" not in result_text
    )
    details = [item.detail for item in base_report.grades if not item.passed]
    return IntegrityAudit(
        result_locked_before_reveal=order,
        sha256_verified=sha,
        separate_reveal_artifact=separate,
        locked_result_immutable=sha and order and commitment,
        trace_and_ledger_verified=commitment and ledger,
        details=details,
    )


def _collect_errors(events: list[TraceEvent]) -> list[str]:
    errors: list[str] = []
    for event in events:
        if event.event_type is TraceEventType.STRUCTURED_OUTPUT_FAILURE:
            errors.append(
                f"sequence {event.sequence}: {event.payload.get('role')} attempt "
                f"{event.payload.get('attempt')}: {event.payload.get('error')}"
            )
        elif event.event_type is TraceEventType.CONTEXT_REJECTED:
            errors.append(f"sequence {event.sequence}: {event.payload.get('reason')}")
        elif event.event_type is TraceEventType.TOOL_RESULT and event.payload.get("status") not in {
            "SUCCESS",
            "PARTIAL",
        }:
            errors.append(
                f"sequence {event.sequence}: tool result status={event.payload.get('status')} "
                f"reason={event.payload.get('reason')}"
            )
    return errors


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "CriticDecisionAudit",
    "DecisionOrigin",
    "LiveRunExpectation",
    "LiveRunReport",
    "LiveValidationReport",
    "ReliabilitySummary",
    "SkepticDecisionAudit",
    "demo_live_expectation",
    "evaluate_live_run",
    "evaluate_live_trials",
    "render_live_validation_markdown",
]
