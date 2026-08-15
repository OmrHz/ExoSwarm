"""Deterministic graders for scientific constraints and outcomes, not one exact path."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from exoswarm.domain.hypotheses import MANDATORY_VETTING
from exoswarm.domain.ledger import EvidenceItem, EvidenceLedger
from exoswarm.domain.models import (
    Candidate,
    CriticDecision,
    ExperimentType,
    LockedInvestigationResult,
    ResultLockReceipt,
    RevealArtifact,
    ScientificDisposition,
    SkepticDecision,
)
from exoswarm.domain.numeric_provenance import NumericProvenanceGuard
from exoswarm.domain.trace import TraceEvent, TraceEventType, TraceRecorder
from exoswarm.security.locking import ResultLocker, ResultLockError

_IDENTITY_TEXT = re.compile(r"\b(?:TIC\s*\d+|TOI[-\s]?\d+(?:\.\d+)?)\b", re.IGNORECASE)

_CANDIDATE_EVIDENCE_ALIASES: dict[str, tuple[str, ...]] = {
    "period_days": ("period_days", "preferred_period_days"),
    "epoch_btjd": ("epoch_btjd", "preferred_epoch_btjd"),
    "transit_depth_ppm": (
        "transit_depth_ppm",
        "preferred_primary_depth_ppm",
    ),
    "duration_hours": ("duration_hours", "preferred_duration_hours"),
    "signal_to_noise": ("signal_to_noise", "preferred_signal_to_noise"),
    "observed_events": ("observed_events", "preferred_observed_events"),
    "search_statistic": ("search_statistic", "bls_global_peak_snr"),
}


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationExpectation(EvalModel):
    opaque_target_id: str
    expected_period_days: float | None = Field(default=None, gt=0)
    period_tolerance_days: float | None = Field(default=None, gt=0)
    accepted_dispositions: set[ScientificDisposition] = Field(default_factory=set)
    expected_adaptive_any_of: set[ExperimentType] = Field(default_factory=set)
    forbidden_adaptive: set[ExperimentType] = Field(default_factory=set)
    negative_control: bool = False
    require_reveal: bool = False


class Grade(EvalModel):
    name: str
    passed: bool
    detail: str


class EvaluationReport(EvalModel):
    opaque_target_id: str
    run_directory: str
    grades: list[Grade]
    measured: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(grade.passed for grade in self.grades)

    @property
    def pass_count(self) -> int:
        return sum(grade.passed for grade in self.grades)


def evaluate_run(
    run_directory: str | Path,
    expectation: EvaluationExpectation,
) -> EvaluationReport:
    directory = Path(run_directory).resolve()
    result_path = directory / "result.json"
    hash_path = directory / "result.json.sha256"
    evidence_path = directory / "evidence.jsonl"
    trace_path = directory / "trace.jsonl"
    result = LockedInvestigationResult.model_validate_json(result_path.read_bytes())
    ledger = EvidenceLedger(evidence_path)
    trace = _load_trace(
        trace_path,
        trace_id=result.trace_id,
        opaque_target_id=result.opaque_target_id,
    )
    events = list(trace.events)
    grades: list[Grade] = []

    calculated_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    stored_hash = hash_path.read_text(encoding="ascii").strip().lower()
    receipt = ResultLockReceipt(
        opaque_target_id=result.opaque_target_id,
        result_path=str(result_path),
        hash_path=str(hash_path),
        sha256=calculated_hash,
        locked_at=datetime.fromtimestamp(hash_path.stat().st_mtime, tz=UTC),
    )
    grades.append(
        Grade(
            name="result_hash_valid",
            passed=calculated_hash == stored_hash,
            detail=f"stored={stored_hash}; calculated={calculated_hash}",
        )
    )
    locker = ResultLocker()
    try:
        locker.verify_trace_commitment(receipt, trace)
        trace_commitment_error = None
    except ResultLockError as exc:
        trace_commitment_error = str(exc)
    grades.append(
        Grade(
            name="locked_result_commits_to_verified_trace",
            passed=trace_commitment_error is None,
            detail=(
                f"pre-lock trace root={result.pre_lock_trace_root_hash}"
                if trace_commitment_error is None
                else trace_commitment_error
            ),
        )
    )

    ledger_ids = [item.id for item in ledger.items]
    ledger_matches_result = (
        result.evidence_ids == ledger_ids and result.evidence_root_hash == ledger.root_hash
    )
    grades.append(
        Grade(
            name="locked_result_matches_evidence_ledger",
            passed=ledger_matches_result,
            detail=(
                f"result_items={len(result.evidence_ids)}; ledger_items={len(ledger_ids)}; "
                f"result_root={result.evidence_root_hash}; ledger_root={ledger.root_hash}"
            ),
        )
    )
    grades.append(
        Grade(
            name="opaque_target_matches",
            passed=(
                result.opaque_target_id == expectation.opaque_target_id
                and all(event.opaque_target_id == expectation.opaque_target_id for event in events)
            ),
            detail=f"result target={result.opaque_target_id}",
        )
    )

    lock_indices = _event_indices(events, TraceEventType.RESULT_LOCKED)
    access_indices = _event_indices(events, TraceEventType.CATALOG_ACCESS_ENABLED)
    reveal_indices = _event_indices(events, TraceEventType.GROUND_TRUTH_REVEALED)
    lock_index = lock_indices[0] if len(lock_indices) == 1 else None
    access_index = access_indices[0] if len(access_indices) == 1 else None
    reveal_index = reveal_indices[0] if len(reveal_indices) == 1 else None
    catalog_order_ok = (
        len(lock_indices) == 1
        and len(access_indices) <= 1
        and len(reveal_indices) <= 1
        and (access_index is None or access_index > lock_index)
        and (reveal_index is None or (access_index is not None and reveal_index > access_index))
    )
    grades.append(
        Grade(
            name="catalog_unavailable_before_result_locked",
            passed=catalog_order_ok,
            detail=f"lock={lock_index}; access={access_index}; reveal={reveal_index}",
        )
    )
    reveal_path = directory / "reveal.json"
    if expectation.require_reveal or reveal_path.exists() or reveal_index is not None:
        reveal_valid, reveal_detail = _validate_reveal(
            directory=directory,
            result=result,
            calculated_result_hash=calculated_hash,
            events=events,
            locker=locker,
        )
        grades.append(
            Grade(
                name="post_lock_reveal_valid_and_ordered",
                passed=reveal_valid,
                detail=reveal_detail,
            )
        )

    pre_reveal_events = events[: access_index if access_index is not None else len(events)]
    leaked = [
        event.sequence
        for event in pre_reveal_events
        if _IDENTITY_TEXT.search(json.dumps(event.payload, default=str))
    ]
    result_text = result_path.read_text(encoding="utf-8")
    grades.append(
        Grade(
            name="target_identity_not_exposed",
            passed=not leaked and not _IDENTITY_TEXT.search(result_text),
            detail=f"pre-reveal identity-like trace events={leaked}",
        )
    )

    grades.append(
        Grade(
            name="candidate_recovered",
            passed=result.candidate is not None,
            detail="locked result contains a deterministic candidate"
            if result.candidate
            else "no candidate",
        )
    )
    if expectation.expected_period_days is not None:
        measured_period = result.candidate.period_days if result.candidate else None
        error = (
            abs(measured_period - expectation.expected_period_days)
            if measured_period is not None
            else None
        )
        tolerance = expectation.period_tolerance_days
        grades.append(
            Grade(
                name="period_within_tolerance",
                passed=(error is not None and tolerance is not None and error <= tolerance),
                detail=(
                    f"measured={measured_period}; expected={expectation.expected_period_days}; "
                    f"absolute_error={error}; tolerance={tolerance}"
                ),
            )
        )

    completed = set(result.completed_tests)
    missing_mandatory = MANDATORY_VETTING - completed
    grades.append(
        Grade(
            name="mandatory_diagnostics_executed",
            passed=not missing_mandatory,
            detail=(
                "all mandatory vetting present"
                if not missing_mandatory
                else "missing: " + ", ".join(sorted(item.value for item in missing_mandatory))
            ),
        )
    )
    adaptive_completed = completed & {
        ExperimentType.HARMONIC_TEST,
        ExperimentType.CENTROID_LOCALIZATION,
        ExperimentType.ALTERNATE_DETRENDING,
        ExperimentType.ALTERNATE_APERTURE,
    }
    if expectation.expected_adaptive_any_of:
        grades.append(
            Grade(
                name="evidence_driven_adaptive_branch",
                passed=bool(adaptive_completed & expectation.expected_adaptive_any_of),
                detail="completed adaptive: "
                + ", ".join(sorted(item.value for item in adaptive_completed)),
            )
        )
    forbidden = adaptive_completed & expectation.forbidden_adaptive
    grades.append(
        Grade(
            name="forbidden_or_unnecessary_branch_avoided",
            passed=not forbidden,
            detail=(
                "none" if not forbidden else ", ".join(sorted(item.value for item in forbidden))
            ),
        )
    )

    if expectation.accepted_dispositions:
        grades.append(
            Grade(
                name="scientific_disposition_allowed",
                passed=result.disposition in expectation.accepted_dispositions,
                detail=f"disposition={result.disposition.value}",
            )
        )
    if expectation.negative_control:
        grades.append(
            Grade(
                name="negative_control_not_given_strong_planet_disposition",
                passed=result.disposition
                not in {
                    ScientificDisposition.PLANETARY_INTERPRETATION_PLAUSIBLE,
                    ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING,
                },
                detail=f"disposition={result.disposition.value}",
            )
        )

    initial = next(
        (event for event in events if event.event_type is TraceEventType.INVESTIGATION_INITIALIZED),
        None,
    )
    max_turns = int(initial.payload.get("settings", {}).get("max_agent_turns", 0)) if initial else 0
    decision_turns = sum(
        event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}
        for event in events
    )
    grades.append(
        Grade(
            name="max_agent_turn_budget_respected",
            passed=max_turns > 0 and decision_turns <= max_turns,
            detail=f"decision_turns={decision_turns}; limit={max_turns}",
        )
    )

    provenance_violations = _agent_numeric_violations(events, ledger)
    grades.append(
        Grade(
            name="no_unsupported_numerical_measurements",
            passed=not provenance_violations,
            detail=(
                "all user-visible agent numbers trace to evidence"
                if not provenance_violations
                else "; ".join(provenance_violations)
            ),
        )
    )
    candidate_violations = _candidate_numeric_violations(result.candidate, ledger)
    grades.append(
        Grade(
            name="candidate_measurements_trace_to_evidence",
            passed=not candidate_violations,
            detail=(
                "all locked candidate measurements trace to source evidence"
                if not candidate_violations
                else "; ".join(candidate_violations)
            ),
        )
    )
    grades.append(
        Grade(
            name="tool_schemas_and_ledger_valid",
            passed=ledger.verify(),
            detail=f"evidence_items={len(ledger)}; root_hash={ledger.root_hash}",
        )
    )
    measured = {
        "period_days": result.candidate.period_days if result.candidate else None,
        "disposition": result.disposition.value,
        "completed_tests": [item.value for item in result.completed_tests],
        "evidence_items": len(ledger),
        "agent_decision_turns": decision_turns,
        "trace_events": len(events),
    }
    return EvaluationReport(
        opaque_target_id=result.opaque_target_id,
        run_directory=str(directory),
        grades=grades,
        measured=measured,
    )


def evaluate_trajectory_diversity(
    reports: list[EvaluationReport],
) -> Grade:
    if len(reports) < 2:
        return Grade(
            name="different_cases_produce_different_branches",
            passed=False,
            detail="at least two run reports are required",
        )
    branches = {
        tuple(
            sorted(
                item
                for item in report.measured.get("completed_tests", [])
                if item
                in {
                    ExperimentType.HARMONIC_TEST.value,
                    ExperimentType.CENTROID_LOCALIZATION.value,
                    ExperimentType.ALTERNATE_DETRENDING.value,
                    ExperimentType.ALTERNATE_APERTURE.value,
                }
            )
        )
        for report in reports
    }
    dispositions = {report.measured.get("disposition") for report in reports}
    passed = len(branches) > 1 and len(dispositions) > 1
    return Grade(
        name="different_cases_produce_different_branches",
        passed=passed,
        detail=f"adaptive_branches={sorted(map(list, branches))}; dispositions={sorted(dispositions)}",
    )


def render_markdown(reports: list[EvaluationReport], diversity: Grade | None = None) -> str:
    lines = ["# ExoSwarm evaluation report", ""]
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        lines.extend(
            [
                f"## {report.opaque_target_id} — {status}",
                "",
                "| Grader | Result | Detail |",
                "|---|---:|---|",
            ]
        )
        for grade in report.grades:
            detail = grade.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {grade.name} | {'PASS' if grade.passed else 'FAIL'} | {detail} |")
        lines.append("")
    if diversity is not None:
        lines.extend(
            [
                "## Cross-case trajectory diversity",
                "",
                f"**{'PASS' if diversity.passed else 'FAIL'}** — {diversity.detail}",
                "",
            ]
        )
    return "\n".join(lines)


def _load_trace(
    path: Path,
    *,
    trace_id: str,
    opaque_target_id: str,
) -> TraceRecorder:
    """Load and cryptographically verify every event in a persisted trace."""

    return TraceRecorder(
        trace_id=trace_id,
        opaque_target_id=opaque_target_id,
        path=path,
    )


def _event_indices(events: list[TraceEvent], kind: TraceEventType) -> list[int]:
    return [index for index, event in enumerate(events) if event.event_type is kind]


def _validate_reveal(
    *,
    directory: Path,
    result: LockedInvestigationResult,
    calculated_result_hash: str,
    events: list[TraceEvent],
    locker: ResultLocker,
) -> tuple[bool, str]:
    reveal_path = directory / ResultLocker.REVEAL_NAME
    if not reveal_path.exists():
        return False, "reveal.json is missing"
    try:
        reveal = RevealArtifact.model_validate_json(reveal_path.read_bytes())
        locker.verify_artifact_order(directory)
    except Exception as exc:
        return False, f"invalid reveal artifact: {exc}"

    lock_events = [event for event in events if event.event_type is TraceEventType.RESULT_LOCKED]
    access_events = [
        event for event in events if event.event_type is TraceEventType.CATALOG_ACCESS_ENABLED
    ]
    reveal_events = [
        event for event in events if event.event_type is TraceEventType.GROUND_TRUTH_REVEALED
    ]
    if len(lock_events) != 1 or len(access_events) != 1 or len(reveal_events) != 1:
        return (
            False,
            "reveal requires exactly one lock, catalog-access, and reveal trace event",
        )
    lock_event, access_event, reveal_event = (
        lock_events[0],
        access_events[0],
        reveal_events[0],
    )
    if not (lock_event.sequence < access_event.sequence < reveal_event.sequence):
        return False, "reveal trace events are out of order"
    if reveal.opaque_target_id != result.opaque_target_id:
        return False, "reveal target does not match locked result"
    if reveal.locked_result_sha256 != calculated_result_hash:
        return False, "reveal digest does not match result.json"
    if access_event.payload.get("locked_result_sha256") != calculated_result_hash:
        return False, "catalog-access event references a different result"
    if access_event.payload.get("pre_lock_trace_root_hash") != result.pre_lock_trace_root_hash:
        return False, "catalog-access event does not preserve the trace commitment"
    if reveal_event.payload.get("locked_result_sha256") != calculated_result_hash:
        return False, "reveal event references a different result"
    if reveal_event.payload.get("reveal_artifact") != ResultLocker.REVEAL_NAME:
        return False, "reveal event references a non-canonical artifact"
    if reveal_event.payload.get("ground_truth") != reveal.ground_truth.model_dump(mode="json"):
        return False, "reveal event and reveal.json contain different catalog data"
    if reveal.revealed_at < access_event.timestamp:
        return False, "reveal artifact timestamp precedes catalog access"
    if reveal_event.timestamp < reveal.revealed_at:
        return False, "reveal trace event timestamp precedes reveal artifact"
    return True, "reveal schema, result hash, and artifact/trace ordering verify"


def _candidate_numeric_violations(
    candidate: Candidate | None,
    ledger: EvidenceLedger,
) -> list[str]:
    if candidate is None:
        return []
    violations: list[str] = []
    if not candidate.source_evidence_ids:
        return ["candidate has no source_evidence_ids"]
    if len(candidate.source_evidence_ids) != len(set(candidate.source_evidence_ids)):
        violations.append("candidate contains duplicate source_evidence_ids")

    source_items: list[EvidenceItem] = []
    ledger_ids = {item.id for item in ledger.items}
    for evidence_id in candidate.source_evidence_ids:
        if evidence_id not in ledger_ids:
            violations.append(f"unknown candidate source evidence {evidence_id}")
            continue
        source_items.append(ledger.get(evidence_id))

    for field_name, aliases in _CANDIDATE_EVIDENCE_ALIASES.items():
        candidate_value = getattr(candidate, field_name)
        if candidate_value is None:
            continue
        supported = any(
            alias in item.numerical_results
            and _numbers_equal(candidate_value, item.numerical_results[alias])
            for item in source_items
            for alias in aliases
        )
        if not supported:
            violations.append(
                f"{field_name}={candidate_value} is absent from candidate source evidence"
            )

    for field_name, uncertainty in candidate.uncertainties.items():
        aliases = _CANDIDATE_EVIDENCE_ALIASES.get(field_name, (field_name,))
        supported = any(
            alias in item.uncertainties
            and item.uncertainties[alias].model_dump(mode="json")
            == uncertainty.model_dump(mode="json")
            for item in source_items
            for alias in aliases
        )
        if not supported:
            violations.append(f"uncertainty {field_name} is absent from candidate source evidence")
    return violations


def _numbers_equal(left: int | float, right: int | float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _agent_numeric_violations(events: list[TraceEvent], ledger: EvidenceLedger) -> list[str]:
    guard = NumericProvenanceGuard(ledger)
    violations: list[str] = []
    for event in events:
        texts: list[tuple[str, str]] = []
        if event.event_type is TraceEventType.AGENT_DECISION:
            decision = SkepticDecision.model_validate(event.payload["decision"])
            texts.extend(
                [
                    ("explanation", decision.explanation),
                    ("expected_discriminating_result", decision.expected_discriminating_result),
                    *(
                        (f"predicted_outcomes.{key}", value)
                        for key, value in decision.predicted_outcomes.items()
                    ),
                ]
            )
            if decision.stop_if:
                texts.append(("stop_if", decision.stop_if))
        elif event.event_type is TraceEventType.CRITIC_DECISION:
            decision = CriticDecision.model_validate(event.payload["decision"])
            texts.append(("reason", decision.reason))
        for field, text in texts:
            report = guard.validate(text)
            if report.violations:
                violations.append(
                    f"trace sequence {event.sequence} {field}: "
                    + ", ".join(item.raw for item in report.violations)
                )
    return violations


__all__ = [
    "EvaluationExpectation",
    "EvaluationReport",
    "Grade",
    "evaluate_run",
    "evaluate_trajectory_diversity",
    "render_markdown",
]
