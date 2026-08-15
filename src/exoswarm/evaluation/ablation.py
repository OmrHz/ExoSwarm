"""Measured adaptive-versus-fixed policy comparison over completed run artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from exoswarm.domain.models import (
    ExperimentType,
    LockedInvestigationResult,
    ResultLockReceipt,
    ScientificDisposition,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder
from exoswarm.security.locking import ResultLocker

from .graders import EvalModel

_ADAPTIVE = {
    ExperimentType.HARMONIC_TEST,
    ExperimentType.CENTROID_LOCALIZATION,
    ExperimentType.ALTERNATE_DETRENDING,
    ExperimentType.ALTERNATE_APERTURE,
}


class PolicyRunMetrics(EvalModel):
    policy: str
    opaque_target_id: str
    final_disposition: ScientificDisposition
    experiments_executed: int
    adaptive_experiments: list[ExperimentType] = Field(default_factory=list)
    repeated_experiments: int
    agent_decision_turns: int
    schema_valid_decisions: int
    structured_output_failures: int
    fallback_events: int
    total_tokens: int | None = None
    latency_seconds: float
    expected_action_selected: bool | None = None


class AblationComparison(EvalModel):
    adaptive: PolicyRunMetrics
    fixed: PolicyRunMetrics
    notes: list[str]


def measure_policy_run(
    run_directory: str | Path,
    *,
    expected_best_action: ExperimentType | None = None,
) -> PolicyRunMetrics:
    directory = Path(run_directory).resolve()
    result = LockedInvestigationResult.model_validate_json((directory / "result.json").read_bytes())
    result_path = directory / ResultLocker.RESULT_NAME
    hash_path = directory / ResultLocker.HASH_NAME
    trace = TraceRecorder(
        trace_id=result.trace_id,
        opaque_target_id=result.opaque_target_id,
        path=directory / "trace.jsonl",
    )
    receipt = ResultLockReceipt(
        opaque_target_id=result.opaque_target_id,
        result_path=str(result_path),
        hash_path=str(hash_path),
        sha256=hash_path.read_text(encoding="ascii").strip().lower(),
        locked_at=datetime.fromtimestamp(hash_path.stat().st_mtime, tz=UTC),
    )
    ResultLocker().verify_trace_commitment(receipt, trace)
    events = list(trace.events)
    initial = next(
        event for event in events if event.event_type is TraceEventType.INVESTIGATION_INITIALIZED
    )
    policy = str(initial.payload.get("investigation_policy", "unknown"))
    completed = result.completed_tests
    adaptive = [item for item in completed if item in _ADAPTIVE]
    repeated = len(completed) - len(set(completed))
    decision_events = [
        event
        for event in events
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}
    ]
    valid_decisions = 0
    for event in decision_events:
        decision = event.payload.get("decision")
        if isinstance(decision, dict):
            valid_decisions += 1
    usage_values: list[int] = []
    for event in events:
        if event.event_type is not TraceEventType.AGENT_RESPONSE:
            continue
        usage = event.payload.get("usage", {})
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            usage_values.append(int(usage["total_tokens"]))
    start = min(event.timestamp for event in events)
    end = max(event.timestamp for event in events)
    return PolicyRunMetrics(
        policy=policy,
        opaque_target_id=result.opaque_target_id,
        final_disposition=result.disposition,
        experiments_executed=len(completed),
        adaptive_experiments=adaptive,
        repeated_experiments=repeated,
        agent_decision_turns=len(decision_events),
        schema_valid_decisions=valid_decisions,
        structured_output_failures=sum(
            event.event_type is TraceEventType.STRUCTURED_OUTPUT_FAILURE for event in events
        ),
        fallback_events=sum(event.event_type is TraceEventType.FALLBACK for event in events),
        total_tokens=sum(usage_values) if usage_values else None,
        latency_seconds=max(0.0, (end - start).total_seconds()),
        expected_action_selected=(
            expected_best_action in adaptive if expected_best_action is not None else None
        ),
    )


def compare_policy_runs(
    adaptive_directory: str | Path,
    fixed_directory: str | Path,
    *,
    expected_best_action: ExperimentType | None = None,
) -> AblationComparison:
    adaptive = measure_policy_run(adaptive_directory, expected_best_action=expected_best_action)
    fixed = measure_policy_run(fixed_directory, expected_best_action=expected_best_action)
    if adaptive.opaque_target_id != fixed.opaque_target_id:
        raise ValueError("ablation runs must use the same opaque target")
    notes = [
        "Expected-best-action labels are curated scientific test assumptions, not calibrated probabilities.",
        "Latency includes local scientific computation and artifact I/O for each recorded run.",
        "No efficiency advantage is inferred when measured counts are equal.",
    ]
    if adaptive.experiments_executed == fixed.experiments_executed:
        notes.append("The two policies executed the same number of experiments on this target.")
    if adaptive.final_disposition == fixed.final_disposition:
        notes.append("The two policies reached the same categorical disposition on this target.")
    return AblationComparison(adaptive=adaptive, fixed=fixed, notes=notes)


def render_ablation_markdown(comparisons: list[AblationComparison]) -> str:
    lines = [
        "# Adaptive vs fixed-checklist ablation",
        "",
        "These are measured run artifacts. Equal results remain equal; no advantage is manufactured.",
        "",
        "| Target | Policy | Disposition | Experiments | Adaptive tests | Repeats | Valid decisions | Failures | Fallbacks | Tokens | Latency (s) | Expected action |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        for item in (comparison.adaptive, comparison.fixed):
            action = (
                "n/a"
                if item.expected_action_selected is None
                else ("yes" if item.expected_action_selected else "no")
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        item.opaque_target_id,
                        item.policy,
                        item.final_disposition.value,
                        str(item.experiments_executed),
                        ", ".join(value.value for value in item.adaptive_experiments) or "none",
                        str(item.repeated_experiments),
                        str(item.schema_valid_decisions),
                        str(item.structured_output_failures),
                        str(item.fallback_events),
                        str(item.total_tokens) if item.total_tokens is not None else "n/a",
                        f"{item.latency_seconds:.3f}",
                        action,
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    for note in dict.fromkeys(note for comparison in comparisons for note in comparison.notes):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "AblationComparison",
    "PolicyRunMetrics",
    "compare_policy_runs",
    "measure_policy_run",
    "render_ablation_markdown",
]
