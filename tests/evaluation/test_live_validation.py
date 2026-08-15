from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from exoswarm.domain.models import GroundTruthRecord, LockedInvestigationResult, ProvenanceRecord
from exoswarm.domain.trace import TraceEventType, TraceRecorder
from exoswarm.evaluation.live_validation import (
    DecisionOrigin,
    evaluate_live_run,
    evaluate_live_trials,
    render_live_validation_markdown,
)
from exoswarm.security.locking import ResultLocker

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "featherless"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _live_copy(source: Path, destination: Path, *, repaired_role: str | None = None) -> Path:
    """Replay a verified reference trajectory with live-provider trace metadata."""

    source_result = LockedInvestigationResult.model_validate_json(
        (source / "result.json").read_bytes()
    )
    source_trace = TraceRecorder(
        trace_id=source_result.trace_id,
        opaque_target_id=source_result.opaque_target_id,
        path=source / "trace.jsonl",
    )
    destination.mkdir(parents=True)
    shutil.copy2(source / "evidence.jsonl", destination / "evidence.jsonl")
    trace_id = f"TRACE-{uuid4().hex.upper()}"
    trace = TraceRecorder(
        trace_id=trace_id,
        opaque_target_id=source_result.opaque_target_id,
        path=destination / "trace.jsonl",
    )
    for event in source_trace.events:
        if event.event_type is TraceEventType.RESULT_LOCKED:
            break
        if event.event_type in {
            TraceEventType.STRUCTURED_OUTPUT_FAILURE,
            TraceEventType.STRUCTURED_OUTPUT_REPAIRED,
        }:
            continue
        if (
            event.event_type is TraceEventType.FALLBACK
            and event.payload.get("source_event") == "agent_fallback"
        ):
            continue
        payload = event.model_dump(mode="json")["payload"]
        if event.event_type is TraceEventType.INVESTIGATION_INITIALIZED:
            payload["settings"].update(
                {"provider": PROVIDER, "model": MODEL, "provider_enabled": True}
            )
        if event.event_type is TraceEventType.AGENT_REQUEST:
            trace.append(event.event_type, payload)
            role = str(payload["role"])
            attempts = 2 if repaired_role == role else 1
            trace.append(
                TraceEventType.AGENT_RESPONSE,
                {
                    "source_event": "agent_response",
                    "role": role,
                    "attempt": 1,
                    "provider": PROVIDER,
                    "model": MODEL,
                    "request_id": f"live-{role.lower()}-1",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "total_tokens": 125,
                    },
                },
            )
            if attempts == 2:
                trace.append(
                    TraceEventType.STRUCTURED_OUTPUT_FAILURE,
                    {
                        "source_event": "structured_output_failure",
                        "role": role,
                        "attempt": 1,
                        "error": "ValidationError: repairable fixture",
                    },
                )
                trace.append(
                    TraceEventType.AGENT_RESPONSE,
                    {
                        "source_event": "agent_response",
                        "role": role,
                        "attempt": 2,
                        "provider": PROVIDER,
                        "model": MODEL,
                        "request_id": f"live-{role.lower()}-2",
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 25,
                            "total_tokens": 125,
                        },
                    },
                )
                trace.append(
                    TraceEventType.STRUCTURED_OUTPUT_REPAIRED,
                    {
                        "source_event": "structured_output_repaired",
                        "role": role,
                        "attempt": 2,
                    },
                )
            continue
        if event.event_type in {
            TraceEventType.AGENT_DECISION,
            TraceEventType.CRITIC_DECISION,
        }:
            role = str(payload["role"])
            payload.update(
                {
                    "used_fallback": False,
                    "repaired": repaired_role == role,
                    "attempts": 2 if repaired_role == role else 1,
                    "decision_origin": (
                        DecisionOrigin.REPAIRED_LIVE_MODEL.value
                        if repaired_role == role
                        else DecisionOrigin.LIVE_MODEL.value
                    ),
                }
            )
        trace.append(event.event_type, payload)

    locked = source_result.model_copy(
        update={"trace_id": trace_id, "pre_lock_trace_root_hash": trace.root_hash}
    )
    locker = ResultLocker()
    receipt = locker.lock(destination, locked, trace=trace)
    trace.append(
        TraceEventType.CATALOG_ACCESS_ENABLED,
        {
            "locked_result_sha256": receipt.sha256,
            "pre_lock_trace_root_hash": locked.pre_lock_trace_root_hash,
        },
    )
    locker.write_reveal(
        receipt,
        ground_truth=GroundTruthRecord(
            actual_target_identity="TIC 999999999",
            catalog_name="post-lock test catalog",
            catalog_status="TEST STATUS",
            provenance=[ProvenanceRecord(source="unit test")],
        ),
        trace=trace,
    )
    return destination


def test_offline_reference_is_not_misreported_as_live() -> None:
    report = evaluate_live_run(ROOT / "runs" / "TARGET-X17")

    assert not report.live_validation_passed
    assert report.deterministic_fallbacks == 2
    assert {item.origin for item in report.skeptic_decisions} == {
        DecisionOrigin.DETERMINISTIC_FALLBACK
    }
    assert not report.skeptic_decisions[0].live_provider_verified


def test_live_run_audit_classifies_first_pass_and_repaired_decisions(tmp_path: Path) -> None:
    first_pass = _live_copy(ROOT / "runs" / "TARGET-X17", tmp_path / "first-pass")
    repaired = _live_copy(
        ROOT / "runs" / "TARGET-X17", tmp_path / "repaired", repaired_role="SKEPTIC"
    )

    first_report = evaluate_live_run(first_pass)
    repaired_report = evaluate_live_run(repaired)

    assert first_report.live_validation_passed
    assert first_report.first_pass_valid_responses == 2
    assert first_report.live_llm_calls == 2
    assert first_report.token_usage.total_tokens == 250
    assert first_report.skeptic_decisions[0].experiment_executed.value == ("centroid_localization")
    assert repaired_report.live_validation_passed
    assert repaired_report.repair_attempts == 1
    assert repaired_report.successful_repairs == 1
    assert repaired_report.skeptic_decisions[0].origin is DecisionOrigin.REPAIRED_LIVE_MODEL


def test_six_trial_aggregate_requires_independent_live_runs_and_diverse_paths(
    tmp_path: Path,
) -> None:
    directories = []
    for target in ("TARGET-X17", "TARGET-X42"):
        for trial in range(3):
            directories.append(
                _live_copy(
                    ROOT / "runs" / target,
                    tmp_path / target / f"trial-{trial + 1}",
                )
            )

    report = evaluate_live_trials(directories)
    markdown = render_live_validation_markdown(report)

    assert report.passed
    assert report.independent_run_ids
    assert report.required_trial_counts_met
    assert report.different_evidence_changed_trajectory
    assert report.overall_live_model_valid_trajectories == 6
    assert report.reliability.total_structured_requests == 12
    assert report.reliability.deterministic_fallbacks == 0
    assert "Overall: PASS" in markdown

    duplicate = evaluate_live_trials([directories[0]] * 3 + directories[3:])
    assert not duplicate.independent_run_ids
    assert not duplicate.passed
