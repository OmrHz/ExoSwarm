from __future__ import annotations

import hashlib

import pytest

from exoswarm.domain import (
    ArtifactRef,
    CatalogMeasurement,
    ExperimentType,
    GroundTruthRecord,
    LockedInvestigationResult,
    ScientificDisposition,
    TraceEventType,
    TraceRecorder,
)
from exoswarm.security import (
    GroundTruthGate,
    OpaqueTargetVault,
    ResultLocker,
    ResultLockError,
    RevealOrderError,
)


def result(
    disposition=ScientificDisposition.PLANETARY_INTERPRETATION_PLAUSIBLE,
    *,
    pre_lock_trace_root_hash: str = "0" * 64,
):
    return LockedInvestigationResult(
        opaque_target_id="TARGET-X17",
        trace_id="TRACE-LOCK",
        disposition=disposition,
        completed_tests=[
            ExperimentType.TRANSIT_SEARCH,
            ExperimentType.ODD_EVEN,
            ExperimentType.SECONDARY_ECLIPSE,
        ],
        evidence_ids=["EV-1", "EV-2"],
        evidence_root_hash="b" * 64,
        pre_lock_trace_root_hash=pre_lock_trace_root_hash,
        limitations=["Photometric vetting alone does not constitute confirmation."],
    )


def truth() -> GroundTruthRecord:
    return GroundTruthRecord(
        actual_target_identity="TIC 999",
        catalog_name="NASA Exoplanet Archive",
        catalog_status="CONFIRMED PLANET",
        measurements={"period": CatalogMeasurement(value=3.14, unit="day")},
    )


def test_atomic_result_hash_then_reveal_order(tmp_path) -> None:
    run_dir = tmp_path / "TARGET-X17"
    trace = TraceRecorder(
        trace_id="TRACE-LOCK",
        opaque_target_id="TARGET-X17",
        path=run_dir / "trace.jsonl",
    )
    locker = ResultLocker()
    receipt = locker.lock(run_dir, result(), trace=trace)
    result_bytes = (run_dir / "result.json").read_bytes()
    assert hashlib.sha256(result_bytes).hexdigest() == receipt.sha256
    assert (run_dir / "result.json.sha256").read_text().strip() == receipt.sha256
    assert not (run_dir / "reveal.json").exists()
    assert trace.events[-1].event_type is TraceEventType.RESULT_LOCKED
    locked = locker.verify_trace_commitment(receipt, trace)
    assert trace.events[-1].previous_hash == locked.pre_lock_trace_root_hash
    assert trace.events[-1].payload["pre_lock_trace_root_hash"] == locked.pre_lock_trace_root_hash

    vault = OpaqueTargetVault()
    vault.register(
        opaque_target_id="TARGET-X17",
        real_target_identity="TIC 999",
        artifacts=[ArtifactRef(artifact_id="LC", path="cache/lc.fits")],
        ground_truth=truth(),
    )
    gate = GroundTruthGate(vault, locker)
    gate.unlock_after_result_lock(receipt, trace=trace)
    artifact = gate.create_reveal_artifact(receipt, trace=trace)
    assert artifact.locked_result_sha256 == receipt.sha256
    assert (run_dir / "reveal.json").exists()
    assert locker.verify_artifact_order(run_dir)
    event_types = [event.event_type for event in trace.events]
    assert (
        event_types.index(TraceEventType.RESULT_LOCKED)
        < event_types.index(TraceEventType.CATALOG_ACCESS_ENABLED)
        < event_types.index(TraceEventType.GROUND_TRUTH_REVEALED)
    )


def test_result_and_reveal_are_immutable(tmp_path) -> None:
    locker = ResultLocker()
    trace = TraceRecorder(
        trace_id="TRACE-LOCK",
        opaque_target_id="TARGET-X17",
        path=tmp_path / "trace.jsonl",
    )
    receipt = locker.lock(tmp_path, result(), trace=trace)
    with pytest.raises(ResultLockError, match="immutable"):
        locker.lock(
            tmp_path,
            result(ScientificDisposition.PLANETARY_INTERPRETATION_WEAK),
        )

    trace.append(
        TraceEventType.CATALOG_ACCESS_ENABLED,
        {
            "locked_result_sha256": receipt.sha256,
            "pre_lock_trace_root_hash": "0" * 64,
        },
    )
    locker.write_reveal(receipt, ground_truth=truth(), trace=trace)
    changed_truth = truth().model_copy(update={"catalog_status": "FALSE POSITIVE"})
    with pytest.raises(RevealOrderError, match="immutable"):
        locker.write_reveal(receipt, ground_truth=changed_truth, trace=trace)


def test_preexisting_reveal_blocks_lock(tmp_path) -> None:
    (tmp_path / "reveal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RevealOrderError, match="before result lock"):
        ResultLocker().lock(tmp_path, result())


def test_lock_rejects_a_result_that_does_not_commit_to_trace(tmp_path) -> None:
    trace = TraceRecorder(
        trace_id="TRACE-LOCK",
        opaque_target_id="TARGET-X17",
        path=tmp_path / "trace.jsonl",
    )
    trace.append(TraceEventType.INVESTIGATION_INITIALIZED, {"blind": True})

    with pytest.raises(ResultLockError, match="pre-lock trace root"):
        ResultLocker().lock(tmp_path, result(), trace=trace)
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.json.sha256").exists()
