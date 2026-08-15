from __future__ import annotations

import json

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
    GroundTruthAccessDenied,
    GroundTruthGate,
    OpaqueTargetVault,
    ResultLocker,
)


def ground_truth() -> GroundTruthRecord:
    return GroundTruthRecord(
        actual_target_identity="TIC 123456789",
        catalog_name="NASA Exoplanet Archive",
        catalog_status="CONFIRMED PLANET",
        measurements={"period": CatalogMeasurement(value=6.2679, unit="day")},
    )


def register_target() -> tuple[OpaqueTargetVault, object]:
    vault = OpaqueTargetVault()
    mapping = vault.register(
        opaque_target_id="TARGET-X17",
        real_target_identity="TIC 123456789",
        artifacts=[
            ArtifactRef(
                artifact_id="LC-1", path="cache/TARGET-X17/lightcurve.fits", role="light_curve"
            )
        ],
        ground_truth=ground_truth(),
    )
    return vault, mapping


def locked_result() -> LockedInvestigationResult:
    return LockedInvestigationResult(
        opaque_target_id="TARGET-X17",
        trace_id="TRACE-1",
        disposition=ScientificDisposition.INCONCLUSIVE,
        completed_tests=[ExperimentType.TRANSIT_SEARCH],
        evidence_ids=["EV-1"],
        evidence_root_hash="a" * 64,
        pre_lock_trace_root_hash="0" * 64,
        limitations=["Photometric vetting is not professional confirmation."],
    )


def test_catalog_unreachable_before_lock(tmp_path) -> None:
    vault, _ = register_target()
    gate = GroundTruthGate(vault, ResultLocker())
    assert not gate.is_available("TARGET-X17")
    with pytest.raises(GroundTruthAccessDenied, match="before RESULT_LOCKED"):
        gate.lookup("TARGET-X17")
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "reveal.json").exists()


def test_target_identity_not_exposed() -> None:
    vault, mapping = register_target()
    agent_json = vault.agent_context("TARGET-X17").model_dump_json()
    science_json = vault.science_data("TARGET-X17").model_dump_json()
    mapping_json = mapping.model_dump_json()
    combined = " ".join((agent_json, science_json, mapping_json, repr(vault)))
    assert "123456789" not in combined
    assert "confirmed" not in combined.casefold()
    assert "catalog" not in agent_json.casefold()
    assert not hasattr(vault, "resolve_identity")


def test_gate_unlocks_only_with_verified_lock_receipt(tmp_path) -> None:
    vault, _ = register_target()
    locker = ResultLocker()
    gate = GroundTruthGate(vault, locker)
    trace = TraceRecorder(
        trace_id="TRACE-1",
        opaque_target_id="TARGET-X17",
        path=tmp_path / "trace.jsonl",
    )
    receipt = locker.lock(tmp_path, locked_result(), trace=trace)
    gate.unlock_after_result_lock(receipt, trace=trace)
    record = gate.lookup("TARGET-X17")
    assert record.actual_target_identity == "TIC 123456789"

    # Changing the locked result invalidates the capability on every lookup.
    result_path = tmp_path / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["limitations"] = ["tampered"]
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="SHA-256"):
        gate.lookup("TARGET-X17")


def test_gate_rejects_lock_without_a_verified_trace_commitment(tmp_path) -> None:
    vault, _ = register_target()
    locker = ResultLocker()
    gate = GroundTruthGate(vault, locker)
    receipt = locker.lock(tmp_path, locked_result())

    with pytest.raises(GroundTruthAccessDenied, match="committed trace"):
        gate.unlock_after_result_lock(receipt)
    assert not gate.is_available("TARGET-X17")


def test_gate_rejects_catalog_event_with_wrong_trace_commitment(tmp_path) -> None:
    vault, _ = register_target()
    locker = ResultLocker()
    gate = GroundTruthGate(vault, locker)
    trace = TraceRecorder(
        trace_id="TRACE-1",
        opaque_target_id="TARGET-X17",
        path=tmp_path / "trace.jsonl",
    )
    receipt = locker.lock(tmp_path, locked_result(), trace=trace)
    trace.append(
        TraceEventType.CATALOG_ACCESS_ENABLED,
        {
            "locked_result_sha256": receipt.sha256,
            "pre_lock_trace_root_hash": "f" * 64,
        },
    )

    with pytest.raises(GroundTruthAccessDenied, match="different result"):
        gate.unlock_after_result_lock(receipt, trace=trace)
    assert not gate.is_available("TARGET-X17")
