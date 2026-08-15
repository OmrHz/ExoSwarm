from __future__ import annotations

import pytest

from exoswarm.domain import (
    TraceEventType,
    TraceRecorder,
    TraceSecurityError,
)


def test_trace_records_hash_chain_and_catalog_boundary(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(trace_id="TRACE-1", opaque_target_id="TARGET-X17", path=path)
    trace.append(TraceEventType.INVESTIGATION_INITIALIZED, {"experiment_budget": 8})
    with pytest.raises(TraceSecurityError, match="sensitive key"):
        trace.append(TraceEventType.AGENT_REQUEST, {"tic_id": "123"})
    with pytest.raises(TraceSecurityError, match="before RESULT_LOCKED"):
        trace.append(TraceEventType.CATALOG_ACCESS_ENABLED, {})

    trace.append(TraceEventType.RESULT_LOCKED, {"result_sha256": "a" * 64})
    trace.append(TraceEventType.CATALOG_ACCESS_ENABLED, {"result_sha256": "a" * 64})
    trace.append(
        TraceEventType.GROUND_TRUTH_REVEALED,
        {"ground_truth": {"actual_target_identity": "TIC 123"}},
    )
    assert trace.verify()
    restored = TraceRecorder(trace_id="TRACE-1", opaque_target_id="TARGET-X17", path=path)
    assert restored.catalog_access_enabled
    assert len(restored.events) == 4
