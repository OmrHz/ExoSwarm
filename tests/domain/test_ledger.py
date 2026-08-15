from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from exoswarm.domain import (
    EvidenceIntegrityError,
    EvidenceLedger,
    ExperimentType,
    InterpretationCode,
    ScientificResult,
    ScientificStatus,
)


def result(period: float = 6.26814) -> ScientificResult:
    return ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.TRANSIT_SEARCH,
        tool_name="astropy_bls",
        tool_version="7.0",
        parameters={"min_period_days": 0.5},
        numerical_results={"period_days": period, "snr": 12.8},
        result_units={"period_days": "day"},
        interpretation_code=InterpretationCode.DETECTED,
        limitations=["BLS precision is not a confirmation."],
    )


def test_ledger_is_append_only_hash_chained_and_persistent(tmp_path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    first = ledger.append_result(
        result(),
        evidence_id="EV-ONE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = ledger.append_result(
        result(3.14),
        evidence_id="EV-TWO",
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert second.previous_hash == first.record_hash
    assert ledger.verify()
    assert EvidenceLedger(path).root_hash == second.record_hash
    assert isinstance(ledger.items, tuple)


def test_nested_mutation_of_returned_item_does_not_modify_ledger() -> None:
    ledger = EvidenceLedger()
    ledger.append_result(result(), evidence_id="EV-ONE")
    returned = ledger.items[0]
    returned.numerical_results["period_days"] = 999
    assert ledger.get("EV-ONE").numerical_results["period_days"] == 6.26814
    assert ledger.verify()


def test_tampering_is_detected_on_reload(tmp_path) -> None:
    path = tmp_path / "evidence.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append_result(result(), evidence_id="EV-ONE")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["numerical_results"]["period_days"] = 99
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError, match="hash mismatch"):
        EvidenceLedger(path)


def test_duplicate_evidence_id_rejected() -> None:
    ledger = EvidenceLedger()
    ledger.append_result(result(), evidence_id="EV-ONE")
    with pytest.raises(ValueError, match="duplicate evidence"):
        ledger.append_result(result(), evidence_id="EV-ONE")
