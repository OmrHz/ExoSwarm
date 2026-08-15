"""Append-only, hash-chained Evidence Ledger.

Every UI measurement and hypothesis update should ultimately point to an item in
this ledger.  The hash chain does not make local files magically immutable, but it
makes deletion, reordering, or mutation detectable and therefore auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import Field, JsonValue

from .models import (
    ArtifactRef,
    ExperimentType,
    FrozenDomainModel,
    InterpretationCode,
    MeasurementUncertainty,
    ProvenanceRecord,
    QualityFlag,
    ScientificResult,
    utc_now,
)

GENESIS_HASH = "0" * 64


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON-compatible data in the canonical form used for hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class EvidenceItem(FrozenDomainModel):
    id: str = Field(pattern=r"^EV-[A-Z0-9]+$")
    sequence: int = Field(ge=1)
    experiment_type: ExperimentType
    tool_name: str
    tool_version: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    input_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    numerical_results: dict[str, int | float] = Field(default_factory=dict)
    result_units: dict[str, str] = Field(default_factory=dict)
    uncertainties: dict[str, MeasurementUncertainty] = Field(default_factory=dict)
    interpretation_code: InterpretationCode
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    timestamp: datetime
    agent_request_id: str | None = None
    critic_decision_id: str | None = None
    previous_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    record_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"record_hash"})


class EvidenceIntegrityError(RuntimeError):
    """The persisted ledger is malformed or its hash chain has been changed."""


class EvidenceLedger:
    """Thread-safe append-only collection with optional durable JSONL storage."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).resolve() if path is not None else None
        self._items: list[EvidenceItem] = []
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            self._items = list(self._read_items(self._path))
            self.verify()

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        # Deep copies ensure a caller cannot mutate a nested dictionary in-place.
        with self._lock:
            return tuple(item.model_copy(deep=True) for item in self._items)

    @property
    def root_hash(self) -> str:
        with self._lock:
            return self._items[-1].record_hash if self._items else GENESIS_HASH

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self):
        return iter(self.items)

    def get(self, evidence_id: str) -> EvidenceItem:
        with self._lock:
            for item in self._items:
                if item.id == evidence_id:
                    return item.model_copy(deep=True)
        raise KeyError(evidence_id)

    def append_result(
        self,
        result: ScientificResult,
        *,
        agent_request_id: str | None = None,
        critic_decision_id: str | None = None,
        evidence_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> EvidenceItem:
        """Append one validated deterministic result and return its immutable record."""

        with self._lock:
            if evidence_id is None:
                evidence_id = f"EV-{uuid4().hex.upper()}"
            if any(existing.id == evidence_id for existing in self._items):
                raise ValueError(f"duplicate evidence id: {evidence_id}")

            input_refs = [artifact.model_dump(mode="json") for artifact in result.input_artifacts]
            input_hash = sha256_json(
                {
                    "input_artifacts": input_refs,
                    "parameters": result.parameters,
                    "tool_name": result.tool_name,
                    "tool_version": result.tool_version,
                }
            )
            payload: dict[str, object] = {
                "id": evidence_id,
                "sequence": len(self._items) + 1,
                "experiment_type": result.experiment_type.value,
                "tool_name": result.tool_name,
                "tool_version": result.tool_version,
                "input_artifact_ids": [a.artifact_id for a in result.input_artifacts],
                "input_hash": input_hash,
                "parameters": result.parameters,
                "numerical_results": result.numerical_results,
                "result_units": result.result_units,
                "uncertainties": {
                    name: uncertainty.model_dump(mode="json")
                    for name, uncertainty in result.uncertainties.items()
                },
                "interpretation_code": result.interpretation_code.value,
                "quality_flags": [flag.model_dump(mode="json") for flag in result.quality_flags],
                "limitations": result.limitations,
                "output_artifact_ids": [a.artifact_id for a in result.output_artifacts],
                "provenance": [p.model_dump(mode="json") for p in result.provenance],
                "timestamp": (timestamp or utc_now()).isoformat(),
                "agent_request_id": agent_request_id,
                "critic_decision_id": critic_decision_id,
                "previous_hash": self.root_hash,
            }
            # Validate first so datetime/enum normalization is part of the exact
            # canonical representation that gets hashed.
            unhashed_item = EvidenceItem.model_validate({**payload, "record_hash": GENESIS_HASH})
            record_hash = sha256_json(unhashed_item.hash_payload())
            item = unhashed_item.model_copy(update={"record_hash": record_hash})
            self._persist(item)
            self._items.append(item)
            return item.model_copy(deep=True)

    def verify(self) -> bool:
        """Verify ordering and every link/hash; raise on the first integrity error."""

        with self._lock:
            previous = GENESIS_HASH
            identifiers: set[str] = set()
            for expected_sequence, item in enumerate(self._items, start=1):
                if item.sequence != expected_sequence:
                    raise EvidenceIntegrityError(
                        f"unexpected sequence {item.sequence}; expected {expected_sequence}"
                    )
                if item.id in identifiers:
                    raise EvidenceIntegrityError(f"duplicate evidence id: {item.id}")
                identifiers.add(item.id)
                if item.previous_hash != previous:
                    raise EvidenceIntegrityError(f"broken previous hash at evidence item {item.id}")
                expected_hash = sha256_json(item.hash_payload())
                if item.record_hash != expected_hash:
                    raise EvidenceIntegrityError(f"record hash mismatch at {item.id}")
                previous = item.record_hash
        return True

    def _persist(self, item: EvidenceItem) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = canonical_json_bytes(item.model_dump(mode="json")) + b"\n"
        # O_APPEND plus fsync means a returned append is durable and never rewrites
        # prior entries. The in-process lock prevents interleaved local writers.
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_items(path: Path) -> Iterable[EvidenceItem]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise EvidenceIntegrityError(f"blank line in ledger at line {line_number}")
                    try:
                        yield EvidenceItem.model_validate_json(line)
                    except Exception as exc:  # validation details are wrapped with location
                        raise EvidenceIntegrityError(
                            f"invalid evidence item at line {line_number}: {exc}"
                        ) from exc
        except UnicodeDecodeError as exc:
            raise EvidenceIntegrityError("ledger is not valid UTF-8") from exc


def artifact_hashes(artifacts: Iterable[ArtifactRef]) -> dict[str, str | None]:
    """Small helper useful to science tools constructing provenance summaries."""

    return {artifact.artifact_id: artifact.sha256 for artifact in artifacts}
