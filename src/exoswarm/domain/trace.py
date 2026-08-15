"""Tamper-evident investigation trace with pre-reveal identity redaction checks."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import Field, JsonValue

from .ledger import GENESIS_HASH, canonical_json_bytes, sha256_json
from .models import FrozenDomainModel, utc_now


class TraceEventType(StrEnum):
    INVESTIGATION_INITIALIZED = "INVESTIGATION_INITIALIZED"
    AGENT_REQUEST = "AGENT_REQUEST"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    AGENT_DECISION = "AGENT_DECISION"
    CRITIC_DECISION = "CRITIC_DECISION"
    TOOL_REQUESTED = "TOOL_REQUESTED"
    TOOL_RESULT = "TOOL_RESULT"
    EVIDENCE_APPENDED = "EVIDENCE_APPENDED"
    HYPOTHESIS_UPDATED = "HYPOTHESIS_UPDATED"
    STRUCTURED_OUTPUT_FAILURE = "STRUCTURED_OUTPUT_FAILURE"
    STRUCTURED_OUTPUT_REPAIRED = "STRUCTURED_OUTPUT_REPAIRED"
    CONTEXT_REJECTED = "CONTEXT_REJECTED"
    FALLBACK = "FALLBACK"
    BUDGET_UPDATED = "BUDGET_UPDATED"
    RESULT_LOCKED = "RESULT_LOCKED"
    CATALOG_ACCESS_ENABLED = "CATALOG_ACCESS_ENABLED"
    GROUND_TRUTH_REVEALED = "GROUND_TRUTH_REVEALED"


class TraceEvent(FrozenDomainModel):
    event_id: str = Field(pattern=r"^TE-[A-Z0-9]+$")
    trace_id: str
    sequence: int = Field(ge=1)
    opaque_target_id: str = Field(pattern=r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
    event_type: TraceEventType
    timestamp: datetime
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    previous_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"event_hash"})


class TraceIntegrityError(RuntimeError):
    pass


class TraceSecurityError(RuntimeError):
    pass


_SENSITIVE_KEYS = frozenset(
    {
        "actual_target_identity",
        "real_target_id",
        "tic_id",
        "toi_id",
        "target_name",
        "catalog_status",
        "catalog_measurements",
        "known_period",
        "confirmation_status",
        "ground_truth",
    }
)


def _contains_sensitive_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _SENSITIVE_KEYS:
                return str(key)
            nested = _contains_sensitive_key(child)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _contains_sensitive_key(child)
            if nested:
                return nested
    return None


class TraceRecorder:
    """Append-only JSONL trace that records the catalog capability transition."""

    def __init__(
        self,
        *,
        trace_id: str,
        opaque_target_id: str,
        path: str | Path | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.opaque_target_id = opaque_target_id
        self.path = Path(path).resolve() if path is not None else None
        self._events: list[TraceEvent] = []
        self._lock = threading.RLock()
        self._catalog_access_enabled = False
        if self.path and self.path.exists():
            self._load()
            self.verify()

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(event.model_copy(deep=True) for event in self._events)

    @property
    def root_hash(self) -> str:
        return self._events[-1].event_hash if self._events else GENESIS_HASH

    @property
    def catalog_access_enabled(self) -> bool:
        return self._catalog_access_enabled

    def append(
        self,
        event_type: TraceEventType,
        payload: dict[str, JsonValue] | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> TraceEvent:
        payload = payload or {}
        with self._lock:
            if event_type is TraceEventType.CATALOG_ACCESS_ENABLED:
                if not any(
                    event.event_type is TraceEventType.RESULT_LOCKED for event in self._events
                ):
                    raise TraceSecurityError(
                        "catalog access cannot be enabled before RESULT_LOCKED"
                    )
            elif not self._catalog_access_enabled:
                sensitive_key = _contains_sensitive_key(payload)
                if sensitive_key:
                    raise TraceSecurityError(
                        f"pre-lock trace payload contains sensitive key {sensitive_key!r}"
                    )
            raw: dict[str, object] = {
                "event_id": f"TE-{uuid4().hex.upper()}",
                "trace_id": self.trace_id,
                "sequence": len(self._events) + 1,
                "opaque_target_id": self.opaque_target_id,
                "event_type": event_type.value,
                "timestamp": (timestamp or utc_now()).isoformat(),
                "payload": payload,
                "previous_hash": self.root_hash,
            }
            unhashed_event = TraceEvent.model_validate({**raw, "event_hash": GENESIS_HASH})
            event = unhashed_event.model_copy(
                update={"event_hash": sha256_json(unhashed_event.hash_payload())}
            )
            self._persist(event)
            self._events.append(event)
            if event_type is TraceEventType.CATALOG_ACCESS_ENABLED:
                self._catalog_access_enabled = True
            return event.model_copy(deep=True)

    def verify(self) -> bool:
        previous = GENESIS_HASH
        saw_lock = False
        catalog_enabled = False
        for sequence, event in enumerate(self._events, start=1):
            if event.sequence != sequence or event.previous_hash != previous:
                raise TraceIntegrityError(f"broken trace chain at event {event.event_id}")
            if sha256_json(event.hash_payload()) != event.event_hash:
                raise TraceIntegrityError(f"trace hash mismatch at event {event.event_id}")
            if event.event_type is TraceEventType.RESULT_LOCKED:
                saw_lock = True
            if event.event_type is TraceEventType.CATALOG_ACCESS_ENABLED:
                if not saw_lock:
                    raise TraceIntegrityError("catalog access enabled before result lock")
                catalog_enabled = True
            elif not catalog_enabled:
                sensitive_key = _contains_sensitive_key(event.payload)
                if sensitive_key:
                    raise TraceIntegrityError(
                        f"pre-lock trace contains sensitive key {sensitive_key!r}"
                    )
            previous = event.event_hash
        self._catalog_access_enabled = catalog_enabled
        return True

    def _persist(self, event: TraceEvent) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(
                descriptor,
                canonical_json_bytes(event.model_dump(mode="json")) + b"\n",
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = TraceEvent.model_validate_json(line)
                except Exception as exc:
                    raise TraceIntegrityError(
                        f"invalid trace event at line {line_number}: {exc}"
                    ) from exc
                if event.trace_id != self.trace_id:
                    raise TraceIntegrityError("trace id does not match recorder")
                if event.opaque_target_id != self.opaque_target_id:
                    raise TraceIntegrityError("opaque target id does not match recorder")
                self._events.append(event)
