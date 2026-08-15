"""Atomic pre-reveal result locking and ordered ground-truth reveal artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from exoswarm.domain.ledger import canonical_json_bytes
from exoswarm.domain.models import (
    GroundTruthRecord,
    LockedInvestigationResult,
    ResultLockReceipt,
    RevealArtifact,
    utc_now,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder


class ResultLockError(RuntimeError):
    pass


class RevealOrderError(ResultLockError):
    pass


_IDENTITY_KEYS = frozenset(
    {
        "actual_target_identity",
        "real_target_identity",
        "real_target_id",
        "tic_id",
        "toi_id",
        "target_name",
        "catalog_status",
        "ground_truth",
        "known_period",
        "confirmation_status",
    }
)


def _find_forbidden_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _IDENTITY_KEYS:
                return str(key)
            nested = _find_forbidden_key(child)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _find_forbidden_key(child)
            if nested:
                return nested
    return None


class ResultLocker:
    """Writes an immutable ``result.json`` + SHA-256 before any reveal is possible."""

    RESULT_NAME = "result.json"
    HASH_NAME = "result.json.sha256"
    REVEAL_NAME = "reveal.json"

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def lock(
        self,
        run_directory: str | Path,
        result: LockedInvestigationResult,
        *,
        trace: TraceRecorder | None = None,
        locked_at: datetime | None = None,
    ) -> ResultLockReceipt:
        directory = Path(run_directory).resolve()
        result_path = directory / self.RESULT_NAME
        hash_path = directory / self.HASH_NAME
        reveal_path = directory / self.REVEAL_NAME
        payload = result.model_dump(mode="json")
        forbidden = _find_forbidden_key(payload)
        if forbidden:
            raise ResultLockError(
                f"pre-reveal result contains forbidden identity/catalog key {forbidden!r}"
            )
        serialized = canonical_json_bytes(payload) + b"\n"
        digest = hashlib.sha256(serialized).hexdigest()

        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            if reveal_path.exists():
                raise RevealOrderError("reveal.json exists before result lock")
            if hash_path.exists() and not result_path.exists():
                raise ResultLockError("orphaned result hash without result.json")

            existing_lock_events = ()
            if trace is not None:
                self._validate_trace_identity(result, trace)
                trace.verify()
                existing_lock_events = tuple(
                    event
                    for event in trace.events
                    if event.event_type is TraceEventType.RESULT_LOCKED
                )
                if existing_lock_events:
                    if len(existing_lock_events) != 1:
                        raise ResultLockError("trace must contain exactly one result-lock event")
                    if not result_path.exists() or not hash_path.exists():
                        raise ResultLockError(
                            "trace records RESULT_LOCKED before lock artifacts exist"
                        )
                elif trace.root_hash != result.pre_lock_trace_root_hash:
                    raise ResultLockError(
                        "locked result does not commit to the current pre-lock trace root"
                    )

            self._atomic_write_once(result_path, serialized)
            # The digest file is the commit marker: a result is not considered
            # locked until both files exist and verify.
            self._atomic_write_once(hash_path, f"{digest}\n".encode("ascii"))
            receipt = ResultLockReceipt(
                opaque_target_id=result.opaque_target_id,
                result_path=str(result_path),
                hash_path=str(hash_path),
                sha256=digest,
                locked_at=locked_at or utc_now(),
            )
            self.verify_receipt(receipt)
            if trace is not None:
                if not existing_lock_events:
                    trace.append(
                        TraceEventType.RESULT_LOCKED,
                        {
                            "result_sha256": digest,
                            "pre_lock_trace_root_hash": result.pre_lock_trace_root_hash,
                            "result_artifact": self.RESULT_NAME,
                            "hash_artifact": self.HASH_NAME,
                        },
                    )
                self.verify_trace_commitment(receipt, trace)
            return receipt

    def verify_receipt(self, receipt: ResultLockReceipt) -> LockedInvestigationResult:
        result_path = Path(receipt.result_path).resolve()
        hash_path = Path(receipt.hash_path).resolve()
        if result_path.name != self.RESULT_NAME or hash_path.name != self.HASH_NAME:
            raise ResultLockError("receipt does not reference canonical lock artifact names")
        if result_path.parent != hash_path.parent:
            raise ResultLockError("result and hash must reside in the same run directory")
        try:
            serialized = result_path.read_bytes()
            stored_hash = hash_path.read_text(encoding="ascii").strip().lower()
        except FileNotFoundError as exc:
            raise ResultLockError("result lock artifacts are incomplete") from exc
        calculated = hashlib.sha256(serialized).hexdigest()
        if stored_hash != calculated or receipt.sha256.lower() != calculated:
            raise ResultLockError("result SHA-256 verification failed")
        try:
            result = LockedInvestigationResult.model_validate_json(serialized)
        except Exception as exc:
            raise ResultLockError(f"locked result schema is invalid: {exc}") from exc
        if result.opaque_target_id != receipt.opaque_target_id:
            raise ResultLockError("lock receipt target does not match result target")
        forbidden = _find_forbidden_key(result.model_dump(mode="json"))
        if forbidden:
            raise ResultLockError(f"locked result exposes forbidden key {forbidden!r}")
        return result

    def verify_trace_commitment(
        self,
        receipt: ResultLockReceipt,
        trace: TraceRecorder,
    ) -> LockedInvestigationResult:
        """Verify that ``result.json`` commits to the trace prefix before lock.

        The lock event is deliberately outside the committed prefix because its
        payload contains the digest of ``result.json``.  Its ``previous_hash``
        therefore provides the non-circular link back to the trace root stored in
        the locked result.
        """

        result = self.verify_receipt(receipt)
        self._validate_trace_identity(result, trace)
        trace.verify()
        lock_events = [
            event for event in trace.events if event.event_type is TraceEventType.RESULT_LOCKED
        ]
        if len(lock_events) != 1:
            raise ResultLockError("trace must contain exactly one RESULT_LOCKED event")
        lock_event = lock_events[0]
        if lock_event.previous_hash != result.pre_lock_trace_root_hash:
            raise ResultLockError("RESULT_LOCKED does not follow the committed trace root")
        expected_payload = {
            "result_sha256": receipt.sha256,
            "pre_lock_trace_root_hash": result.pre_lock_trace_root_hash,
            "result_artifact": self.RESULT_NAME,
            "hash_artifact": self.HASH_NAME,
        }
        if any(lock_event.payload.get(key) != value for key, value in expected_payload.items()):
            raise ResultLockError("RESULT_LOCKED payload does not match lock artifacts")
        return result

    def write_reveal(
        self,
        receipt: ResultLockReceipt,
        *,
        ground_truth: GroundTruthRecord,
        trace: TraceRecorder | None = None,
        revealed_at: datetime | None = None,
    ) -> RevealArtifact:
        """Persist reveal.json; verification ensures lock files already exist."""

        if trace is None:
            raise RevealOrderError("a verified committed trace is required for reveal")
        try:
            locked_result = self.verify_trace_commitment(receipt, trace)
        except ResultLockError as exc:
            raise RevealOrderError(str(exc)) from exc
        if not trace.catalog_access_enabled:
            raise RevealOrderError("trace has no post-lock CATALOG_ACCESS_ENABLED event")
        access_events = [
            event
            for event in trace.events
            if event.event_type is TraceEventType.CATALOG_ACCESS_ENABLED
        ]
        if len(access_events) != 1 or (
            access_events[0].payload.get("locked_result_sha256") != receipt.sha256
            or access_events[0].payload.get("pre_lock_trace_root_hash")
            != locked_result.pre_lock_trace_root_hash
        ):
            raise RevealOrderError("trace catalog-access event does not match the locked result")
        artifact = RevealArtifact(
            opaque_target_id=receipt.opaque_target_id,
            locked_result_sha256=receipt.sha256,
            ground_truth=ground_truth,
            revealed_at=revealed_at or utc_now(),
        )
        if locked_result.opaque_target_id != artifact.opaque_target_id:
            raise RevealOrderError("reveal target does not match locked result")
        reveal_path = Path(receipt.result_path).resolve().parent / self.REVEAL_NAME
        serialized = canonical_json_bytes(artifact.model_dump(mode="json")) + b"\n"
        with self._lock:
            reveal_events = [
                event
                for event in trace.events
                if event.event_type is TraceEventType.GROUND_TRUTH_REVEALED
            ]
            if reveal_events and (
                len(reveal_events) != 1
                or reveal_events[0].payload.get("locked_result_sha256") != receipt.sha256
                or reveal_events[0].payload.get("reveal_artifact") != self.REVEAL_NAME
                or reveal_events[0].payload.get("ground_truth")
                != ground_truth.model_dump(mode="json")
                or not reveal_path.exists()
            ):
                raise RevealOrderError("reveal artifact/event already exists and is immutable")
            if reveal_path.exists():
                existing = reveal_path.read_bytes()
                if existing != serialized:
                    # Never revise history after catalog truth has been exposed.
                    raise RevealOrderError("reveal artifact already exists and is immutable")
            else:
                self._atomic_write_once(reveal_path, serialized)
            self.verify_artifact_order(reveal_path.parent)
            if not reveal_events:
                trace.append(
                    TraceEventType.GROUND_TRUTH_REVEALED,
                    {
                        "locked_result_sha256": receipt.sha256,
                        "ground_truth": ground_truth.model_dump(mode="json"),
                        "reveal_artifact": self.REVEAL_NAME,
                    },
                )
            trace.verify()
        return artifact

    def verify_artifact_order(self, run_directory: str | Path) -> bool:
        directory = Path(run_directory).resolve()
        result_path = directory / self.RESULT_NAME
        hash_path = directory / self.HASH_NAME
        reveal_path = directory / self.REVEAL_NAME
        if reveal_path.exists() and (not result_path.exists() or not hash_path.exists()):
            raise RevealOrderError("reveal exists without complete lock artifacts")
        if reveal_path.exists():
            # Creation is code-gated; timestamps provide a second audit signal.
            reveal_mtime = reveal_path.stat().st_mtime_ns
            if result_path.stat().st_mtime_ns > reveal_mtime:
                raise RevealOrderError("result.json is newer than reveal.json")
            if hash_path.stat().st_mtime_ns > reveal_mtime:
                raise RevealOrderError("result hash is newer than reveal.json")
            reveal = RevealArtifact.model_validate_json(reveal_path.read_bytes())
            serialized = result_path.read_bytes()
            calculated = hashlib.sha256(serialized).hexdigest()
            stored = hash_path.read_text(encoding="ascii").strip().lower()
            if calculated != stored or calculated != reveal.locked_result_sha256:
                raise RevealOrderError("reveal references a different locked result")
            result = LockedInvestigationResult.model_validate_json(serialized)
            if reveal.opaque_target_id != result.opaque_target_id:
                raise RevealOrderError("reveal target does not match locked result")
        return True

    @staticmethod
    def _validate_trace_identity(
        result: LockedInvestigationResult,
        trace: TraceRecorder,
    ) -> None:
        if trace.opaque_target_id != result.opaque_target_id:
            raise ResultLockError("trace target does not match result target")
        if trace.trace_id != result.trace_id:
            raise ResultLockError("trace id does not match locked result")

    @staticmethod
    def _atomic_write_once(path: Path, content: bytes) -> None:
        if path.exists():
            if path.read_bytes() != content:
                raise ResultLockError(f"immutable artifact already exists: {path.name}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            # os.replace is atomic within one filesystem. The process lock and
            # existence check enforce write-once semantics for this application.
            if path.exists():
                if path.read_bytes() != content:
                    raise ResultLockError(f"immutable artifact concurrently created: {path.name}")
            else:
                os.replace(temporary_path, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path.exists():
                temporary_path.unlink()
