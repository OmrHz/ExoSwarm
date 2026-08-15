"""Opaque target registry and mechanically gated ground-truth capability."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from exoswarm.domain.models import (
    ArtifactRef,
    BackendTargetMappingRef,
    FrozenDomainModel,
    GroundTruthRecord,
    LockedInvestigationResult,
    ResultLockReceipt,
    RevealArtifact,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder

if TYPE_CHECKING:
    from .locking import ResultLocker


class GroundTruthAccessDenied(PermissionError):
    pass


class UnknownOpaqueTarget(KeyError):
    pass


class OpaqueTargetContext(FrozenDomainModel):
    """The only target metadata that may be placed in an agent prompt."""

    opaque_target_id: str = Field(pattern=r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
    available_artifact_ids: list[str]
    available_product_roles: list[str]


class ScienceDataHandle(FrozenDomainModel):
    """Cached inputs for deterministic code; it intentionally has no identity field."""

    opaque_target_id: str
    artifacts: list[ArtifactRef]


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateTargetRecord:
    real_target_identity: str
    artifacts: tuple[ArtifactRef, ...]
    ground_truth: GroundTruthRecord
    mapping_key: str


@dataclass(frozen=True, slots=True)
class _UnlockedResult:
    receipt: ResultLockReceipt
    trace: TraceRecorder


class OpaqueTargetVault:
    """Backend-only owner of the opaque-to-real target mapping.

    There is intentionally no public ``resolve_identity`` method.  Before lock,
    callers can retrieve cached science products or an identity-free agent view.
    Only :class:`GroundTruthGate` can invoke the name-mangled reveal operation.
    """

    __slots__ = ("__records", "__lock")

    def __init__(self) -> None:
        self.__records: dict[str, _PrivateTargetRecord] = {}
        self.__lock = threading.RLock()

    def __repr__(self) -> str:
        return f"OpaqueTargetVault(targets={len(self.__records)}, mappings=<redacted>)"

    def register(
        self,
        *,
        opaque_target_id: str,
        real_target_identity: str,
        artifacts: list[ArtifactRef],
        ground_truth: GroundTruthRecord,
    ) -> BackendTargetMappingRef:
        if not opaque_target_id.startswith("TARGET-"):
            raise ValueError("opaque_target_id must use the TARGET- prefix")
        if ground_truth.actual_target_identity != real_target_identity:
            raise ValueError("ground-truth identity does not match private mapping")
        with self.__lock:
            if opaque_target_id in self.__records:
                raise ValueError(f"opaque target already registered: {opaque_target_id}")
            mapping_key = secrets.token_urlsafe(24)
            self.__records[opaque_target_id] = _PrivateTargetRecord(
                real_target_identity=real_target_identity,
                artifacts=tuple(artifact.model_copy(deep=True) for artifact in artifacts),
                ground_truth=ground_truth.model_copy(deep=True),
                mapping_key=mapping_key,
            )
            return BackendTargetMappingRef(mapping_key=mapping_key)

    def agent_context(self, opaque_target_id: str) -> OpaqueTargetContext:
        record = self.__get(opaque_target_id)
        return OpaqueTargetContext(
            opaque_target_id=opaque_target_id,
            available_artifact_ids=[item.artifact_id for item in record.artifacts],
            available_product_roles=sorted(
                {item.role for item in record.artifacts if item.role is not None}
            ),
        )

    def science_data(self, opaque_target_id: str) -> ScienceDataHandle:
        """Return cached artifacts, never catalog facts or target identity."""

        record = self.__get(opaque_target_id)
        return ScienceDataHandle(
            opaque_target_id=opaque_target_id,
            artifacts=[item.model_copy(deep=True) for item in record.artifacts],
        )

    def mapping_ref(self, opaque_target_id: str) -> BackendTargetMappingRef:
        return BackendTargetMappingRef(mapping_key=self.__get(opaque_target_id).mapping_key)

    def contains(self, opaque_target_id: str) -> bool:
        with self.__lock:
            return opaque_target_id in self.__records

    def __get(self, opaque_target_id: str) -> _PrivateTargetRecord:
        with self.__lock:
            try:
                return self.__records[opaque_target_id]
            except KeyError as exc:
                raise UnknownOpaqueTarget(opaque_target_id) from exc

    def __reveal_after_verified_lock(self, opaque_target_id: str) -> GroundTruthRecord:
        """Private friend API used only by GroundTruthGate after verification."""

        return self.__get(opaque_target_id).ground_truth.model_copy(deep=True)


class GroundTruthGate:
    """Catalog capability that is unusable until a result lock verifies on disk."""

    def __init__(self, vault: OpaqueTargetVault, result_locker: ResultLocker) -> None:
        self._vault = vault
        self._locker = result_locker
        self._unlocked_results: dict[str, _UnlockedResult] = {}
        self._lock = threading.RLock()

    def is_available(self, opaque_target_id: str) -> bool:
        with self._lock:
            return opaque_target_id in self._unlocked_results

    def lookup(self, opaque_target_id: str) -> GroundTruthRecord:
        """Return catalog data only after :meth:`unlock_after_result_lock`."""

        with self._lock:
            unlocked = self._unlocked_results.get(opaque_target_id)
            if unlocked is None:
                raise GroundTruthAccessDenied("ground truth is unavailable before RESULT_LOCKED")
            try:
                result = self._locker.verify_trace_commitment(
                    unlocked.receipt,
                    unlocked.trace,
                )
                self._validate_catalog_access(
                    unlocked.receipt,
                    result,
                    unlocked.trace,
                )
            except Exception as exc:
                raise GroundTruthAccessDenied(
                    f"ground truth lock or trace commitment no longer verifies: {exc}"
                ) from exc
        reveal_method = self._vault._OpaqueTargetVault__reveal_after_verified_lock
        return reveal_method(opaque_target_id)

    def unlock_after_result_lock(
        self,
        receipt: ResultLockReceipt,
        *,
        trace: TraceRecorder | None = None,
    ) -> None:
        """Enable one target only after both result.json and its hash verify."""

        if trace is None:
            raise GroundTruthAccessDenied(
                "a verified committed trace is required to unlock ground truth"
            )
        try:
            result = self._locker.verify_trace_commitment(receipt, trace)
        except Exception as exc:
            raise GroundTruthAccessDenied("result lock is not bound to the supplied trace") from exc
        access_events = [
            event
            for event in trace.events
            if event.event_type is TraceEventType.CATALOG_ACCESS_ENABLED
        ]
        if access_events and (
            len(access_events) != 1
            or access_events[0].payload.get("locked_result_sha256") != receipt.sha256
            or access_events[0].payload.get("pre_lock_trace_root_hash")
            != result.pre_lock_trace_root_hash
        ):
            raise GroundTruthAccessDenied(
                "trace contains a catalog-access event for a different result"
            )
        if not access_events:
            trace.append(
                TraceEventType.CATALOG_ACCESS_ENABLED,
                {
                    "locked_result_sha256": receipt.sha256,
                    "pre_lock_trace_root_hash": result.pre_lock_trace_root_hash,
                },
            )
        self._validate_catalog_access(receipt, result, trace)
        with self._lock:
            self._unlocked_results[receipt.opaque_target_id] = _UnlockedResult(
                receipt=receipt,
                trace=trace,
            )

    def create_reveal_artifact(
        self,
        receipt: ResultLockReceipt,
        *,
        trace: TraceRecorder | None = None,
    ) -> RevealArtifact:
        """Read catalog truth and persist ``reveal.json`` after the verified lock."""

        if trace is None:
            raise GroundTruthAccessDenied(
                "a verified committed trace is required to create reveal.json"
            )
        with self._lock:
            unlocked = self._unlocked_results.get(receipt.opaque_target_id)
            if (
                unlocked is None
                or unlocked.receipt.sha256 != receipt.sha256
                or unlocked.trace.trace_id != trace.trace_id
            ):
                raise GroundTruthAccessDenied(
                    "ground truth capability has not been unlocked for this result"
                )
        try:
            self._locker.verify_trace_commitment(receipt, trace)
        except Exception as exc:
            raise GroundTruthAccessDenied("result lock is not bound to the supplied trace") from exc
        ground_truth = self.lookup(receipt.opaque_target_id)
        return self._locker.write_reveal(receipt, ground_truth=ground_truth, trace=trace)

    @staticmethod
    def _validate_catalog_access(
        receipt: ResultLockReceipt,
        result: LockedInvestigationResult,
        trace: TraceRecorder,
    ) -> None:
        access_events = [
            event
            for event in trace.events
            if event.event_type is TraceEventType.CATALOG_ACCESS_ENABLED
        ]
        if (
            not trace.catalog_access_enabled
            or len(access_events) != 1
            or access_events[0].payload.get("locked_result_sha256") != receipt.sha256
            or access_events[0].payload.get("pre_lock_trace_root_hash")
            != result.pre_lock_trace_root_hash
        ):
            raise GroundTruthAccessDenied(
                "catalog-access event does not match the verified result lock"
            )
