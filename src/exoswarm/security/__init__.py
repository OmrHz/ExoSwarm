"""Blindness, result-lock, and reveal security boundary."""

from .blindness import (
    GroundTruthAccessDenied,
    GroundTruthGate,
    OpaqueTargetContext,
    OpaqueTargetVault,
    ScienceDataHandle,
    UnknownOpaqueTarget,
)
from .boundary import (
    ImportBoundaryViolation,
    assert_catalog_import_boundary,
    catalog_import_violations,
)
from .locking import ResultLocker, ResultLockError, RevealOrderError

__all__ = [
    "GroundTruthAccessDenied",
    "GroundTruthGate",
    "ImportBoundaryViolation",
    "OpaqueTargetContext",
    "OpaqueTargetVault",
    "ResultLockError",
    "ResultLocker",
    "RevealOrderError",
    "ScienceDataHandle",
    "UnknownOpaqueTarget",
    "assert_catalog_import_boundary",
    "catalog_import_violations",
]
