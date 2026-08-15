"""Integrity-aware loading of persisted mission-control artifacts.

This module is deliberately independent of Streamlit so that the blindness,
result-lock, and artifact-path boundaries can be tested without launching a UI.
Only sanitized run-local science products are exposed.  Cached FITS files are
never read here because their headers may reveal the target identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from exoswarm.domain.ledger import EvidenceIntegrityError, EvidenceItem, EvidenceLedger
from exoswarm.domain.models import LockedInvestigationResult, RevealArtifact
from exoswarm.domain.trace import TraceEvent, TraceIntegrityError, TraceRecorder

TARGET_PATTERN = re.compile(r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
CORE_ARTIFACTS = frozenset(
    {"result.json", "result.json.sha256", "evidence.jsonl", "trace.jsonl", "reveal.json"}
)
KNOWN_SCIENCE_ROLES = (
    "raw_light_curve",
    "normalized_light_curve",
    "cleaned_light_curve",
    "bls_periodogram",
    "phase_folded",
    "odd_even",
    "secondary_eclipse",
    "harmonic_test",
    "centroid_localization",
)
MAX_SCIENCE_ARTIFACT_BYTES = 512 * 1024 * 1024


class RunPhase(StrEnum):
    EMPTY = "EMPTY"
    ACTIVE = "ACTIVE"
    RESULT_LOCKED = "RESULT_LOCKED"
    GROUND_TRUTH_REVEALED = "GROUND_TRUTH_REVEALED"
    CORRUPT = "CORRUPT"


class IssueSeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class LoadIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.WARNING


@dataclass(frozen=True, slots=True)
class ScienceArtifact:
    """A safe run-local artifact selected from the science manifest."""

    artifact_id: str
    role: str
    path: Path
    media_type: str | None
    sha256: str | None
    integrity_verified: bool

    @property
    def suffix(self) -> str:
        return self.path.suffix.casefold()


@dataclass(frozen=True, slots=True)
class MissionControlRun:
    run_directory: Path
    opaque_target_id: str
    phase: RunPhase
    result: LockedInvestigationResult | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    trace: tuple[TraceEvent, ...] = ()
    reveal: RevealArtifact | None = None
    lock_sha256: str | None = None
    artifacts: tuple[ScienceArtifact, ...] = ()
    issues: tuple[LoadIssue, ...] = ()

    @property
    def ground_truth_visible(self) -> bool:
        return self.phase is RunPhase.GROUND_TRUTH_REVEALED and self.reveal is not None

    @property
    def lock_verified(self) -> bool:
        return self.result is not None and self.lock_sha256 is not None

    @property
    def trace_id(self) -> str | None:
        if self.result is not None:
            return self.result.trace_id
        return self.trace[0].trace_id if self.trace else None

    def artifact(self, role: str, suffix: str | None = None) -> ScienceArtifact | None:
        normalized_suffix = suffix.casefold() if suffix else None
        ledger_artifact_ids = {
            artifact_id
            for evidence_item in self.evidence
            for artifact_id in evidence_item.output_artifact_ids
        }
        matches = [
            item
            for item in self.artifacts
            if item.role == role
            and item.artifact_id in ledger_artifact_ids
            and (normalized_suffix is None or item.suffix == normalized_suffix)
        ]
        if not matches:
            return None
        # Prefer hash-verified products, then compact machine-readable artifacts.
        return sorted(
            matches,
            key=lambda item: (
                not item.integrity_verified,
                {".npz": 0, ".csv": 1, ".png": 2}.get(item.suffix, 3),
                item.path.name,
            ),
        )[0]

    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return {item.id: item for item in self.evidence}


@dataclass(frozen=True, slots=True)
class ScienceProduct:
    role: str
    source: ScienceArtifact
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)


def available_target_ids(
    runs_directory: str | Path,
    *,
    required_demo_targets: Iterable[str] = ("TARGET-X17", "TARGET-X42"),
) -> tuple[str, ...]:
    """Return opaque IDs only; no target catalog is consulted."""

    root = Path(runs_directory)
    targets = {item for item in required_demo_targets if TARGET_PATTERN.fullmatch(item)}
    if root.is_dir():
        targets.update(
            child.name
            for child in root.iterdir()
            if child.is_dir() and TARGET_PATTERN.fullmatch(child.name)
        )
    return tuple(sorted(targets))


def resolve_run_directory(runs_directory: str | Path, opaque_target_id: str) -> Path:
    """Resolve the current run while remaining inside the configured runs root.

    The runtime currently writes ``runs/TARGET-X17`` directly.  Selecting the most
    recently modified nested run as a fallback keeps the UI compatible with future
    timestamped runs without broad recursive filesystem access.
    """

    if not TARGET_PATTERN.fullmatch(opaque_target_id):
        raise ValueError("invalid opaque target ID")
    root = Path(runs_directory).resolve()
    direct = (root / opaque_target_id).resolve()
    if not direct.is_relative_to(root):
        raise ValueError("target path escaped the configured runs directory")
    if _looks_like_run(direct) or not direct.is_dir():
        return direct
    nested = [child for child in direct.iterdir() if child.is_dir() and _looks_like_run(child)]
    return max(nested, key=lambda child: child.stat().st_mtime_ns) if nested else direct


def load_run(
    run_directory: str | Path, *, opaque_target_id: str | None = None
) -> MissionControlRun:
    """Load one run, enforcing the lock-before-reveal boundary.

    Invalid core artifacts produce a ``CORRUPT`` view and are not silently trusted.
    Plot artifacts are independently verified and omitted when their manifest entry
    is unsafe or fails its hash.
    """

    directory = Path(run_directory).resolve()
    target = opaque_target_id or _target_from_path(directory)
    if not TARGET_PATTERN.fullmatch(target):
        raise ValueError("a valid opaque target ID is required")
    if not directory.exists():
        return MissionControlRun(
            run_directory=directory,
            opaque_target_id=target,
            phase=RunPhase.EMPTY,
        )
    if not directory.is_dir():
        raise ValueError("run path is not a directory")

    issues: list[LoadIssue] = []
    result, lock_digest, lock_valid = _load_locked_result(directory, target, issues)
    evidence = _load_evidence(directory, issues)
    trace = _load_trace(directory, target, result, issues)
    trace_claims_lock = any(event.event_type.value == "RESULT_LOCKED" for event in trace)
    if (
        trace_claims_lock
        and not lock_valid
        and not any(
            item.code in {"LOCK_INCOMPLETE", "LOCK_HASH_MISMATCH", "RESULT_INVALID"}
            for item in issues
        )
    ):
        issues.append(
            LoadIssue(
                "LOCK_INCOMPLETE",
                "The trace records RESULT_LOCKED, but the committed result files are unavailable.",
                IssueSeverity.ERROR,
            )
        )
    reveal = _load_reveal(
        directory,
        target,
        result=result,
        lock_digest=lock_digest,
        lock_valid=lock_valid,
        trace=trace,
        issues=issues,
    )
    _validate_cross_artifact_links(
        result=result if lock_valid else None,
        evidence=evidence,
        trace=trace,
        lock_digest=lock_digest if lock_valid else None,
        issues=issues,
    )
    artifacts = _load_science_manifest(directory, target, issues)

    core_error = any(
        issue.severity is IssueSeverity.ERROR
        and issue.code
        in {
            "RESULT_INVALID",
            "LOCK_INCOMPLETE",
            "LOCK_HASH_MISMATCH",
            "LEDGER_INVALID",
            "TRACE_INVALID",
            "REVEAL_BEFORE_LOCK",
            "REVEAL_INVALID",
            "REVEAL_TRACE_INVALID",
            "CROSS_ARTIFACT_MISMATCH",
        }
        for issue in issues
    )
    has_activity = bool(
        evidence or trace or any((directory / name).exists() for name in CORE_ARTIFACTS)
    )
    if core_error:
        phase = RunPhase.CORRUPT
    elif reveal is not None:
        phase = RunPhase.GROUND_TRUTH_REVEALED
    elif lock_valid and result is not None:
        phase = RunPhase.RESULT_LOCKED
    elif has_activity:
        phase = RunPhase.ACTIVE
    else:
        phase = RunPhase.EMPTY

    return MissionControlRun(
        run_directory=directory,
        opaque_target_id=target,
        phase=phase,
        result=result if lock_valid and not core_error else None,
        evidence=evidence,
        trace=trace,
        reveal=reveal if not core_error else None,
        lock_sha256=lock_digest if lock_valid and not core_error else None,
        artifacts=artifacts,
        issues=tuple(issues),
    )


def load_science_product(run: MissionControlRun, role: str) -> ScienceProduct | None:
    """Load a numeric NPZ/CSV science product selected through the safe manifest."""

    source = run.artifact(role, ".npz") or run.artifact(role, ".csv")
    if source is None:
        return None
    _check_science_file(source.path)
    if source.suffix == ".npz":
        with np.load(source.path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    else:
        # ``genfromtxt`` avoids a second representation dependency in the loader.
        table = np.genfromtxt(source.path, delimiter=",", names=True, dtype=None, encoding=None)
        names = table.dtype.names or ()
        arrays = {name: np.atleast_1d(table[name]) for name in names}
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"science artifact {source.path.name} contains object array {key!r}")
    return ScienceProduct(role=role, source=source, arrays=arrays)


def image_artifact(run: MissionControlRun, role: str) -> ScienceArtifact | None:
    return run.artifact(role, ".png")


def _load_locked_result(
    directory: Path,
    target: str,
    issues: list[LoadIssue],
) -> tuple[LockedInvestigationResult | None, str | None, bool]:
    result_path = directory / "result.json"
    hash_path = directory / "result.json.sha256"
    if not result_path.exists() and not hash_path.exists():
        return None, None, False
    if not result_path.exists() or not hash_path.exists():
        issues.append(
            LoadIssue(
                "LOCK_INCOMPLETE",
                "The result lock is incomplete; both result.json and result.json.sha256 are required.",
                IssueSeverity.ERROR,
            )
        )
        return None, None, False
    try:
        serialized = result_path.read_bytes()
        stored = hash_path.read_text(encoding="ascii").strip().casefold()
        calculated = hashlib.sha256(serialized).hexdigest()
        if not re.fullmatch(r"[a-f0-9]{64}", stored) or stored != calculated:
            issues.append(
                LoadIssue(
                    "LOCK_HASH_MISMATCH",
                    "The pre-reveal result does not match its recorded SHA-256.",
                    IssueSeverity.ERROR,
                )
            )
            return None, None, False
        result = LockedInvestigationResult.model_validate_json(serialized)
        if result.opaque_target_id != target:
            raise ValueError("locked result target does not match the selected opaque target")
        return result, calculated, True
    except Exception as exc:
        if not any(item.code == "LOCK_HASH_MISMATCH" for item in issues):
            issues.append(
                LoadIssue(
                    "RESULT_INVALID",
                    f"The locked result could not be validated: {exc}",
                    IssueSeverity.ERROR,
                )
            )
        return None, None, False


def _load_evidence(directory: Path, issues: list[LoadIssue]) -> tuple[EvidenceItem, ...]:
    path = directory / "evidence.jsonl"
    if not path.exists():
        return ()
    try:
        ledger = EvidenceLedger(path)
        ledger.verify()
        return ledger.items
    except (EvidenceIntegrityError, OSError, ValueError) as exc:
        issues.append(
            LoadIssue(
                "LEDGER_INVALID",
                f"Evidence Ledger integrity validation failed: {exc}",
                IssueSeverity.ERROR,
            )
        )
        return ()


def _load_trace(
    directory: Path,
    target: str,
    result: LockedInvestigationResult | None,
    issues: list[LoadIssue],
) -> tuple[TraceEvent, ...]:
    path = directory / "trace.jsonl"
    if not path.exists():
        return ()
    try:
        first = _first_jsonl_object(path)
        trace_id = str(first["trace_id"])
        trace_target = str(first["opaque_target_id"])
        if trace_target != target:
            raise ValueError("trace target does not match the selected opaque target")
        if result is not None and trace_id != result.trace_id:
            raise ValueError("trace ID does not match the locked result")
        recorder = TraceRecorder(trace_id=trace_id, opaque_target_id=target, path=path)
        recorder.verify()
        return recorder.events
    except (TraceIntegrityError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        issues.append(
            LoadIssue(
                "TRACE_INVALID",
                f"Investigation trace integrity validation failed: {exc}",
                IssueSeverity.ERROR,
            )
        )
        return ()


def _load_reveal(
    directory: Path,
    target: str,
    *,
    result: LockedInvestigationResult | None,
    lock_digest: str | None,
    lock_valid: bool,
    trace: tuple[TraceEvent, ...],
    issues: list[LoadIssue],
) -> RevealArtifact | None:
    path = directory / "reveal.json"
    if not path.exists():
        return None
    if not lock_valid or result is None or lock_digest is None:
        issues.append(
            LoadIssue(
                "REVEAL_BEFORE_LOCK",
                "Ground truth was withheld because no valid pre-reveal result lock exists.",
                IssueSeverity.ERROR,
            )
        )
        return None
    try:
        reveal = RevealArtifact.model_validate_json(path.read_bytes())
        if reveal.opaque_target_id != target:
            raise ValueError("reveal target does not match the selected opaque target")
        if reveal.locked_result_sha256.casefold() != lock_digest.casefold():
            raise ValueError("reveal references a different locked result")
        result_mtime = (directory / "result.json").stat().st_mtime_ns
        hash_mtime = (directory / "result.json.sha256").stat().st_mtime_ns
        if path.stat().st_mtime_ns < max(result_mtime, hash_mtime):
            raise ValueError("reveal predates the complete result lock")
        if not _trace_proves_reveal_order(trace, lock_digest, result.pre_lock_trace_root_hash):
            issues.append(
                LoadIssue(
                    "REVEAL_TRACE_INVALID",
                    "Ground truth was withheld because the trace does not prove RESULT_LOCKED < CATALOG_ACCESS_ENABLED < GROUND_TRUTH_REVEALED ordering for this result hash.",
                    IssueSeverity.ERROR,
                )
            )
            return None
        return reveal
    except (OSError, ValueError) as exc:
        issues.append(
            LoadIssue(
                "REVEAL_INVALID",
                f"Ground-truth reveal validation failed: {exc}",
                IssueSeverity.ERROR,
            )
        )
        return None


def _trace_proves_reveal_order(
    trace: tuple[TraceEvent, ...],
    lock_digest: str,
    pre_lock_trace_root_hash: str,
) -> bool:
    lock_sequences = [
        event.sequence
        for event in trace
        if event.event_type.value == "RESULT_LOCKED"
        and event.payload.get("result_sha256") == lock_digest
        and event.payload.get("pre_lock_trace_root_hash") == pre_lock_trace_root_hash
        and event.previous_hash == pre_lock_trace_root_hash
    ]
    catalog_sequences = [
        event.sequence
        for event in trace
        if event.event_type.value == "CATALOG_ACCESS_ENABLED"
        and event.payload.get("locked_result_sha256") == lock_digest
        and event.payload.get("pre_lock_trace_root_hash") == pre_lock_trace_root_hash
    ]
    reveal_sequences = [
        event.sequence
        for event in trace
        if event.event_type.value == "GROUND_TRUTH_REVEALED"
        and event.payload.get("locked_result_sha256") == lock_digest
    ]
    return any(
        lock_sequence < catalog_sequence < reveal_sequence
        for lock_sequence in lock_sequences
        for catalog_sequence in catalog_sequences
        for reveal_sequence in reveal_sequences
    )


def _load_science_manifest(
    directory: Path, target: str, issues: list[LoadIssue]
) -> tuple[ScienceArtifact, ...]:
    candidates = (directory / "artifacts" / "artifacts.json", directory / "artifacts.json")
    manifest_path = next((path for path in candidates if path.exists()), None)
    if manifest_path is None:
        return _discover_conventional_artifacts(directory, issues)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(
            LoadIssue(
                "MANIFEST_INVALID",
                f"Science artifact manifest could not be read: {exc}",
                IssueSeverity.WARNING,
            )
        )
        return ()
    raw_entries = payload.get("artifacts", []) if isinstance(payload, dict) else payload
    if isinstance(payload, dict):
        manifest_target = payload.get("opaque_target_id")
        if manifest_target is not None and manifest_target != target:
            issues.append(
                LoadIssue(
                    "MANIFEST_TARGET_MISMATCH",
                    "Science artifact manifest belongs to a different opaque target.",
                    IssueSeverity.ERROR,
                )
            )
            return ()
    if not isinstance(raw_entries, list):
        issues.append(
            LoadIssue(
                "MANIFEST_INVALID",
                "Science artifact manifest must contain an artifacts list.",
                IssueSeverity.WARNING,
            )
        )
        return ()

    artifacts: list[ScienceArtifact] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            issues.append(
                LoadIssue("MANIFEST_ENTRY_INVALID", f"Manifest entry {index} is not an object.")
            )
            continue
        entry = _validated_manifest_entry(directory, manifest_path, target, raw, index, issues)
        if entry is None:
            continue
        fingerprint = (entry.artifact_id, str(entry.path))
        if fingerprint not in seen:
            artifacts.append(entry)
            seen.add(fingerprint)
    return tuple(artifacts)


def _validated_manifest_entry(
    directory: Path,
    manifest_path: Path,
    target: str,
    raw: dict[str, Any],
    index: int,
    issues: list[LoadIssue],
) -> ScienceArtifact | None:
    artifact_id = str(raw.get("artifact_id") or raw.get("id") or "").strip()
    manifest_role = str(raw.get("role") or "").strip()
    role = _canonical_role(manifest_role)
    relative = str(raw.get("relative_path") or raw.get("path") or "").strip()
    digest_value = raw.get("sha256")
    digest = str(digest_value).casefold() if digest_value else None
    media_type_value = raw.get("media_type")
    media_type = str(media_type_value) if media_type_value else None
    if not artifact_id or role not in KNOWN_SCIENCE_ROLES or not relative:
        issues.append(
            LoadIssue(
                "MANIFEST_ENTRY_INVALID",
                f"Manifest entry {index} lacks a supported artifact ID, role, or path.",
            )
        )
        return None
    if not artifact_id.startswith(f"{target}:") or not re.fullmatch(
        r"[A-Za-z0-9:_.-]+", artifact_id
    ):
        issues.append(
            LoadIssue(
                "ARTIFACT_ID_REJECTED",
                f"Manifest entry {index} has an invalid or cross-target artifact ID.",
            )
        )
        return None
    supplied = Path(relative)
    if supplied.is_absolute():
        issues.append(
            LoadIssue("ARTIFACT_PATH_REJECTED", f"Artifact {artifact_id} uses an absolute path.")
        )
        return None
    # The agreed contract is run-relative.  Manifest-relative paths are accepted
    # only when they remain inside the same run, easing migration from prototypes.
    run_relative = (directory / supplied).resolve()
    manifest_relative = (manifest_path.parent / supplied).resolve()
    path = run_relative if run_relative.exists() else manifest_relative
    if not path.is_relative_to(directory):
        issues.append(
            LoadIssue(
                "ARTIFACT_PATH_REJECTED", f"Artifact {artifact_id} escapes the run directory."
            )
        )
        return None
    if not path.is_file():
        issues.append(
            LoadIssue("ARTIFACT_MISSING", f"Artifact {artifact_id} is listed but not present.")
        )
        return None
    if path.suffix.casefold() not in {".npz", ".csv", ".png"}:
        issues.append(
            LoadIssue(
                "ARTIFACT_TYPE_REJECTED", f"Artifact {artifact_id} has an unsupported format."
            )
        )
        return None
    if path.stem != role:
        issues.append(
            LoadIssue(
                "ARTIFACT_NAME_REJECTED",
                f"Artifact {artifact_id} does not use the canonical filename for its role.",
            )
        )
        return None
    expected_media_types = {
        ".npz": {"application/x-npz", "application/octet-stream"},
        ".csv": {"text/csv", "application/csv"},
        ".png": {"image/png"},
    }
    if (
        media_type is not None
        and media_type.casefold() not in expected_media_types[path.suffix.casefold()]
    ):
        issues.append(
            LoadIssue(
                "ARTIFACT_MEDIA_MISMATCH",
                f"Artifact {artifact_id} media type does not match its file format.",
            )
        )
        return None
    if path.stat().st_size > MAX_SCIENCE_ARTIFACT_BYTES:
        issues.append(
            LoadIssue("ARTIFACT_TOO_LARGE", f"Artifact {artifact_id} exceeds the UI safety limit.")
        )
        return None
    verified = False
    if digest is not None:
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            issues.append(
                LoadIssue(
                    "ARTIFACT_HASH_INVALID", f"Artifact {artifact_id} has an invalid SHA-256."
                )
            )
            return None
        calculated = _sha256_file(path)
        if calculated != digest:
            issues.append(
                LoadIssue(
                    "ARTIFACT_HASH_MISMATCH", f"Artifact {artifact_id} failed SHA-256 verification."
                )
            )
            return None
        verified = True
    else:
        issues.append(
            LoadIssue("ARTIFACT_HASH_MISSING", f"Artifact {artifact_id} has no recorded SHA-256.")
        )
    return ScienceArtifact(
        artifact_id=artifact_id,
        role=role,
        path=path,
        media_type=media_type,
        sha256=digest,
        integrity_verified=verified,
    )


def _discover_conventional_artifacts(
    directory: Path, issues: list[LoadIssue]
) -> tuple[ScienceArtifact, ...]:
    science_directory = directory / "artifacts" / "science"
    if not science_directory.is_dir():
        return ()
    artifacts: list[ScienceArtifact] = []
    for role in KNOWN_SCIENCE_ROLES:
        for suffix in (".npz", ".csv", ".png"):
            path = science_directory / f"{role}{suffix}"
            if path.is_file() and path.stat().st_size <= MAX_SCIENCE_ARTIFACT_BYTES:
                artifacts.append(
                    ScienceArtifact(
                        artifact_id=f"conventional:{role}:{suffix[1:]}",
                        role=role,
                        path=path.resolve(),
                        media_type=None,
                        sha256=None,
                        integrity_verified=False,
                    )
                )
    if artifacts:
        issues.append(
            LoadIssue(
                "MANIFEST_MISSING",
                "Science products were found by convention but lack a hashed artifact manifest.",
            )
        )
    return tuple(artifacts)


def _validate_cross_artifact_links(
    *,
    result: LockedInvestigationResult | None,
    evidence: tuple[EvidenceItem, ...],
    trace: tuple[TraceEvent, ...],
    lock_digest: str | None,
    issues: list[LoadIssue],
) -> None:
    if result is None or lock_digest is None:
        return
    evidence_ids = [item.id for item in evidence]
    evidence_root = evidence[-1].record_hash if evidence else "0" * 64
    if result.evidence_ids != evidence_ids or result.evidence_root_hash != evidence_root:
        issues.append(
            LoadIssue(
                "CROSS_ARTIFACT_MISMATCH",
                "The locked result does not reference the validated Evidence Ledger exactly.",
                IssueSeverity.ERROR,
            )
        )
    lock_events = [event for event in trace if event.event_type.value == "RESULT_LOCKED"]
    if len(lock_events) != 1:
        issues.append(
            LoadIssue(
                "CROSS_ARTIFACT_MISMATCH",
                "The validated trace must contain exactly one RESULT_LOCKED event.",
                IssueSeverity.ERROR,
            )
        )
        return
    lock_event = lock_events[0]
    if lock_event.payload.get("result_sha256") != lock_digest:
        issues.append(
            LoadIssue(
                "CROSS_ARTIFACT_MISMATCH",
                "The trace lock event references a different pre-reveal result hash.",
                IssueSeverity.ERROR,
            )
        )
    if lock_event.previous_hash != result.pre_lock_trace_root_hash:
        issues.append(
            LoadIssue(
                "CROSS_ARTIFACT_MISMATCH",
                "The locked result does not commit to the exact trace prefix immediately before RESULT_LOCKED.",
                IssueSeverity.ERROR,
            )
        )
    if lock_event.payload.get("pre_lock_trace_root_hash") != result.pre_lock_trace_root_hash:
        issues.append(
            LoadIssue(
                "CROSS_ARTIFACT_MISMATCH",
                "The RESULT_LOCKED payload does not repeat the committed pre-lock trace root.",
                IssueSeverity.ERROR,
            )
        )


def _check_science_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_SCIENCE_ARTIFACT_BYTES:
        raise ValueError("science artifact exceeds the UI safety limit")


def _first_jsonl_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("first trace record is not an object")
                return value
    raise ValueError("trace is empty")


def _target_from_path(directory: Path) -> str:
    for part in reversed(directory.parts):
        if TARGET_PATTERN.fullmatch(part):
            return part
    return directory.name


def _looks_like_run(directory: Path) -> bool:
    return directory.is_dir() and any((directory / name).exists() for name in CORE_ARTIFACTS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_role(role: str) -> str:
    """Map science-store data/plot roles onto one UI product role."""

    for suffix in ("_data", "_plot"):
        if role.endswith(suffix):
            return role[: -len(suffix)]
    return role


__all__ = [
    "IssueSeverity",
    "KNOWN_SCIENCE_ROLES",
    "LoadIssue",
    "MissionControlRun",
    "RunPhase",
    "ScienceArtifact",
    "ScienceProduct",
    "available_target_ids",
    "image_artifact",
    "load_run",
    "load_science_product",
    "resolve_run_directory",
]
