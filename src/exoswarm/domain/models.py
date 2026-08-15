"""Typed domain contracts for an ExoSwarm investigation.

The models in this module intentionally contain no astronomy implementation and no
model-provider code.  They are the boundary between those two systems: scientific
software produces :class:`ScientificResult` objects and agents may only request
experiments through the bounded decision models below.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp (kept injectable at service boundaries)."""

    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Strict base model used by every externally exchanged domain object."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class FrozenDomainModel(DomainModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ScientificStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    ERROR = "ERROR"


class ExperimentType(StrEnum):
    LOAD_CACHED_DATA = "load_cached_data"
    QUALITY_INSPECTION = "quality_inspection"
    NORMALIZATION = "normalization"
    DETRENDING = "detrending"
    TRANSIT_SEARCH = "transit_search"
    PHASE_FOLD = "phase_fold"
    SIGNAL_QUALITY = "signal_quality"
    ODD_EVEN = "odd_even"
    SECONDARY_ECLIPSE = "secondary_eclipse"
    CONTAMINATION_SCREEN = "contamination_screen"
    HARMONIC_TEST = "harmonic_test"
    CENTROID_LOCALIZATION = "centroid_localization"
    ALTERNATE_DETRENDING = "alternate_detrending"
    ALTERNATE_APERTURE = "alternate_aperture"


class InterpretationCode(StrEnum):
    LOADED = "LOADED"
    PROCESSED = "PROCESSED"
    ACCEPTABLE = "ACCEPTABLE"
    POOR_QUALITY = "POOR_QUALITY"
    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    PASS = "PASS"
    FAIL = "FAIL"
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    SIGNIFICANT = "SIGNIFICANT"
    NOT_SIGNIFICANT = "NOT_SIGNIFICANT"
    NEIGHBOR_DETECTED = "NEIGHBOR_DETECTED"
    NO_NEARBY_SOURCE = "NO_NEARBY_SOURCE"
    TARGET_CONSISTENT = "TARGET_CONSISTENT"
    OFFSET_DETECTED = "OFFSET_DETECTED"
    PREFERRED_HALF_PERIOD = "PREFERRED_HALF_PERIOD"
    PREFERRED_NOMINAL_PERIOD = "PREFERRED_NOMINAL_PERIOD"
    PREFERRED_DOUBLE_PERIOD = "PREFERRED_DOUBLE_PERIOD"
    ROBUST = "ROBUST"
    PREPROCESSING_SENSITIVE = "PREPROCESSING_SENSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"
    PRECONDITION_NOT_MET = "PRECONDITION_NOT_MET"
    TOOL_ERROR = "TOOL_ERROR"


class QualitySeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityFlag(FrozenDomainModel):
    code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    severity: QualitySeverity = QualitySeverity.INFO
    detail: str | None = None


class ArtifactRef(FrozenDomainModel):
    """Reference to an artifact, without embedding its potentially huge contents."""

    artifact_id: str = Field(min_length=1)
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    media_type: str | None = None
    role: str | None = None
    source_uri: str | None = None


class MeasurementUncertainty(FrozenDomainModel):
    """Uncertainty or explicit tolerance attached to a named measurement."""

    value: float = Field(ge=0)
    unit: str | None = None
    method: str = Field(min_length=1)
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    kind: Literal["standard_uncertainty", "tolerance", "resolution"] = "standard_uncertainty"


class ProvenanceRecord(FrozenDomainModel):
    source: str = Field(min_length=1)
    source_uri: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    retrieved_at: datetime | None = None
    software: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ScientificResult(FrozenDomainModel):
    """Successful or explicitly partial output from deterministic science code."""

    status: Literal[ScientificStatus.SUCCESS, ScientificStatus.PARTIAL]
    experiment_type: ExperimentType
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    numerical_results: dict[str, int | float] = Field(default_factory=dict)
    result_units: dict[str, str] = Field(default_factory=dict)
    uncertainties: dict[str, MeasurementUncertainty] = Field(default_factory=dict)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    interpretation_code: InterpretationCode
    limitations: list[str] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def units_only_reference_results(self) -> ScientificResult:
        unknown = self.result_units.keys() - self.numerical_results.keys()
        if unknown:
            raise ValueError(f"units reference unknown numerical results: {sorted(unknown)}")
        unknown_uncertainties = self.uncertainties.keys() - self.numerical_results.keys()
        if unknown_uncertainties:
            raise ValueError(
                "uncertainties reference unknown numerical results: "
                f"{sorted(unknown_uncertainties)}"
            )
        return self


class ScientificFailure(FrozenDomainModel):
    """Structured failure that allows the runtime to re-plan safely."""

    status: Literal[
        ScientificStatus.PRECONDITION_FAILED,
        ScientificStatus.INVALID_REQUEST,
        ScientificStatus.ERROR,
    ]
    experiment_type: ExperimentType
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    reason_code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    suggested_alternatives: list[ExperimentType] = Field(default_factory=list)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    interpretation_code: Literal[
        InterpretationCode.PRECONDITION_NOT_MET, InterpretationCode.TOOL_ERROR
    ]


ScientificToolResponse = Annotated[
    ScientificResult | ScientificFailure, Field(discriminator="status")
]


class Candidate(FrozenDomainModel):
    candidate_id: str = Field(default_factory=lambda: f"CAND-{uuid4().hex[:12].upper()}")
    period_days: float = Field(gt=0)
    epoch_btjd: float
    transit_depth_ppm: float = Field(gt=0)
    duration_hours: float = Field(gt=0)
    signal_to_noise: float
    observed_events: int = Field(ge=1)
    uncertainties: dict[str, MeasurementUncertainty] = Field(default_factory=dict)
    search_statistic: float | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)


class Hypothesis(StrEnum):
    PLANETARY_TRANSIT = "H1_PLANETARY_TRANSIT"
    ECLIPSING_BINARY = "H2_ECLIPSING_BINARY"
    BACKGROUND_CONTAMINANT = "H3_BACKGROUND_CONTAMINANT"
    STELLAR_VARIABILITY = "H4_STELLAR_VARIABILITY"
    INSTRUMENTAL_SYSTEMATIC = "H5_INSTRUMENTAL_SYSTEMATIC"
    PERIOD_ALIAS_HARMONIC = "H6_PERIOD_ALIAS_HARMONIC"


class EvidenceState(StrEnum):
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    UNRESOLVED = "UNRESOLVED"
    WEAKENED = "WEAKENED"
    DISFAVORED = "DISFAVORED"


class HypothesisState(FrozenDomainModel):
    hypothesis: Hypothesis
    evidence_state: EvidenceState = EvidenceState.UNRESOLVED
    heuristic_evidence_weight: float = Field(
        default=0,
        description="Declared, uncalibrated evidence weight; never a probability.",
    )
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    opposing_evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def initial_hypotheses() -> dict[Hypothesis, HypothesisState]:
    return {hypothesis: HypothesisState(hypothesis=hypothesis) for hypothesis in Hypothesis}


class InvestigationStatus(StrEnum):
    ACTIVE = "ACTIVE"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class LockState(StrEnum):
    UNLOCKED = "UNLOCKED"
    READY_TO_LOCK = "READY_TO_LOCK"
    RESULT_LOCKED = "RESULT_LOCKED"
    GROUND_TRUTH_REVEALED = "GROUND_TRUTH_REVEALED"


class ScientificDisposition(StrEnum):
    NO_CREDIBLE_PERIODIC_SIGNAL = "NO CREDIBLE PERIODIC SIGNAL"
    TRANSIT_LIKE_SIGNAL = "TRANSIT-LIKE SIGNAL"
    PLANETARY_INTERPRETATION_WEAK = "PLANETARY INTERPRETATION WEAK"
    PLANETARY_INTERPRETATION_PLAUSIBLE = "PLANETARY INTERPRETATION PLAUSIBLE"
    PLANETARY_INTERPRETATION_SURVIVES_VETTING = (
        "PLANETARY INTERPRETATION SURVIVES IMPLEMENTED VETTING"
    )
    INCONCLUSIVE = "INCONCLUSIVE — ADDITIONAL DATA REQUIRED"


class Budget(FrozenDomainModel):
    limit: int = Field(ge=0)
    used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def usage_does_not_exceed_limit(self) -> Budget:
        if self.used > self.limit:
            raise ValueError("used budget cannot exceed limit")
        return self

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self, amount: int = 1) -> Budget:
        if amount < 0:
            raise ValueError("budget amount must be non-negative")
        if amount > self.remaining:
            raise ValueError("budget exhausted")
        return self.model_copy(update={"used": self.used + amount})


class DataQualitySummary(FrozenDomainModel):
    total_cadences: int = Field(ge=0)
    usable_cadences: int = Field(ge=0)
    rejected_cadences: int = Field(ge=0)
    median_cadence_minutes: float | None = Field(default=None, gt=0)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    evidence_id: str | None = None

    @model_validator(mode="after")
    def cadence_counts_are_consistent(self) -> DataQualitySummary:
        if self.usable_cadences + self.rejected_cadences > self.total_cadences:
            raise ValueError("usable + rejected cadences exceeds total cadences")
        return self


class PreprocessingRun(FrozenDomainModel):
    run_id: str = Field(default_factory=lambda: f"PREP-{uuid4().hex[:12].upper()}")
    method: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    evidence_id: str | None = None


class BackendTargetMappingRef(FrozenDomainModel):
    """Opaque handle only; the real mapping is owned by ``exoswarm.security``."""

    mapping_key: str = Field(min_length=16)


class AgentCandidateView(FrozenDomainModel):
    candidate_id: str
    period_days: float
    epoch_btjd: float
    transit_depth_ppm: float
    duration_hours: float
    signal_to_noise: float
    observed_events: int
    uncertainties: dict[str, MeasurementUncertainty] = Field(default_factory=dict)
    source_evidence_ids: list[str] = Field(default_factory=list)


class AgentInvestigationView(FrozenDomainModel):
    """Compact and identity-safe state packet supplied to scientific agents."""

    opaque_target_id: str
    status: InvestigationStatus
    candidates: list[AgentCandidateView] = Field(default_factory=list)
    hypotheses: dict[Hypothesis, HypothesisState] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    completed_tests: list[ExperimentType] = Field(default_factory=list)
    available_tests: list[ExperimentType] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    experiment_budget_remaining: int
    agent_turn_budget_remaining: int
    lock_state: LockState
    final_disposition: ScientificDisposition | None = None


class InvestigationState(DomainModel):
    """Application-owned state; agents receive only :meth:`to_agent_view`."""

    opaque_target_id: str = Field(pattern=r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
    backend_target_mapping: BackendTargetMappingRef = Field(repr=False, exclude=True)
    trace_id: str = Field(default_factory=lambda: f"TRACE-{uuid4().hex.upper()}")
    status: InvestigationStatus = InvestigationStatus.ACTIVE
    available_data_products: list[ArtifactRef] = Field(default_factory=list)
    data_quality: DataQualitySummary | None = None
    preprocessing_runs: list[PreprocessingRun] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    hypotheses: dict[Hypothesis, HypothesisState] = Field(default_factory=initial_hypotheses)
    evidence: list[str] = Field(default_factory=list)
    completed_tests: list[ExperimentType] = Field(default_factory=list)
    available_tests: list[ExperimentType] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    experiment_budget: Budget = Field(default_factory=lambda: Budget(limit=12))
    agent_turn_budget: Budget = Field(default_factory=lambda: Budget(limit=15))
    lock_state: LockState = LockState.UNLOCKED
    final_disposition: ScientificDisposition | None = None
    _real_identity: None = PrivateAttr(default=None)

    @field_validator("hypotheses")
    @classmethod
    def hypothesis_keys_match_values(
        cls, value: dict[Hypothesis, HypothesisState]
    ) -> dict[Hypothesis, HypothesisState]:
        for key, state in value.items():
            if key != state.hypothesis:
                raise ValueError(f"hypothesis key {key} does not match its state")
        return value

    @model_validator(mode="after")
    def lock_and_disposition_are_consistent(self) -> InvestigationState:
        if (
            self.lock_state in {LockState.READY_TO_LOCK, LockState.RESULT_LOCKED}
            and self.final_disposition is None
        ):
            raise ValueError("a final disposition is required before result locking")
        if self.lock_state is LockState.GROUND_TRUTH_REVEALED and self.final_disposition is None:
            raise ValueError("a revealed investigation must have a final disposition")
        return self

    def to_agent_view(self) -> AgentInvestigationView:
        """Return a deliberately small view with no backend mapping or artifact paths."""

        return AgentInvestigationView(
            opaque_target_id=self.opaque_target_id,
            status=self.status,
            candidates=[
                AgentCandidateView(
                    candidate_id=candidate.candidate_id,
                    period_days=candidate.period_days,
                    epoch_btjd=candidate.epoch_btjd,
                    transit_depth_ppm=candidate.transit_depth_ppm,
                    duration_hours=candidate.duration_hours,
                    signal_to_noise=candidate.signal_to_noise,
                    observed_events=candidate.observed_events,
                    uncertainties=candidate.uncertainties,
                    source_evidence_ids=candidate.source_evidence_ids,
                )
                for candidate in self.candidates
            ],
            hypotheses=self.hypotheses,
            evidence_ids=self.evidence,
            completed_tests=self.completed_tests,
            available_tests=self.available_tests,
            unresolved_questions=self.unresolved_questions,
            experiment_budget_remaining=self.experiment_budget.remaining,
            agent_turn_budget_remaining=self.agent_turn_budget.remaining,
            lock_state=self.lock_state,
            final_disposition=self.final_disposition,
        )

    def transition_lock(self, next_state: LockState) -> InvestigationState:
        allowed: dict[LockState, set[LockState]] = {
            LockState.UNLOCKED: {LockState.READY_TO_LOCK},
            LockState.READY_TO_LOCK: {LockState.RESULT_LOCKED},
            LockState.RESULT_LOCKED: {LockState.GROUND_TRUTH_REVEALED},
            LockState.GROUND_TRUTH_REVEALED: set(),
        }
        if next_state not in allowed[self.lock_state]:
            raise ValueError(f"invalid lock transition: {self.lock_state} -> {next_state}")
        self.lock_state = next_state
        return self


class DecisionPriority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SkepticAction(StrEnum):
    REQUEST_EXPERIMENT = "REQUEST_EXPERIMENT"
    STOP = "STOP"


class SkepticDecision(FrozenDomainModel):
    decision_id: str = Field(default_factory=lambda: f"SK-{uuid4().hex.upper()}")
    action: SkepticAction = SkepticAction.REQUEST_EXPERIMENT
    hypothesis_under_test: Hypothesis | None = None
    requested_experiment: ExperimentType | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reason_code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    explanation: str = Field(min_length=1)
    expected_discriminating_result: str = Field(min_length=1)
    predicted_outcomes: dict[str, str] = Field(default_factory=dict)
    expected_information_value: float = Field(
        ge=0,
        le=1,
        description="Uncalibrated decision utility, not an astrophysical probability.",
    )
    stop_if: str | None = None
    priority: DecisionPriority = DecisionPriority.MEDIUM

    @model_validator(mode="after")
    def action_has_matching_fields(self) -> SkepticDecision:
        if self.action is SkepticAction.REQUEST_EXPERIMENT:
            if self.requested_experiment is None or self.hypothesis_under_test is None:
                raise ValueError(
                    "experiment decisions require requested_experiment and hypothesis_under_test"
                )
        elif self.requested_experiment is not None or self.parameters:
            raise ValueError("STOP decisions cannot request an experiment or parameters")
        return self


class ToolRequest(FrozenDomainModel):
    request_id: str = Field(default_factory=lambda: f"REQ-{uuid4().hex.upper()}")
    experiment_type: ExperimentType
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    adaptive: bool = False
    requested_by: str = Field(min_length=1)
    justification: str | None = None
    agent_decision_id: str | None = None


class CriticVerdict(StrEnum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    VETO = "VETO"


class CriticDecision(FrozenDomainModel):
    critic_decision_id: str = Field(default_factory=lambda: f"CR-{uuid4().hex.upper()}")
    reviewed_request_id: str
    verdict: CriticVerdict
    reason_code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    reason: str = Field(min_length=1)
    revised_request: ToolRequest | None = None

    @model_validator(mode="after")
    def revision_matches_verdict(self) -> CriticDecision:
        if self.verdict is CriticVerdict.REVISE and self.revised_request is None:
            raise ValueError("REVISE requires exactly one revised_request")
        if self.verdict is not CriticVerdict.REVISE and self.revised_request is not None:
            raise ValueError("only REVISE may include a revised_request")
        return self


class ReviewStatus(StrEnum):
    ALLOWED = "ALLOWED"
    REJECTED = "REJECTED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ToolRequestReview(FrozenDomainModel):
    status: ReviewStatus
    request_id: str
    experiment_type: ExperimentType
    normalized_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reason_code: str | None = None
    reason: str | None = None
    suggested_alternatives: list[ExperimentType] = Field(default_factory=list)


class LockedInvestigationResult(FrozenDomainModel):
    """Pre-reveal result.  It deliberately has no field capable of storing identity."""

    schema_version: str = "1.0"
    opaque_target_id: str = Field(pattern=r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
    trace_id: str
    disposition: ScientificDisposition
    candidate: Candidate | None = None
    completed_tests: list[ExperimentType]
    evidence_ids: list[str]
    evidence_root_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    pre_lock_trace_root_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    limitations: list[str]
    created_at: datetime = Field(default_factory=utc_now)
    lock_state: Literal[LockState.RESULT_LOCKED] = LockState.RESULT_LOCKED


class ResultLockReceipt(FrozenDomainModel):
    opaque_target_id: str
    result_path: str
    hash_path: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    locked_at: datetime


class CatalogMeasurement(FrozenDomainModel):
    value: float
    unit: str
    uncertainty: float | None = Field(default=None, ge=0)
    source_field: str | None = None


class GroundTruthRecord(FrozenDomainModel):
    actual_target_identity: str = Field(min_length=1)
    catalog_name: str = Field(min_length=1)
    catalog_status: str = Field(min_length=1)
    measurements: dict[str, CatalogMeasurement] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class RevealArtifact(FrozenDomainModel):
    schema_version: str = "1.0"
    opaque_target_id: str
    locked_result_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    ground_truth: GroundTruthRecord
    revealed_at: datetime = Field(default_factory=utc_now)
    lock_state: Literal[LockState.GROUND_TRUTH_REVEALED] = LockState.GROUND_TRUTH_REVEALED
