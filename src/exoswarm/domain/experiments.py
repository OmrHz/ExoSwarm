"""Bounded scientific experiment registry and deterministic request validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .hypotheses import MANDATORY_VETTING
from .models import (
    DomainModel,
    ExperimentType,
    InterpretationCode,
    InvestigationState,
    LockState,
    ReviewStatus,
    ScientificFailure,
    ScientificStatus,
    ToolRequest,
    ToolRequestReview,
)


class NoParameters(DomainModel):
    pass


class CachedDataParameters(DomainModel):
    product_id: str | None = None


class CandidateParameters(DomainModel):
    candidate_id: str = Field(min_length=1)


class DetrendingParameters(DomainModel):
    method: Literal["median_filter", "savgol"] = "median_filter"
    window_hours: float = Field(default=24.0, ge=12, le=72)
    sigma_clip: float = Field(default=5.0, ge=2, le=10)


class TransitSearchParameters(DomainModel):
    min_period_days: float = Field(default=0.5, ge=0.1, le=100)
    max_period_days: float = Field(default=20.0, ge=0.2, le=100)
    durations_hours: list[float] = Field(
        default_factory=lambda: [1.0, 2.0, 4.0, 6.0], min_length=1, max_length=20
    )

    @model_validator(mode="after")
    def sensible_search_range(self) -> TransitSearchParameters:
        if self.max_period_days <= self.min_period_days:
            raise ValueError("max_period_days must exceed min_period_days")
        if any(duration <= 0 or duration > 24 for duration in self.durations_hours):
            raise ValueError("durations_hours must be within (0, 24]")
        return self


class HarmonicTestParameters(CandidateParameters):
    base_period_days: float = Field(gt=0, le=100)
    factors: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])

    @field_validator("factors")
    @classmethod
    def only_declared_harmonics(cls, value: list[float]) -> list[float]:
        if value != [0.5, 1.0, 2.0]:
            raise ValueError("harmonic test must compare exactly P/2, P, and 2P")
        return value


class CentroidParameters(CandidateParameters):
    aperture_id: Literal["pipeline", "spoc"] | None = None
    transit_window_scale: float = Field(default=1.0, gt=0.25, le=3)


class AlternateDetrendingParameters(CandidateParameters):
    method: Literal["median_filter", "savgol"]
    window_hours: float = Field(ge=12, le=72)


class AlternateApertureParameters(CandidateParameters):
    aperture_id: str = Field(min_length=1)


class ExperimentRegistryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    experiment_type: ExperimentType
    tool_name: str
    parameter_model: type[BaseModel]
    mandatory: bool = False
    adaptive: bool = False
    cost: int = 1
    required_tests: tuple[ExperimentType, ...] = ()
    requires_candidate: bool = False
    minimum_events: int | None = None
    required_artifact_role: str | None = None
    max_executions: int = 1


class ExperimentRegistry:
    """The only source of executable experiment names and their preconditions."""

    def __init__(self, definitions: list[ExperimentDefinition] | None = None) -> None:
        self._definitions: dict[ExperimentType, ExperimentDefinition] = {}
        for definition in definitions or default_experiment_definitions():
            self.register(definition)

    def register(self, definition: ExperimentDefinition) -> None:
        if definition.experiment_type in self._definitions:
            raise ExperimentRegistryError(
                f"duplicate experiment: {definition.experiment_type.value}"
            )
        if definition.cost <= 0 or definition.max_executions <= 0:
            raise ExperimentRegistryError("cost and max_executions must be positive")
        self._definitions[definition.experiment_type] = definition

    @property
    def experiment_types(self) -> tuple[ExperimentType, ...]:
        return tuple(self._definitions)

    @property
    def mandatory_experiments(self) -> frozenset[ExperimentType]:
        return frozenset(
            definition.experiment_type
            for definition in self._definitions.values()
            if definition.mandatory
        )

    def definition(self, experiment_type: ExperimentType) -> ExperimentDefinition:
        try:
            return self._definitions[experiment_type]
        except KeyError as exc:
            raise ExperimentRegistryError(f"unknown experiment: {experiment_type.value}") from exc

    def initialize_available_tests(self, state: InvestigationState) -> None:
        if state.lock_state in {LockState.RESULT_LOCKED, LockState.GROUND_TRUTH_REVEALED}:
            state.available_tests = []
            return
        available: list[ExperimentType] = []
        for experiment_type in self.experiment_types:
            synthetic = ToolRequest(
                experiment_type=experiment_type,
                parameters=self._default_parameters(experiment_type, state),
                adaptive=self.definition(experiment_type).adaptive,
                requested_by="runtime",
                justification="availability probe for a bounded repeat",
            )
            review = self.validate(synthetic, state, enforce_available_list=False)
            if review.status is ReviewStatus.ALLOWED:
                available.append(experiment_type)
        state.available_tests = available

    def validate_raw(self, payload: dict[str, Any], state: InvestigationState) -> ToolRequestReview:
        request_id = str(payload.get("request_id", "INVALID-REQUEST"))
        raw_experiment = payload.get("experiment_type", "invalid")
        try:
            request = ToolRequest.model_validate(payload)
        except ValidationError as exc:
            try:
                experiment = ExperimentType(str(raw_experiment))
            except ValueError:
                # A bounded enum cannot represent an unknown tool, so use a harmless
                # known type solely in the typed rejection object.
                experiment = ExperimentType.QUALITY_INSPECTION
            return ToolRequestReview(
                status=ReviewStatus.REJECTED,
                request_id=request_id,
                experiment_type=experiment,
                reason_code="INVALID_TOOL_REQUEST_SCHEMA",
                reason=str(exc),
            )
        return self.validate(request, state)

    def validate(
        self,
        request: ToolRequest,
        state: InvestigationState,
        *,
        enforce_available_list: bool = True,
    ) -> ToolRequestReview:
        definition = self._definitions.get(request.experiment_type)
        if definition is None:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "UNKNOWN_EXPERIMENT",
                "requested experiment is not in the bounded registry",
            )
        if state.lock_state in {LockState.RESULT_LOCKED, LockState.GROUND_TRUTH_REVEALED}:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "RESULT_ALREADY_LOCKED",
                "scientific experiments cannot execute after result lock",
            )
        if enforce_available_list and request.experiment_type not in state.available_tests:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "EXPERIMENT_NOT_AVAILABLE",
                "experiment is not currently exposed to the requesting agent",
                self._precondition_alternatives(state),
            )
        try:
            normalized = definition.parameter_model.model_validate(request.parameters).model_dump(
                mode="json"
            )
        except ValidationError as exc:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "INVALID_EXPERIMENT_PARAMETERS",
                str(exc),
            )
        if state.experiment_budget.remaining < definition.cost:
            return self._rejection(
                request,
                ReviewStatus.BUDGET_EXHAUSTED,
                "EXPERIMENT_BUDGET_EXHAUSTED",
                "insufficient experiment budget",
            )
        execution_count = state.completed_tests.count(request.experiment_type)
        if execution_count >= definition.max_executions:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "MAX_EXECUTIONS_REACHED",
                "the experiment reached its declared execution limit",
            )
        if execution_count and not request.justification:
            return self._rejection(
                request,
                ReviewStatus.REJECTED,
                "REDUNDANT_EXPERIMENT",
                "a repeat requires an explicit scientific justification",
            )

        missing = [
            dependency
            for dependency in definition.required_tests
            if dependency not in state.completed_tests
        ]
        if missing:
            return self._rejection(
                request,
                ReviewStatus.PRECONDITION_FAILED,
                "MISSING_DEPENDENCY",
                "required prior experiments are incomplete: "
                + ", ".join(item.value for item in missing),
                missing,
            )
        candidate = None
        candidate_id = normalized.get("candidate_id")
        if definition.requires_candidate:
            candidate = next(
                (
                    item
                    for item in state.candidates
                    if candidate_id is None or item.candidate_id == candidate_id
                ),
                None,
            )
            if candidate is None:
                return self._rejection(
                    request,
                    ReviewStatus.PRECONDITION_FAILED,
                    "CANDIDATE_REQUIRED",
                    "experiment requires an existing candidate with a matching candidate_id",
                    [ExperimentType.TRANSIT_SEARCH],
                )
        if (
            candidate is not None
            and definition.minimum_events is not None
            and candidate.observed_events < definition.minimum_events
        ):
            alternatives = [
                item
                for item in (
                    ExperimentType.SECONDARY_ECLIPSE,
                    ExperimentType.HARMONIC_TEST,
                )
                if item != request.experiment_type
            ]
            return self._rejection(
                request,
                ReviewStatus.PRECONDITION_FAILED,
                "INSUFFICIENT_OBSERVED_EVENTS",
                f"{request.experiment_type.value} requires >= "
                f"{definition.minimum_events} usable events; found {candidate.observed_events}",
                alternatives,
            )
        if candidate is not None and request.experiment_type is ExperimentType.HARMONIC_TEST:
            base_period = float(normalized["base_period_days"])
            tolerance = max(1e-12, abs(candidate.period_days) * 1e-12)
            if abs(base_period - candidate.period_days) > tolerance:
                return self._rejection(
                    request,
                    ReviewStatus.REJECTED,
                    "CANDIDATE_EPHEMERIS_MISMATCH",
                    "harmonic base_period_days must be the deterministic current candidate period",
                )
        if definition.required_artifact_role and not any(
            artifact.role == definition.required_artifact_role
            for artifact in state.available_data_products
        ):
            return self._rejection(
                request,
                ReviewStatus.PRECONDITION_FAILED,
                "REQUIRED_DATA_PRODUCT_MISSING",
                f"experiment requires an artifact with role {definition.required_artifact_role!r}",
                [ExperimentType.CONTAMINATION_SCREEN],
            )

        return ToolRequestReview(
            status=ReviewStatus.ALLOWED,
            request_id=request.request_id,
            experiment_type=request.experiment_type,
            normalized_parameters=normalized,
        )

    def record_attempt(
        self,
        state: InvestigationState,
        request: ToolRequest,
        *,
        successful: bool,
    ) -> None:
        """Consume budget; only successful attempts satisfy completed diagnostics."""

        definition = self.definition(request.experiment_type)
        state.experiment_budget = state.experiment_budget.consume(definition.cost)
        if successful:
            state.completed_tests = [*state.completed_tests, request.experiment_type]
        self.initialize_available_tests(state)

    @staticmethod
    def consume_agent_turn(state: InvestigationState) -> None:
        state.agent_turn_budget = state.agent_turn_budget.consume(1)

    def missing_mandatory(self, state: InvestigationState) -> frozenset[ExperimentType]:
        return self.mandatory_experiments - set(state.completed_tests)

    def mandatory_complete(self, state: InvestigationState) -> bool:
        return not self.missing_mandatory(state)

    def next_mandatory(self, state: InvestigationState) -> ExperimentType | None:
        missing = self.missing_mandatory(state)
        for experiment in self.experiment_types:
            if experiment not in missing:
                continue
            request = ToolRequest(
                experiment_type=experiment,
                parameters=self._default_parameters(experiment, state),
                requested_by="runtime-mandatory-policy",
            )
            if (
                self.validate(request, state, enforce_available_list=False).status
                is ReviewStatus.ALLOWED
            ):
                return experiment
        return None

    def as_scientific_failure(
        self, review: ToolRequestReview, *, tool_version: str = "registry"
    ) -> ScientificFailure:
        return ScientificFailure(
            status=(
                ScientificStatus.PRECONDITION_FAILED
                if review.status is ReviewStatus.PRECONDITION_FAILED
                else ScientificStatus.INVALID_REQUEST
            ),
            experiment_type=review.experiment_type,
            tool_name=self._definitions.get(
                review.experiment_type,
                ExperimentDefinition(review.experiment_type, "unknown", NoParameters),
            ).tool_name,
            tool_version=tool_version,
            parameters=review.normalized_parameters,
            reason=review.reason or review.status.value,
            reason_code=review.reason_code or review.status.value,
            suggested_alternatives=review.suggested_alternatives,
            interpretation_code=InterpretationCode.PRECONDITION_NOT_MET,
        )

    def _default_parameters(
        self,
        experiment_type: ExperimentType,
        state: InvestigationState | None = None,
    ) -> dict[str, Any]:
        definition = self.definition(experiment_type)
        defaults: dict[str, Any] = {}
        if definition.requires_candidate and state and state.candidates:
            candidate = state.candidates[0]
            defaults["candidate_id"] = candidate.candidate_id
            if experiment_type is ExperimentType.HARMONIC_TEST:
                defaults["base_period_days"] = candidate.period_days
            elif experiment_type is ExperimentType.ALTERNATE_DETRENDING:
                defaults.update({"method": "savgol", "window_hours": 36.0})
            elif experiment_type is ExperimentType.ALTERNATE_APERTURE:
                defaults["aperture_id"] = "alternate"
        try:
            return definition.parameter_model.model_validate(defaults).model_dump(mode="json")
        except ValidationError:
            return defaults

    def _precondition_alternatives(self, state: InvestigationState) -> list[ExperimentType]:
        return [
            experiment
            for experiment in state.available_tests
            if state.completed_tests.count(experiment) < self.definition(experiment).max_executions
        ]

    @staticmethod
    def _rejection(
        request: ToolRequest,
        status: ReviewStatus,
        reason_code: str,
        reason: str,
        alternatives: list[ExperimentType] | tuple[ExperimentType, ...] = (),
    ) -> ToolRequestReview:
        return ToolRequestReview(
            status=status,
            request_id=request.request_id,
            experiment_type=request.experiment_type,
            reason_code=reason_code,
            reason=reason,
            suggested_alternatives=list(alternatives),
        )


def default_experiment_definitions() -> list[ExperimentDefinition]:
    """Return the deliberately small P0/P1 experiment surface."""

    return [
        ExperimentDefinition(
            ExperimentType.LOAD_CACHED_DATA,
            "load_cached_tess_data",
            CachedDataParameters,
            mandatory=True,
        ),
        ExperimentDefinition(
            ExperimentType.QUALITY_INSPECTION,
            "inspect_tess_quality",
            NoParameters,
            mandatory=True,
            required_tests=(ExperimentType.LOAD_CACHED_DATA,),
        ),
        ExperimentDefinition(
            ExperimentType.NORMALIZATION,
            "normalize_light_curve",
            NoParameters,
            mandatory=True,
            required_tests=(ExperimentType.QUALITY_INSPECTION,),
        ),
        ExperimentDefinition(
            ExperimentType.DETRENDING,
            "detrend_light_curve",
            DetrendingParameters,
            mandatory=True,
            required_tests=(ExperimentType.NORMALIZATION,),
        ),
        ExperimentDefinition(
            ExperimentType.TRANSIT_SEARCH,
            "box_least_squares_search",
            TransitSearchParameters,
            mandatory=True,
            required_tests=(ExperimentType.DETRENDING,),
        ),
        ExperimentDefinition(
            ExperimentType.PHASE_FOLD,
            "phase_fold_candidate",
            CandidateParameters,
            mandatory=True,
            requires_candidate=True,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.SIGNAL_QUALITY,
            "evaluate_signal_quality",
            CandidateParameters,
            mandatory=True,
            requires_candidate=True,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.ODD_EVEN,
            "odd_even_transit_test",
            CandidateParameters,
            mandatory=True,
            requires_candidate=True,
            minimum_events=4,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.SECONDARY_ECLIPSE,
            "secondary_eclipse_test",
            CandidateParameters,
            mandatory=True,
            requires_candidate=True,
            minimum_events=2,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.CONTAMINATION_SCREEN,
            "basic_contamination_screen",
            CandidateParameters,
            mandatory=True,
            requires_candidate=True,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.HARMONIC_TEST,
            "period_harmonic_test",
            HarmonicTestParameters,
            adaptive=True,
            requires_candidate=True,
            minimum_events=2,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.CENTROID_LOCALIZATION,
            "centroid_localization",
            CentroidParameters,
            adaptive=True,
            requires_candidate=True,
            required_artifact_role="target_pixel",
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
        ),
        ExperimentDefinition(
            ExperimentType.ALTERNATE_DETRENDING,
            "alternate_detrending_sensitivity",
            AlternateDetrendingParameters,
            adaptive=True,
            requires_candidate=True,
            required_tests=(ExperimentType.TRANSIT_SEARCH,),
            max_executions=2,
        ),
    ]


# Assert that the explicitly required safety baseline cannot drift out of the
# registry due to a future refactor.
assert {
    definition.experiment_type
    for definition in default_experiment_definitions()
    if definition.mandatory
} >= MANDATORY_VETTING
