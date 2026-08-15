"""Application-owned bounded investigation loop.

The Director controls permissions, budgets, mandatory diagnostics, state transitions,
failure handling, evidence updates, and locking. Model output can select an adaptive
experiment but cannot execute tools or mutate scientific state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from exoswarm.agents.factory import build_provider
from exoswarm.agents.guardrails import sanitize_critic_decision, sanitize_skeptic_decision
from exoswarm.agents.provider import InferenceProvider, UnavailableProvider
from exoswarm.agents.roles import CriticAgent, SkepticAgent, build_evidence_packet
from exoswarm.agents.structured import DecisionSource, StructuredAgentRunner
from exoswarm.config import Settings
from exoswarm.domain.experiments import ExperimentRegistry
from exoswarm.domain.hypotheses import DeterministicHypothesisUpdater, derive_disposition
from exoswarm.domain.ledger import EvidenceItem, EvidenceLedger
from exoswarm.domain.models import (
    Budget,
    Candidate,
    CriticDecision,
    CriticVerdict,
    DataQualitySummary,
    ExperimentType,
    InterpretationCode,
    InvestigationState,
    InvestigationStatus,
    LockedInvestigationResult,
    LockState,
    PreprocessingRun,
    ResultLockReceipt,
    RevealArtifact,
    ReviewStatus,
    ScientificDisposition,
    ScientificFailure,
    ScientificResult,
    ScientificToolResponse,
    SkepticAction,
    SkepticDecision,
    ToolRequest,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder
from exoswarm.security.blindness import GroundTruthGate, OpaqueTargetVault
from exoswarm.security.locking import ResultLocker


class ScientificToolbox(Protocol):
    """The runtime-facing surface implemented by deterministic science code."""

    def execute(
        self, request: ToolRequest, state: InvestigationState
    ) -> ScientificToolResponse: ...


class InvestigationRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InvestigationOutcome:
    state: InvestigationState
    ledger: EvidenceLedger
    trace: TraceRecorder
    receipt: ResultLockReceipt
    reveal: RevealArtifact | None
    run_directory: Path


class ScientificDirector:
    """Run one opaque target through mandatory vetting and bounded adaptive inquiry."""

    def __init__(
        self,
        *,
        settings: Settings,
        toolbox: ScientificToolbox,
        vault: OpaqueTargetVault,
        provider: InferenceProvider | None = None,
        registry: ExperimentRegistry | None = None,
        locker: ResultLocker | None = None,
        gate: GroundTruthGate | None = None,
    ) -> None:
        self.settings = settings
        self.toolbox = toolbox
        self.vault = vault
        self.registry = registry or ExperimentRegistry()
        self.locker = locker or ResultLocker()
        self.gate = gate or GroundTruthGate(vault, self.locker)
        self.provider = provider or build_provider(settings)
        self.updater = DeterministicHypothesisUpdater()

    def investigate(
        self,
        opaque_target_id: str,
        *,
        run_directory: Path | None = None,
        reveal: bool = False,
        policy: Literal["adaptive", "fixed"] = "adaptive",
    ) -> InvestigationOutcome:
        if not self.vault.contains(opaque_target_id):
            raise InvestigationRuntimeError(f"unknown opaque target: {opaque_target_id}")
        directory = (run_directory or (self.settings.runs_dir / opaque_target_id)).resolve()
        self._assert_new_run_directory(directory)
        directory.mkdir(parents=True, exist_ok=True)

        state = InvestigationState(
            opaque_target_id=opaque_target_id,
            backend_target_mapping=self.vault.mapping_ref(opaque_target_id),
            available_data_products=self.vault.science_data(opaque_target_id).artifacts,
            experiment_budget=Budget(limit=self.settings.experiment_budget),
            agent_turn_budget=Budget(limit=self.settings.max_agent_turns),
        )
        ledger = EvidenceLedger(directory / "evidence.jsonl")
        trace = TraceRecorder(
            trace_id=state.trace_id,
            opaque_target_id=opaque_target_id,
            path=directory / "trace.jsonl",
        )
        trace.append(
            TraceEventType.INVESTIGATION_INITIALIZED,
            {
                "settings": self.settings.safe_summary(),
                "available_product_roles": self.vault.agent_context(
                    opaque_target_id
                ).available_product_roles,
                "ground_truth_available": False,
                "investigation_policy": policy,
            },
        )
        self.registry.initialize_available_tests(state)

        runner = StructuredAgentRunner(
            self.provider,
            trace=self._agent_trace_adapter(trace),
        )
        skeptic = SkepticAgent(runner)
        critic = CriticAgent(runner)

        mandatory_ok = self._run_mandatory(state, ledger, trace)
        if mandatory_ok and state.candidates and policy == "adaptive":
            self._run_adaptive(state, ledger, trace, skeptic, critic)
        elif mandatory_ok and state.candidates:
            self._run_fixed_baseline(state, ledger, trace)

        outcome = self._lock(state, ledger, trace, directory)
        reveal_artifact: RevealArtifact | None = None
        if reveal:
            reveal_artifact = self.reveal_locked(outcome.receipt, trace=trace)
            state.transition_lock(LockState.GROUND_TRUTH_REVEALED)
        return InvestigationOutcome(
            state=state,
            ledger=ledger,
            trace=trace,
            receipt=outcome.receipt,
            reveal=reveal_artifact,
            run_directory=directory,
        )

    def reveal_locked(
        self,
        receipt: ResultLockReceipt,
        *,
        trace: TraceRecorder | None = None,
    ) -> RevealArtifact:
        """Cross the catalog boundary only after the on-disk lock verifies."""

        self.gate.unlock_after_result_lock(receipt, trace=trace)
        return self.gate.create_reveal_artifact(receipt, trace=trace)

    def _run_mandatory(
        self,
        state: InvestigationState,
        ledger: EvidenceLedger,
        trace: TraceRecorder,
    ) -> bool:
        failed_experiments: set[ExperimentType] = set()
        while not self.registry.mandatory_complete(state):
            if state.experiment_budget.remaining == 0:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "EXPERIMENT_BUDGET_EXHAUSTED",
                        "phase": "mandatory",
                    },
                )
                return False
            experiment = self.registry.next_mandatory(state)
            if experiment is None or experiment in failed_experiments:
                missing = sorted(item.value for item in self.registry.missing_mandatory(state))
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "MANDATORY_PRECONDITION_UNRESOLVED",
                        "missing_experiments": missing,
                    },
                )
                return False
            request = ToolRequest(
                experiment_type=experiment,
                parameters=self._mandatory_parameters(experiment, state),
                adaptive=False,
                requested_by="scientific-director/mandatory-policy",
                justification="Code-enforced scientific baseline from the ExoSwarm specification.",
            )
            if not self._execute_request(state, ledger, trace, request):
                failed_experiments.add(experiment)
                return False
        return True

    def _run_adaptive(
        self,
        state: InvestigationState,
        ledger: EvidenceLedger,
        trace: TraceRecorder,
        skeptic: SkepticAgent,
        critic: CriticAgent,
    ) -> None:
        seen: set[tuple[ExperimentType, tuple[tuple[str, str], ...]]] = set()
        while state.status is InvestigationStatus.ACTIVE:
            if state.agent_turn_budget.remaining == 0 or state.experiment_budget.remaining == 0:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "ADAPTIVE_BUDGET_EXHAUSTED",
                        "agent_turns_remaining": state.agent_turn_budget.remaining,
                        "experiments_remaining": state.experiment_budget.remaining,
                    },
                )
                break
            if (
                ExperimentType.HARMONIC_TEST in state.completed_tests
                and self._strong_rejection_evidence(state)
            ):
                # Once a strong non-planetary alternative already determines the conservative
                # disposition, additional optional work is not scientifically economical.
                break

            packet = build_evidence_packet(state, ledger)
            self.registry.consume_agent_turn(state)
            call = skeptic.decide(packet)
            decision, decision_guard = sanitize_skeptic_decision(call.value, ledger)
            if decision_guard["changed"]:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "AGENT_PROSE_GUARDRAIL_REPAIR",
                        "role": "SKEPTIC",
                        "repair_summary": decision_guard,
                    },
                )
            trace.append(
                TraceEventType.AGENT_DECISION,
                {
                    "role": "SKEPTIC",
                    "decision": decision.model_dump(mode="json"),
                    "used_fallback": call.used_fallback,
                    "repaired": call.repaired,
                    "attempts": call.attempts,
                    "decision_source": self._decision_source(
                        call.decision_source, decision_guard
                    ).value,
                    "provider": self.provider.name,
                    "model": self.provider.model,
                    "provider_request_ids": list(call.provider_request_ids),
                },
            )
            self._trace_budget(state, trace)
            if decision.action is SkepticAction.STOP:
                break

            request = self._request_from_skeptic(decision, state)
            review = self.registry.validate(request, state)
            if review.status is not ReviewStatus.ALLOWED:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "INVALID_AGENT_TOOL_REQUEST",
                        "review": review.model_dump(mode="json"),
                    },
                )
                request = self._safe_request_after_invalid(packet, decision, state)
                if request is None:
                    break
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "DETERMINISTIC_TOOL_SELECTION_FALLBACK",
                        "request": request.model_dump(mode="json"),
                    },
                )

            fingerprint = (
                request.experiment_type,
                tuple(sorted((str(key), str(value)) for key, value in request.parameters.items())),
            )
            if fingerprint in seen:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "REPEATED_LOW_VALUE_ACTION",
                        "experiment": request.experiment_type.value,
                    },
                )
                break
            seen.add(fingerprint)

            if state.agent_turn_budget.remaining == 0:
                trace.append(
                    TraceEventType.FALLBACK,
                    {"reason_code": "CRITIC_TURN_UNAVAILABLE"},
                )
                break
            self.registry.consume_agent_turn(state)
            critic_call = critic.review(packet, request)
            critic_decision, critic_guard = sanitize_critic_decision(critic_call.value, ledger)
            if critic_guard["changed"]:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "AGENT_PROSE_GUARDRAIL_REPAIR",
                        "role": "CRITIC",
                        "repair_summary": critic_guard,
                    },
                )
            trace.append(
                TraceEventType.CRITIC_DECISION,
                {
                    "role": "CRITIC",
                    "decision": critic_decision.model_dump(mode="json"),
                    "used_fallback": critic_call.used_fallback,
                    "repaired": critic_call.repaired,
                    "attempts": critic_call.attempts,
                    "decision_source": self._decision_source(
                        critic_call.decision_source, critic_guard
                    ).value,
                    "provider": self.provider.name,
                    "model": self.provider.model,
                    "provider_request_ids": list(critic_call.provider_request_ids),
                    "revision_round": 0,
                },
            )
            self._trace_budget(state, trace)
            if critic_decision.reviewed_request_id != request.request_id:
                trace.append(
                    TraceEventType.FALLBACK,
                    {
                        "reason_code": "CRITIC_REVIEW_TARGET_MISMATCH",
                        "expected_request_id": request.request_id,
                        "reviewed_request_id": critic_decision.reviewed_request_id,
                    },
                )
                break
            if critic_decision.verdict is CriticVerdict.VETO:
                break
            if critic_decision.verdict is CriticVerdict.REVISE:
                # Exactly one revision is accepted; it is validated but not recursively reviewed.
                assert critic_decision.revised_request is not None
                request = self._bind_adaptive_parameters(critic_decision.revised_request, state)
                revised_review = self.registry.validate(request, state)
                if revised_review.status is not ReviewStatus.ALLOWED:
                    trace.append(
                        TraceEventType.FALLBACK,
                        {
                            "reason_code": "CRITIC_REVISION_INVALID",
                            "review": revised_review.model_dump(mode="json"),
                            "revision_round": 1,
                        },
                    )
                    break

            if not self._execute_request(
                state,
                ledger,
                trace,
                request,
                agent_request_id=request.request_id,
                critic_decision=critic_decision,
            ):
                break
            if self._adaptive_evidence_sufficient(state):
                break

    def _run_fixed_baseline(
        self,
        state: InvestigationState,
        ledger: EvidenceLedger,
        trace: TraceRecorder,
    ) -> None:
        """Execute the declared non-agent ablation: mandatory checks then centroid."""

        if ExperimentType.CENTROID_LOCALIZATION not in state.available_tests:
            trace.append(
                TraceEventType.FALLBACK,
                {
                    "reason_code": "FIXED_BASELINE_CENTROID_UNAVAILABLE",
                    "policy": "BLS_ODD_EVEN_SECONDARY_CENTROID",
                },
            )
            return
        candidate = state.candidates[0]
        request = ToolRequest(
            experiment_type=ExperimentType.CENTROID_LOCALIZATION,
            parameters={"candidate_id": candidate.candidate_id, "transit_window_scale": 1.0},
            adaptive=False,
            requested_by="fixed-checklist-baseline",
            justification="Declared ablation policy: BLS -> odd/even -> secondary -> centroid -> result.",
        )
        self._execute_request(state, ledger, trace, request)

    def _execute_request(
        self,
        state: InvestigationState,
        ledger: EvidenceLedger,
        trace: TraceRecorder,
        request: ToolRequest,
        *,
        agent_request_id: str | None = None,
        critic_decision: CriticDecision | None = None,
    ) -> bool:
        review = self.registry.validate(request, state)
        trace.append(
            TraceEventType.TOOL_REQUESTED,
            {
                "request": request.model_dump(mode="json"),
                "validation": review.model_dump(mode="json"),
            },
        )
        if review.status is not ReviewStatus.ALLOWED:
            failure = self.registry.as_scientific_failure(review)
            trace.append(TraceEventType.TOOL_RESULT, _compact_failure(failure))
            return False

        normalized_request = request.model_copy(update={"parameters": review.normalized_parameters})
        try:
            response = self.toolbox.execute(normalized_request, state)
        except Exception as exc:
            trace.append(
                TraceEventType.TOOL_RESULT,
                {
                    "status": "ERROR",
                    "experiment_type": request.experiment_type.value,
                    "reason_code": "UNHANDLED_TOOL_EXCEPTION",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1_000],
                },
            )
            self.registry.record_attempt(state, request, successful=False)
            self._trace_budget(state, trace)
            return False

        if isinstance(response, ScientificFailure):
            trace.append(TraceEventType.TOOL_RESULT, _compact_failure(response))
            self.registry.record_attempt(state, request, successful=False)
            self._trace_budget(state, trace)
            return False
        if not isinstance(response, ScientificResult):
            raise InvestigationRuntimeError(
                f"toolbox returned unsupported result type {type(response).__name__}"
            )

        evidence = ledger.append_result(
            response,
            agent_request_id=agent_request_id,
            critic_decision_id=(critic_decision.critic_decision_id if critic_decision else None),
        )
        self._reduce_result(state, response, evidence)
        update = self.updater.apply_to_investigation(state, evidence)
        self.registry.record_attempt(state, request, successful=True)
        trace.append(TraceEventType.TOOL_RESULT, _compact_result(response))
        trace.append(
            TraceEventType.EVIDENCE_APPENDED,
            {
                "evidence_id": evidence.id,
                "record_hash": evidence.record_hash,
                "experiment_type": evidence.experiment_type.value,
                "interpretation_code": evidence.interpretation_code.value,
                "agent_request_id": evidence.agent_request_id,
                "critic_decision_id": evidence.critic_decision_id,
            },
        )
        trace.append(
            TraceEventType.HYPOTHESIS_UPDATED,
            {
                "report": update.model_dump(mode="json"),
                "realized_evidence_state_change": len(update.updates),
            },
        )
        self._trace_budget(state, trace)
        return True

    def _reduce_result(
        self,
        state: InvestigationState,
        result: ScientificResult,
        evidence: EvidenceItem,
    ) -> None:
        if result.experiment_type is ExperimentType.QUALITY_INSPECTION:
            values = result.numerical_results
            total = int(values.get("total_cadences", 0))
            usable = int(values.get("usable_cadences", total))
            rejected = int(values.get("rejected_cadences", max(0, total - usable)))
            cadence = values.get("median_cadence_minutes")
            state.data_quality = DataQualitySummary(
                total_cadences=total,
                usable_cadences=usable,
                rejected_cadences=rejected,
                median_cadence_minutes=float(cadence) if cadence else None,
                quality_flags=result.quality_flags,
                evidence_id=evidence.id,
            )
        elif result.experiment_type in {
            ExperimentType.NORMALIZATION,
            ExperimentType.DETRENDING,
            ExperimentType.ALTERNATE_DETRENDING,
        }:
            state.preprocessing_runs = [
                *state.preprocessing_runs,
                PreprocessingRun(
                    method=str(result.parameters.get("method", result.experiment_type.value)),
                    parameters=result.parameters,
                    input_artifact_ids=[item.artifact_id for item in result.input_artifacts],
                    output_artifact_ids=[item.artifact_id for item in result.output_artifacts],
                    evidence_id=evidence.id,
                ),
            ]
        elif result.experiment_type is ExperimentType.TRANSIT_SEARCH:
            values = result.numerical_results
            required = {
                "period_days",
                "epoch_btjd",
                "transit_depth_ppm",
                "duration_hours",
                "signal_to_noise",
                "observed_events",
            }
            missing = required - values.keys()
            if result.interpretation_code is InterpretationCode.DETECTED and missing:
                raise InvestigationRuntimeError(
                    f"detected transit result lacks candidate measurements: {sorted(missing)}"
                )
            # Search tools report their measured best peak even when it does not
            # clear the declared detection gate.  Those diagnostic numbers are
            # valid evidence, but they must not be promoted into a viable
            # Candidate unless the deterministic interpretation is DETECTED.
            if result.interpretation_code is InterpretationCode.DETECTED and not missing:
                state.candidates = [
                    Candidate(
                        period_days=float(values["period_days"]),
                        epoch_btjd=float(values["epoch_btjd"]),
                        transit_depth_ppm=float(values["transit_depth_ppm"]),
                        duration_hours=float(values["duration_hours"]),
                        signal_to_noise=float(values["signal_to_noise"]),
                        observed_events=int(values["observed_events"]),
                        uncertainties={
                            key: value
                            for key, value in result.uncertainties.items()
                            if key in required
                        },
                        search_statistic=(
                            float(values["search_statistic"])
                            if "search_statistic" in values
                            else None
                        ),
                        source_evidence_ids=[evidence.id],
                        artifact_ids=[item.artifact_id for item in result.output_artifacts],
                    )
                ]
        elif (
            result.experiment_type is ExperimentType.HARMONIC_TEST
            and state.candidates
            and "preferred_period_days" in result.numerical_results
            and result.interpretation_code
            in {
                InterpretationCode.PREFERRED_HALF_PERIOD,
                InterpretationCode.PREFERRED_NOMINAL_PERIOD,
                InterpretationCode.PREFERRED_DOUBLE_PERIOD,
            }
        ):
            current = state.candidates[0]
            uncertainties = dict(current.uncertainties)
            uncertainty_mapping = {
                "preferred_period_days": "period_days",
                "preferred_epoch_btjd": "epoch_btjd",
                "preferred_duration_hours": "duration_hours",
                "preferred_primary_depth_ppm": "transit_depth_ppm",
            }
            for source_name, candidate_name in uncertainty_mapping.items():
                preferred_uncertainty = result.uncertainties.get(source_name)
                if preferred_uncertainty is not None:
                    uncertainties[candidate_name] = preferred_uncertainty
            values = result.numerical_results
            state.candidates = [
                current.model_copy(
                    update={
                        "period_days": float(values["preferred_period_days"]),
                        "epoch_btjd": float(values.get("preferred_epoch_btjd", current.epoch_btjd)),
                        "duration_hours": float(
                            values.get("preferred_duration_hours", current.duration_hours)
                        ),
                        "transit_depth_ppm": float(
                            values.get(
                                "preferred_primary_depth_ppm",
                                current.transit_depth_ppm,
                            )
                        ),
                        "signal_to_noise": float(
                            values.get(
                                "preferred_signal_to_noise",
                                current.signal_to_noise,
                            )
                        ),
                        "observed_events": int(
                            values.get("preferred_observed_events", current.observed_events)
                        ),
                        "uncertainties": uncertainties,
                        "source_evidence_ids": [
                            *current.source_evidence_ids,
                            evidence.id,
                        ],
                        "artifact_ids": list(
                            dict.fromkeys(
                                [
                                    *current.artifact_ids,
                                    *(item.artifact_id for item in result.output_artifacts),
                                ]
                            )
                        ),
                    }
                )
            ]

        existing_ids = {item.artifact_id for item in state.available_data_products}
        new_artifacts = [
            item for item in result.output_artifacts if item.artifact_id not in existing_ids
        ]
        if new_artifacts:
            state.available_data_products = [*state.available_data_products, *new_artifacts]

    def _lock(
        self,
        state: InvestigationState,
        ledger: EvidenceLedger,
        trace: TraceRecorder,
        directory: Path,
    ) -> InvestigationOutcome:
        state.final_disposition = derive_disposition(state)
        state.status = InvestigationStatus.COMPLETE
        state.transition_lock(LockState.READY_TO_LOCK)
        limitations = _collect_limitations(ledger)
        result = LockedInvestigationResult(
            opaque_target_id=state.opaque_target_id,
            trace_id=state.trace_id,
            disposition=state.final_disposition,
            candidate=state.candidates[0] if state.candidates else None,
            completed_tests=state.completed_tests,
            evidence_ids=state.evidence,
            evidence_root_hash=ledger.root_hash,
            pre_lock_trace_root_hash=trace.root_hash,
            limitations=limitations,
        )
        receipt = self.locker.lock(directory, result, trace=trace)
        state.transition_lock(LockState.RESULT_LOCKED)
        return InvestigationOutcome(
            state=state,
            ledger=ledger,
            trace=trace,
            receipt=receipt,
            reveal=None,
            run_directory=directory,
        )

    def _mandatory_parameters(
        self, experiment: ExperimentType, state: InvestigationState
    ) -> dict[str, Any]:
        candidate_id = state.candidates[0].candidate_id if state.candidates else None
        if experiment is ExperimentType.DETRENDING:
            return {"method": "median_filter", "window_hours": 24.0, "sigma_clip": 5.0}
        if experiment is ExperimentType.TRANSIT_SEARCH:
            return {
                "min_period_days": 0.5,
                "max_period_days": 15.0,
                "durations_hours": [1.0, 2.0, 3.0, 4.0, 6.0],
            }
        if candidate_id and experiment in {
            ExperimentType.PHASE_FOLD,
            ExperimentType.SIGNAL_QUALITY,
            ExperimentType.ODD_EVEN,
            ExperimentType.SECONDARY_ECLIPSE,
            ExperimentType.CONTAMINATION_SCREEN,
        }:
            return {"candidate_id": candidate_id}
        return {}

    @staticmethod
    def _request_from_skeptic(decision: SkepticDecision, state: InvestigationState) -> ToolRequest:
        assert decision.requested_experiment is not None
        request = ToolRequest(
            experiment_type=decision.requested_experiment,
            parameters=decision.parameters,
            adaptive=True,
            requested_by="skeptic-agent",
            justification=decision.explanation,
            agent_decision_id=decision.decision_id,
        )
        return ScientificDirector._bind_adaptive_parameters(request, state)

    @staticmethod
    def _bind_adaptive_parameters(request: ToolRequest, state: InvestigationState) -> ToolRequest:
        """Bind deterministic candidate identity/ephemeris after the model chooses a tool."""

        parameters = dict(request.parameters)
        candidate = state.candidates[0] if state.candidates else None
        if candidate is not None and request.experiment_type in {
            ExperimentType.HARMONIC_TEST,
            ExperimentType.CENTROID_LOCALIZATION,
            ExperimentType.ALTERNATE_DETRENDING,
        }:
            parameters["candidate_id"] = candidate.candidate_id
        if candidate is not None and request.experiment_type is ExperimentType.HARMONIC_TEST:
            parameters["base_period_days"] = candidate.period_days
            parameters["factors"] = [0.5, 1.0, 2.0]
        elif request.experiment_type is ExperimentType.CENTROID_LOCALIZATION:
            parameters.setdefault("aperture_id", None)
            parameters.setdefault("transit_window_scale", 1.0)
        elif request.experiment_type is ExperimentType.ALTERNATE_DETRENDING:
            parameters.setdefault("method", "savgol")
            parameters.setdefault("window_hours", 36.0)
        return request.model_copy(update={"parameters": parameters, "adaptive": True})

    @staticmethod
    def _safe_request_after_invalid(
        packet: dict[str, Any],
        original: SkepticDecision,
        state: InvestigationState,
    ) -> ToolRequest | None:
        fallback_runner = StructuredAgentRunner(UnavailableProvider("runtime validation fallback"))
        fallback_decision = SkepticAgent(fallback_runner).decide(packet).value
        if fallback_decision.action is SkepticAction.STOP:
            return None
        request = ScientificDirector._request_from_skeptic(fallback_decision, state)
        return request.model_copy(
            update={
                "justification": (
                    f"Agent request {original.decision_id} failed validation; "
                    + (request.justification or "declared safe fallback")
                )
            }
        )

    @staticmethod
    def _decision_source(source: DecisionSource, guard_report: dict[str, Any]) -> DecisionSource:
        if source is DecisionSource.DETERMINISTIC_FALLBACK:
            return source
        if bool(guard_report.get("changed")):
            return DecisionSource.REPAIRED_LIVE_MODEL
        return source

    @staticmethod
    def _strong_rejection_evidence(state: InvestigationState) -> bool:
        return derive_disposition(state) is ScientificDisposition.PLANETARY_INTERPRETATION_WEAK

    @staticmethod
    def _adaptive_evidence_sufficient(state: InvestigationState) -> bool:
        if ExperimentType.HARMONIC_TEST in state.completed_tests:
            return ScientificDirector._strong_rejection_evidence(state)
        if ExperimentType.CENTROID_LOCALIZATION in state.completed_tests:
            return derive_disposition(state) in {
                ScientificDisposition.PLANETARY_INTERPRETATION_WEAK,
                ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING,
            }
        return False

    @staticmethod
    def _trace_budget(state: InvestigationState, trace: TraceRecorder) -> None:
        trace.append(
            TraceEventType.BUDGET_UPDATED,
            {
                "experiment_budget": {
                    "used": state.experiment_budget.used,
                    "remaining": state.experiment_budget.remaining,
                    "limit": state.experiment_budget.limit,
                },
                "agent_turn_budget": {
                    "used": state.agent_turn_budget.used,
                    "remaining": state.agent_turn_budget.remaining,
                    "limit": state.agent_turn_budget.limit,
                },
            },
        )

    @staticmethod
    def _agent_trace_adapter(trace: TraceRecorder):
        mapping = {
            "agent_request": TraceEventType.AGENT_REQUEST,
            "agent_response": TraceEventType.AGENT_RESPONSE,
            "structured_output_failure": TraceEventType.STRUCTURED_OUTPUT_FAILURE,
            "structured_output_repaired": TraceEventType.STRUCTURED_OUTPUT_REPAIRED,
            "agent_context_rejected": TraceEventType.CONTEXT_REJECTED,
            "agent_fallback": TraceEventType.FALLBACK,
        }

        def record(kind: str, payload: dict[str, Any]) -> None:
            event_type = mapping.get(kind, TraceEventType.FALLBACK)
            safe_payload = {"source_event": kind, **payload}
            trace.append(event_type, safe_payload)

        return record

    @staticmethod
    def _assert_new_run_directory(directory: Path) -> None:
        protected = [
            directory / "result.json",
            directory / "result.json.sha256",
            directory / "reveal.json",
            directory / "evidence.jsonl",
            directory / "trace.jsonl",
        ]
        existing = [path.name for path in protected if path.exists()]
        artifacts_directory = directory / "artifacts"
        if artifacts_directory.is_dir() and any(
            path.is_file() for path in artifacts_directory.rglob("*")
        ):
            existing.append("artifacts/")
        if existing:
            raise InvestigationRuntimeError(
                "refusing to overwrite an existing investigation: " + ", ".join(existing)
            )


def _compact_result(result: ScientificResult) -> dict[str, Any]:
    """Trace deterministic output without leaking source paths or target identities."""

    return {
        "status": result.status.value,
        "experiment_type": result.experiment_type.value,
        "tool_name": result.tool_name,
        "tool_version": result.tool_version,
        "parameters": result.parameters,
        "numerical_results": result.numerical_results,
        "result_units": result.result_units,
        "uncertainties": {
            key: value.model_dump(mode="json") for key, value in result.uncertainties.items()
        },
        "quality_flags": [item.model_dump(mode="json") for item in result.quality_flags],
        "interpretation_code": result.interpretation_code.value,
        "limitations": result.limitations,
        "input_artifact_ids": [item.artifact_id for item in result.input_artifacts],
        "output_artifact_ids": [item.artifact_id for item in result.output_artifacts],
    }


def _compact_failure(result: ScientificFailure) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "experiment_type": result.experiment_type.value,
        "tool_name": result.tool_name,
        "tool_version": result.tool_version,
        "parameters": result.parameters,
        "reason": result.reason,
        "reason_code": result.reason_code,
        "suggested_alternatives": [item.value for item in result.suggested_alternatives],
        "quality_flags": [item.model_dump(mode="json") for item in result.quality_flags],
        "interpretation_code": result.interpretation_code.value,
        "limitations": result.limitations,
        "input_artifact_ids": [item.artifact_id for item in result.input_artifacts],
    }


def _collect_limitations(ledger: EvidenceLedger) -> list[str]:
    mandatory = [
        "Photometric and centroid vetting do not constitute professional planet confirmation.",
        "The implemented diagnostics cannot exclude every blended or astrophysical false-positive scenario.",
    ]
    observed = [limitation for item in ledger.items for limitation in item.limitations]
    return list(dict.fromkeys([*mandatory, *observed]))


__all__ = [
    "InvestigationOutcome",
    "InvestigationRuntimeError",
    "ScientificDirector",
    "ScientificToolbox",
]
