from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from exoswarm.agents.provider import (
    InferenceResponse,
    InferenceUsage,
    UnavailableProvider,
)
from exoswarm.agents.structured import StructuredCall
from exoswarm.config import Settings
from exoswarm.domain.models import (
    ArtifactRef,
    CatalogMeasurement,
    CriticDecision,
    CriticVerdict,
    ExperimentType,
    GroundTruthRecord,
    InterpretationCode,
    InvestigationState,
    LockState,
    MeasurementUncertainty,
    ProvenanceRecord,
    ScientificDisposition,
    ScientificResult,
    ScientificStatus,
    ToolRequest,
)
from exoswarm.domain.trace import TraceEventType, TraceIntegrityError
from exoswarm.evaluation.ablation import compare_policy_runs
from exoswarm.evaluation.graders import (
    EvaluationExpectation,
    evaluate_run,
    evaluate_trajectory_diversity,
)
from exoswarm.runtime.director import InvestigationRuntimeError, ScientificDirector
from exoswarm.security.blindness import GroundTruthAccessDenied, OpaqueTargetVault


class FakeScienceToolbox:
    """Schema-valid deterministic fixture; it is not used for showcased science."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ExperimentType]] = []

    def execute(self, request: ToolRequest, state: InvestigationState) -> ScientificResult:
        target = state.opaque_target_id
        experiment = request.experiment_type
        self.calls.append((target, experiment))
        interpretation = InterpretationCode.PROCESSED
        numbers: dict[str, int | float] = {}
        units: dict[str, str] = {}
        uncertainties: dict[str, MeasurementUncertainty] = {}

        if experiment is ExperimentType.LOAD_CACHED_DATA:
            interpretation = InterpretationCode.LOADED
        elif experiment is ExperimentType.QUALITY_INSPECTION:
            interpretation = InterpretationCode.ACCEPTABLE
            numbers = {
                "total_cadences": 10_000,
                "usable_cadences": 9_500,
                "rejected_cadences": 500,
                "median_cadence_minutes": 2.0,
            }
        elif experiment is ExperimentType.TRANSIT_SEARCH:
            interpretation = InterpretationCode.DETECTED
            period = 3.25 if target == "TARGET-X17" else 2.0
            numbers = {
                "period_days": period,
                "epoch_btjd": 1450.5,
                "transit_depth_ppm": 1_200 if target == "TARGET-X17" else 80_000,
                "duration_hours": 2.2,
                "signal_to_noise": 12.0 if target == "TARGET-X17" else 35.0,
                "observed_events": 6,
                "search_statistic": 15.0,
            }
            units = {
                "period_days": "d",
                "epoch_btjd": "BTJD",
                "transit_depth_ppm": "ppm",
                "duration_hours": "h",
                "signal_to_noise": "dimensionless",
                "observed_events": "count",
                "search_statistic": "dimensionless",
            }
            uncertainties = {
                "period_days": MeasurementUncertainty(
                    value=0.01, unit="d", method="grid resolution", kind="resolution"
                ),
                "epoch_btjd": MeasurementUncertainty(
                    value=0.02, unit="BTJD", method="cadence tolerance", kind="tolerance"
                ),
                "transit_depth_ppm": MeasurementUncertainty(
                    value=100, unit="ppm", method="out-of-transit scatter"
                ),
                "duration_hours": MeasurementUncertainty(
                    value=0.2, unit="h", method="duration grid", kind="resolution"
                ),
            }
        elif experiment is ExperimentType.SIGNAL_QUALITY:
            interpretation = InterpretationCode.PASS
        elif experiment is ExperimentType.ODD_EVEN:
            interpretation = (
                InterpretationCode.CONSISTENT
                if target == "TARGET-X17"
                else InterpretationCode.INCONSISTENT
            )
        elif experiment is ExperimentType.SECONDARY_ECLIPSE:
            interpretation = (
                InterpretationCode.NOT_SIGNIFICANT
                if target == "TARGET-X17"
                else InterpretationCode.SIGNIFICANT
            )
        elif experiment is ExperimentType.CONTAMINATION_SCREEN:
            interpretation = (
                InterpretationCode.NEIGHBOR_DETECTED
                if target == "TARGET-X17"
                else InterpretationCode.NO_NEARBY_SOURCE
            )
        elif experiment is ExperimentType.CENTROID_LOCALIZATION:
            interpretation = InterpretationCode.TARGET_CONSISTENT
            numbers = {"centroid_offset_pixels": 0.02, "offset_significance_sigma": 0.3}
            units = {
                "centroid_offset_pixels": "pixel",
                "offset_significance_sigma": "sigma",
            }
        elif experiment is ExperimentType.HARMONIC_TEST:
            interpretation = InterpretationCode.PREFERRED_DOUBLE_PERIOD
            numbers = {
                "half_period_score": 2.0,
                "nominal_period_score": 5.0,
                "double_period_score": 15.0,
                "preferred_period_days": 4.0,
                "preferred_observed_events": 3,
            }
        return ScientificResult(
            status=ScientificStatus.SUCCESS,
            experiment_type=experiment,
            tool_name=f"fixture_{experiment.value}",
            tool_version="test-only",
            parameters=request.parameters,
            numerical_results=numbers,
            result_units=units,
            uncertainties=uncertainties,
            interpretation_code=interpretation,
            limitations=["Deterministic unit-test fixture; not astronomical evidence."],
            provenance=[ProvenanceRecord(source="unit test")],
        )


class NoSignalScienceToolbox(FakeScienceToolbox):
    """Return a measured BLS peak that fails the deterministic detection gate."""

    def execute(self, request: ToolRequest, state: InvestigationState) -> ScientificResult:
        result = super().execute(request, state)
        if request.experiment_type is ExperimentType.TRANSIT_SEARCH:
            return result.model_copy(
                update={"interpretation_code": InterpretationCode.NOT_DETECTED}
            )
        return result


class AlternateDetrendingProvider:
    """Scripted valid model output proving inference can change the runtime branch."""

    def __init__(self) -> None:
        self.roles: list[str] = []
        self.skeptic_calls = 0

    @property
    def name(self) -> str:
        return "scripted-test-provider"

    @property
    def model(self) -> str:
        return "scripted-test-model"

    def complete(self, *, system: str, user: str) -> InferenceResponse:
        packet = json.loads(user)
        if "ROLE: SKEPTIC" in system:
            self.roles.append("SKEPTIC")
            self.skeptic_calls += 1
            if self.skeptic_calls == 1:
                candidate_id = packet["current_candidate"]["candidate_id"]
                payload = {
                    "action": "REQUEST_EXPERIMENT",
                    "hypothesis_under_test": "H4_STELLAR_VARIABILITY",
                    "requested_experiment": "alternate_detrending",
                    "parameters": {
                        "candidate_id": candidate_id,
                        "method": "savgol",
                        "window_hours": 36.0,
                    },
                    "reason_code": "TEST_PREPROCESSING_SENSITIVITY",
                    "explanation": "Challenge whether preprocessing determines the signal.",
                    "expected_discriminating_result": (
                        "A stable or sensitive deterministic re-search distinguishes the alternatives."
                    ),
                    "predicted_outcomes": {
                        "ROBUST": "The signal persists under the permitted alternate treatment.",
                        "PREPROCESSING_SENSITIVE": (
                            "The signal changes materially under the alternate treatment."
                        ),
                    },
                    "expected_information_value": 0.8,
                    "priority": "HIGH",
                }
            else:
                payload = {
                    "action": "STOP",
                    "reason_code": "NO_FURTHER_DISCRIMINATING_TEST",
                    "explanation": "The requested robustness challenge is complete.",
                    "expected_discriminating_result": (
                        "No additional deterministic result is requested."
                    ),
                    "expected_information_value": 0.0,
                    "priority": "LOW",
                }
        else:
            self.roles.append("CRITIC")
            payload = {
                "reviewed_request_id": packet["proposal"]["request_id"],
                "verdict": "APPROVE",
                "reason_code": "APPROVE_DISCRIMINATING",
                "reason": "The proposed robustness test is permitted, unused, and discriminating.",
            }
        return InferenceResponse(
            content=json.dumps(payload),
            provider=self.name,
            model=self.model,
            request_id=f"provider-request-{len(self.roles)}",
            finish_reason="stop",
            usage=InferenceUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )


def _settings(tmp_path: Path, *, turns: int = 4) -> Settings:
    return Settings(
        provider="offline",
        model="none",
        api_base="https://invalid.example/v1",
        api_key=None,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        max_agent_turns=turns,
        experiment_budget=14,
        request_timeout_seconds=1,
    )


def _vault() -> OpaqueTargetVault:
    vault = OpaqueTargetVault()
    artifacts = [
        ArtifactRef(artifact_id="cached-lc", path="lightcurve.fits", role="light_curve"),
        ArtifactRef(artifact_id="cached-tpf", path="pixels.fits", role="target_pixel"),
    ]
    for opaque, identity, period, status in [
        ("TARGET-X17", "Private Planet Target", 3.24, "CONFIRMED PLANET"),
        ("TARGET-X42", "Private EB Target", 4.0, "ECLIPSING BINARY"),
    ]:
        vault.register(
            opaque_target_id=opaque,
            real_target_identity=identity,
            artifacts=artifacts,
            ground_truth=GroundTruthRecord(
                actual_target_identity=identity,
                catalog_name="test catalog",
                catalog_status=status,
                measurements={"period": CatalogMeasurement(value=period, unit="d")},
                provenance=[ProvenanceRecord(source="unit test")],
            ),
        )
    return vault


def test_planet_path_runs_centroid_then_locks_and_reveals(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )
    with pytest.raises(GroundTruthAccessDenied):
        director.gate.lookup("TARGET-X17")

    outcome = director.investigate("TARGET-X17", reveal=True)

    experiments = [item for target, item in toolbox.calls if target == "TARGET-X17"]
    assert ExperimentType.CENTROID_LOCALIZATION in experiments
    assert ExperimentType.HARMONIC_TEST not in experiments
    assert (
        outcome.state.final_disposition
        is ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING
    )
    assert outcome.state.lock_state is LockState.GROUND_TRUTH_REVEALED
    assert outcome.reveal is not None
    assert Path(outcome.receipt.result_path).exists()
    assert Path(outcome.receipt.hash_path).exists()
    assert (outcome.run_directory / "reveal.json").exists()
    assert director.locker.verify_artifact_order(outcome.run_directory)

    event_types = [event.event_type for event in outcome.trace.events]
    assert event_types.index(TraceEventType.RESULT_LOCKED) < event_types.index(
        TraceEventType.CATALOG_ACCESS_ENABLED
    )
    assert event_types.index(TraceEventType.CATALOG_ACCESS_ENABLED) < event_types.index(
        TraceEventType.GROUND_TRUTH_REVEALED
    )
    lock_event = next(
        event for event in outcome.trace.events if event.event_type is TraceEventType.RESULT_LOCKED
    )
    locked_result = director.locker.verify_trace_commitment(
        outcome.receipt,
        outcome.trace,
    )
    assert lock_event.previous_hash == locked_result.pre_lock_trace_root_hash


def test_negative_control_runs_harmonic_and_gets_weak_disposition(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )
    outcome = director.investigate("TARGET-X42")

    experiments = [item for target, item in toolbox.calls if target == "TARGET-X42"]
    assert ExperimentType.HARMONIC_TEST in experiments
    assert ExperimentType.CENTROID_LOCALIZATION not in experiments
    assert outcome.state.final_disposition is ScientificDisposition.PLANETARY_INTERPRETATION_WEAK
    assert outcome.state.candidates[0].period_days == 4.0
    assert len(outcome.state.candidates[0].source_evidence_ids) == 2
    assert outcome.state.lock_state is LockState.RESULT_LOCKED
    assert not (outcome.run_directory / "reveal.json").exists()


def test_valid_model_decision_changes_the_adaptive_trajectory(tmp_path: Path) -> None:
    provider = AlternateDetrendingProvider()
    toolbox = FakeScienceToolbox()
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=provider,
    ).investigate("TARGET-X17")

    experiments = [item for target, item in toolbox.calls if target == "TARGET-X17"]
    assert ExperimentType.ALTERNATE_DETRENDING in experiments
    assert ExperimentType.CENTROID_LOCALIZATION not in experiments
    assert ExperimentType.HARMONIC_TEST not in experiments
    assert provider.roles == ["SKEPTIC", "CRITIC", "SKEPTIC"]
    assert any(
        event.event_type is TraceEventType.AGENT_RESPONSE
        and event.payload.get("provider") == provider.name
        for event in outcome.trace.events
    )
    decisions = [
        event
        for event in outcome.trace.events
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}
    ]
    assert decisions
    assert all(event.payload.get("decision_source") == "LIVE_MODEL" for event in decisions)
    assert all(event.payload.get("provider") == provider.name for event in decisions)
    assert all(event.payload.get("model") == provider.model for event in decisions)


def test_mandatory_baseline_is_code_enforced(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")
    mandatory = {
        ExperimentType.SIGNAL_QUALITY,
        ExperimentType.ODD_EVEN,
        ExperimentType.SECONDARY_ECLIPSE,
        ExperimentType.CONTAMINATION_SCREEN,
    }
    assert mandatory <= set(outcome.state.completed_tests)


def test_subthreshold_bls_peak_is_not_promoted_to_candidate(tmp_path: Path) -> None:
    toolbox = NoSignalScienceToolbox()
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")

    assert not outcome.state.candidates
    assert outcome.state.final_disposition is ScientificDisposition.NO_CREDIBLE_PERIODIC_SIGNAL
    assert ExperimentType.TRANSIT_SEARCH in outcome.state.completed_tests
    assert ExperimentType.PHASE_FOLD not in outcome.state.completed_tests


def test_max_agent_turn_budget_is_respected(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    outcome = ScientificDirector(
        settings=_settings(tmp_path, turns=1),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")
    assert outcome.state.agent_turn_budget.used == 1
    assert outcome.state.agent_turn_budget.remaining == 0
    assert ExperimentType.CENTROID_LOCALIZATION not in outcome.state.completed_tests


def test_mismatched_critic_review_never_executes_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched_review(_critic, _packet, _request):
        return StructuredCall(
            value=CriticDecision(
                reviewed_request_id="REQ-UNRELATED",
                verdict=CriticVerdict.APPROVE,
                reason_code="APPROVE_DISCRIMINATING",
                reason="The proposed spatial test is permitted and non-redundant.",
            ),
            used_fallback=False,
            repaired=False,
            attempts=1,
            provider_request_ids=(),
        )

    monkeypatch.setattr(
        "exoswarm.runtime.director.CriticAgent.review",
        mismatched_review,
    )
    toolbox = FakeScienceToolbox()
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")

    assert ExperimentType.CENTROID_LOCALIZATION not in outcome.state.completed_tests
    mismatch = [
        event
        for event in outcome.trace.events
        if event.event_type is TraceEventType.FALLBACK
        and event.payload.get("reason_code") == "CRITIC_REVIEW_TARGET_MISMATCH"
    ]
    assert len(mismatch) == 1


def test_existing_locked_run_is_never_overwritten(tmp_path: Path) -> None:
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )
    director.investigate("TARGET-X17")
    with pytest.raises(InvestigationRuntimeError, match="refusing to overwrite"):
        director.investigate("TARGET-X17")


def test_preexisting_science_artifacts_are_never_reused(tmp_path: Path) -> None:
    run_directory = tmp_path / "artifact-only-run"
    artifact_directory = run_directory / "artifacts" / "science"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "stale.npz").write_bytes(b"stale")
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )

    with pytest.raises(InvestigationRuntimeError, match="artifacts/"):
        director.investigate("TARGET-X17", run_directory=run_directory)


def test_locked_result_contains_only_opaque_identity(tmp_path: Path) -> None:
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")
    payload = json.loads(Path(outcome.receipt.result_path).read_text())
    serialized = json.dumps(payload).lower()
    assert payload["opaque_target_id"] == "TARGET-X17"
    assert "private planet target" not in serialized
    assert "ground_truth" not in serialized
    assert "catalog_status" not in serialized


def test_constraint_evaluators_and_cross_case_diversity(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )
    planet = director.investigate("TARGET-X17", run_directory=tmp_path / "planet", reveal=True)
    negative = director.investigate("TARGET-X42", run_directory=tmp_path / "negative", reveal=False)
    planet_report = evaluate_run(
        planet.run_directory,
        EvaluationExpectation(
            opaque_target_id="TARGET-X17",
            expected_period_days=3.25,
            period_tolerance_days=0.001,
            accepted_dispositions={ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING},
            expected_adaptive_any_of={ExperimentType.CENTROID_LOCALIZATION},
            forbidden_adaptive={ExperimentType.HARMONIC_TEST},
            require_reveal=True,
        ),
    )
    negative_report = evaluate_run(
        negative.run_directory,
        EvaluationExpectation(
            opaque_target_id="TARGET-X42",
            expected_period_days=4.0,
            period_tolerance_days=0.001,
            accepted_dispositions={ScientificDisposition.PLANETARY_INTERPRETATION_WEAK},
            expected_adaptive_any_of={ExperimentType.HARMONIC_TEST},
            forbidden_adaptive={ExperimentType.CENTROID_LOCALIZATION},
            negative_control=True,
        ),
    )
    assert planet_report.passed, [grade for grade in planet_report.grades if not grade.passed]
    assert negative_report.passed, [grade for grade in negative_report.grades if not grade.passed]
    assert next(
        grade
        for grade in negative_report.grades
        if grade.name == "candidate_measurements_trace_to_evidence"
    ).passed
    assert next(
        grade
        for grade in planet_report.grades
        if grade.name == "post_lock_reveal_valid_and_ordered"
    ).passed
    assert evaluate_trajectory_diversity([planet_report, negative_report]).passed


def test_fixed_ablation_executes_declared_non_agent_checklist(tmp_path: Path) -> None:
    toolbox = FakeScienceToolbox()
    director = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=toolbox,
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    )
    outcome = director.investigate("TARGET-X42", run_directory=tmp_path / "fixed", policy="fixed")
    adaptive = director.investigate(
        "TARGET-X42", run_directory=tmp_path / "adaptive", policy="adaptive"
    )
    assert ExperimentType.CENTROID_LOCALIZATION in outcome.state.completed_tests
    assert ExperimentType.HARMONIC_TEST not in outcome.state.completed_tests
    assert outcome.state.agent_turn_budget.used == 0
    initial = next(
        event
        for event in outcome.trace.events
        if event.event_type is TraceEventType.INVESTIGATION_INITIALIZED
    )
    assert initial.payload["investigation_policy"] == "fixed"
    comparison = compare_policy_runs(
        adaptive.run_directory,
        outcome.run_directory,
        expected_best_action=ExperimentType.HARMONIC_TEST,
    )
    assert comparison.adaptive.expected_action_selected is True
    assert comparison.fixed.expected_action_selected is False
    assert comparison.adaptive.experiments_executed == comparison.fixed.experiments_executed

    trace_path = adaptive.run_directory / "trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(lines[0])
    first_event["payload"]["investigation_policy"] = "fixed"
    lines[0] = json.dumps(first_event)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="trace hash mismatch"):
        compare_policy_runs(adaptive.run_directory, outcome.run_directory)


def test_evaluator_detects_candidate_number_not_present_in_evidence(tmp_path: Path) -> None:
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X42")
    result_path = Path(outcome.receipt.result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["candidate"]["period_days"] = 9.99
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    Path(outcome.receipt.hash_path).write_text(f"{digest}\n", encoding="ascii")

    report = evaluate_run(
        outcome.run_directory,
        EvaluationExpectation(opaque_target_id="TARGET-X42"),
    )
    grades = {grade.name: grade for grade in report.grades}
    assert grades["result_hash_valid"].passed
    assert grades["locked_result_matches_evidence_ledger"].passed
    assert not grades["locked_result_commits_to_verified_trace"].passed
    assert not grades["candidate_measurements_trace_to_evidence"].passed


def test_evaluator_cross_checks_locked_evidence_ids_and_root(tmp_path: Path) -> None:
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")
    result_path = Path(outcome.receipt.result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["evidence_ids"] = payload["evidence_ids"][:-1]
    payload["evidence_root_hash"] = "f" * 64
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    Path(outcome.receipt.hash_path).write_text(f"{digest}\n", encoding="ascii")

    report = evaluate_run(
        outcome.run_directory,
        EvaluationExpectation(opaque_target_id="TARGET-X17"),
    )
    grades = {grade.name: grade for grade in report.grades}
    assert grades["result_hash_valid"].passed
    assert not grades["locked_result_commits_to_verified_trace"].passed
    assert not grades["locked_result_matches_evidence_ledger"].passed


def test_evaluator_rejects_tampered_trace_hash(tmp_path: Path) -> None:
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17")
    trace_path = outcome.run_directory / "trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(lines[0])
    first_event["payload"]["investigation_policy"] = "fixed"
    lines[0] = json.dumps(first_event)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TraceIntegrityError, match="trace hash mismatch"):
        evaluate_run(
            outcome.run_directory,
            EvaluationExpectation(opaque_target_id="TARGET-X17"),
        )


def test_evaluator_validates_reveal_schema_hash_and_order(tmp_path: Path) -> None:
    outcome = ScientificDirector(
        settings=_settings(tmp_path),
        toolbox=FakeScienceToolbox(),
        vault=_vault(),
        provider=UnavailableProvider("unit test"),
    ).investigate("TARGET-X17", reveal=True)
    reveal_path = outcome.run_directory / "reveal.json"
    payload = json.loads(reveal_path.read_text(encoding="utf-8"))
    payload["locked_result_sha256"] = "f" * 64
    reveal_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = evaluate_run(
        outcome.run_directory,
        EvaluationExpectation(
            opaque_target_id="TARGET-X17",
            require_reveal=True,
        ),
    )
    grade = next(
        item for item in report.grades if item.name == "post_lock_reveal_valid_and_ordered"
    )
    assert not grade.passed
