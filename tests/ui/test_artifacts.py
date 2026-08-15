from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from exoswarm.domain.ledger import EvidenceLedger, canonical_json_bytes
from exoswarm.domain.models import (
    ArtifactRef,
    Candidate,
    CatalogMeasurement,
    ExperimentType,
    GroundTruthRecord,
    Hypothesis,
    InterpretationCode,
    LockedInvestigationResult,
    MeasurementUncertainty,
    ProvenanceRecord,
    ScientificDisposition,
    ScientificResult,
    ScientificStatus,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder
from exoswarm.security.locking import ResultLocker
from exoswarm.ui.artifacts import RunPhase, load_run, load_science_product
from exoswarm.ui.viewmodels import (
    candidate_measurements,
    hypothesis_views,
    latest_critic_decision,
    latest_skeptic_decision,
)


def _locked_run(
    directory: Path,
    *,
    reveal: bool = False,
    harmonic: bool = False,
    agent_fallback: bool = False,
) -> tuple[Path, str]:
    target = "TARGET-X17"
    directory.mkdir(parents=True)
    ledger = EvidenceLedger(directory / "evidence.jsonl")
    result = ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.TRANSIT_SEARCH,
        tool_name="deterministic_bls",
        tool_version="test",
        numerical_results={
            "period_days": 3.1415,
            "epoch_btjd": 2042.25,
            "transit_depth_ppm": 1250.0,
            "duration_hours": 2.5,
            "signal_to_noise": 12.0,
            "observed_events": 7,
        },
        result_units={
            "period_days": "d",
            "epoch_btjd": "BTJD",
            "transit_depth_ppm": "ppm",
            "duration_hours": "h",
            "signal_to_noise": "dimensionless",
            "observed_events": "count",
        },
        uncertainties={
            "period_days": MeasurementUncertainty(
                value=0.001,
                unit="d",
                method="BLS grid resolution",
                kind="resolution",
            )
        },
        interpretation_code=InterpretationCode.DETECTED,
        limitations=["Test-only deterministic fixture."],
        output_artifacts=[
            ArtifactRef(
                artifact_id="TARGET-X17:bls_periodogram_data:npz",
                path="artifacts/science/bls_periodogram.npz",
                role="bls_periodogram_data",
            )
        ],
        provenance=[ProvenanceRecord(source="test fixture")],
    )
    evidence = ledger.append_result(result)
    candidate_values: dict[str, int | float] = {
        "period_days": 3.1415,
        "epoch_btjd": 2042.25,
        "transit_depth_ppm": 1250.0,
        "duration_hours": 2.5,
        "signal_to_noise": 12.0,
        "observed_events": 7,
    }
    source_evidence_ids = [evidence.id]
    completed_tests = [ExperimentType.TRANSIT_SEARCH]
    if harmonic:
        harmonic_evidence = ledger.append_result(
            ScientificResult(
                status=ScientificStatus.SUCCESS,
                experiment_type=ExperimentType.HARMONIC_TEST,
                tool_name="deterministic_harmonic_test",
                tool_version="test",
                numerical_results={
                    "preferred_period_days": 6.283,
                    "preferred_epoch_btjd": 2043.0,
                    "preferred_primary_depth_ppm": 2500.0,
                    "preferred_duration_hours": 4.0,
                    "preferred_signal_to_noise": 18.0,
                    "preferred_observed_events": 4,
                },
                result_units={
                    "preferred_period_days": "d",
                    "preferred_epoch_btjd": "BTJD",
                    "preferred_primary_depth_ppm": "ppm",
                    "preferred_duration_hours": "h",
                    "preferred_signal_to_noise": "dimensionless",
                    "preferred_observed_events": "count",
                },
                interpretation_code=InterpretationCode.PREFERRED_DOUBLE_PERIOD,
            )
        )
        candidate_values = {
            "period_days": 6.283,
            "epoch_btjd": 2043.0,
            "transit_depth_ppm": 2500.0,
            "duration_hours": 4.0,
            "signal_to_noise": 18.0,
            "observed_events": 4,
        }
        source_evidence_ids.append(harmonic_evidence.id)
        completed_tests.append(ExperimentType.HARMONIC_TEST)
    trace = TraceRecorder(
        trace_id="TRACE-UI-TEST",
        opaque_target_id=target,
        path=directory / "trace.jsonl",
    )
    trace.append(
        TraceEventType.INVESTIGATION_INITIALIZED,
        {"ground_truth_available": False},
    )
    trace.append(
        TraceEventType.HYPOTHESIS_UPDATED,
        {
            "report": {
                "evidence_id": evidence.id,
                "rule_key": "transit_search:DETECTED",
                "applied": True,
                "updates": [
                    {
                        "hypothesis": Hypothesis.PLANETARY_TRANSIT.value,
                        "delta": 0.75,
                        "previous_weight": 0.0,
                        "updated_weight": 0.75,
                        "previous_state": "UNRESOLVED",
                        "updated_state": "SUPPORTED",
                        "evidence_id": evidence.id,
                    }
                ],
            }
        },
    )
    trace.append(
        TraceEventType.AGENT_DECISION,
        {
            "decision": {
                "decision_id": "SK-UI",
                "action": "REQUEST_EXPERIMENT",
                "hypothesis_under_test": Hypothesis.PERIOD_ALIAS_HARMONIC.value,
                "requested_experiment": ExperimentType.HARMONIC_TEST.value,
                "reason_code": "TEST_ALIAS",
                "explanation": "An unsupported 9.99 hour claim must not reach the UI.",
                "expected_discriminating_result": "Compare the recorded aliases.",
                "predicted_outcomes": {},
                "expected_information_value": 0.8,
                "priority": "HIGH",
            },
            "decision_source": ("DETERMINISTIC_FALLBACK" if agent_fallback else "LIVE_MODEL"),
            "provider": "unavailable" if agent_fallback else "featherless",
            "model": "none" if agent_fallback else "test-live-model",
            "provider_request_ids": [] if agent_fallback else ["provider-skeptic-1"],
            "attempts": 2 if agent_fallback else 1,
            "repaired": False,
            "used_fallback": agent_fallback,
        },
    )
    trace.append(
        TraceEventType.CRITIC_DECISION,
        {
            "decision": {
                "critic_decision_id": "CR-UI",
                "reviewed_request_id": "REQ-UI",
                "verdict": "APPROVE",
                "reason_code": "DISCRIMINATING_AND_NONREDUNDANT",
                "reason": "The requested experiment targets unresolved recorded evidence.",
                "revised_request": None,
            },
            "decision_source": (
                "DETERMINISTIC_FALLBACK" if agent_fallback else "REPAIRED_LIVE_MODEL"
            ),
            "provider": "unavailable" if agent_fallback else "featherless",
            "model": "none" if agent_fallback else "test-live-model",
            "provider_request_ids": (
                [] if agent_fallback else ["provider-critic-1", "provider-critic-2"]
            ),
            "attempts": 2,
            "repaired": not agent_fallback,
            "used_fallback": agent_fallback,
        },
    )
    candidate = Candidate(
        period_days=float(candidate_values["period_days"]),
        epoch_btjd=float(candidate_values["epoch_btjd"]),
        transit_depth_ppm=float(candidate_values["transit_depth_ppm"]),
        duration_hours=float(candidate_values["duration_hours"]),
        signal_to_noise=float(candidate_values["signal_to_noise"]),
        observed_events=int(candidate_values["observed_events"]),
        uncertainties=result.uncertainties,
        source_evidence_ids=source_evidence_ids,
    )
    locked = LockedInvestigationResult(
        opaque_target_id=target,
        trace_id=trace.trace_id,
        disposition=ScientificDisposition.TRANSIT_LIKE_SIGNAL,
        candidate=candidate,
        completed_tests=completed_tests,
        evidence_ids=[item.id for item in ledger.items],
        evidence_root_hash=ledger.root_hash,
        pre_lock_trace_root_hash=trace.root_hash,
        limitations=["Test-only deterministic fixture."],
    )
    locker = ResultLocker()
    receipt = locker.lock(directory, locked, trace=trace)
    if reveal:
        trace.append(
            TraceEventType.CATALOG_ACCESS_ENABLED,
            {
                "locked_result_sha256": receipt.sha256,
                "pre_lock_trace_root_hash": locked.pre_lock_trace_root_hash,
            },
        )
        locker.write_reveal(
            receipt,
            ground_truth=GroundTruthRecord(
                actual_target_identity="Post-lock identity",
                catalog_name="External test catalog",
                catalog_status="TEST STATUS",
                measurements={"period": CatalogMeasurement(value=3.14, unit="d")},
                provenance=[ProvenanceRecord(source="test catalog")],
            ),
            trace=trace,
        )
    return directory, evidence.id


def test_missing_run_is_graceful_empty_state(tmp_path: Path) -> None:
    run = load_run(tmp_path / "TARGET-X17", opaque_target_id="TARGET-X17")
    assert run.phase is RunPhase.EMPTY
    assert run.result is None
    assert not run.ground_truth_visible


def test_active_run_exposes_candidate_metrics_from_ledger(tmp_path: Path) -> None:
    directory = tmp_path / "TARGET-X17"
    directory.mkdir()
    ledger = EvidenceLedger(directory / "evidence.jsonl")
    evidence = ledger.append_result(
        ScientificResult(
            status=ScientificStatus.SUCCESS,
            experiment_type=ExperimentType.TRANSIT_SEARCH,
            tool_name="deterministic_bls",
            tool_version="test",
            numerical_results={
                "period_days": 3.1415,
                "epoch_btjd": 2042.25,
                "transit_depth_ppm": 1250.0,
                "duration_hours": 2.5,
                "signal_to_noise": 12.0,
                "observed_events": 7,
            },
            result_units={
                "period_days": "d",
                "epoch_btjd": "BTJD",
                "transit_depth_ppm": "ppm",
                "duration_hours": "h",
                "signal_to_noise": "dimensionless",
                "observed_events": "count",
            },
            interpretation_code=InterpretationCode.DETECTED,
        )
    )
    trace = TraceRecorder(
        trace_id="TRACE-ACTIVE-UI",
        opaque_target_id="TARGET-X17",
        path=directory / "trace.jsonl",
    )
    trace.append(
        TraceEventType.INVESTIGATION_INITIALIZED,
        {"ground_truth_available": False},
    )

    run = load_run(directory)
    measurements = candidate_measurements(run)

    assert run.phase is RunPhase.ACTIVE
    assert len(measurements) == 6
    assert {source for item in measurements for source in item.source_ids} == {evidence.id}


def test_locked_run_loads_candidate_with_numeric_sources(tmp_path: Path) -> None:
    directory, evidence_id = _locked_run(tmp_path / "TARGET-X17")
    run = load_run(directory)

    assert run.phase is RunPhase.RESULT_LOCKED
    assert run.lock_verified
    assert run.reveal is None
    measurements = candidate_measurements(run)
    assert len(measurements) == 6
    period = next(item for item in measurements if item.key == "period_days")
    assert period.source_ids == ("result.json", evidence_id)
    assert all(item.source_ids for item in measurements)
    assert "Post-lock identity" not in repr(run)


def test_harmonic_resolved_candidate_links_preferred_measurements(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", harmonic=True)
    run = load_run(directory)
    harmonic_evidence = next(
        item for item in run.evidence if item.experiment_type is ExperimentType.HARMONIC_TEST
    )

    measurements = candidate_measurements(run)

    assert len(measurements) == 6
    assert all(harmonic_evidence.id in item.source_ids for item in measurements)


def test_reveal_is_visible_only_after_matching_lock(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", reveal=True)
    run = load_run(directory)

    assert run.phase is RunPhase.GROUND_TRUTH_REVEALED
    assert run.ground_truth_visible
    assert run.reveal is not None
    assert run.reveal.ground_truth.actual_target_identity == "Post-lock identity"


def test_reveal_is_withheld_when_trace_capability_events_are_missing(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", reveal=True)
    trace_path = directory / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [item["event_type"] for item in events[-2:]] == [
        "CATALOG_ACCESS_ENABLED",
        "GROUND_TRUTH_REVEALED",
    ]
    trace_path.write_text(
        "\n".join(json.dumps(item, separators=(",", ":"), sort_keys=True) for item in events[:-2])
        + "\n",
        encoding="utf-8",
    )

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert run.reveal is None
    assert not run.ground_truth_visible
    assert any(item.code == "REVEAL_TRACE_INVALID" for item in run.issues)


def test_reveal_with_tampered_locked_hash_is_withheld(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", reveal=True)
    reveal_path = directory / "reveal.json"
    payload = json.loads(reveal_path.read_text(encoding="utf-8"))
    payload["locked_result_sha256"] = "f" * 64
    reveal_path.write_text(json.dumps(payload), encoding="utf-8")

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert run.reveal is None
    assert not run.ground_truth_visible
    assert any(item.code == "REVEAL_INVALID" for item in run.issues)


def test_reveal_without_lock_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "TARGET-X17"
    directory.mkdir()
    (directory / "reveal.json").write_text("{}", encoding="utf-8")

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert not run.ground_truth_visible
    assert any(item.code == "REVEAL_BEFORE_LOCK" for item in run.issues)


def test_tampered_locked_result_is_not_displayed(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", reveal=True)
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    payload["candidate"]["period_days"] = 99.0
    (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert run.result is None
    assert run.reveal is None
    assert not run.ground_truth_visible
    assert any(item.code == "LOCK_HASH_MISMATCH" for item in run.issues)


def test_locked_result_requires_trace_lock_event(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17")
    trace_path = directory / "trace.jsonl"
    events = trace_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(events[-1])["event_type"] == "RESULT_LOCKED"
    trace_path.write_text("\n".join(events[:-1]) + "\n", encoding="utf-8")

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert run.result is None
    assert any(item.code == "CROSS_ARTIFACT_MISMATCH" for item in run.issues)


def test_locked_result_commits_exact_pre_lock_trace_root(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17")
    result_path = directory / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["pre_lock_trace_root_hash"] = "f" * 64
    serialized = canonical_json_bytes(payload) + b"\n"
    result_path.write_bytes(serialized)
    (directory / "result.json.sha256").write_text(
        hashlib.sha256(serialized).hexdigest() + "\n", encoding="ascii"
    )

    run = load_run(directory)

    assert run.phase is RunPhase.CORRUPT
    assert run.result is None
    assert any(
        item.code == "CROSS_ARTIFACT_MISMATCH" and "exact trace prefix" in item.message
        for item in run.issues
    )


def test_hashed_manifest_product_loads_and_traversal_is_rejected(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17")
    science = directory / "artifacts" / "science"
    science.mkdir(parents=True)
    artifact = science / "bls_periodogram.npz"
    np.savez(artifact, period_days=np.array([1.0, 2.0]), power_snr=np.array([3.0, 8.0]))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    outside = directory.parent / "leak.npz"
    np.savez(outside, secret=np.array([42.0]))
    manifest = {
        "artifacts": [
            {
                "artifact_id": "TARGET-X17:bls_periodogram_data:npz",
                "relative_path": "artifacts/science/bls_periodogram.npz",
                "sha256": digest,
                "media_type": "application/x-npz",
                "role": "bls_periodogram_data",
            },
            {
                "artifact_id": "TARGET-X17:raw_light_curve:npz",
                "relative_path": "../leak.npz",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                "media_type": "application/x-npz",
                "role": "raw_light_curve_data",
            },
        ]
    }
    (directory / "artifacts" / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")

    run = load_run(directory)
    product = load_science_product(run, "bls_periodogram")

    assert product is not None
    assert product.source.integrity_verified
    assert product.arrays["power_snr"].tolist() == [3.0, 8.0]
    assert run.artifact("raw_light_curve") is None
    assert any(item.code == "ARTIFACT_PATH_REJECTED" for item in run.issues)


def test_viewmodels_replay_hypotheses_and_repair_agent_numbers(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17")
    run = load_run(directory)

    hypotheses = hypothesis_views(run.trace)
    planet = next(item for item in hypotheses if item.code == Hypothesis.PLANETARY_TRANSIT.value)
    skeptic = latest_skeptic_decision(run)
    critic = latest_critic_decision(run)

    assert planet.state == "SUPPORTED"
    assert planet.weight == 0.75
    assert skeptic is not None
    assert skeptic.decision_source == "LIVE_MODEL"
    assert skeptic.provider == "featherless"
    assert skeptic.model == "test-live-model"
    assert skeptic.provider_request_ids == ("provider-skeptic-1",)
    assert skeptic.attempts == 1
    assert critic is not None
    assert critic.decision_source == "REPAIRED_LIVE_MODEL"
    assert critic.provider_request_ids == ("provider-critic-1", "provider-critic-2")
    assert critic.attempts == 2
    assert "9.99" not in skeptic.explanation
    assert "[measurement unavailable]" in skeptic.explanation


def test_viewmodels_label_declared_deterministic_fallback(tmp_path: Path) -> None:
    directory, _ = _locked_run(tmp_path / "TARGET-X17", agent_fallback=True)
    run = load_run(directory)

    skeptic = latest_skeptic_decision(run)
    critic = latest_critic_decision(run)

    assert skeptic is not None
    assert critic is not None
    assert skeptic.decision_source == "DETERMINISTIC_FALLBACK"
    assert critic.decision_source == "DETERMINISTIC_FALLBACK"
    assert skeptic.used_fallback
    assert critic.used_fallback
