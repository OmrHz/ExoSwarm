from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from exoswarm.agents.provider import InferenceResponse, InferenceUsage, ProviderError
from exoswarm.agents.structured import DecisionSource, StructuredAgentRunner


class DecisionStub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requested_experiment: str
    priority: int


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: Iterable[str | Exception]) -> None:
        self.responses = iter(responses)

    def complete(self, *, system: str, user: str) -> InferenceResponse:
        assert "OUTPUT JSON SCHEMA" in system
        assert user
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return InferenceResponse(
            content=value,
            provider=self.name,
            model=self.model,
            request_id="request-1",
            finish_reason="stop",
            usage=InferenceUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def fallback(_packet: dict[str, object], _reason: str) -> DecisionStub:
    return DecisionStub(requested_experiment="stop", priority=0)


def test_valid_json_is_schema_validated() -> None:
    runner = StructuredAgentRunner(
        FakeProvider(['{"requested_experiment":"harmonic_test","priority":2}'])
    )
    result = runner.request(
        role="Skeptic",
        objective="choose an experiment",
        packet={"available_experiments": ["harmonic_test", "stop"]},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.value.requested_experiment == "harmonic_test"
    assert not result.used_fallback
    assert result.attempts == 1
    assert result.decision_source is DecisionSource.LIVE_MODEL


def test_one_repair_is_allowed() -> None:
    events: list[str] = []
    runner = StructuredAgentRunner(
        FakeProvider(
            [
                "not json",
                '{"requested_experiment":"centroid_localization","priority":1}',
            ]
        ),
        trace=lambda kind, _payload: events.append(kind),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose an experiment",
        packet={"available_experiments": ["centroid_localization"]},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.repaired
    assert result.attempts == 2
    assert result.decision_source is DecisionSource.REPAIRED_LIVE_MODEL
    assert "structured_output_repaired" in events


def test_second_invalid_output_uses_declared_fallback() -> None:
    events: list[str] = []
    runner = StructuredAgentRunner(
        FakeProvider(["{}", "still invalid"]),
        trace=lambda kind, _payload: events.append(kind),
    )
    result = runner.request(
        role="Critic",
        objective="review",
        packet={"proposal": "duplicate"},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.used_fallback
    assert result.decision_source is DecisionSource.DETERMINISTIC_FALLBACK
    assert result.value.requested_experiment == "stop"
    assert events.count("structured_output_failure") == 2
    assert events[-1] == "agent_fallback"


def test_provider_failure_is_not_silent() -> None:
    events: list[str] = []
    runner = StructuredAgentRunner(
        FakeProvider([ProviderError("down"), ProviderError("still down")]),
        trace=lambda kind, _payload: events.append(kind),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.used_fallback
    assert "agent_fallback" in events


def test_raw_or_oversize_context_is_rejected_before_provider_call() -> None:
    events: list[str] = []
    runner = StructuredAgentRunner(
        FakeProvider([]),
        trace=lambda kind, _payload: events.append(kind),
        max_packet_chars=20,
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={"raw_flux": [1.0] * 100},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.used_fallback
    assert result.attempts == 0
    assert events == ["agent_context_rejected", "agent_fallback"]


def test_prose_outside_json_is_rejected() -> None:
    runner = StructuredAgentRunner(
        FakeProvider(
            [
                '{"requested_experiment":"harmonic_test","priority":2} because useful',
                '{"requested_experiment":"stop","priority":0}',
            ]
        )
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.repaired
    assert result.value.requested_experiment == "stop"


def test_observed_missing_opening_brace_is_conservatively_repaired() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    runner = StructuredAgentRunner(
        FakeProvider(['"requested_experiment":"harmonic_test","priority":2}']),
        trace=lambda kind, payload: events.append((kind, payload)),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.value.requested_experiment == "harmonic_test"
    assert result.attempts == 1
    assert result.decision_source is DecisionSource.REPAIRED_LIVE_MODEL
    repaired = next(payload for kind, payload in events if kind == "structured_output_repaired")
    assert repaired["repair_kind"] == "MISSING_OPENING_OBJECT_BRACE"


def test_missing_opening_brace_with_trailing_prose_is_still_rejected() -> None:
    runner = StructuredAgentRunner(
        FakeProvider(
            [
                '"requested_experiment":"harmonic_test","priority":2} because useful',
                '{"requested_experiment":"stop","priority":0}',
            ]
        )
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.attempts == 2
    assert result.value.requested_experiment == "stop"


def test_validation_error_trace_redacts_guessed_catalog_identity() -> None:
    payloads: list[dict[str, object]] = []
    runner = StructuredAgentRunner(
        FakeProvider(
            [
                '{"requested_experiment":"TIC 123456","priority":"bad"}',
                '{"requested_experiment":"stop","priority":0}',
            ]
        ),
        trace=lambda kind, payload: payloads.append({"kind": kind, **payload}),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.repaired
    assert "TIC 123456" not in repr(payloads)


def test_agent_request_records_safe_replay_metadata_and_context_digest() -> None:
    payloads: list[dict[str, object]] = []
    runner = StructuredAgentRunner(
        FakeProvider(['{"requested_experiment":"harmonic_test","priority":2}']),
        trace=lambda kind, payload: payloads.append({"kind": kind, **payload}),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={
            "opaque_target_id": "TARGET-X17",
            "lock_state": "UNLOCKED",
            "evidence": [{"id": "EV-1"}],
            "available_experiments": ["harmonic_test"],
            "completed_tests": ["odd_even"],
        },
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert not result.used_fallback
    request = next(payload for payload in payloads if payload["kind"] == "agent_request")
    assert request["context_preflight"] == "PASS"
    assert request["opaque_target_id"] == "TARGET-X17"
    assert request["evidence_ids"] == ["EV-1"]
    assert request["available_experiments"] == ["harmonic_test"]
    assert len(str(request["context_sha256"])) == 64


def test_catalog_identity_context_is_rejected_before_provider_call() -> None:
    events: list[str] = []
    runner = StructuredAgentRunner(
        FakeProvider([]),
        trace=lambda kind, _payload: events.append(kind),
    )
    result = runner.request(
        role="Skeptic",
        objective="choose",
        packet={"opaque_target_id": "TARGET-X17", "note": "TIC 123456"},
        response_model=DecisionStub,
        fallback=fallback,
    )
    assert result.decision_source is DecisionSource.DETERMINISTIC_FALLBACK
    assert result.attempts == 0
    assert events == ["agent_context_rejected", "agent_fallback"]
