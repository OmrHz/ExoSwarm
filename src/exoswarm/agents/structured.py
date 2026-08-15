"""Strict structured-output execution with one repair and a declared fallback."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .provider import InferenceProvider, InferenceResponse, ProviderError

T = TypeVar("T", bound=BaseModel)
TraceHook = Callable[[str, dict[str, Any]], None]
FallbackFactory = Callable[[dict[str, Any], str], T]
_SENSITIVE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:TIC\s*\d+|TOI[-\s]?\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_PROHIBITED_CONTEXT_KEYS = frozenset(
    {
        "actual_target_identity",
        "backend_target_mapping",
        "catalog_measurements",
        "catalog_status",
        "confirmation_status",
        "fits_path",
        "ground_truth",
        "known_parameters",
        "known_period",
        "pixel_cube",
        "private_manifest",
        "raw_flux",
        "raw_light_curve",
        "real_target_id",
        "reveal",
        "reveal_artifact",
        "source_identity",
        "target_name",
        "tic_id",
        "toi_id",
    }
)


class DecisionSource(StrEnum):
    """Origin of a validated agent decision, recorded explicitly in every live trace."""

    LIVE_MODEL = "LIVE_MODEL"
    REPAIRED_LIVE_MODEL = "REPAIRED_LIVE_MODEL"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


@dataclass(frozen=True, slots=True)
class StructuredCall[T: BaseModel]:
    value: T
    used_fallback: bool
    repaired: bool
    attempts: int
    provider_request_ids: tuple[str, ...]

    @property
    def decision_source(self) -> DecisionSource:
        if self.used_fallback:
            return DecisionSource.DETERMINISTIC_FALLBACK
        if self.repaired:
            return DecisionSource.REPAIRED_LIVE_MODEL
        return DecisionSource.LIVE_MODEL


class StructuredAgentRunner:
    """Validate all model decisions and never silently continue after invalid output."""

    def __init__(
        self,
        provider: InferenceProvider,
        *,
        trace: TraceHook | None = None,
        max_packet_chars: int = 20_000,
    ) -> None:
        self.provider = provider
        self.trace = trace or (lambda _kind, _payload: None)
        self.max_packet_chars = max_packet_chars

    def request(
        self,
        *,
        role: str,
        objective: str,
        packet: dict[str, Any],
        response_model: type[T],
        fallback: FallbackFactory[T],
    ) -> StructuredCall[T]:
        user = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str)
        context_violation = _context_violation(packet)
        if context_violation is not None:
            reason = f"agent context failed the blindness/data-minimization preflight: {context_violation}"
            self.trace("agent_context_rejected", {"role": role, "reason": reason})
            value = fallback(packet, reason)
            self.trace(
                "agent_fallback",
                {"role": role, "reason": reason, "output": value.model_dump(mode="json")},
            )
            return StructuredCall(value, True, False, 0, ())
        if len(user) > self.max_packet_chars:
            reason = (
                f"compact evidence packet exceeded {self.max_packet_chars} characters; "
                "raw scientific arrays are not valid agent context"
            )
            self.trace("agent_context_rejected", {"role": role, "reason": reason})
            value = fallback(packet, reason)
            self.trace(
                "agent_fallback",
                {"role": role, "reason": reason, "output": value.model_dump(mode="json")},
            )
            return StructuredCall(value, True, False, 0, ())

        schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
        system = (
            f"ROLE: {role}\n"
            f"SCIENTIFIC OBJECTIVE: {objective}\n"
            "BOUNDARIES: Choose only from the supplied experiments. Never invent or infer "
            "measurements. Supplied numbers are deterministic evidence, not permission to "
            "create new numbers. Do not assign astrophysical probabilities. Return exactly "
            "one JSON object and no markdown.\n"
            f"OUTPUT JSON SCHEMA: {schema}"
        )
        self.trace(
            "agent_request",
            {
                "role": role,
                "provider": self.provider.name,
                "model": self.provider.model,
                "context_characters": len(user),
                "context_keys": sorted(packet),
                "context_sha256": sha256(user.encode("utf-8")).hexdigest(),
                "context_preflight": "PASS",
                "opaque_target_id": packet.get("opaque_target_id"),
                "lock_state": packet.get("lock_state"),
                "evidence_ids": [
                    str(item.get("id"))
                    for item in packet.get("evidence", [])
                    if isinstance(item, dict) and item.get("id") is not None
                ],
                "available_experiments": _string_list(packet.get("available_experiments", [])),
                "completed_tests": _string_list(packet.get("completed_tests", [])),
                "proposal_request_id": (
                    packet.get("proposal", {}).get("request_id")
                    if isinstance(packet.get("proposal"), dict)
                    else None
                ),
                "proposal_experiment": (
                    packet.get("proposal", {}).get("experiment_type")
                    if isinstance(packet.get("proposal"), dict)
                    else None
                ),
            },
        )

        request_ids: list[str] = []
        first_error = ""
        first_content = ""
        try:
            first = self.provider.complete(system=system, user=user)
            self._trace_response(role, 1, first)
            if first.request_id:
                request_ids.append(first.request_id)
            first_content = first.content
            raw_object, adapter_repaired = _extract_json_object(first.content)
            parsed = response_model.model_validate(raw_object)
            if adapter_repaired:
                self.trace(
                    "structured_output_repaired",
                    {
                        "role": role,
                        "attempt": 1,
                        "repair_kind": "MISSING_OPENING_OBJECT_BRACE",
                    },
                )
            return StructuredCall(parsed, False, adapter_repaired, 1, tuple(request_ids))
        except (ProviderError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            first_error = _safe_error(exc)
            self.trace(
                "structured_output_failure",
                {"role": role, "attempt": 1, "error": first_error},
            )

        repair_system = (
            system
            + "\nREPAIR: The previous response failed validation. Correct it once. Return only "
            "a complete JSON object that conforms exactly to the schema."
        )
        repair_user = json.dumps(
            {
                "evidence_packet": packet,
                "previous_output": first_content[:6_000],
                "validation_error": first_error,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        try:
            second = self.provider.complete(system=repair_system, user=repair_user)
            self._trace_response(role, 2, second)
            if second.request_id:
                request_ids.append(second.request_id)
            raw_object, adapter_repaired = _extract_json_object(second.content)
            parsed = response_model.model_validate(raw_object)
            self.trace(
                "structured_output_repaired",
                {
                    "role": role,
                    "attempt": 2,
                    "repair_kind": (
                        "MODEL_REPAIR_AND_MISSING_OPENING_OBJECT_BRACE"
                        if adapter_repaired
                        else "MODEL_REPAIR"
                    ),
                },
            )
            return StructuredCall(parsed, False, True, 2, tuple(request_ids))
        except (ProviderError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            second_error = _safe_error(exc)
            self.trace(
                "structured_output_failure",
                {"role": role, "attempt": 2, "error": second_error},
            )
            reason = f"structured decision unavailable after one repair: {second_error}"
            value = fallback(packet, reason)
            self.trace(
                "agent_fallback",
                {"role": role, "reason": reason, "output": value.model_dump(mode="json")},
            )
            return StructuredCall(value, True, False, 2, tuple(request_ids))

    def _trace_response(self, role: str, attempt: int, response: InferenceResponse) -> None:
        self.trace(
            "agent_response",
            {
                "role": role,
                "attempt": attempt,
                "provider": response.provider,
                "model": response.model,
                "request_id": response.request_id,
                "finish_reason": response.finish_reason,
                "usage": asdict(response.usage),
            },
        )


def _extract_json_object(text: str) -> tuple[dict[str, Any], bool]:
    """Extract one object, repairing only Featherless' observed missing leading brace."""

    cleaned = text.strip()
    cleaned = re.sub(r"^<think>[\s\S]*?</think>\s*", "", cleaned, flags=re.IGNORECASE)
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    adapter_repaired = False
    if cleaned.startswith('"') and cleaned.endswith("}"):
        # DeepSeek-V4-Flash on Featherless JSON mode can return an otherwise complete
        # object body with the opening brace omitted.  This exact one-character repair
        # remains subject to full JSON decoding, no-trailing-prose, and Pydantic validation.
        cleaned = "{" + cleaned
        adapter_repaired = True
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(cleaned)
    if cleaned[end:].strip():
        raise ValueError("model returned prose outside the JSON object")
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value, adapter_repaired


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        message = json.dumps(exc.errors(include_url=False), default=str)[:4_000]
    else:
        message = f"{type(exc).__name__}: {str(exc)[:1_000]}"
    return _SENSITIVE_IDENTIFIER_PATTERN.sub("[identity withheld until result lock]", message)


def _context_violation(value: object) -> str | None:
    """Return a safe reason without echoing prohibited context values."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _PROHIBITED_CONTEXT_KEYS:
                return f"prohibited context field {normalized!r}"
            nested = _context_violation(child)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _context_violation(child)
            if nested is not None:
                return nested
    elif isinstance(value, str) and _SENSITIVE_IDENTIFIER_PATTERN.search(value):
        return "recognizable catalog identifier detected"
    return None


def _string_list(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values]


__all__ = ["DecisionSource", "StructuredAgentRunner", "StructuredCall"]
