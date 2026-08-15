"""Model-provider boundary for ExoSwarm.

Only this module knows how an external inference API is called. Scientific tools,
state, evidence, and lock semantics are provider-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import httpx


class ProviderErrorCategory(StrEnum):
    """Credential-safe failure categories exposed to tracing and validation."""

    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT = "RATE_LIMIT"
    SERVICE = "SERVICE"
    HTTP_STATUS = "HTTP_STATUS"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    UNAVAILABLE = "UNAVAILABLE"


class ProviderError(RuntimeError):
    """A model provider could not produce a usable response.

    Error details deliberately exclude response bodies, request headers, and URLs because
    any of those may contain provider-supplied diagnostics or credentials.  The bounded
    category and optional HTTP status are sufficient for live-path debugging and reports.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ProviderErrorCategory = ProviderErrorCategory.UNAVAILABLE,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class InferenceUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> InferenceUsage:
        data = value if isinstance(value, dict) else {}
        return cls(
            prompt_tokens=_optional_int(data.get("prompt_tokens")),
            completion_tokens=_optional_int(data.get("completion_tokens")),
            total_tokens=_optional_int(data.get("total_tokens")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    content: str
    provider: str
    model: str
    request_id: str | None
    finish_reason: str | None
    usage: InferenceUsage


class InferenceProvider(Protocol):
    """Minimal common provider interface used by the structured agent harness."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, user: str) -> InferenceResponse: ...


class FeatherlessProvider:
    """Small OpenAI-compatible Featherless adapter using direct HTTP.

    The adapter intentionally exposes neither tools nor arbitrary code execution. Agents
    return one bounded JSON decision, which the application validates separately.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str = "https://api.featherless.ai/v1",
        timeout_seconds: float = 45.0,
        max_tokens: int = 1_200,
    ) -> None:
        if not api_key:
            raise ValueError("A non-empty Featherless API key is required")
        if not model:
            raise ValueError("A non-empty model id is required")
        self._api_key = api_key
        self._model = model
        self._url = f"{api_base.rstrip('/')}/chat/completions"
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "featherless"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> InferenceResponse:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "seed": 17,
            # Featherless recommends JSON-object mode for prompt-driven structured
            # output. Pydantic still performs the authoritative schema validation.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/exoswarm/exoswarm",
            "X-Title": "ExoSwarm",
        }
        try:
            response = httpx.post(
                self._url,
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "Featherless request timed out",
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.HTTPStatusError as exc:
            # Do not include the response body, request URL, or headers: they may
            # contain provider diagnostics or the Authorization credential.
            status_code = exc.response.status_code
            category = _http_error_category(status_code)
            raise ProviderError(
                f"Featherless request failed ({category.value}, HTTP {status_code})",
                category=category,
                status_code=status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                f"Featherless request failed ({ProviderErrorCategory.TRANSPORT.value})",
                category=ProviderErrorCategory.TRANSPORT,
            ) from exc

        try:
            payload = response.json()
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty completion content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                "Featherless returned an invalid completion envelope",
                category=ProviderErrorCategory.INVALID_RESPONSE,
            ) from exc

        return InferenceResponse(
            content=content,
            provider=self.name,
            model=str(payload.get("model") or self._model),
            request_id=str(payload["id"]) if payload.get("id") is not None else None,
            finish_reason=(
                str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None
            ),
            usage=InferenceUsage.from_mapping(payload.get("usage")),
        )


class UnavailableProvider:
    """Explicit provider used to exercise the trace-recorded safe fallback path."""

    def __init__(self, reason: str = "no model provider configured") -> None:
        self.reason = reason

    @property
    def name(self) -> str:
        return "unavailable"

    @property
    def model(self) -> str:
        return "none"

    def complete(self, *, system: str, user: str) -> InferenceResponse:
        del system, user
        raise ProviderError(self.reason, category=ProviderErrorCategory.UNAVAILABLE)


def _http_error_category(status_code: int) -> ProviderErrorCategory:
    if status_code == 401:
        return ProviderErrorCategory.AUTHENTICATION
    if status_code == 403:
        return ProviderErrorCategory.AUTHORIZATION
    if status_code == 404:
        return ProviderErrorCategory.NOT_FOUND
    if status_code == 429:
        return ProviderErrorCategory.RATE_LIMIT
    if status_code >= 500:
        return ProviderErrorCategory.SERVICE
    return ProviderErrorCategory.HTTP_STATUS
