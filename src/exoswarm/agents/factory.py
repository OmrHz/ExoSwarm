"""Construct the configured inference boundary without leaking provider concerns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from exoswarm.config import Settings

from .provider import FeatherlessProvider, InferenceProvider, UnavailableProvider
from .structured import StructuredAgentRunner


def build_provider(settings: Settings) -> InferenceProvider:
    if settings.provider_enabled:
        if settings.provider not in {"auto", "featherless"}:
            return UnavailableProvider(
                f"unsupported EXOSWARM_PROVIDER={settings.provider!r}; safe fallback required"
            )
        assert settings.api_key is not None  # narrowed by provider_enabled
        return FeatherlessProvider(
            api_key=settings.api_key,
            model=settings.model,
            api_base=settings.api_base,
            timeout_seconds=settings.request_timeout_seconds,
        )
    return UnavailableProvider(
        "no model API key configured; using the explicit deterministic policy fallback"
    )


def build_agent_runner(
    settings: Settings,
    *,
    trace: Callable[[str, dict[str, Any]], None] | None = None,
) -> StructuredAgentRunner:
    return StructuredAgentRunner(build_provider(settings), trace=trace)


__all__ = ["build_agent_runner", "build_provider"]
