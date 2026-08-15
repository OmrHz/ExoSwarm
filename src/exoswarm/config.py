"""Environment-driven application configuration.

Secrets are read only by the provider adapter. They are never serialized into traces,
investigation state, scientific artifacts, or the mission-control UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str
    model: str
    api_base: str
    api_key: str | None = field(repr=False)
    data_dir: Path
    runs_dir: Path
    max_agent_turns: int
    experiment_budget: int
    request_timeout_seconds: int

    @classmethod
    def from_env(cls, root: Path | None = None) -> Settings:
        workspace = (root or Path.cwd()).resolve()
        # Process-level environment variables win; .env is a local convenience
        # and remains excluded from version control.
        load_dotenv(workspace / ".env", override=False)
        data_dir = Path(os.getenv("EXOSWARM_DATA_DIR", "data"))
        runs_dir = Path(os.getenv("EXOSWARM_RUNS_DIR", "runs"))
        if not data_dir.is_absolute():
            data_dir = workspace / data_dir
        if not runs_dir.is_absolute():
            runs_dir = workspace / runs_dir
        key = os.getenv("EXOSWARM_API_KEY") or None
        return cls(
            provider=os.getenv("EXOSWARM_PROVIDER", "auto").strip().lower(),
            model=os.getenv("EXOSWARM_MODEL", "deepseek-ai/DeepSeek-V4-Flash").strip(),
            api_base=os.getenv("EXOSWARM_API_BASE", "https://api.featherless.ai/v1").rstrip("/"),
            api_key=key,
            data_dir=data_dir.resolve(),
            runs_dir=runs_dir.resolve(),
            max_agent_turns=_positive_int("EXOSWARM_MAX_AGENT_TURNS", 4),
            experiment_budget=_positive_int("EXOSWARM_EXPERIMENT_BUDGET", 14),
            request_timeout_seconds=_positive_int("EXOSWARM_REQUEST_TIMEOUT_SECONDS", 45),
        )

    @property
    def provider_enabled(self) -> bool:
        return self.provider not in {"offline", "deterministic", "none"} and bool(self.api_key)

    def safe_summary(self) -> dict[str, object]:
        """Return trace-safe settings, deliberately excluding the API key."""

        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "max_agent_turns": self.max_agent_turns,
            "experiment_budget": self.experiment_budget,
            "request_timeout_seconds": self.request_timeout_seconds,
            "provider_enabled": self.provider_enabled,
        }
