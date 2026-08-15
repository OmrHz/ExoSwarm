from __future__ import annotations

from pathlib import Path

from exoswarm.agents.factory import build_provider
from exoswarm.agents.provider import FeatherlessProvider, UnavailableProvider
from exoswarm.config import Settings


def test_api_secret_never_appears_in_safe_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXOSWARM_API_KEY", "super-secret")
    monkeypatch.setenv("EXOSWARM_PROVIDER", "featherless")
    settings = Settings.from_env(tmp_path)
    assert settings.api_key == "super-secret"
    assert "super-secret" not in repr(settings.safe_summary())
    assert "api_key" not in settings.safe_summary()
    assert "data_dir" not in settings.safe_summary()
    assert "runs_dir" not in settings.safe_summary()
    assert str(tmp_path) not in repr(settings.safe_summary())
    assert "super-secret" not in repr(settings)
    assert isinstance(build_provider(settings), FeatherlessProvider)


def test_missing_key_selects_explicit_unavailable_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EXOSWARM_API_KEY", raising=False)
    settings = Settings.from_env(tmp_path)
    assert not settings.provider_enabled
    assert isinstance(build_provider(settings), UnavailableProvider)


def test_local_dotenv_is_loaded_without_overriding_process_environment(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "EXOSWARM_API_KEY=from-file\nEXOSWARM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EXOSWARM_API_KEY", raising=False)
    monkeypatch.setenv("EXOSWARM_MODEL", "process-model")
    settings = Settings.from_env(tmp_path)
    assert settings.api_key == "from-file"
    assert settings.model == "process-model"
