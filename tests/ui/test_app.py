from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.test_artifacts import _locked_run


def _rendered_text(app: AppTest) -> str:
    element_types = ("markdown", "caption", "info", "success", "warning", "error")
    return "\n".join(
        str(element.value)
        for element_type in element_types
        for element in getattr(app, element_type)
    )


def _frontend_payload(app: AppTest) -> bytes:
    chunks = [_rendered_text(app).encode("utf-8")]
    for element in app:
        proto = getattr(element, "proto", None)
        serialize = getattr(proto, "SerializeToString", None)
        if serialize is not None:
            chunks.append(serialize())
    return b"\n".join(chunks)


def test_app_renders_no_run_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXOSWARM_RUNS_DIR", str(tmp_path / "runs"))
    app_path = Path(__file__).parents[2] / "src" / "exoswarm" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert app.selectbox[0].value == "TARGET-X17"
    assert any("No persisted investigation" in markdown.value for markdown in app.markdown)


@pytest.mark.parametrize(
    ("agent_fallback", "expected_sources"),
    [
        (False, {"LIVE_MODEL", "REPAIRED_LIVE_MODEL"}),
        (True, {"DETERMINISTIC_FALLBACK"}),
    ],
)
def test_app_renders_trace_backed_agent_source(
    tmp_path: Path,
    monkeypatch,
    agent_fallback: bool,
    expected_sources: set[str],
) -> None:
    runs_root = tmp_path / "runs"
    _locked_run(runs_root / "TARGET-X17", agent_fallback=agent_fallback)
    monkeypatch.setenv("EXOSWARM_RUNS_DIR", str(runs_root))
    credential_sentinel = "ui-test-credential-must-not-render"
    monkeypatch.setenv("EXOSWARM_API_KEY", credential_sentinel)
    app_path = Path(__file__).parents[2] / "src" / "exoswarm" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15)
    app.query_params["target"] = "TARGET-X17"
    app.run()
    rendered = _rendered_text(app)

    assert not app.exception
    assert "RESULT LOCKED" in rendered
    assert "DECISION SOURCE" in rendered
    assert all(source in rendered for source in expected_sources)
    if agent_fallback:
        assert "unavailable" in rendered
        assert "none" in rendered
    else:
        assert "featherless" in rendered
        assert "test-live-model" in rendered
    assert "Post-lock identity" not in rendered
    assert credential_sentinel.encode() not in _frontend_payload(app)
