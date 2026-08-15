from pathlib import Path

import exoswarm.cli as cli
from exoswarm.domain.trace import TraceEventType, TraceRecorder


def test_project_root_prefers_checkout_in_working_directory(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "data").mkdir()
    (checkout / "pyproject.toml").touch()
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(
        cli,
        "__file__",
        str(tmp_path / "venv" / "Lib" / "site-packages" / "exoswarm" / "cli.py"),
    )

    assert cli._project_root() == checkout.resolve()


def test_agent_execution_summary_distinguishes_repaired_live_from_fallback() -> None:
    live_trace = TraceRecorder(trace_id="TRACE-LIVE", opaque_target_id="TARGET-X17")
    live_trace.append(
        TraceEventType.AGENT_RESPONSE,
        {"role": "SKEPTIC", "provider": "featherless", "model": "model"},
    )
    live_trace.append(
        TraceEventType.AGENT_DECISION,
        {
            "role": "SKEPTIC",
            "decision_source": "REPAIRED_LIVE_MODEL",
            "used_fallback": False,
            "repaired": True,
        },
    )
    live = cli._agent_execution_summary(live_trace)
    assert live["agent_mode"] == "REPAIRED_LIVE_MODEL"
    assert live["live_agent_success"] is True

    fallback_trace = TraceRecorder(trace_id="TRACE-FALLBACK", opaque_target_id="TARGET-X17")
    fallback_trace.append(
        TraceEventType.AGENT_DECISION,
        {
            "role": "SKEPTIC",
            "decision_source": "DETERMINISTIC_FALLBACK",
            "used_fallback": True,
            "repaired": False,
        },
    )
    fallback = cli._agent_execution_summary(fallback_trace)
    assert fallback["agent_mode"] == "DETERMINISTIC_FALLBACK"
    assert fallback["live_agent_success"] is False


def test_require_live_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_one",
        lambda *_args, **_kwargs: {"live_agent_success": False},
    )
    assert cli.main(["run", "TARGET-X17", "--require-live"]) == 3
    assert cli.main(["run", "TARGET-X17", "--offline", "--require-live"]) == 2
