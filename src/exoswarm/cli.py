"""Command-line entry points for running, revealing, reproducing, and evaluating ExoSwarm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from exoswarm.agents.provider import UnavailableProvider
from exoswarm.config import Settings
from exoswarm.domain.models import (
    ExperimentType,
    LockedInvestigationResult,
    ResultLockReceipt,
    ScientificDisposition,
)
from exoswarm.domain.trace import TraceEventType, TraceRecorder
from exoswarm.evaluation.ablation import compare_policy_runs, render_ablation_markdown
from exoswarm.evaluation.graders import (
    EvaluationExpectation,
    EvaluationReport,
    evaluate_run,
    evaluate_trajectory_diversity,
    render_markdown,
)
from exoswarm.runtime.director import ScientificDirector
from exoswarm.runtime.targets import blind_target_summaries, load_demo_vault
from exoswarm.science import ScienceToolbox
from exoswarm.science.common import (
    load_science_manifest,
    product_path,
    verify_cached_product,
)
from exoswarm.security.blindness import GroundTruthGate
from exoswarm.security.locking import ResultLocker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exoswarm",
        description="Blinded, evidence-led investigation of cached real TESS observations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("targets", help="List only identity-safe demo target metadata.")

    run = subparsers.add_parser("run", help="Run and lock one investigation.")
    run.add_argument("target", choices=["TARGET-X17", "TARGET-X42"])
    run.add_argument("--runs-root", type=Path, default=None)
    run.add_argument("--reveal", action="store_true", help="Reveal catalog truth after lock.")
    run.add_argument("--policy", choices=["adaptive", "fixed"], default="adaptive")
    run.add_argument(
        "--offline",
        action="store_true",
        help="Use the explicitly traced deterministic agent fallback even if an API key exists.",
    )
    run.add_argument(
        "--require-live",
        action="store_true",
        help="Return a failure status unless every agent decision came from the configured model.",
    )

    reveal = subparsers.add_parser(
        "reveal", help="Reveal catalog truth for an already locked result."
    )
    reveal.add_argument("target", choices=["TARGET-X17", "TARGET-X42"])
    reveal.add_argument("--runs-root", type=Path, default=None)

    verify = subparsers.add_parser("verify", help="Run deterministic graders on a run.")
    verify.add_argument("target", choices=["TARGET-X17", "TARGET-X42"])
    verify.add_argument("--runs-root", type=Path, default=None)
    verify.add_argument("--policy", choices=["adaptive", "fixed"], default="adaptive")

    cache = subparsers.add_parser(
        "verify-cache", help="Verify cached source sizes and SHA-256 digests."
    )
    cache.add_argument("target", nargs="?", choices=["TARGET-X17", "TARGET-X42"])

    reproduce = subparsers.add_parser(
        "reproduce", help="Rerun both deterministic investigations offline and grade them."
    )
    reproduce.add_argument("--runs-root", type=Path, default=None)

    ablation = subparsers.add_parser(
        "ablation", help="Measure adaptive and fixed policies over both real targets."
    )
    ablation.add_argument("--runs-root", type=Path, default=None)

    ui = subparsers.add_parser("ui", help="Launch the Streamlit mission-control interface.")
    ui.add_argument("--runs-root", type=Path, default=None)
    ui.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(_project_root())
    try:
        if args.command == "targets":
            print(json.dumps(blind_target_summaries(settings.data_dir), indent=2))
            return 0
        if args.command == "verify-cache":
            return _verify_cache(settings, args.target)
        if args.command == "run":
            if args.offline and args.require_live:
                raise ValueError("--offline and --require-live are mutually exclusive")
            runs_root = (args.runs_root or settings.runs_dir).resolve()
            outcome = _run_one(
                settings,
                target=args.target,
                runs_root=runs_root,
                policy=args.policy,
                reveal=args.reveal,
                offline=args.offline,
            )
            print(json.dumps(outcome, indent=2, default=str))
            return 0 if not args.require_live or outcome["live_agent_success"] else 3
        if args.command == "reveal":
            runs_root = (args.runs_root or settings.runs_dir).resolve()
            artifact = _reveal_existing(settings, args.target, runs_root)
            print(artifact.model_dump_json(indent=2))
            return 0
        if args.command == "verify":
            runs_root = (args.runs_root or settings.runs_dir).resolve()
            report = evaluate_run(
                runs_root / args.target,
                _expectation(
                    args.target,
                    policy=args.policy,
                    require_reveal=(runs_root / args.target / "reveal.json").exists(),
                ),
            )
            print(render_markdown([report]))
            return 0 if report.passed else 1
        if args.command == "reproduce":
            root = _unique_output_root(args.runs_root or settings.runs_dir / "reproductions")
            reports = _reproduce(settings, root)
            diversity = evaluate_trajectory_diversity(reports)
            markdown = render_markdown(reports, diversity)
            _write_report(root, reports, markdown, diversity.model_dump(mode="json"))
            print(markdown)
            print(f"Artifacts: {root}")
            return 0 if all(report.passed for report in reports) and diversity.passed else 1
        if args.command == "ablation":
            root = _unique_output_root(args.runs_root or settings.runs_dir / "ablations")
            markdown = _run_ablation(settings, root)
            print(markdown)
            print(f"Artifacts: {root}")
            return 0
        if args.command == "ui":
            return _launch_ui(settings, args.runs_root, args.port)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ExoSwarm error: {exc}", file=sys.stderr)
        return 2
    return 2


def _run_one(
    settings: Settings,
    *,
    target: str,
    runs_root: Path,
    policy: str,
    reveal: bool,
    offline: bool,
) -> dict[str, object]:
    run_settings = replace(settings, runs_dir=runs_root)
    if offline:
        run_settings = replace(
            run_settings,
            provider="offline",
            model="none",
            api_key=None,
        )
    vault, targets = load_demo_vault(run_settings.data_dir)
    if target not in targets:
        raise ValueError(f"target is not in the curated registry: {target}")
    toolbox = ScienceToolbox(
        data_root=run_settings.data_dir / "tess",
        runs_root=runs_root,
    )
    provider = UnavailableProvider("offline reproducibility mode") if offline else None
    director = ScientificDirector(
        settings=run_settings,
        toolbox=toolbox,
        vault=vault,
        provider=provider,
    )
    outcome = director.investigate(target, policy=policy, reveal=reveal)
    agent_summary = _agent_execution_summary(outcome.trace)
    summary: dict[str, object] = {
        "opaque_target_id": target,
        "trace_id": outcome.state.trace_id,
        "policy": policy,
        **agent_summary,
        "disposition": outcome.state.final_disposition.value
        if outcome.state.final_disposition
        else None,
        "candidate": outcome.state.candidates[0].model_dump(mode="json")
        if outcome.state.candidates
        else None,
        "completed_tests": [item.value for item in outcome.state.completed_tests],
        "result_sha256": outcome.receipt.sha256,
        "run_directory": str(outcome.run_directory),
        "ground_truth_revealed": outcome.reveal is not None,
    }
    if outcome.reveal is not None:
        summary["reveal"] = outcome.reveal.model_dump(mode="json")
    return summary


def _agent_execution_summary(trace: TraceRecorder) -> dict[str, object]:
    decision_events = [
        event
        for event in trace.events
        if event.event_type in {TraceEventType.AGENT_DECISION, TraceEventType.CRITIC_DECISION}
    ]
    sources: list[str] = []
    for event in decision_events:
        explicit = event.payload.get("decision_source")
        if isinstance(explicit, str):
            sources.append(explicit)
        elif bool(event.payload.get("used_fallback")):
            sources.append("DETERMINISTIC_FALLBACK")
        elif bool(event.payload.get("repaired")):
            sources.append("REPAIRED_LIVE_MODEL")
        else:
            sources.append("LIVE_MODEL")
    response_events = [
        event for event in trace.events if event.event_type is TraceEventType.AGENT_RESPONSE
    ]
    fallback_reason_codes = [
        str(event.payload.get("reason_code") or event.payload.get("source_event") or "UNKNOWN")
        for event in trace.events
        if event.event_type is TraceEventType.FALLBACK
    ]
    deterministic_selection_fallback = any(
        reason
        in {
            "agent_fallback",
            "DETERMINISTIC_TOOL_SELECTION_FALLBACK",
            "INVALID_AGENT_TOOL_REQUEST",
        }
        for reason in fallback_reason_codes
    )
    live_agent_success = (
        bool(decision_events)
        and bool(response_events)
        and all(source in {"LIVE_MODEL", "REPAIRED_LIVE_MODEL"} for source in sources)
        and not deterministic_selection_fallback
    )
    if not sources or "DETERMINISTIC_FALLBACK" in sources or deterministic_selection_fallback:
        agent_mode = "DETERMINISTIC_FALLBACK"
    elif "REPAIRED_LIVE_MODEL" in sources:
        agent_mode = "REPAIRED_LIVE_MODEL"
    else:
        agent_mode = "LIVE_MODEL"
    return {
        "agent_mode": agent_mode,
        "live_agent_success": live_agent_success,
        "decision_sources": sources,
        "live_model_calls": len(response_events),
        "fallback_reason_codes": fallback_reason_codes,
    }


def _reveal_existing(settings: Settings, target: str, runs_root: Path):
    directory = (runs_root / target).resolve()
    reveal_path = directory / "reveal.json"
    if reveal_path.exists():
        raise RuntimeError("reveal.json already exists and is immutable")
    result_path = directory / "result.json"
    hash_path = directory / "result.json.sha256"
    result = LockedInvestigationResult.model_validate_json(result_path.read_bytes())
    digest = hash_path.read_text(encoding="ascii").strip().lower()
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("locked result hash verification failed")
    receipt = ResultLockReceipt(
        opaque_target_id=target,
        result_path=str(result_path),
        hash_path=str(hash_path),
        sha256=digest,
        locked_at=datetime.fromtimestamp(hash_path.stat().st_mtime, tz=UTC),
    )
    trace = TraceRecorder(
        trace_id=result.trace_id,
        opaque_target_id=target,
        path=directory / "trace.jsonl",
    )
    vault, _ = load_demo_vault(settings.data_dir)
    locker = ResultLocker()
    gate = GroundTruthGate(vault, locker)
    gate.unlock_after_result_lock(receipt, trace=trace)
    return gate.create_reveal_artifact(receipt, trace=trace)


def _verify_cache(settings: Settings, selected: str | None) -> int:
    targets = [selected] if selected else ["TARGET-X17", "TARGET-X42"]
    rows: list[dict[str, object]] = []
    for target in targets:
        manifest = load_science_manifest(settings.data_dir / "tess", target)
        for role, product in manifest.products.items():
            path = product_path(settings.data_dir / "tess", manifest, role)
            verify_cached_product(path, product)
            rows.append(
                {
                    "opaque_target_id": target,
                    "role": role,
                    "bytes": product.size_bytes,
                    "sha256": product.sha256,
                    "status": "VERIFIED",
                }
            )
    print(json.dumps(rows, indent=2))
    return 0


def _reproduce(settings: Settings, root: Path) -> list[EvaluationReport]:
    offline = replace(
        settings,
        provider="offline",
        model="none",
        api_key=None,
        runs_dir=root,
    )
    reports: list[EvaluationReport] = []
    for target in ("TARGET-X17", "TARGET-X42"):
        _run_one(
            offline,
            target=target,
            runs_root=root,
            policy="adaptive",
            reveal=True,
            offline=True,
        )
        reports.append(
            evaluate_run(
                root / target,
                _expectation(target, policy="adaptive", require_reveal=True),
            )
        )
    return reports


def _run_ablation(settings: Settings, root: Path) -> str:
    offline = replace(settings, provider="offline", model="none", api_key=None)
    comparisons = []
    expected = {
        "TARGET-X17": ExperimentType.CENTROID_LOCALIZATION,
        "TARGET-X42": ExperimentType.HARMONIC_TEST,
    }
    for target in ("TARGET-X17", "TARGET-X42"):
        adaptive_root = root / "adaptive"
        fixed_root = root / "fixed"
        _run_one(
            offline,
            target=target,
            runs_root=adaptive_root,
            policy="adaptive",
            reveal=False,
            offline=True,
        )
        _run_one(
            offline,
            target=target,
            runs_root=fixed_root,
            policy="fixed",
            reveal=False,
            offline=True,
        )
        comparisons.append(
            compare_policy_runs(
                adaptive_root / target,
                fixed_root / target,
                expected_best_action=expected[target],
            )
        )
    markdown = render_ablation_markdown(comparisons)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation.md").write_text(markdown, encoding="utf-8")
    (root / "ablation.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in comparisons],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown


def _expectation(
    target: str,
    *,
    policy: str,
    require_reveal: bool,
) -> EvaluationExpectation:
    if target == "TARGET-X17":
        return EvaluationExpectation(
            opaque_target_id=target,
            expected_period_days=3.739494,
            period_tolerance_days=0.01,
            accepted_dispositions={
                ScientificDisposition.PLANETARY_INTERPRETATION_PLAUSIBLE,
                ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING,
            },
            expected_adaptive_any_of={ExperimentType.CENTROID_LOCALIZATION},
            forbidden_adaptive={ExperimentType.HARMONIC_TEST},
            require_reveal=require_reveal,
        )
    return EvaluationExpectation(
        opaque_target_id=target,
        expected_period_days=3.0319175718,
        period_tolerance_days=0.02,
        accepted_dispositions={ScientificDisposition.PLANETARY_INTERPRETATION_WEAK},
        expected_adaptive_any_of=(
            {ExperimentType.HARMONIC_TEST}
            if policy == "adaptive"
            else {ExperimentType.CENTROID_LOCALIZATION}
        ),
        forbidden_adaptive=(
            {ExperimentType.CENTROID_LOCALIZATION} if policy == "adaptive" else set()
        ),
        negative_control=True,
        require_reveal=require_reveal,
    )


def _write_report(
    root: Path,
    reports: list[EvaluationReport],
    markdown: str,
    cross_case: dict[str, object],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "evaluation.md").write_text(markdown, encoding="utf-8")
    (root / "evaluation.json").write_text(
        json.dumps(
            {
                "reports": [item.model_dump(mode="json") for item in reports],
                "cross_case": cross_case,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unique_output_root(base: Path) -> Path:
    base = base.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return base / stamp


def _launch_ui(settings: Settings, runs_root: Path | None, port: int) -> int:
    if runs_root is not None:
        import os

        os.environ["EXOSWARM_RUNS_DIR"] = str(runs_root.resolve())
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    return int(streamlit_cli.main() or 0)


def _project_root() -> Path:
    """Locate a source checkout without making a wheel depend on ``site-packages/data``."""

    working_directory = Path.cwd().resolve()
    source_checkout = Path(__file__).resolve().parents[2]
    for candidate in (working_directory, source_checkout):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    return working_directory


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
