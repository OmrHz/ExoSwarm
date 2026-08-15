"""ExoSwarm mission-control Streamlit application.

Run with::

    streamlit run src/exoswarm/ui/app.py

The application is an audit surface over persisted artifacts.  It never imports
the target vault, opens cached FITS products, or performs scientific measurements.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from exoswarm.config import Settings
from exoswarm.domain.models import ExperimentType
from exoswarm.ui.artifacts import (
    IssueSeverity,
    MissionControlRun,
    RunPhase,
    ScienceProduct,
    available_target_ids,
    image_artifact,
    load_run,
    load_science_product,
    resolve_run_directory,
)
from exoswarm.ui.charts import (
    AMBER,
    CYAN,
    TEAL,
    bls_figure,
    centroid_figure,
    folded_figure,
    harmonic_figure,
    light_curve_figure,
    odd_even_figure,
    secondary_figure,
)
from exoswarm.ui.theme import MISSION_CONTROL_CSS, PLOTLY_CONFIG
from exoswarm.ui.viewmodels import (
    MeasurementVM,
    adaptive_experiments,
    candidate_measurements,
    catalog_measurements,
    evidence_headline,
    evidence_tone,
    hypothesis_views,
    latest_critic_decision,
    latest_skeptic_decision,
    working_hypotheses,
)


def main() -> None:
    st.set_page_config(
        page_title="ExoSwarm Mission Control",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "ExoSwarm is a blinded, evidence-led investigation interface. "
                "Agents select permitted experiments; deterministic Python creates measurements."
            )
        },
    )
    st.markdown(MISSION_CONTROL_CSS, unsafe_allow_html=True)

    settings = Settings.from_env()
    targets = available_target_ids(settings.runs_dir)
    requested = st.query_params.get("target")
    initial_index = targets.index(requested) if requested in targets else 0

    with st.sidebar:
        _render_brand()
        st.caption("BLINDED INVESTIGATION CONSOLE")
        target = st.selectbox(
            "Opaque target",
            targets,
            index=initial_index,
            help="Only opaque target IDs are available before the reveal boundary.",
        )
        auto_refresh = st.toggle(
            "Follow artifact updates",
            value=False,
            help="Replays newly persisted evidence every two seconds; this is not a simulated feed.",
        )
        if st.button("Refresh artifacts", width="stretch", type="secondary"):
            st.rerun()
        st.divider()
        st.markdown("**Trust boundary**")
        st.caption(
            "Read-only UI · sanitized run artifacts · verified hashes · no cached FITS headers · "
            "no catalog import"
        )
        st.caption(f"Runs root · `{_safe_relative(settings.runs_dir)}`")

    if st.query_params.get("target") != target:
        st.query_params["target"] = target

    run_directory = resolve_run_directory(settings.runs_dir, target)

    initial_run = load_run(run_directory, opaque_target_id=target)
    _render_sidebar_health(initial_run)

    def render_surface() -> None:
        run = load_run(run_directory, opaque_target_id=target)
        _render_run(run)

    if auto_refresh:

        @st.fragment(run_every="2s")
        def live_artifact_replay() -> None:
            render_surface()

        live_artifact_replay()
    else:
        render_surface()


def _render_brand() -> None:
    st.markdown(
        """
        <div class="exo-brand">
          <div class="exo-mark"></div>
          <div>
            <div class="exo-brand-name">EXOSWARM</div>
            <div class="exo-brand-sub">Mission Control</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_health(run: MissionControlRun) -> None:
    with st.sidebar:
        st.divider()
        st.markdown("**Artifact state**")
        if run.phase is RunPhase.CORRUPT:
            st.error("Integrity check failed", icon="⚠️")
        elif run.lock_verified:
            st.success("Result hash verified", icon="✅")
        elif run.phase is RunPhase.ACTIVE:
            st.info("Investigation artifacts active", icon="🔄")
        else:
            st.caption("No run artifacts yet")
        st.caption(f"Evidence records · {len(run.evidence)}")
        st.caption(f"Trace events · {len(run.trace)}")
        verified_science = sum(item.integrity_verified for item in run.artifacts)
        st.caption(f"Verified science products · {verified_science}")


def _render_run(run: MissionControlRun) -> None:
    _render_hero(run)
    _render_issues(run)
    if run.phase is RunPhase.EMPTY:
        _render_empty(run)
        return
    if run.phase is RunPhase.CORRUPT:
        _render_corrupt_stop()
        return

    _render_workflow(run)
    _section(
        "OBSERVATION 01",
        "Real TESS photometry",
        "Sanitized arrays exported by deterministic science code. Source FITS headers never cross into the UI.",
    )
    _render_light_curves(run)

    _section(
        "DETECTION 02",
        "Transit search and candidate",
        "Every displayed measurement is read from the locked result and its Evidence Ledger provenance.",
    )
    _render_candidate_metrics(run)
    _render_detection_plots(run)

    _section(
        "ADVERSARIAL INQUIRY 03",
        "Competing explanations",
        "Evidence states and weights come from declared deterministic update rules; weights are uncalibrated and are not probabilities.",
    )
    _render_hypotheses_and_agents(run)

    _section(
        "VETTING 04",
        "Falsification diagnostics",
        "Mandatory baseline tests execute in code. The adaptive panel appears only when the recorded trajectory requested it.",
    )
    _render_vetting(run)

    _section(
        "AUDIT 05",
        "Evidence Ledger and trajectory",
        f"{len(run.evidence)} hash-chained scientific records currently anchor the investigation.",
    )
    _render_audit(run)

    _section(
        "COMMIT 06",
        "Pre-reveal result lock",
        "The scientific disposition is serialized and hashed before any external catalog fact becomes visible.",
    )
    _render_lock(run)

    _section(
        "REVEAL 07",
        "Independent catalog comparison",
        "Catalog status is an external fact, kept separate from ExoSwarm’s locked photometric disposition.",
    )
    _render_reveal(run)

    _section(
        "LIMITS 08",
        "What this investigation cannot establish",
        "Photometric vetting is useful evidence, not professional confirmation.",
    )
    _render_limitations(run)


def _render_hero(run: MissionControlRun) -> None:
    if run.phase is RunPhase.GROUND_TRUTH_REVEALED:
        gate_label, gate_class = "Ground truth · revealed", "good"
    else:
        gate_label, gate_class = "Ground truth · locked", "sealed"
    investigation_label = {
        RunPhase.EMPTY: "Investigation · awaiting run",
        RunPhase.ACTIVE: "Investigation · active",
        RunPhase.RESULT_LOCKED: "Investigation · result locked",
        RunPhase.GROUND_TRUTH_REVEALED: "Investigation · complete",
        RunPhase.CORRUPT: "Investigation · integrity alert",
    }[run.phase]
    investigation_class = {
        RunPhase.EMPTY: "warn",
        RunPhase.ACTIVE: "live",
        RunPhase.RESULT_LOCKED: "good",
        RunPhase.GROUND_TRUTH_REVEALED: "good",
        RunPhase.CORRUPT: "bad",
    }[run.phase]
    trace = run.trace_id or "trace pending"
    replay = "AUDIT REPLAY" if run.lock_verified else "PERSISTED MISSION FEED"
    st.markdown(
        f"""
        <div class="exo-hero">
          <div>
            <div class="exo-eyebrow">{replay}</div>
            <div class="exo-target">{html.escape(run.opaque_target_id)}</div>
            <div class="exo-trace">{html.escape(trace)}</div>
          </div>
          <div class="exo-statuses">
            <span class="exo-chip {gate_class}">{gate_label}</span>
            <span class="exo-chip {investigation_class}">{investigation_label}</span>
            <span class="exo-chip">science · deterministic</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_issues(run: MissionControlRun) -> None:
    errors = [item for item in run.issues if item.severity is IssueSeverity.ERROR]
    warnings = [item for item in run.issues if item.severity is IssueSeverity.WARNING]
    for issue in errors:
        st.error(f"{issue.code} · {issue.message}", icon="⚠️")
    if warnings:
        with st.expander(f"Artifact notices · {len(warnings)}", expanded=False):
            for issue in warnings:
                st.warning(f"{issue.code} · {issue.message}", icon="ℹ️")


def _render_empty(run: MissionControlRun) -> None:
    st.markdown(
        f"""
        <div class="exo-empty">
          <div class="exo-empty-orbit">◎</div>
          <div class="exo-empty-title">No persisted investigation for {html.escape(run.opaque_target_id)}</div>
          <div class="exo-empty-copy">
            Run the deterministic investigation from the ExoSwarm CLI, then refresh this console.
            Mission Control will populate only after real artifacts appear in the configured runs directory.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_corrupt_stop() -> None:
    st.markdown(
        """
        <div class="exo-empty">
          <div class="exo-empty-orbit">◇</div>
          <div class="exo-empty-title">Core run artifacts are not trustworthy</div>
          <div class="exo-empty-copy">
            Scientific values and catalog identity are withheld. Repair or reproduce the run before using this investigation in a demo.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow(run: MissionControlRun) -> None:
    experiments = {item.experiment_type.value for item in run.evidence}
    mandatory = {
        ExperimentType.SIGNAL_QUALITY.value,
        ExperimentType.ODD_EVEN.value,
        ExperimentType.SECONDARY_ECLIPSE.value,
        ExperimentType.CONTAMINATION_SCREEN.value,
    }
    event_types = {item.event_type.value for item in run.trace}
    conditions = (
        bool(experiments & {"load_cached_data", "quality_inspection", "normalization"}),
        ExperimentType.TRANSIT_SEARCH.value in experiments,
        mandatory <= experiments,
        "AGENT_DECISION" in event_types,
        "CRITIC_DECISION" in event_types,
        bool(adaptive_experiments(run)),
        run.lock_verified,
        run.ground_truth_visible,
    )
    labels = (
        "Observation",
        "Transit search",
        "Mandatory vetting",
        "Skeptic choice",
        "Critic review",
        "Adaptive test",
        "Result lock",
        "Catalog reveal",
    )
    first_incomplete = next((index for index, done in enumerate(conditions) if not done), None)
    steps = []
    for index, (label, done) in enumerate(zip(labels, conditions, strict=True), start=1):
        state = "done" if done else "active" if first_incomplete == index - 1 else "pending"
        steps.append(
            f'<div class="exo-step {state}"><span class="exo-step-index">{index:02d}</span>{label}</div>'
        )
    st.markdown(f'<div class="exo-workflow">{"".join(steps)}</div>', unsafe_allow_html=True)


def _render_light_curves(run: MissionControlRun) -> None:
    roles = (
        ("raw_light_curve", "Raw observation", CYAN, "Flux [native product units]"),
        ("normalized_light_curve", "Quality-filtered / normalized", AMBER, "Relative flux"),
        ("cleaned_light_curve", "Detrended / cleaned", TEAL, "Relative flux"),
    )
    tabs = st.tabs([label for _, label, _, _ in roles])
    for tab, (role, label, color, y_title) in zip(tabs, roles, strict=True):
        with tab:
            _render_plot(
                run,
                role,
                lambda product, label=label, color=color, y_title=y_title: light_curve_figure(
                    product, title=label, color=color, y_title=y_title
                ),
                empty_message=f"{label} artifact has not been persisted for this run.",
            )


def _render_candidate_metrics(run: MissionControlRun) -> None:
    measurements = candidate_measurements(run)
    if not measurements:
        st.info("No viable candidate measurement is present in the persisted result.", icon="ℹ️")
        return
    first_row = st.columns(3)
    second_row = st.columns(3)
    for column, measurement in zip((*first_row, *second_row), measurements, strict=True):
        with column:
            _metric_card(measurement)
    candidate = run.result.candidate if run.result else None
    if candidate is not None:
        with st.expander("Measurement methods and provenance", expanded=False):
            rows = []
            for measurement in measurements:
                rows.append(
                    {
                        "measurement": measurement.label,
                        "uncertainty_kind": measurement.uncertainty_kind or "not recorded",
                        "uncertainty_method": measurement.uncertainty_method or "not recorded",
                        "sources": ", ".join(measurement.source_ids),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")


def _metric_card(measurement: MeasurementVM) -> None:
    value = _format_measurement(measurement.value, measurement.uncertainty)
    if measurement.uncertainty is None:
        detail = "Uncertainty not recorded for this metric"
    else:
        uncertainty = _format_number(measurement.uncertainty)
        uncertainty_unit = measurement.uncertainty_unit or measurement.unit
        kind = (measurement.uncertainty_kind or "uncertainty").replace("_", " ")
        detail = f"{kind} ± {uncertainty} {uncertainty_unit}"
    sources = " · ".join(measurement.source_ids)
    st.markdown(
        f"""
        <div class="exo-metric">
          <div class="exo-metric-label">{html.escape(measurement.label)}</div>
          <div class="exo-metric-value">{html.escape(value)}<span class="exo-metric-unit">{html.escape(measurement.unit)}</span></div>
          <div class="exo-metric-detail">{html.escape(detail)}</div>
          <div class="exo-provenance">SOURCE · {html.escape(sources)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detection_plots(run: MissionControlRun) -> None:
    columns = st.columns(2)
    period_measurement = next(
        (item for item in candidate_measurements(run) if item.key == "period_days"), None
    )
    candidate_period = float(period_measurement.value) if period_measurement is not None else None
    with columns[0]:
        _render_plot(
            run,
            "bls_periodogram",
            lambda product: bls_figure(product, candidate_period_days=candidate_period),
            empty_message="The BLS periodogram artifact has not been persisted.",
        )
    with columns[1]:
        _render_plot(
            run,
            "phase_folded",
            folded_figure,
            empty_message="The phase-folded candidate artifact has not been persisted.",
        )
        harmonic_resolution = next(
            (
                item
                for item in reversed(run.evidence)
                if item.experiment_type is ExperimentType.HARMONIC_TEST
                and item.interpretation_code.value
                in {"PREFERRED_HALF_PERIOD", "PREFERRED_DOUBLE_PERIOD"}
            ),
            None,
        )
        if harmonic_resolution is not None:
            st.caption(
                "This fold preserves the initial search ephemeris. The later harmonic experiment supersedes the locked candidate metrics shown above."
            )


def _render_hypotheses_and_agents(run: MissionControlRun) -> None:
    hypotheses = hypothesis_views(run.trace)
    current, alternative = working_hypotheses(hypotheses)
    left, right = st.columns([0.9, 1.35])
    with left:
        current_title = current.label if current else "Awaiting evidence"
        alternative_title = alternative.label if alternative else "Awaiting evidence"
        st.markdown(
            f"""
            <div class="exo-panel">
              <div class="exo-panel-label">Current working interpretation</div>
              <div class="exo-panel-title">{html.escape(current_title)}</div>
              <div class="exo-panel-label" style="margin-top:.9rem">Strongest non-planetary alternative</div>
              <div class="exo-panel-title">{html.escape(alternative_title)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        rows = []
        for hypothesis in hypotheses:
            rows.append(
                f"""
                <div class="exo-hypothesis-row">
                  <span class="exo-h-name">{html.escape(hypothesis.label)}</span>
                  <span class="exo-h-state">{html.escape(hypothesis.state.replace("_", " "))}</span>
                  <span class="exo-h-weight">{hypothesis.weight:+.2f}</span>
                </div>
                """
            )
        st.markdown(
            '<div class="exo-panel" style="padding:.35rem .45rem">'
            + "".join(rows)
            + '<div class="exo-provenance" style="padding:.45rem .7rem">DECLARED HEURISTIC WEIGHT · NOT A PROBABILITY</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    skeptic = latest_skeptic_decision(run)
    critic = latest_critic_decision(run)
    director_column, skeptic_column, critic_column = st.columns(3)
    with director_column:
        disposition = run.result.disposition.value if run.result else "Investigation active"
        st.markdown(
            f"""
            <div class="exo-panel">
              <div class="exo-panel-label">Scientific Director · deterministic runtime</div>
              <div class="exo-panel-title">Owns permissions + budgets</div>
              <div class="exo-panel-copy">Enforces mandatory tests, validates tool requests, records evidence, and controls stopping and lock transitions.</div>
              <div class="exo-panel-code">{html.escape(disposition)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with skeptic_column:
        if skeptic is None:
            _pending_agent_card("Skeptic", "Awaiting a complete mandatory baseline")
        else:
            experiment = (skeptic.experiment or skeptic.action).replace("_", " ")
            st.markdown(
                f"""
                <div class="exo-panel">
                  <div class="exo-panel-label">Skeptic · challenge selected</div>
                  <div class="exo-panel-title">{html.escape(skeptic.hypothesis_label)}</div>
                  <div class="exo-panel-copy"><b>{html.escape(experiment.title())}</b><br>{html.escape(skeptic.explanation)}</div>
                  <div class="exo-panel-code">{html.escape(skeptic.reason_code)}</div>
                  <div class="exo-provenance">DECISION SOURCE · {html.escape(skeptic.decision_source)}<br>{html.escape(skeptic.provider)} · {html.escape(skeptic.model)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Decision utility · {skeptic.decision_utility:.2f} · uncalibrated · priority {skeptic.priority}"
            )
            st.caption(_agent_call_caption(skeptic.attempts, skeptic.provider_request_ids))
            with st.expander("Expected discriminating result", expanded=False):
                st.write(skeptic.expected_result)
                for outcome, implication in skeptic.predicted_outcomes:
                    st.markdown(f"`{outcome}` → {implication}")
    with critic_column:
        if critic is None:
            _pending_agent_card("Critic", "No adaptive proposal has required review")
        else:
            verdict_class = (
                critic.verdict if critic.verdict in {"APPROVE", "REVISE", "VETO"} else ""
            )
            st.markdown(
                f"""
                <div class="exo-panel">
                  <div class="exo-panel-label">Critic · experiment review</div>
                  <div class="exo-verdict {verdict_class}">{html.escape(critic.verdict)}</div>
                  <div class="exo-panel-copy">{html.escape(critic.reason)}</div>
                  <div class="exo-panel-code">{html.escape(critic.reason_code)}</div>
                  <div class="exo-provenance">DECISION SOURCE · {html.escape(critic.decision_source)}<br>{html.escape(critic.provider)} · {html.escape(critic.model)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(_agent_call_caption(critic.attempts, critic.provider_request_ids))


def _pending_agent_card(role: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="exo-panel">
          <div class="exo-panel-label">{html.escape(role)}</div>
          <div class="exo-panel-title">Standing by</div>
          <div class="exo-panel-copy">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _agent_call_caption(attempts: int, request_ids: tuple[str, ...]) -> str:
    request_text = ", ".join(request_ids) if request_ids else "none recorded"
    return f"Structured attempts · {attempts} · provider request IDs · {request_text}"


def _render_vetting(run: MissionControlRun) -> None:
    completed = {item.experiment_type for item in run.evidence}
    baseline = (
        (ExperimentType.ODD_EVEN, "Odd / even"),
        (ExperimentType.SECONDARY_ECLIPSE, "Secondary eclipse"),
        (ExperimentType.CONTAMINATION_SCREEN, "Contamination screen"),
        (ExperimentType.SIGNAL_QUALITY, "Signal quality"),
    )
    columns = st.columns(4)
    for column, (experiment, label) in zip(columns, baseline, strict=True):
        status = "COMPLETE" if experiment in completed else "PENDING"
        chip_class = "good" if experiment in completed else "warn"
        with column:
            st.markdown(
                f"""
                <div class="exo-panel" style="min-height:88px">
                  <div class="exo-panel-label">Mandatory</div>
                  <div class="exo-panel-title" style="font-size:.85rem">{label}</div>
                  <span class="exo-chip {chip_class}" style="display:inline-block;margin-top:.45rem">{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tabs = st.tabs(["Odd / even", "Secondary event", "Harmonic branch", "Centroid branch"])
    with tabs[0]:
        _render_plot(
            run,
            "odd_even",
            odd_even_figure,
            empty_message="No odd/even plot artifact is available.",
        )
    with tabs[1]:
        _render_plot(
            run,
            "secondary_eclipse",
            secondary_figure,
            empty_message="No secondary-event plot artifact is available.",
        )
    with tabs[2]:
        _render_conditional_plot(
            run,
            "harmonic_test",
            harmonic_figure,
            "The harmonic test was not selected on this trajectory.",
        )
    with tabs[3]:
        _render_conditional_plot(
            run,
            "centroid_localization",
            centroid_figure,
            "Centroid localization was not selected on this trajectory.",
        )

    branches = adaptive_experiments(run)
    if branches:
        branch_labels = " → ".join(item.replace("_", " ").upper() for item in branches)
        st.info(f"Recorded adaptive branch · {branch_labels}", icon="🧭")


def _render_conditional_plot(
    run: MissionControlRun,
    role: str,
    builder: Callable[[ScienceProduct], Any],
    unselected_message: str,
) -> None:
    evidence_for_role = any(item.experiment_type.value == role for item in run.evidence)
    if not evidence_for_role:
        st.caption(unselected_message)
        return
    _render_plot(
        run,
        role,
        builder,
        empty_message=f"The {role} result exists but its plot artifact is unavailable.",
    )


def _render_plot(
    run: MissionControlRun,
    role: str,
    builder: Callable[[ScienceProduct], Any],
    *,
    empty_message: str,
) -> None:
    product = None
    error: Exception | None = None
    try:
        product = load_science_product(run, role)
        if product is not None:
            figure = builder(product)
            st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)
            _artifact_caption(product.source)
            return
    except (OSError, ValueError, TypeError) as exc:
        error = exc
    fallback = image_artifact(run, role)
    if fallback is not None:
        st.image(str(fallback.path), width="stretch")
        _artifact_caption(fallback)
        if error is not None:
            st.caption(
                f"Interactive artifact unavailable; showing deterministic PNG ({type(error).__name__})."
            )
        return
    st.markdown(
        f"""
        <div class="exo-sealed" style="border-color:rgba(117,157,197,.18);background:none">
          <div class="exo-sealed-title" style="color:#8190a9">Artifact unavailable</div>
          <div class="exo-sealed-copy">{html.escape(empty_message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _artifact_caption(artifact: Any) -> None:
    integrity = "SHA-256 verified" if artifact.integrity_verified else "manifest hash unavailable"
    st.caption(f"Deterministic artifact · `{artifact.artifact_id}` · {integrity}")


def _render_audit(run: MissionControlRun) -> None:
    ledger_tab, trace_tab = st.tabs(["Evidence board", "Trajectory trace"])
    with ledger_tab:
        if not run.evidence:
            st.info("The Evidence Ledger has no validated records yet.", icon="ℹ️")
        for item in run.evidence:
            tone = evidence_tone(item)
            headline = evidence_headline(item)
            agent_link = "adaptive" if item.agent_request_id else "mandatory / director"
            st.markdown(
                f"""
                <div class="exo-evidence {tone}">
                  <div class="exo-evidence-head">
                    <span class="exo-evidence-title">{html.escape(headline)}</span>
                    <span class="exo-evidence-id">{html.escape(item.id)}</span>
                  </div>
                  <div class="exo-evidence-meta">{html.escape(item.tool_name)} · v{html.escape(item.tool_version)} · {agent_link}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Inspect {item.id}", expanded=False):
                rows = []
                for name, value in item.numerical_results.items():
                    uncertainty = item.uncertainties.get(name)
                    rows.append(
                        {
                            "measurement": name,
                            "value": value,
                            "unit": item.result_units.get(name, ""),
                            "uncertainty/tolerance": uncertainty.value if uncertainty else None,
                            "uncertainty method": uncertainty.method if uncertainty else "",
                        }
                    )
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch")
                else:
                    st.caption("Categorical diagnostic; no numerical measurement was recorded.")
                if item.quality_flags:
                    st.markdown(
                        "Quality flags · "
                        + ", ".join(f"`{flag.code}`" for flag in item.quality_flags)
                    )
                if item.limitations:
                    st.markdown("Limitations")
                    for limitation in item.limitations:
                        st.markdown(f"- {limitation}")
                st.caption(f"Record hash · {item.record_hash}")
    with trace_tab:
        safe_events = [
            event
            for event in run.trace
            if event.event_type.value not in {"AGENT_REQUEST", "AGENT_RESPONSE"}
        ]
        rows = [
            {
                "seq": event.sequence,
                "time (UTC)": event.timestamp.isoformat(),
                "event": event.event_type.value,
                "event id": event.event_id,
            }
            for event in safe_events
        ]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", height=390)
            st.caption(
                "Audit replay omits raw provider request/response text. Structured decisions are shown in the agent panels above."
            )
        else:
            st.info("No validated trace events are available.", icon="ℹ️")


def _render_lock(run: MissionControlRun) -> None:
    if run.lock_verified and run.result is not None and run.lock_sha256 is not None:
        created = run.result.created_at.isoformat()
        st.markdown(
            f"""
            <div class="exo-lock">
              <div class="exo-lock-title">✓ RESULT LOCKED</div>
              <div class="exo-lock-copy">Pre-reveal disposition serialized · Evidence Ledger root committed · catalog capability remained sealed at commit time.</div>
              <div class="exo-hash">SHA-256 · {html.escape(run.lock_sha256)}</div>
              <div class="exo-provenance">CREATED · {html.escape(created)} · {html.escape(run.result.trace_id)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Locked disposition:** `{run.result.disposition.value}`  \n"
            f"Evidence root: `{run.result.evidence_root_hash}`"
        )
    elif run.phase is RunPhase.CORRUPT:
        st.error("Result lock could not be verified. Reveal remains unavailable.", icon="⚠️")
    else:
        st.markdown(
            """
            <div class="exo-sealed">
              <div class="exo-sealed-title">Lock pending</div>
              <div class="exo-sealed-copy">The investigation is still building its evidence record. No pre-reveal scientific result has been committed.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_reveal(run: MissionControlRun) -> None:
    if not run.ground_truth_visible or run.reveal is None or run.result is None:
        message = (
            "A verified result exists, but no separately persisted reveal artifact is present. "
            "Identity and catalog measurements remain hidden."
            if run.lock_verified
            else "Identity, catalog status, and known parameters are mechanically unavailable before a valid result lock."
        )
        st.markdown(
            f"""
            <div class="exo-sealed">
              <div class="exo-sealed-title">◆ Catalog capability sealed</div>
              <div class="exo-sealed-copy">{html.escape(message)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    truth = run.reveal.ground_truth
    st.markdown(
        f"""
        <div class="exo-reveal">
          <div class="exo-reveal-kicker">Ground truth revealed after lock</div>
          <div class="exo-reveal-name">{html.escape(truth.actual_target_identity)}</div>
          <div class="exo-reveal-status">{html.escape(truth.catalog_name)} · {html.escape(truth.catalog_status)}</div>
          <div class="exo-provenance">LOCKED RESULT · {html.escape(run.reveal.locked_result_sha256)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    exoswarm_column, catalog_column = st.columns(2)
    with exoswarm_column:
        st.markdown("#### Locked ExoSwarm measurements")
        exoswarm_rows = [
            {
                "measurement": measurement.label,
                "value": measurement.value,
                "unit": measurement.unit,
                "source": "result.json / " + ", ".join(measurement.source_ids[1:]),
            }
            for measurement in candidate_measurements(run)
        ]
        if exoswarm_rows:
            st.dataframe(exoswarm_rows, hide_index=True, width="stretch")
        else:
            st.caption("The locked result contains no candidate measurements.")
        st.caption(f"Independent disposition · {run.result.disposition.value}")
    with catalog_column:
        st.markdown("#### External catalog record")
        catalog_rows = [
            {
                "measurement": measurement.name,
                "value": measurement.value,
                "unit": measurement.unit,
                "uncertainty": measurement.uncertainty,
                "source": measurement.source_field or "reveal.json",
            }
            for measurement in catalog_measurements(run)
        ]
        if catalog_rows:
            st.dataframe(catalog_rows, hide_index=True, width="stretch")
        else:
            st.caption("No comparable numerical catalog measurements were recorded.")
        st.caption(f"External status · {truth.catalog_status}")

    st.caption(
        "Transit epochs may name different transit cycles. Compare epochs modulo the recorded orbital period rather than treating a raw epoch subtraction as a disagreement."
    )

    if "SURVIVES IMPLEMENTED VETTING" in run.result.disposition.value:
        st.success(
            "ExoSwarm independently recovered the transit-like signal and found that the planetary interpretation survived its implemented photometric vetting. Only afterward was the external catalog status revealed.",
            icon="✅",
        )
    else:
        st.info(
            "ExoSwarm locked its independent photometric disposition before external catalog status was accessed. The catalog record shown here was not an input to the investigation.",
            icon="ℹ️",
        )


def _render_limitations(run: MissionControlRun) -> None:
    limitations = list(run.result.limitations) if run.result else []
    defaults = [
        "Limited photometric and centroid vetting cannot exclude every astrophysical false-positive scenario.",
        "This investigation does not replace follow-up observations or professional statistical validation.",
        "External catalog confirmation and ExoSwarm’s locked disposition are separate concepts.",
    ]
    for limitation in defaults:
        if limitation not in limitations:
            limitations.append(limitation)
    for limitation in limitations:
        st.markdown(f"- {limitation}")


def _section(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="exo-section">
          <div class="exo-section-kicker">{html.escape(kicker)}</div>
          <div class="exo-section-title">{html.escape(title)}</div>
          <div class="exo-section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_measurement(value: int | float, uncertainty: float | None) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if uncertainty is not None and uncertainty > 0 and math.isfinite(uncertainty):
        places = max(0, min(9, -math.floor(math.log10(uncertainty)) + 1))
        return f"{value:,.{places}f}"
    return _format_number(value)


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.7g}"


def _safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


if __name__ == "__main__":
    main()
