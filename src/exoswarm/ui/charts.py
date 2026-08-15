"""Plotly figures built exclusively from deterministic science artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .artifacts import ScienceProduct

INK = "#E9F2FF"
MUTED = "#8190A9"
GRID = "rgba(120, 153, 190, 0.12)"
CYAN = "#41D9FF"
TEAL = "#49F2C2"
AMBER = "#FFBE55"
CORAL = "#FF6577"
PURPLE = "#A58BFF"


def light_curve_figure(
    product: ScienceProduct,
    *,
    title: str,
    color: str = CYAN,
    y_title: str = "Relative flux",
) -> go.Figure:
    time = _array(product, "time_btjd", "time", "btjd")
    flux = _array(product, "flux", "normalized_flux", "clean_flux", "detrended_flux")
    time, flux = _finite_pairs(time, flux)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=flux,
            mode="markers",
            marker={"size": 2.4, "color": color, "opacity": 0.58},
            name=title,
            hovertemplate="BTJD %{x:.5f}<br>Flux %{y:.7g}<extra></extra>",
        )
    )
    trend = _array(product, "trend")
    if trend is not None and len(trend) == len(_array(product, "time_btjd", "time", "btjd")):
        raw_time = _array(product, "time_btjd", "time", "btjd")
        mask = np.isfinite(raw_time) & np.isfinite(trend)
        figure.add_trace(
            go.Scattergl(
                x=raw_time[mask],
                y=trend[mask],
                mode="lines",
                line={"width": 1.5, "color": AMBER},
                name="Recorded trend",
                hovertemplate="BTJD %{x:.5f}<br>Trend %{y:.7f}<extra></extra>",
            )
        )
    _style(figure, title=title, x_title="Time [BTJD]", y_title=y_title)
    return figure


def bls_figure(product: ScienceProduct, *, candidate_period_days: float | None) -> go.Figure:
    period = _array(product, "period_days", "period")
    power = _array(product, "power_snr", "power", "bls_power", "snr")
    period, power = _finite_pairs(period, power)
    figure = go.Figure(
        go.Scattergl(
            x=period,
            y=power,
            mode="lines",
            line={"width": 1.35, "color": CYAN},
            fill="tozeroy",
            fillcolor="rgba(65, 217, 255, 0.08)",
            hovertemplate="Period %{x:.7g} d<br>Search statistic %{y:.6g}<extra></extra>",
            name="BLS search",
        )
    )
    if candidate_period_days is not None:
        figure.add_vline(
            x=candidate_period_days,
            line_width=1.4,
            line_dash="dash",
            line_color=AMBER,
            annotation_text="locked candidate",
            annotation_font_color=AMBER,
            annotation_position="top right",
        )
    _style(
        figure,
        title="Box Least Squares search",
        x_title="Trial period [days]",
        y_title="Recorded BLS statistic",
    )
    return figure


def folded_figure(product: ScienceProduct) -> go.Figure:
    phase = _array(product, "phase")
    flux = _array(product, "flux", "folded_flux")
    phase, flux = _finite_pairs(phase, flux)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=phase,
            y=flux,
            mode="markers",
            marker={"size": 2.7, "color": "rgba(129, 144, 169, 0.34)"},
            name="Cadences",
            hovertemplate="Phase %{x:.5f}<br>Relative flux %{y:.7f}<extra></extra>",
        )
    )
    bin_phase = _array(product, "bin_phase", "binned_phase")
    bin_flux = _array(product, "bin_flux", "binned_flux")
    if bin_phase is not None and bin_flux is not None:
        bin_phase, bin_flux = _finite_pairs(bin_phase, bin_flux)
        figure.add_trace(
            go.Scatter(
                x=bin_phase,
                y=bin_flux,
                mode="markers+lines",
                marker={"size": 6, "color": CYAN, "line": {"width": 1, "color": "#C9F6FF"}},
                line={"width": 1, "color": "rgba(65, 217, 255, 0.48)"},
                name="Binned data",
                hovertemplate="Phase %{x:.5f}<br>Binned flux %{y:.7f}<extra></extra>",
            )
        )
    model = _array(product, "model_flux", "model")
    if model is not None:
        model_phase = None
        if bin_phase is not None and len(model) == len(bin_phase):
            model_phase = bin_phase
        elif len(model) == len(phase):
            model_phase = phase
        if model_phase is not None:
            order = np.argsort(model_phase)
            figure.add_trace(
                go.Scatter(
                    x=model_phase[order],
                    y=model[order],
                    mode="lines",
                    line={"width": 2.0, "color": AMBER},
                    name="Deterministic BLS model",
                    hovertemplate="Phase %{x:.5f}<br>Model flux %{y:.7f}<extra></extra>",
                )
            )
    _style(
        figure,
        title="Initial search-ephemeris phase fold",
        x_title="Orbital phase",
        y_title="Relative flux",
    )
    return figure


def odd_even_figure(product: ScienceProduct) -> go.Figure:
    events = _array(product, "event_number", "transit_number")
    depth = _array(product, "event_depth_ppm", "depth_ppm")
    uncertainty = _array(product, "event_depth_uncertainty_ppm", "depth_uncertainty_ppm")
    parity = _array(product, "event_parity", "parity")
    if events is None or depth is None:
        raise ValueError("odd/even artifact lacks event number or depth")
    if parity is None or len(parity) != len(events):
        parity = np.where(np.asarray(events, dtype=int) % 2 == 0, "even", "odd")
    figure = go.Figure()
    parity_values = np.asarray(parity)
    if np.issubdtype(parity_values.dtype, np.number):
        parity_text = np.where(parity_values.astype(int) % 2 == 0, "even", "odd")
    else:
        parity_text = parity_values.astype(str)
    for label, color in (("odd", PURPLE), ("even", TEAL)):
        mask = np.char.lower(parity_text) == label
        error_y = None
        if uncertainty is not None and len(uncertainty) == len(events):
            error_y = {"type": "data", "array": uncertainty[mask], "visible": True}
        figure.add_trace(
            go.Scatter(
                x=events[mask],
                y=depth[mask],
                error_y=error_y,
                mode="markers",
                marker={"size": 8, "color": color},
                name=label.title(),
                hovertemplate="Event %{x}<br>Depth %{y:.7g} ppm<extra></extra>",
            )
        )
    _style(
        figure,
        title="Odd / even transit comparison",
        x_title="Observed event number",
        y_title="Recorded depth [ppm]",
    )
    return figure


def secondary_figure(product: ScienceProduct) -> go.Figure:
    phase = _array(product, "phase")
    flux = _array(product, "flux")
    phase, flux = _finite_pairs(phase, flux)
    figure = go.Figure(
        go.Scattergl(
            x=phase,
            y=flux,
            mode="markers",
            marker={"size": 2.5, "color": "rgba(129, 144, 169, 0.30)"},
            name="Cadences",
            hovertemplate="Phase %{x:.5f}<br>Relative flux %{y:.7f}<extra></extra>",
        )
    )
    bin_phase = _array(product, "bin_phase", "binned_phase")
    bin_flux = _array(product, "bin_flux", "binned_flux")
    if bin_phase is not None and bin_flux is not None:
        bin_phase, bin_flux = _finite_pairs(bin_phase, bin_flux)
        figure.add_trace(
            go.Scatter(
                x=bin_phase,
                y=bin_flux,
                mode="lines+markers",
                line={"width": 1.4, "color": CYAN},
                marker={"size": 5, "color": CYAN},
                name="Binned data",
                hovertemplate="Phase %{x:.5f}<br>Binned flux %{y:.7f}<extra></extra>",
            )
        )
    secondary_phase = -0.5 if float(np.nanmin(phase)) < 0 else 0.5
    figure.add_vline(
        x=secondary_phase,
        line_width=1.2,
        line_dash="dot",
        line_color=AMBER,
        annotation_text="secondary phase",
        annotation_font_color=AMBER,
    )
    _style(
        figure,
        title="Secondary-event test",
        x_title="Orbital phase",
        y_title="Relative flux",
    )
    return figure


def harmonic_figure(product: ScienceProduct) -> go.Figure:
    periods = _array(product, "tested_period_days", "period_days")
    scores = _array(product, "bls_snr", "score", "power_snr")
    periods, scores = _finite_pairs(periods, scores)
    figure = go.Figure(
        go.Bar(
            x=periods,
            y=scores,
            marker={
                "color": [PURPLE, CYAN, AMBER][: len(periods)],
                "line": {"width": 1, "color": "rgba(255,255,255,.25)"},
            },
            hovertemplate="Tested period %{x:.7g} d<br>Recorded score %{y:.6g}<extra></extra>",
            name="Harmonic comparison",
        )
    )
    _style(
        figure,
        title="P/2 · P · 2P alias test",
        x_title="Tested period [days]",
        y_title="Recorded BLS score",
    )
    return figure


def centroid_figure(product: ScienceProduct) -> go.Figure:
    panels: list[tuple[str, np.ndarray, str]] = []
    for title, key, colorscale in (
        ("Out of transit", "out_of_transit_image", "Viridis"),
        ("In transit", "in_transit_image", "Viridis"),
        ("Difference", "difference_image", "RdBu"),
    ):
        image = _array(product, key)
        if image is not None and image.ndim == 2:
            panels.append((title, image, colorscale))
    if not panels:
        raise ValueError("centroid artifact has no image planes")
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[item[0] for item in panels],
        horizontal_spacing=0.045,
    )
    for index, (_, image, colorscale) in enumerate(panels, start=1):
        figure.add_trace(
            go.Heatmap(
                z=image,
                colorscale=colorscale,
                showscale=index == len(panels),
                colorbar={"title": "flux"} if index == len(panels) else None,
                hovertemplate="column %{x}<br>row %{y}<br>value %{z:.6g}<extra></extra>",
            ),
            row=1,
            col=index,
        )
    target = _array(product, "target_xy")
    centroid = _array(product, "difference_centroid_xy", "centroid_xy")
    target_column = 1
    difference_column = len(panels)
    if target is not None and target.size >= 2:
        figure.add_trace(
            go.Scatter(
                x=[target.flat[0]],
                y=[target.flat[1]],
                mode="markers",
                marker={"symbol": "x", "size": 12, "color": TEAL, "line": {"width": 2}},
                name="Target position",
                hovertemplate="Target x %{x:.4g}<br>Target y %{y:.4g}<extra></extra>",
            ),
            row=1,
            col=target_column,
        )
        if difference_column != target_column:
            figure.add_trace(
                go.Scatter(
                    x=[target.flat[0]],
                    y=[target.flat[1]],
                    mode="markers",
                    marker={"symbol": "x", "size": 12, "color": TEAL, "line": {"width": 2}},
                    name="Target position",
                    showlegend=False,
                    hovertemplate="Target x %{x:.4g}<br>Target y %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=difference_column,
            )
    if centroid is not None and centroid.size >= 2:
        figure.add_trace(
            go.Scatter(
                x=[centroid.flat[0]],
                y=[centroid.flat[1]],
                mode="markers",
                marker={"symbol": "circle-open", "size": 13, "color": AMBER, "line": {"width": 2}},
                name="Difference centroid",
                hovertemplate="Centroid x %{x:.4g}<br>Centroid y %{y:.4g}<extra></extra>",
            ),
            row=1,
            col=difference_column,
        )
    figure.update_yaxes(scaleanchor="x")
    _style(figure, title="Transit-associated centroid localization", height=430)
    return figure


def _array(product: ScienceProduct, *names: str) -> np.ndarray | None:
    for name in names:
        if name in product.arrays:
            return np.asarray(product.arrays[name]).squeeze()
    return None


def _finite_pairs(
    first: np.ndarray | None, second: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if first is None or second is None:
        raise ValueError("science artifact lacks a required array")
    first = np.asarray(first)
    second = np.asarray(second)
    if first.ndim != 1 or second.ndim != 1 or len(first) != len(second):
        raise ValueError("science artifact arrays have incompatible shapes")
    mask = np.isfinite(first.astype(float)) & np.isfinite(second.astype(float))
    if not np.any(mask):
        raise ValueError("science artifact has no finite samples")
    return first[mask], second[mask]


def _style(
    figure: go.Figure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 390,
) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.015, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 56, "r": 26, "t": 58, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3, 8, 16, 0.20)",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": INK, "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 0.99,
            "font": {"color": MUTED, "size": 11},
        },
        hoverlabel={"bgcolor": "#0B1422", "font_color": INK, "bordercolor": CYAN},
        hovermode="closest",
    )
    figure.update_xaxes(
        title_text=x_title,
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="rgba(129,144,169,.22)",
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
    )
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="rgba(129,144,169,.22)",
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
    )


def has_required_arrays(product: ScienceProduct, names: Iterable[str]) -> bool:
    return all(name in product.arrays for name in names)


__all__ = [
    "AMBER",
    "CORAL",
    "CYAN",
    "MUTED",
    "PURPLE",
    "TEAL",
    "bls_figure",
    "centroid_figure",
    "folded_figure",
    "harmonic_figure",
    "has_required_arrays",
    "light_curve_figure",
    "odd_even_figure",
    "secondary_figure",
]
