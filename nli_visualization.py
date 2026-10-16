"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import nli_constants as C
from nli_evaluation import Analysis, de, rho_label


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Wahrheit x": a.x[:k], "Start": a.x0[:k], "Lösung": a.run.x[:k],
                         "linearisiertes Modell": a.x_lin[:k], "Daten b": a.b[:k]})


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Start, Lösung des gewählten Lösers, Lösung des Nullstarts und das linearisierte Modell; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    if a.settings.amp > 0.0:
        fig.add_trace(go.Scatter(x=a.s, y=a.x0, mode="lines", name="Start", line=dict(color=C.COLOR_START, width=1.5, dash="dot"), hoverinfo="skip"))
        if a.settings.solver != "lm" or a.settings.iters < 200:
            fig.add_trace(go.Scatter(x=a.s, y=a.ref.x, mode="lines", name="Levenberg-Marquardt vom Nullstart", line=dict(color=C.COLOR_EXACT, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_lin, mode="lines", name=f"linearisiertes Modell (λ = {de(a.lam_lin, 2)})", line=dict(color=C.COLOR_LINEAR, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.run.x, mode="lines+markers", name=f"Lösung ({C.SOLVER_SHORT[a.settings.solver]})", line=dict(color=C.COLOR_SOLUTION, width=2.5),
                             marker=dict(size=5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def _convergence(a: Analysis, key: str, title: str, floor: float) -> go.Figure:
    fig = go.Figure()
    for mode in C.SOLVERS:
        r = a.runs[mode]
        vals = r.phi if key == "phi" else [g / r.gnorm[0] for g in r.gnorm]
        fig.add_trace(go.Scatter(x=list(range(len(vals))), y=[max(v, floor) for v in vals], mode="lines+markers" if len(vals) < 40 else "lines", name=C.SOLVER_SHORT[mode],
                                 line=dict(color=C.SOLVER_COLORS[mode], width=3 if mode == a.settings.solver else 1.5), marker=dict(size=5)))
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title=title, type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_phi_chart(a: Analysis) -> go.Figure:
    """Φ über die Iterationen für alle vier Löser vom selben Start (der gewählte dick)."""
    return _convergence(a, "phi", "Φ(x)", 1e-12)


def build_grad_chart(a: Analysis) -> go.Figure:
    """Gradientennorm relativ zum Start über die Iterationen; der Abbruch liegt bei 10⁻⁸."""
    fig = _convergence(a, "grad", "|∇Φ| / |∇Φ₀|", 1e-14)
    fig.add_hline(y=C.GN_TOL, line=dict(color=C.COLOR_GREY, dash="dot"), annotation_text="Abbruch", annotation_position="bottom right")
    return fig


def build_gradcheck_chart(a: Analysis) -> go.Figure:
    """Adjungierter Gradient und Differenzenquotient je Koordinate am Prüfpunkt 0,8 · Wahrheit + 0,05."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.g_fd, mode="markers", name="Differenzenquotient (2n Vorwärtsläufe)", marker=dict(color=C.COLOR_LINEAR, size=9, symbol="circle-open", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=a.s, y=a.g_adj, mode="lines", name="adjungiert (1 Vorwärts- und 1 Rückwärtslauf)", line=dict(color=C.COLOR_SOLUTION, width=2.5)))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="∂(½|F(x) − b|²)/∂xᵢ")
    return _fixed(fig, 340)


def build_adjoint_chart(res: dict) -> go.Figure:
    """Vorwärtsläufe je Gradient über n: adjungiert gegen Differenzenquotient (logarithmisch)."""
    rows = [r for r in res["rows"] if r["rho"] == res["rows"][0]["rho"]]
    xs = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["nf_adj"] for r in rows], name="adjungiert", marker_color=C.COLOR_SOLUTION, text=[str(r["nf_adj"]) for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=xs, y=[r["nf_fd"] for r in rows], name="Differenzenquotient", marker_color=C.COLOR_LINEAR, text=[str(r["nf_fd"]) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Zahl der Unbekannten n", type="category")
    fig.update_yaxes(title="Vorwärtsläufe je Gradient", type="log", range=[-0.2, 3.2])
    return _fixed(fig, 320)


def build_ignore_chart(res: dict) -> go.Figure:
    """Bester Fehler über λ je Reaktionsstärke: linearisiertes gegen nichtlineares Modell."""
    rows = res["rows"]
    xs = [rho_label(r["rho"]) for r in rows]
    fig = go.Figure()
    for key, name, color in (("lin", "linearisiertes Modell (ρ = 0 angenommen)", C.COLOR_LINEAR), ("nl", "nichtlineares Modell (Levenberg-Marquardt)", C.COLOR_SOLUTION)):
        fig.add_trace(go.Bar(x=xs, y=[100.0 * r[key] for r in rows], name=name, marker_color=color, text=[f"{de(100 * r[key], 1)} %" for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Reaktionsstärke ρ", type="category")
    fig.update_yaxes(title="Bester RMS-Fehler über λ (% der Signalnorm)", range=[0, 80])
    return _fixed(fig, 360)


def build_solver_chart(res: dict) -> go.Figure:
    """Iterationen bis zum Abbruch je Löser und Reaktionsstärke (logarithmisch); nicht konvergierte Läufe sind schraffiert."""
    fig = go.Figure()
    rhos = sorted({r["rho"] for r in res["rows"]})
    for mode in C.SOLVERS:
        rows = [next(r for r in res["rows"] if r["rho"] == rho and r["solver"] == mode) for rho in rhos]
        fig.add_trace(go.Bar(x=[rho_label(v) for v in rhos], y=[r["iterations"] for r in rows], name=C.SOLVER_SHORT[mode], marker_color=C.SOLVER_COLORS[mode],
                             marker_pattern_shape=["" if r["converged"] else "/" for r in rows], text=[str(r["iterations"]) for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Reaktionsstärke ρ (Nullstart, λ = 10⁻³)", type="category")
    fig.update_yaxes(title="Iterationen bis zum Abbruch", type="log", range=[0, 3.2])
    return _fixed(fig, 360)


def build_basin_chart(res: dict, rho: float) -> go.Figure:
    """Anteil der Starts je Amplitude, die dasselbe Minimum wie der Nullstart erreichen."""
    rows = [r for r in res["basin"] if r["rho"] == rho]
    fig = go.Figure()
    xs = [de(r["amp"], 2) for r in rows]
    for mode, name in (("lm", "Levenberg-Marquardt"), ("gn", "Gauss-Newton mit Schrittweitensuche"), ("gn_pur", "Gauss-Newton pur")):
        fig.add_trace(go.Scatter(x=xs, y=[r[mode] / r["starts"] for r in rows], mode="lines+markers", name=name, line=dict(color=C.SOLVER_COLORS[mode], width=3 if mode == "lm" else 2),
                                 marker=dict(size=8)))
    fig.update_xaxes(title=f"Amplitude des Starts (ρ = {rho_label(rho)})", type="category")
    fig.update_yaxes(title="Anteil im selben Minimum", range=[-0.05, 1.05], tickformat=".0%")
    return _fixed(fig, 340)


def build_laplace_chart(res: dict) -> go.Figure:
    """sd der Laplace-Näherung geteilt durch sd der Ketten je Reaktionsstärke (Median; Balken: kleinster bis größter Wert)."""
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[rho_label(r["rho"]) for r in rows], y=[r["ratio_median"] for r in rows], marker_color=C.COLOR_LAPLACE,
                         error_y=dict(type="data", symmetric=False, array=[r["ratio_max"] - r["ratio_median"] for r in rows], arrayminus=[r["ratio_median"] - r["ratio_min"] for r in rows],
                                      color="#555555"),
                         text=[de(r["ratio_median"], 2) for r in rows], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Laplace-sd = Ketten-sd", annotation_position="top left")
    fig.update_xaxes(title="Reaktionsstärke ρ", type="category")
    fig.update_yaxes(title="sd der Laplace-Näherung / sd der Ketten", range=[0, 1.4])
    return _fixed(fig, 340, showlegend=False)
