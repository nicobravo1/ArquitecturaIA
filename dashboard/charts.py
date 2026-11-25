"""Gráficos Plotly del dashboard.

Color por función (paleta de referencia validada):
* Decisiones = **estado** (APPROVE=good, REVIEW=warning, REJECT=critical),
  siempre acompañadas de ícono + etiqueta, nunca solo color.
* Contribuciones SHAP = **divergente** (rojo aumenta el riesgo / azul lo reduce).
* Distribución de probabilidades = una sola serie azul.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

STATUS = {"APPROVE": "#0aac23", "REVIEW": "#fbb315", "REJECT": "#dc3c3d"}
ICON = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "⛔"}
LABEL = {"APPROVE": "Aprobada", "REVIEW": "Revisión", "REJECT": "Rechazada"}
ORDER = ["APPROVE", "REVIEW", "REJECT"]
RISK_UP, RISK_DOWN = "#ec4647", "#257de0"     # polos del par divergente
SERIES = "#257de0"
GRID, AXIS = "rgba(128,128,128,0.18)", "rgba(128,128,128,0.45)"


def badge(decision: str) -> str:
    return f"{ICON[decision]} {LABEL[decision]}"


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8), bargap=0.25,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hoverlabel=dict(namelength=-1), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


BUCKETS = ["10s", "30s", "1min", "5min", "15min", "1h", "6h", "1D"]


def pick_bucket(times: pd.Series, max_buckets: int = 40) -> str:
    """El intervalo más fino que deja como máximo ``max_buckets`` puntos."""
    span = (times.max() - times.min()) if len(times) else pd.Timedelta(0)
    for b in BUCKETS:
        if span / pd.Timedelta(b) <= max_buckets:
            return b
    return BUCKETS[-1]


def decisions_over_time(txs: list[dict]) -> tuple[go.Figure, str]:
    """Decisiones por intervalo, una línea por decisión (eje único, conteos).
    Devuelve la figura y el intervalo elegido."""
    fig = go.Figure()
    freq = "1min"
    if txs:
        df = pd.DataFrame(txs)
        df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True, format="ISO8601")
        freq = pick_bucket(df["scored_at"])
        counts = (df.groupby([pd.Grouper(key="scored_at", freq=freq), "decision"]).size()
                  .unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
                  .asfreq(freq, fill_value=0))          # intervalos sin tráfico = 0
        for d in ORDER:
            fig.add_scatter(x=counts.index, y=counts[d], name=badge(d), mode="lines+markers",
                            line=dict(color=STATUS[d], width=2), marker=dict(size=8, color=STATUS[d]),
                            hovertemplate=LABEL[d] + ": %{y}<extra></extra>")
    fig.update_yaxes(title="Transacciones", rangemode="tozero")
    fig.update_xaxes(tickformat="%H:%M:%S" if freq.endswith("s") else "%d/%m %H:%M")
    return _layout(fig, hovermode="x unified"), freq


def probability_histogram(txs: list[dict], threshold: float, review_threshold: float) -> go.Figure:
    fig = go.Figure()
    probs = [t["fraud_probability"] for t in txs]
    fig.add_histogram(x=probs, xbins=dict(start=0, end=1.0001, size=0.05), name="Transacciones",
                      marker=dict(color=SERIES, line=dict(width=2, color="rgba(0,0,0,0)")),
                      hovertemplate="Probabilidad %{x}<br>%{y} transacciones<extra></extra>")
    for x, text in ((review_threshold, "revisión"), (threshold, "rechazo")):
        fig.add_vline(x=x, line=dict(color=AXIS, width=2, dash="dash"),
                      annotation_text=f"umbral {text} {x:.2f}", annotation_position="top")
    fig.update_xaxes(title="Probabilidad de fraude", range=[0, 1])
    fig.update_yaxes(title="Transacciones", rangemode="tozero")
    return _layout(fig, showlegend=False)


def shap_contributions(explanation: dict, top: int = 10, min_abs: float = 0.005) -> go.Figure:
    """Barras horizontales de las contribuciones SHAP más grandes (en log-odds).
    Se omiten las despreciables (|SHAP| < ``min_abs``)."""
    shap_values: dict = explanation.get("shap_values") or {}
    features: dict = explanation.get("features") or {}
    s = pd.Series(shap_values, dtype=float)
    s = s[s.abs() >= min_abs]
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top)[::-1]
    hour = None
    if "Time" in features:
        hour = int((features["Time"] % 86400) // 3600)
    values = [hour if f == "hour" else features.get(f) for f in s.index]
    names = ["Monto" if f == "Amount" else "Hora" if f == "hour" else f for f in s.index]
    labels = [f"{n} = {v:.2f}" if isinstance(v, (int, float)) else n for n, v in zip(names, values)]
    fig = go.Figure(go.Bar(
        x=s.values, y=labels, orientation="h",
        marker=dict(color=[RISK_UP if v > 0 else RISK_DOWN for v in s.values],
                    line=dict(width=2, color="rgba(0,0,0,0)")),
        text=[f"{v:+.2f}" for v in s.values], textposition="outside", cliponaxis=False,
        customdata=["aumenta el riesgo" if v > 0 else "reduce el riesgo" for v in s.values],
        hovertemplate="%{y}<br>SHAP %{x:+.3f} (%{customdata})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Contribución SHAP (log-odds) · rojo aumenta el riesgo · azul lo reduce")
    return _layout(fig, height=max(220, 34 * len(s) + 80), showlegend=False)
