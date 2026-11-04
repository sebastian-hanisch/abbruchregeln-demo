"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import abr_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def build_peek(a):
    """Anteil der Läufe, in denen die Wahrheit bei irgendeinem n außerhalb liegt, je Folge; gestrichelt das Niveau 0,05."""
    fig = go.Figure(go.Bar(x=[C.SEQ_LABELS[k] for k in C.SEQ], y=[max(a.peek[k]["ever"], 1e-6) for k in C.SEQ], marker_color=[C.SEQ_COLORS[k] for k in C.SEQ],
                           text=[f"{a.peek[k]['ever']:.3f}".replace(".", ",") for k in C.SEQ], textposition="outside"))
    fig.add_hline(y=C.ALPHA_CI, line=dict(color=C.COLORS["ref"], dash="dash"), annotation_text="nominell 0,05", annotation_position="top right")
    fig.update_yaxes(title_text="Anteil „irgendwann außerhalb“", range=[0, min(1.05, max(0.2, 1.15 * max(a.peek[k]["ever"] for k in C.SEQ)))])
    fig.update_layout(showlegend=False)
    return _base(fig, 360)


def build_width(a):
    """Halbbreite gegen n, doppelt logarithmisch, für alle fünf Folgen."""
    ns = [w["n"] for w in a.widths]
    fig = go.Figure()
    dash = {"wald": "solid", "wilson": "dash", "beta": "solid", "hoeffding": "dot", "asym": "dashdot"}
    for k in C.SEQ:
        fig.add_trace(go.Scatter(x=ns, y=[w[k] if w[k] > 0 else None for w in a.widths], mode="lines+markers", name=C.SEQ_LABELS[k], line=dict(color=C.SEQ_COLORS[k], dash=dash[k])))
    fig.update_xaxes(title_text="Ziehungen n", type="log", tickvals=ns, ticktext=[f"{n:,}".replace(",", ".") for n in ns])
    fig.update_yaxes(title_text="Halbbreite", type="log")
    return _base(fig, 340)


def build_stop(ex):
    """Mittleres n beim Stopp geteilt durch den Orakelumfang, je Regel, mit der Abdeckung als Beschriftung."""
    from abr_evaluation import RULE_LABELS, RULES
    rows = ex["rows"]
    ys = [rows[k]["n"] / rows[k]["n_ref"] if np.isfinite(rows[k]["n"]) else None for k in RULES]
    fig = go.Figure(go.Bar(x=[RULE_LABELS[k] for k in RULES], y=ys, marker_color=[C.SEQ_COLORS["wald"], C.SEQ_COLORS["wilson"], C.SEQ_COLORS["beta"], C.COLOR_BAD],
                           text=[f"Abdeckung {rows[k]['cov']:.3f}".replace(".", ",") for k in RULES], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLORS["ref"], dash="dot"), annotation_text="Orakelumfang", annotation_position="top left")
    fig.update_yaxes(title_text="mittleres n / Orakelumfang", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig, 340)


def build_inverse(ex):
    """Verzerrung von k/n und (k − 1)/(n − 1) gegen k."""
    ks = [r["k"] for r in ex["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[100 * r["bias_naive"] for r in ex["rows"]], mode="lines+markers", name="k/n", line=dict(color=C.COLORS["naive"])))
    fig.add_trace(go.Scatter(x=ks, y=[100 * r["bias_unbiased"] for r in ex["rows"]], mode="lines+markers", name="(k − 1)/(n − 1)", line=dict(color=C.COLORS["unbiased"])))
    fig.update_xaxes(title_text="Ereignisse k beim Stopp", type="log", tickvals=ks, ticktext=[str(k) for k in ks])
    fig.update_yaxes(title_text="relative Verzerrung (%)")
    return _base(fig, 320)


def build_mctest(ex):
    """Entscheidungsrisiko gegen π/α für die sechs Verfahren (links) und mittlere Zahl der Simulationen (rechts)."""
    rows = sorted(ex["rows"], key=lambda r: r["factor"])
    xs = [r["factor"] for r in rows]
    risk, cost = go.Figure(), go.Figure()
    for k in C.MC_DESIGNS:
        risk.add_trace(go.Scatter(x=xs, y=[r["cells"][k]["risk"] for r in rows], mode="lines+markers", name=C.MC_LABELS[k], line=dict(color=C.COLORS[k])))
        cost.add_trace(go.Scatter(x=xs, y=[r["cells"][k]["n"] for r in rows], mode="lines+markers", name=C.MC_LABELS[k], line=dict(color=C.COLORS[k])))
    for fig, title, kind in ((risk, "Entscheidungsrisiko", "linear"), (cost, "mittlere Zahl der Simulationen", "log")):
        fig.add_vline(x=1.0, line=dict(color=C.COLORS["ref"], dash="dot"))
        fig.add_vline(x=ex["own"]["factor"], line=dict(color=C.COLOR_OK, dash="dash"), annotation_text="gewähltes Netz", annotation_position="top")
        fig.update_xaxes(title_text="π / α (exakter p-Wert geteilt durch das Testniveau)", type="log")
        fig.update_yaxes(title_text=title, type=kind)
    return _base(risk, 360), _base(cost, 360)
