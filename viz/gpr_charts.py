import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

DARK_BG   = "#0E1117"
DARK_CARD = "#1A1D27"
TEXT      = "#FAFAFA"
GRID      = "#2a2a3a"
BLUE      = "#378ADD"
RED       = "#E24B4A"
AMBER     = "#EF9F27"
GREEN     = "#1D9E75"


def plot_gpr_timeline(gpr_df: pd.DataFrame, start: str, end: str) -> go.Figure:
    """
    Plot the GPR index over the selected period with regime bands.
    """
    mask = (gpr_df["date"] >= pd.Timestamp(start)) & (gpr_df["date"] <= pd.Timestamp(end))
    df = gpr_df[mask].copy()
    if df.empty:
        df = gpr_df.tail(500).copy()

    threshold_75 = gpr_df["gpr"].quantile(0.75)
    threshold_90 = gpr_df["gpr"].quantile(0.90)

    fig = go.Figure()

    # Crisis band
    fig.add_hrect(
        y0=threshold_75, y1=df["gpr"].max() * 1.1,
        fillcolor=RED, opacity=0.07,
        line_width=0,
        annotation_text="Crisis zone",
        annotation_position="top left",
        annotation_font=dict(color=RED, size=10),
    )

    # Elevated band
    fig.add_hrect(
        y0=threshold_75 * 0.75, y1=threshold_75,
        fillcolor=AMBER, opacity=0.05,
        line_width=0,
    )

    # GPR line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["gpr"],
        mode="lines",
        name="GPR index",
        line=dict(color=BLUE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(55,138,221,0.08)",
    ))

    # Threshold lines
    fig.add_hline(y=threshold_75, line_dash="dot", line_color=RED,
                  annotation_text=f"Crisis threshold ({threshold_75:.0f})",
                  annotation_font=dict(color=RED, size=10),
                  annotation_position="bottom right")

    # Major event annotations
    events = [
        ("2020-03-15", "COVID crash"),
        ("2022-02-24", "Ukraine invasion"),
        ("2023-10-08", "Middle East crisis"),
        ("2021-08-15", "Afghanistan"),
    ]
    for date_str, label in events:
        dt = pd.Timestamp(date_str)
        if df["date"].min() <= dt <= df["date"].max():
            row = df[df["date"] >= dt].head(1)
            if not row.empty:
                fig.add_annotation(
                    x=dt,
                    y=float(row["gpr"].values[0]),
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=AMBER,
                    ax=40, ay=-30,
                    font=dict(size=9, color=AMBER),
                    bgcolor=DARK_CARD,
                    bordercolor=AMBER,
                    borderwidth=1,
                    borderpad=3,
                )

    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID, title=""),
        yaxis=dict(gridcolor=GRID, title="GPR index"),
        showlegend=False,
    )
    return fig


def plot_rolling_tail_dependence(
    pseudo_obs: pd.DataFrame,
    returns_df: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.05,
) -> go.Figure:
    """
    Rolling lower tail dependence (λL) between each asset pair.
    Shows how crash correlations evolved over time — the key proof
    that diversification fails exactly when you need it most.
    """
    u = pseudo_obs.values
    cols = list(pseudo_obs.columns)
    n = len(u)
    dates = returns_df.index[-n:]
    pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    fig = go.Figure()

    colors = [BLUE, RED, AMBER, GREEN, "#AFA9EC", "#5DCAA5", "#F0997B"]

    for idx, (i, j) in enumerate(pairs[:6]):  # max 6 pairs for readability
        lam_series = []
        for t in range(window, n):
            u_win = u[t - window:t]
            lam = np.mean((u_win[:, i] < threshold) & (u_win[:, j] < threshold)) / threshold
            lam_series.append(lam)

        label = f"{cols[i]} ↔ {cols[j]}"
        fig.add_trace(go.Scatter(
            x=dates[window:],
            y=lam_series,
            mode="lines",
            name=label,
            line=dict(color=colors[idx % len(colors)], width=1.4),
            opacity=0.85,
        ))

 # Use add_shape instead of add_vline — avoids Plotly string date arithmetic bug
    for event_date, label, color, xshift in [
        ("2022-02-24", "Ukraine invasion", RED,   40),
        ("2020-03-15", "COVID crash",      AMBER, -80),
    ]:
        dt = pd.Timestamp(event_date)
        if len(dates) > window and dates[window] <= dt <= dates[-1]:
            fig.add_shape(
                type="line",
                x0=dt, x1=dt, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=color, width=1.2, dash="dash"),
            )
            fig.add_annotation(
                x=dt, y=0.95, yref="paper",
                text=label,
                showarrow=False,
                xshift=xshift,
                font=dict(size=9, color=color),
                bgcolor=DARK_BG,
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title="Rolling tail dep. λL", tickformat=".2f"),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
    )
    return fig


def plot_regime_comparison(
    pnl_calm: np.ndarray,
    pnl_crisis: np.ndarray,
    var_calm: float,
    var_crisis: float,
    cvar_calm: float,
    cvar_crisis: float,
) -> go.Figure:
    """
    Side-by-side PnL distributions for calm vs crisis regimes.
    The visual proof that geopolitical risk matters for your portfolio.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pnl_calm, nbinsx=100, histnorm="probability density",
        name="Calm regime", marker_color=GREEN, opacity=0.55,
    ))
    fig.add_trace(go.Histogram(
        x=pnl_crisis, nbinsx=100, histnorm="probability density",
        name="Crisis regime", marker_color=RED, opacity=0.55,
    ))

    annotations = [
        (var_calm,   f"VaR calm<br>{var_calm:.2%}",   GREEN, "dash",  -160),
        (cvar_calm,  f"CVaR calm<br>{cvar_calm:.2%}",  "#5DCAA5", "dot", -100),
        (var_crisis, f"VaR crisis<br>{var_crisis:.2%}", RED,   "dash",  -50),
        (cvar_crisis,f"CVaR crisis<br>{cvar_crisis:.2%}","#F09595","dot", -10),
    ]

    for x, label, color, dash, ay in annotations:
        fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.5)
        fig.add_annotation(
            x=x, y=1, yref="paper",
            text=label, showarrow=True,
            arrowhead=0, arrowcolor=color,
            ax=28, ay=ay,
            font=dict(size=10, color=color),
            bgcolor=DARK_BG,
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        barmode="overlay",
        height=360,
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID, tickformat=".1%", title="Daily portfolio return"),
        yaxis=dict(gridcolor=GRID, title="Density"),
        legend=dict(orientation="h", y=1.12, font=dict(size=12)),
    )
    return fig


def plot_news_sentiment_gauge(articles: list[dict]) -> go.Figure:
    if not articles:
        avg = 0.0
    else:
        scores = [a.get("sentiment", 0) for a in articles]
        avg = float(np.mean(scores))

    gauge_val = (avg + 1) * 5

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": 5, "valueformat": ".1f"},
        gauge={
            "axis":      {"range": [0, 10], "tickwidth": 1, "tickcolor": "#FAFAFA"},
            "bar":       {"color": "#378ADD"},
            "bgcolor":   "#1A1D27",
            "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [0,   3.5], "color": "#3d1212"},
                {"range": [3.5, 6.5], "color": "#2a2a1a"},
                {"range": [6.5, 10],  "color": "#0d2a1a"},
            ],
            "threshold": {
                "line":      {"color": "#E24B4A", "width": 2},
                "thickness": 0.75,
                "value":     3.5,
            },
        },
        number={"suffix": "/10", "valueformat": ".1f", "font": {"color": "#FAFAFA"}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
    )
    return fig