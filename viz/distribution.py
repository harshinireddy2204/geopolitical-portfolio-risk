import plotly.graph_objects as go

def plot_pnl_distribution(pnl_copula, pnl_gaussian, var_c, var_g, cvar_c, cvar_g):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pnl_copula, nbinsx=120, histnorm="probability density",
        name="Copula model", marker_color="#378ADD", opacity=0.6,
    ))

    fig.add_trace(go.Histogram(
        x=pnl_gaussian, nbinsx=120, histnorm="probability density",
        name="Gaussian (standard)", marker_color="#D85A30", opacity=0.45,
    ))

    # Each line gets a different yref position so labels never overlap
    annotations = [
        dict(
            x=cvar_c, label=f"CVaR copula<br>{cvar_c:.2%}",
            color="#85B7EB", dash="dot", ay=-200
        ),
        dict(
            x=var_c, label=f"VaR copula<br>{var_c:.2%}",
            color="#378ADD", dash="dash", ay=-140
        ),
        dict(
            x=var_g, label=f"VaR Gaussian<br>{var_g:.2%}",
            color="#D85A30", dash="dash", ay=-80
        ),
        dict(
            x=cvar_g, label=f"CVaR Gaussian<br>{cvar_g:.2%}",
            color="#F0997B", dash="dot", ay=-20
        ),
    ]

    for a in annotations:
        fig.add_vline(
            x=a["x"],
            line_dash=a["dash"],
            line_color=a["color"],
            line_width=1.5,
        )
        fig.add_annotation(
            x=a["x"],
            y=1,
            yref="paper",
            text=a["label"],
            showarrow=True,
            arrowhead=0,
            arrowcolor=a["color"],
            ax=28,
            ay=a["ay"],
            font=dict(size=11, color=a["color"]),
            bgcolor="#0E1117",
            bordercolor=a["color"],
            borderwidth=1,
            borderpad=4,
            align="left",
        )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Daily portfolio return",
        yaxis_title="Density",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
            font=dict(size=12),
        ),
        margin=dict(l=40, r=40, t=100, b=40),
        height=420,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(
            gridcolor="#2a2a3a",
            tickformat=".1%",
            zeroline=True,
            zerolinecolor="#444",
        ),
        yaxis=dict(gridcolor="#2a2a3a"),
    )
    return fig