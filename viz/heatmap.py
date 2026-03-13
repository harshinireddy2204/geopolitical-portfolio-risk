import plotly.graph_objects as go

def plot_tail_dependence(td_matrix):
    cols = list(td_matrix.columns)
    z = td_matrix.values.tolist()
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale=[
            [0,   "#1A1D27"],
            [0.3, "#0C447C"],
            [0.6, "#378ADD"],
            [1,   "#85B7EB"],
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(color="#FAFAFA", size=12),
        showscale=True,
        colorbar=dict(
            thickness=12,
            tickfont=dict(color="#FAFAFA"),
            title=dict(
                text="λL",
                font=dict(color="#FAFAFA"),
            ),
        ),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(tickfont=dict(color="#FAFAFA"), gridcolor="#2a2a3a"),
        yaxis=dict(tickfont=dict(color="#FAFAFA"), gridcolor="#2a2a3a"),
    )
    return fig