import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

from config import TICKERS, TICKER_LABELS, DEFAULT_START, DEFAULT_END, N_SIM, CONFIDENCE
from data.loader import get_returns
from data.gpr_loader import (
    load_gpr_index, classify_gpr_regime, get_current_gpr_level,
    fetch_geopolitical_news,
)
from models.marginals import fit_marginals
from models.copula import fit_and_select, simulate
from models.gpr_copula import (
    fit_regime_copulas, simulate_gpr_conditioned, compute_regime_shift,
)
from models.risk import (
    invert_to_returns, portfolio_pnl, compute_var_cvar,
    gaussian_baseline, tail_dependence_matrix,
)
from viz.distribution import plot_pnl_distribution
from viz.heatmap import plot_tail_dependence
from viz.gpr_charts import (
    plot_gpr_timeline, plot_rolling_tail_dependence,
    plot_regime_comparison, plot_news_sentiment_gauge,
)

st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    layout="wide",
    page_icon="🌍",
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🌍 Geopolitical Risk & Portfolio Tail Dashboard")
st.markdown(
    "How does your portfolio's crash risk shift when geopolitics explode? "
    "This dashboard combines **live news**, the **Caldara-Iacoviello GPR index**, "
    "and a **regime-conditioned copula model** to show you the honest answer."
)

with st.expander("📖 What does this dashboard do? (Plain English)", expanded=False):
    st.markdown("""
**The problem with standard portfolio risk tools:** They assume today's crash correlations are the same as during peacetime. They're not.

**What we do differently:**
- We download the **Geopolitical Risk Index (GPR)** — a measure of war threats, terrorism, and conflict built by Federal Reserve economists from 10 major newspapers since 1985
- We split history into **calm periods** and **crisis periods** based on that index
- We fit a *separate* statistical crash model (Clayton copula) for each regime
- We look at today's live news headlines to determine which regime we're currently in
- We show you how much worse your portfolio loss could be in a crisis vs calm environment

**Key terms:**
| Term | Plain English |
|------|--------------|
| **GPR index** | Score of how much war/terror/conflict is in the news. Higher = scarier. |
| **Calm regime** | Bottom 75% of historical GPR — normal times |
| **Crisis regime** | Top 25% of GPR — wars, 9/11, Ukraine, COVID crash |
| **Clayton copula** | A model that captures how strongly assets crash *together* |
| **Tail dependence λL** | Probability that two assets crash at the same time |
| **VaR** | Minimum loss on the worst 1% of days |
| **CVaR** | Average loss across those worst days |
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Portfolio settings")

    selected = st.multiselect(
        "Assets",
        TICKERS,
        default=["SPY", "TLT", "GLD", "QQQ"],
        format_func=lambda t: TICKER_LABELS.get(t, t),
    )

    col1, col2 = st.columns(2)
    start = col1.text_input("From", DEFAULT_START)
    end   = col2.text_input("To",   DEFAULT_END)

    n_sim = st.select_slider(
        "Simulated scenarios",
        options=[10_000, 25_000, 50_000, 100_000],
        value=50_000,
    )
    confidence = st.selectbox(
        "Confidence level",
        [0.99, 0.95],
        format_func=lambda x: f"{x:.0%} — worst {1-x:.0%} of days",
    )

    st.subheader("Portfolio weights")
    weights = []
    for t in selected:
        w = st.slider(
            TICKER_LABELS.get(t, t), 0.0, 1.0,
            round(1.0 / max(len(selected), 1), 2), 0.05,
        )
        weights.append(w)

    if sum(weights) > 0:
        norm_w = [w / sum(weights) for w in weights]
        for t, w in zip(selected, norm_w):
            st.progress(w, text=f"{TICKER_LABELS.get(t, t)}: {w:.0%}")

    st.divider()
    st.subheader("🔑 NewsAPI key (optional)")
    st.caption(
        "Free key from [newsapi.org](https://newsapi.org) — 100 requests/day. "
        "Without it the dashboard uses demo headlines."
    )
    news_api_key = st.text_input("NewsAPI key", value="", type="password",
                                  placeholder="Paste your free key here")

    run = st.button("▶ Run analysis", type="primary", use_container_width=True)

# ── Idle state ────────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Configure your portfolio in the sidebar, then click **Run analysis**.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 🌍 Live GPR")
        st.markdown("Real-time geopolitical risk level from today's news headlines — categorised as calm, elevated, or crisis.")
    with c2:
        st.markdown("### 🔀 Regime copula")
        st.markdown("Two separate crash models: one for peacetime, one for geopolitical crises. Most tools only use one.")
    with c3:
        st.markdown("### 📈 Rolling tail dep.")
        st.markdown("Watch how crash correlations between your assets spiked during Ukraine 2022 and COVID 2020.")
    with c4:
        st.markdown("### 💬 Live headlines")
        st.markdown("Geopolitical headlines scored by risk sentiment, feeding into the regime classification.")
    st.stop()

if len(selected) < 2:
    st.error("Please select at least 2 assets.")
    st.stop()
if sum(weights) == 0:
    st.error("At least one weight must be above 0.")
    st.stop()

# ── Data pipeline ─────────────────────────────────────────────────────────────
progress = st.progress(0, text="Downloading market data...")
returns = get_returns(selected, start, end)

progress.progress(15, text="Loading GPR index...")
gpr_df  = load_gpr_index()
gpr_now = get_current_gpr_level(gpr_df)

progress.progress(25, text="Fetching live geopolitical headlines...")
headlines = fetch_geopolitical_news(news_api_key or "")

progress.progress(35, text="Fitting GARCH volatility models...")
pseudo_obs, _ = fit_marginals(returns)

progress.progress(50, text="Building calm/crisis regime labels...")
# Align GPR to returns index
gpr_aligned = (
    gpr_df.set_index("date")["gpr"]
    .reindex(returns.index, method="ffill")
    .fillna(method="bfill")
)
regime_labels = classify_gpr_regime(gpr_aligned)

progress.progress(60, text="Fitting regime-conditioned copulas...")
regime_copulas = fit_regime_copulas(pseudo_obs, regime_labels)
regime_shift   = compute_regime_shift(regime_copulas)

progress.progress(72, text=f"Simulating {n_sim:,} scenarios for CURRENT regime ({gpr_now['regime']})...")
u_sim_current  = simulate_gpr_conditioned(regime_copulas, gpr_now["regime"], pseudo_obs, n_sim)
sim_ret_current = invert_to_returns(u_sim_current, returns)
pnl_current     = portfolio_pnl(sim_ret_current, weights)
var_c, cvar_c   = compute_var_cvar(pnl_current, confidence)

progress.progress(80, text="Simulating calm regime baseline...")
u_sim_calm   = simulate_gpr_conditioned(regime_copulas, "calm", pseudo_obs, n_sim)
sim_ret_calm = invert_to_returns(u_sim_calm, returns)
pnl_calm     = portfolio_pnl(sim_ret_calm, weights)
var_calm, cvar_calm = compute_var_cvar(pnl_calm, confidence)

progress.progress(88, text="Simulating crisis regime...")
u_sim_crisis   = simulate_gpr_conditioned(regime_copulas, "crisis", pseudo_obs, n_sim)
sim_ret_crisis = invert_to_returns(u_sim_crisis, returns)
pnl_crisis     = portfolio_pnl(sim_ret_crisis, weights)
var_crisis, cvar_crisis = compute_var_cvar(pnl_crisis, confidence)

progress.progress(94, text="Computing Gaussian baseline and tail dependence...")
var_g, cvar_g = gaussian_baseline(returns, weights, n_sim=n_sim, confidence=confidence)
w_arr = np.array(weights) / sum(weights)
pnl_gauss = np.random.multivariate_normal(
    returns.mean().values, returns.cov().values, size=n_sim
) @ w_arr
td = tail_dependence_matrix(pseudo_obs)

progress.progress(100, text="Done!")
progress.empty()

# ── Live GPR status banner ────────────────────────────────────────────────────
regime_color = {"calm": "success", "elevated": "warning", "extreme": "error"}
regime_icon  = {"calm": "✅", "elevated": "⚠️", "extreme": "🚨"}
regime_msg   = {
    "calm":     "Markets are in a **calm geopolitical regime**. Crash correlations are at normal levels.",
    "elevated": "Geopolitical risk is **elevated**. Asset crash correlations are higher than normal — review your tail risk.",
    "extreme":  "🚨 Geopolitical risk is **extreme** — top 10% historically. Your portfolio faces materially higher crash correlation risk.",
}
getattr(st, regime_color[gpr_now["regime"]])(
    f"{regime_icon[gpr_now['regime']]} **GPR index today: {gpr_now['value']} "
    f"(higher than {gpr_now['pct_rank']:.0f}% of all historical readings)** — "
    + regime_msg[gpr_now["regime"]]
)

# Loss vs Gaussian gap
gap = abs(cvar_c) - abs(cvar_g)
gap_pct = gap / abs(cvar_g) * 100 if cvar_g != 0 else 0
if gap > 0.003:
    st.error(
        f"📉 In the current **{gpr_now['regime']} regime**, your true average worst-case loss is "
        f"**{abs(cvar_c):.2%}** — that's **{gap_pct:.0f}% worse** than the standard Gaussian estimate "
        f"of {abs(cvar_g):.2%}. The gap is caused by crash co-movement the Gaussian model ignores."
    )
else:
    st.success(
        "Your portfolio shows limited excess tail risk in the current regime vs the Gaussian baseline."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Live geopolitical risk",
    "📉 Regime risk comparison",
    "🔗 Tail dependence",
    "📊 Full model detail",
])

# ══ TAB 1 — Live geopolitical risk ══════════════════════════════════════════
with tab1:
    st.subheader("Today's geopolitical risk environment")

    col_gpr, col_gauge = st.columns([2, 1])
    with col_gpr:
        st.markdown("**GPR index — your selected period**")
        st.caption("Shaded red zone = crisis regime (top 25% of history). Labelled spikes are major events.")
        st.plotly_chart(plot_gpr_timeline(gpr_df, start, end), use_container_width=True)
        
    with col_gauge:
        st.markdown("#### 📡 News sentiment score")
        st.caption("0 = all crisis headlines · 10 = all calm headlines")
        st.plotly_chart(plot_news_sentiment_gauge(headlines), use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("GPR today", gpr_now["value"],
                  help="Caldara-Iacoviello GPR index, most recent reading")
        m2.metric("Percentile", f"{gpr_now['pct_rank']:.0f}th",
                  help="How this reading ranks vs all history since 1985")

    st.divider()
    st.subheader("📰 Live geopolitical headlines")
    st.caption("Headlines fetched now and scored for risk sentiment. Red = negative/risky, green = calming.")

    for a in headlines:
        s = a.get("sentiment", 0)
        if s <= -0.3:
            icon, color = "🔴", "#3d1212"
        elif s < 0.1:
            icon, color = "🟡", "#2a2a1a"
        else:
            icon, color = "🟢", "#0d2a1a"

        pub = a.get("publishedAt", "")[:10]
        src = a.get("source", "")
        url = a.get("url", "#")
        title = a.get("title", "")

        st.markdown(
            f"""<div style="background:{color};border-radius:8px;padding:10px 14px;margin-bottom:8px;">
            {icon} <b><a href="{url}" target="_blank" style="color:#FAFAFA;text-decoration:none;">{title}</a></b>
            <br><span style="font-size:12px;color:#aaa;">{src} · {pub} · sentiment: {s:+.2f}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if not news_api_key:
        st.info(
            "💡 These are demo headlines. Add your free NewsAPI key in the sidebar "
            "to see real live geopolitical news updated continuously."
        )

# ══ TAB 2 — Regime risk comparison ═══════════════════════════════════════════
with tab2:
    st.subheader("How much worse is your portfolio in a geopolitical crisis?")
    st.caption(
        "We fit two separate Clayton copula models — one on calm-regime data, one on crisis-regime data. "
        "Here's how your portfolio risk differs between them."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VaR — calm regime",   f"{var_calm:.2%}",
              help="Worst-day loss floor when GPR is low")
    m2.metric("CVaR — calm regime",  f"{cvar_calm:.2%}",
              help="Average loss on worst days in calm periods")
    m3.metric("VaR — crisis regime", f"{var_crisis:.2%}",
              delta=f"{var_crisis - var_calm:.2%} vs calm",
              delta_color="inverse",
              help="Worst-day loss floor when GPR is elevated")
    m4.metric("CVaR — crisis regime",f"{cvar_crisis:.2%}",
              delta=f"{cvar_crisis - cvar_calm:.2%} vs calm",
              delta_color="inverse",
              help="Average loss on worst days during crises")

    st.info(
        f"🔬 **The copula shift:** In a geopolitical crisis, the Clayton θ parameter rises from "
        f"**{regime_shift['theta_calm']}** (calm) to **{regime_shift['theta_crisis']}** (crisis) — "
        f"a {regime_shift['theta_pct']:+.0f}% increase. "
        f"Crash tail dependence λL rises from **{regime_shift['td_calm']}** to **{regime_shift['td_crisis']}** "
        f"({regime_shift['td_pct']:+.0f}%). "
        f"This means your assets are {regime_shift['td_pct']:.0f}% more likely to crash together in a crisis."
    )

    st.markdown("---")
    st.subheader("Loss distributions: calm vs crisis regime")
    st.caption(
        "The green distribution is your portfolio in calm times. "
        "The red is during geopolitical crises. Notice the heavier left tail in red — "
        "that's the cost of crash co-movement."
    )
    st.plotly_chart(
        plot_regime_comparison(pnl_calm, pnl_crisis, var_calm, var_crisis, cvar_calm, cvar_crisis),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("💵 What does this mean in dollars?")
    portfolio_size = st.number_input(
        "Portfolio size ($)", min_value=1000, max_value=10_000_000,
        value=100_000, step=10_000,
        help="Enter your portfolio value to see losses in dollar terms"
    )
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Calm CVaR loss",
               f"-${abs(cvar_calm) * portfolio_size:,.0f}",
               help="Average worst-day loss in peacetime")
    dc2.metric("Crisis CVaR loss",
               f"-${abs(cvar_crisis) * portfolio_size:,.0f}",
               delta=f"-${(abs(cvar_crisis) - abs(cvar_calm)) * portfolio_size:,.0f} extra vs calm",
               delta_color="inverse",
               help="Average worst-day loss during geopolitical crises")
    dc3.metric("Gaussian blind spot",
               f"-${(abs(cvar_c) - abs(cvar_g)) * portfolio_size:,.0f}",
               help="Extra loss the standard Gaussian model fails to warn you about")

# ══ TAB 3 — Tail dependence ══════════════════════════════════════════════════
with tab3:
    st.subheader("How crash correlations evolved over time")
    st.caption(
        "Rolling 60-day lower tail dependence (λL) between each asset pair. "
        "Watch it spike during the major geopolitical events marked on the chart. "
        "This is the empirical proof that diversification fails exactly when you need it most."
    )
    st.plotly_chart(
        plot_rolling_tail_dependence(pseudo_obs, returns, window=60),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Current tail dependence matrix")
    st.caption("Based on full-sample data. For regime-specific values, see the Regime comparison tab.")
    st.plotly_chart(plot_tail_dependence(td), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.error("🔴 > 0.4 — These assets crash together often")
    col2.warning("🟡 0.2–0.4 — Moderate crash correlation")
    col3.success("🟢 < 0.2 — Low crash correlation, good diversifier")

    st.markdown("**Regime-specific tail dependence:**")
    dc1, dc2 = st.columns(2)
    dc1.metric("Average λL — calm",   f"{regime_shift['td_calm']:.3f}",
               help="How correlated are crashes when geopolitics are quiet")
    dc2.metric("Average λL — crisis", f"{regime_shift['td_crisis']:.3f}",
               delta=f"+{regime_shift['td_lift']:.3f} vs calm",
               delta_color="inverse",
               help="How correlated are crashes during geopolitical crises")

# ══ TAB 4 — Full model detail ════════════════════════════════════════════════
with tab4:
    st.subheader("Standard model vs copula vs current regime")
    st.caption(f"Based on {len(returns):,} trading days · {n_sim:,} simulations · {confidence:.0%} confidence")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VaR (copula, current regime)",  f"{var_c:.2%}",
              delta=f"{var_c - var_g:.2%} vs Gaussian", delta_color="inverse")
    m2.metric("CVaR (copula, current regime)", f"{cvar_c:.2%}",
              delta=f"{cvar_c - cvar_g:.2%} vs Gaussian", delta_color="inverse")
    m3.metric("VaR (Gaussian)",  f"{var_g:.2%}")
    m4.metric("CVaR (Gaussian)", f"{cvar_g:.2%}")

    st.markdown("---")
    st.subheader("Full PnL distribution — copula vs Gaussian")
    st.plotly_chart(
        plot_pnl_distribution(pnl_current, pnl_gauss, var_c, var_g, cvar_c, cvar_g),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Regime data summary")
    regime_summary = pd.DataFrame([
        {
            "Regime": "Calm",
            "Observations": regime_copulas["calm"]["n_obs"],
            "Clayton θ": regime_copulas["calm"]["theta"],
            "Tail dep. λL": regime_copulas["calm"]["tail_dep"],
            "AIC": regime_copulas["calm"]["aic"],
        },
        {
            "Regime": "Crisis",
            "Observations": regime_copulas["crisis"]["n_obs"],
            "Clayton θ": regime_copulas["crisis"]["theta"],
            "Tail dep. λL": regime_copulas["crisis"]["tail_dep"],
            "AIC": regime_copulas["crisis"]["aic"],
        },
    ])
    st.dataframe(regime_summary, hide_index=True, use_container_width=True)

    st.caption(
        "Clayton θ: higher = stronger crash co-movement. "
        "Tail dep. λL = P(asset A crashes | asset B crashes). "
        "AIC: model fit score, lower is better."
    )