import numpy as np
import pandas as pd
from scipy.stats import norm

def invert_to_returns(u_sim, returns_df):
    if u_sim is None or u_sim.shape[0] == 0:
        # Fallback: return zeros shaped correctly
        return np.zeros((1, len(returns_df.columns)))
    sim_returns = np.zeros_like(u_sim)
    for i, col in enumerate(returns_df.columns):
        empirical = np.sort(returns_df[col].dropna().values)
        if len(empirical) == 0:
            continue
        quantiles = np.linspace(0, 1, len(empirical))
        sim_returns[:, i] = np.interp(u_sim[:, i], quantiles, empirical)
    return sim_returns

def portfolio_pnl(sim_returns, weights):
    w = np.array(weights)
    w = w / w.sum()
    return sim_returns @ w

def compute_var_cvar(pnl, confidence=0.99):
    var = np.quantile(pnl, 1 - confidence)
    cvar = pnl[pnl <= var].mean()
    return float(var), float(cvar)

def gaussian_baseline(returns_df, weights, n_sim=50_000, confidence=0.99):
    w = np.array(weights) / sum(weights)
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    sim = np.random.multivariate_normal(mu, cov, size=n_sim)
    pnl = sim @ w
    return compute_var_cvar(pnl, confidence)

def tail_dependence_matrix(pseudo_obs, threshold=0.05):
    u = pseudo_obs.values
    n, d = u.shape
    cols = pseudo_obs.columns
    td = pd.DataFrame(np.eye(d), index=cols, columns=cols)
    for i in range(d):
        for j in range(i + 1, d):
            both_low = np.mean((u[:, i] < threshold) & (u[:, j] < threshold))
            lambda_l = both_low / threshold
            td.iloc[i, j] = lambda_l
            td.iloc[j, i] = lambda_l
    return td