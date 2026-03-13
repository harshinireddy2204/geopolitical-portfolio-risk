import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau


def _cholesky_safe(corr: np.ndarray) -> np.ndarray:
    """Return a PSD correlation matrix safe for Cholesky decomposition."""
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-8:
        corr += np.eye(corr.shape[0]) * (abs(eigvals.min()) + 1e-6)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _pseudo_log_lik_gaussian(u: np.ndarray, corr: np.ndarray) -> float:
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    try:
        corr_inv = np.linalg.inv(corr)
        _, logdet = np.linalg.slogdet(corr)
        quad = np.einsum("ij,jk,ik->i", z, corr_inv - np.eye(corr.shape[0]), z)
        return float(np.sum(-0.5 * (logdet + quad)))
    except Exception:
        return -np.inf


def fit_regime_copulas(
    pseudo_obs: pd.DataFrame,
    regime_labels: pd.Series,
) -> dict:
    """
    Fit Clayton copulas separately for calm and crisis regimes.
    Returns dict with keys 'calm' and 'crisis', each containing:
        theta, corr, tail_dep, n_obs, aic
    """
    u_all = pseudo_obs.values
    results = {}

    for regime in ("calm", "crisis"):
        mask = (regime_labels == regime).values
        # Align mask length to pseudo_obs (pseudo_obs may be shorter due to GARCH burn-in)
        mask = mask[-len(u_all):]
        u = u_all[mask]

        if len(u) < 30:
            # Not enough data — use full sample as fallback
            u = u_all

        d = u.shape[1]

        # Correlation (Gaussian copula corr)
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        corr = np.corrcoef(z.T)
        corr = _cholesky_safe(corr)

        # Clayton theta via Kendall's tau
        taus = [kendalltau(u[:, i], u[:, j])[0]
                for i in range(d) for j in range(i + 1, d)]
        tau_mean = max(float(np.mean(taus)), 0.01)
        theta = max(2 * tau_mean / (1 - tau_mean), 0.01)

        # Lower tail dependence lambda_L for Clayton = 2^(-1/theta)
        tail_dep = 2 ** (-1.0 / theta)

        # AIC (Clayton log-likelihood)
        eps = 1e-300
        ll = float(np.sum(np.log(np.maximum(
            (theta + 1) * np.prod(u, axis=1) ** (-(theta + 1))
            * (np.sum(u ** (-theta), axis=1) - d + 1) ** (-(2 + 1 / theta)),
            eps
        ))))
        aic = 2 * 1 - 2 * ll

        results[regime] = {
            "theta":    round(theta, 4),
            "corr":     corr,
            "tail_dep": round(tail_dep, 4),
            "n_obs":    int(mask.sum()),
            "aic":      round(aic, 1),
        }

    return results


def simulate_gpr_conditioned(
    regime_copulas: dict,
    current_regime: str,
    pseudo_obs: pd.DataFrame,
    n_sim: int = 50_000,
) -> np.ndarray:
    """
    Simulate from the copula fitted for the CURRENT geopolitical regime.
    current_regime: 'calm', 'elevated', or 'extreme'
    """
    key = "crisis" if current_regime in ("elevated", "extreme") else "calm"
    info = regime_copulas[key]
    d    = pseudo_obs.shape[1]
    theta = info["theta"]

    # Clayton simulation via conditional method
    u = np.random.uniform(size=(n_sim, d))
    v = np.random.gamma(shape=1 / theta, scale=1.0, size=n_sim)
    exp = -np.log(u) / v[:, None]
    samples = (1 + theta * exp) ** (-1.0 / theta)
    return np.clip(samples, 1e-6, 1 - 1e-6)


def compute_regime_shift(regime_copulas: dict) -> dict:
    """
    Quantify how much riskier the crisis regime is vs calm.
    """
    calm   = regime_copulas.get("calm",   {})
    crisis = regime_copulas.get("crisis", {})

    theta_calm   = calm.get("theta",    1.0)
    theta_crisis = crisis.get("theta",  1.0)
    td_calm      = calm.get("tail_dep", 0.0)
    td_crisis    = crisis.get("tail_dep", 0.0)

    return {
        "theta_lift":   round(theta_crisis - theta_calm, 4),
        "theta_pct":    round((theta_crisis / max(theta_calm, 0.01) - 1) * 100, 1),
        "td_lift":      round(td_crisis - td_calm, 4),
        "td_pct":       round((td_crisis / max(td_calm, 0.001) - 1) * 100, 1),
        "theta_calm":   round(theta_calm, 3),
        "theta_crisis": round(theta_crisis, 3),
        "td_calm":      round(td_calm, 3),
        "td_crisis":    round(td_crisis, 3),
    }