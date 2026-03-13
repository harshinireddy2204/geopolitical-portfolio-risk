import numpy as np
import pandas as pd
from scipy.stats import norm

def _gaussian_copula_sim(corr, n_sim):
    L = np.linalg.cholesky(corr)
    z = np.random.standard_normal((n_sim, corr.shape[0]))
    u = norm.cdf(z @ L.T)
    return u

def _t_copula_sim(corr, nu, n_sim):
    from scipy.stats import t as t_dist, chi2
    d = corr.shape[0]
    L = np.linalg.cholesky(corr)
    z = np.random.standard_normal((n_sim, d)) @ L.T
    w = chi2.rvs(nu, size=n_sim) / nu
    x = z / np.sqrt(w)[:, None]
    u = t_dist.cdf(x, df=nu)
    return u

def _clayton_copula_sim(theta, d, n_sim):
    # Conditional sampling for Clayton
    u = np.random.uniform(size=(n_sim, d))
    # Marshall-Olkin for Clayton
    v = np.random.gamma(shape=1/theta, scale=1.0, size=n_sim)
    exp = -np.log(u) / v[:, None]
    result = (1 + theta * exp) ** (-1.0 / theta)
    return np.clip(result, 1e-6, 1 - 1e-6)

def _aic(log_lik, k):
    return 2 * k - 2 * log_lik

def _pseudo_log_lik(u, corr):
    """Gaussian copula log-likelihood on pseudo-obs."""
    from scipy.stats import norm, multivariate_normal
    z = norm.ppf(u)
    try:
        corr_inv = np.linalg.inv(corr)
        sign, logdet = np.linalg.slogdet(corr)
        quad = np.einsum("ij,jk,ik->i", z, corr_inv - np.eye(corr.shape[0]), z)
        ll = -0.5 * (logdet + quad)
        return float(np.sum(ll))
    except Exception:
        return -np.inf

def fit_and_select(pseudo_obs: pd.DataFrame):
    """
    Fit Gaussian, Student-t, Clayton copulas.
    Return best fit name + correlation matrix.
    """
    u = pseudo_obs.values
    d = u.shape[1]

    # Compute linear correlation on normal scores
    z = norm.ppf(u)
    corr = np.corrcoef(z.T)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    results = {}

    # Gaussian
    ll_gauss = _pseudo_log_lik(u, corr)
    k_gauss = d * (d - 1) / 2
    results["Gaussian"] = {
        "aic": _aic(ll_gauss, k_gauss),
        "corr": corr,
        "nu": None,
    }

    # Student-t (grid search over nu)
    from scipy.stats import t as t_dist
    best_t_aic = np.inf
    best_nu = 4
    for nu in [3, 4, 5, 6, 8, 10, 15, 20]:
        z_t = t_dist.ppf(u, df=nu)
        z_t = np.clip(z_t, -10, 10)
        corr_t = np.corrcoef(z_t.T)
        corr_t = (corr_t + corr_t.T) / 2
        np.fill_diagonal(corr_t, 1.0)
        ll_t = _pseudo_log_lik(u, corr_t)
        aic_t = _aic(ll_t, k_gauss + 1)
        if aic_t < best_t_aic:
            best_t_aic = aic_t
            best_nu = nu
            best_corr_t = corr_t

    results["Student-t"] = {
        "aic": best_t_aic,
        "corr": best_corr_t,
        "nu": best_nu,
    }

    # Clayton (bivariate pairs average)
    try:
        from scipy.stats import kendalltau
        taus = []
        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = kendalltau(u[:, i], u[:, j])
                taus.append(tau)
        tau_mean = max(np.mean(taus), 0.01)
        theta_c = 2 * tau_mean / (1 - tau_mean)
        ll_c = np.sum(
            np.log(np.maximum(
                (theta_c + 1) * np.prod(u, axis=1) ** (-(theta_c + 1)) *
                (np.sum(u ** (-theta_c), axis=1) - d + 1) ** (-(2 + 1 / theta_c)),
                1e-300,
            ))
        )
        results["Clayton"] = {"aic": _aic(ll_c, 1), "theta": theta_c, "corr": corr}
    except Exception:
        results["Clayton"] = {"aic": np.inf, "theta": 1.0, "corr": corr}

    best = min(results, key=lambda k: results[k]["aic"])
    return best, results, corr

def simulate(best_name, results, pseudo_obs, n_sim=50_000):
    """
    Simulate joint returns from the selected copula.
    Returns uniform matrix (n_sim x d).
    """
    d = pseudo_obs.shape[1]
    info = results[best_name]
    corr = info["corr"]
    np.fill_diagonal(corr, 1.0)

    if best_name == "Gaussian":
        return _gaussian_copula_sim(corr, n_sim)
    elif best_name == "Student-t":
        nu = info.get("nu", 5) or 5
        return _t_copula_sim(corr, nu, n_sim)
    else:  # Clayton fallback
        theta = info.get("theta", 1.0) or 1.0
        return _clayton_copula_sim(theta, d, n_sim)
import numpy as np
import pandas as pd
from scipy.stats import norm

def _gaussian_copula_sim(corr, n_sim):
    L = np.linalg.cholesky(corr)
    z = np.random.standard_normal((n_sim, corr.shape[0]))
    u = norm.cdf(z @ L.T)
    return u

def _t_copula_sim(corr, nu, n_sim):
    from scipy.stats import t as t_dist, chi2
    d = corr.shape[0]
    L = np.linalg.cholesky(corr)
    z = np.random.standard_normal((n_sim, d)) @ L.T
    w = chi2.rvs(nu, size=n_sim) / nu
    x = z / np.sqrt(w)[:, None]
    u = t_dist.cdf(x, df=nu)
    return u

def _clayton_copula_sim(theta, d, n_sim):
    # Conditional sampling for Clayton
    u = np.random.uniform(size=(n_sim, d))
    # Marshall-Olkin for Clayton
    v = np.random.gamma(shape=1/theta, scale=1.0, size=n_sim)
    exp = -np.log(u) / v[:, None]
    result = (1 + theta * exp) ** (-1.0 / theta)
    return np.clip(result, 1e-6, 1 - 1e-6)

def _aic(log_lik, k):
    return 2 * k - 2 * log_lik

def _pseudo_log_lik(u, corr):
    """Gaussian copula log-likelihood on pseudo-obs."""
    from scipy.stats import norm, multivariate_normal
    z = norm.ppf(u)
    try:
        corr_inv = np.linalg.inv(corr)
        sign, logdet = np.linalg.slogdet(corr)
        quad = np.einsum("ij,jk,ik->i", z, corr_inv - np.eye(corr.shape[0]), z)
        ll = -0.5 * (logdet + quad)
        return float(np.sum(ll))
    except Exception:
        return -np.inf

def fit_and_select(pseudo_obs: pd.DataFrame):
    """
    Fit Gaussian, Student-t, Clayton copulas.
    Return best fit name + correlation matrix.
    """
    u = pseudo_obs.values
    d = u.shape[1]

    # Compute linear correlation on normal scores
    z = norm.ppf(u)
    corr = np.corrcoef(z.T)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    results = {}

    # Gaussian
    ll_gauss = _pseudo_log_lik(u, corr)
    k_gauss = d * (d - 1) / 2
    results["Gaussian"] = {
        "aic": _aic(ll_gauss, k_gauss),
        "corr": corr,
        "nu": None,
    }

    # Student-t (grid search over nu)
    from scipy.stats import t as t_dist
    best_t_aic = np.inf
    best_nu = 4
    for nu in [3, 4, 5, 6, 8, 10, 15, 20]:
        z_t = t_dist.ppf(u, df=nu)
        z_t = np.clip(z_t, -10, 10)
        corr_t = np.corrcoef(z_t.T)
        corr_t = (corr_t + corr_t.T) / 2
        np.fill_diagonal(corr_t, 1.0)
        ll_t = _pseudo_log_lik(u, corr_t)
        aic_t = _aic(ll_t, k_gauss + 1)
        if aic_t < best_t_aic:
            best_t_aic = aic_t
            best_nu = nu
            best_corr_t = corr_t

    results["Student-t"] = {
        "aic": best_t_aic,
        "corr": best_corr_t,
        "nu": best_nu,
    }

    # Clayton (bivariate pairs average)
    try:
        from scipy.stats import kendalltau
        taus = []
        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = kendalltau(u[:, i], u[:, j])
                taus.append(tau)
        tau_mean = max(np.mean(taus), 0.01)
        theta_c = 2 * tau_mean / (1 - tau_mean)
        ll_c = np.sum(
            np.log(np.maximum(
                (theta_c + 1) * np.prod(u, axis=1) ** (-(theta_c + 1)) *
                (np.sum(u ** (-theta_c), axis=1) - d + 1) ** (-(2 + 1 / theta_c)),
                1e-300,
            ))
        )
        results["Clayton"] = {"aic": _aic(ll_c, 1), "theta": theta_c, "corr": corr}
    except Exception:
        results["Clayton"] = {"aic": np.inf, "theta": 1.0, "corr": corr}

    best = min(results, key=lambda k: results[k]["aic"])
    return best, results, corr

def simulate(best_name, results, pseudo_obs, n_sim=50_000):
    """
    Simulate joint returns from the selected copula.
    Returns uniform matrix (n_sim x d).
    """
    d = pseudo_obs.shape[1]
    info = results[best_name]
    corr = info["corr"]
    np.fill_diagonal(corr, 1.0)

    if best_name == "Gaussian":
        return _gaussian_copula_sim(corr, n_sim)
    elif best_name == "Student-t":
        nu = info.get("nu", 5) or 5
        return _t_copula_sim(corr, nu, n_sim)
    else:  # Clayton fallback
        theta = info.get("theta", 1.0) or 1.0
        return _clayton_copula_sim(theta, d, n_sim)