import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

def fit_marginals(returns: pd.DataFrame):
    """
    Fit GARCH(1,1) per asset. Return:
      - pseudo_obs : DataFrame of uniform [0,1] pseudo-observations
      - garch_fits : dict of fitted arch models (for diagnostics)
    """
    pseudo_obs = {}
    garch_fits = {}

    for col in returns.columns:
        r = returns[col].dropna() * 100  # arch expects percentage returns

        try:
            model = arch_model(r, vol="Garch", p=1, q=1, dist="Normal")
            res = model.fit(disp="off", show_warning=False)
            std_resid = res.resid / res.conditional_volatility
        except Exception:
            # Fallback: standardise with rolling std
            std_resid = (r - r.mean()) / r.std()

        std_resid = std_resid.dropna()

        # Empirical CDF → uniform pseudo-observations
        ranks = std_resid.rank(method="average")
        n = len(ranks)
        u = ranks / (n + 1)   # Blom-style to keep strictly in (0,1)

        pseudo_obs[col] = u.values
        garch_fits[col] = res if "res" in dir() else None

    # Align lengths (trim to shortest)
    min_len = min(len(v) for v in pseudo_obs.values())
    pseudo_obs = {k: v[-min_len:] for k, v in pseudo_obs.items()}

    return pd.DataFrame(pseudo_obs), garch_fits