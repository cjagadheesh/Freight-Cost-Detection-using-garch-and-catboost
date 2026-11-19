"""GARCH(1,1) with Student-t innovations for freight-return volatility.

    r_t = mu + e_t,   e_t = sigma_t * z_t,   z_t ~ t(nu) (unit variance)
    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

Used for (a) the volatility term structure that scales prediction intervals and
(b) filtered-historical-simulation Monte Carlo paths.
Returns are modelled in percent (x100) for optimiser stability.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    loglik: float
    aic: float
    bic: float
    n_obs: int
    last_variance: float  # one-step-ahead variance for the day after the last obs, pct^2
    last_resid: float  # pct
    std_resid: np.ndarray  # standardised residuals (for bootstrap)
    cond_vol: pd.Series  # in-sample conditional volatility, decimal daily

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def half_life_days(self) -> float | None:
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else None

    @property
    def long_run_daily_vol(self) -> float:
        p = self.persistence
        return float(np.sqrt(self.omega / (1 - p)) / 100) if p < 1 else float("nan")

    def variance_path(self, horizon: int) -> np.ndarray:
        """Daily variance forecasts (decimal^2) for steps 1..horizon."""
        p = self.persistence
        lr = self.omega / (1 - p) if p < 1 else self.last_variance
        steps = np.arange(horizon)
        v = lr + (p ** steps) * (self.last_variance - lr)
        return v / 1e4

    def cumulative_std(self, horizon: int) -> np.ndarray:
        """Std of the cumulative log return over 1..horizon days (decimal)."""
        return np.sqrt(np.cumsum(self.variance_path(horizon)))

    def summary(self) -> dict:
        return {
            "model": "GARCH(1,1), Student-t innovations",
            "mu_pct": round(self.mu, 5),
            "omega": round(self.omega, 6),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "persistence": round(self.persistence, 4),
            "nu": round(self.nu, 2),
            "half_life_days": round(self.half_life_days, 1) if self.half_life_days else None,
            "current_daily_vol_pct": round(float(np.sqrt(self.last_variance)), 3),
            "long_run_daily_vol_pct": round(self.long_run_daily_vol * 100, 3),
            "current_annual_vol_pct": round(float(np.sqrt(self.last_variance * 252)), 1),
            "log_likelihood": round(self.loglik, 1),
            "aic": round(self.aic, 1),
            "bic": round(self.bic, 1),
            "n_obs": self.n_obs,
        }


def _model(returns_pct: pd.Series):
    return arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)


def fit_garch(log_returns: pd.Series) -> GarchFit:
    y = log_returns.dropna() * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(y).fit(disp="off", show_warning=False)
    prm = res.params
    mu, omega, alpha, beta, nu = (float(prm[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]", "nu"))
    cv = np.asarray(res.conditional_volatility)
    resid = np.asarray(res.resid)
    last_var = omega + alpha * resid[-1] ** 2 + beta * cv[-1] ** 2
    std_resid = resid / cv
    return GarchFit(
        mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu,
        loglik=float(res.loglikelihood), aic=float(res.aic), bic=float(res.bic), n_obs=int(res.nobs),
        last_variance=float(last_var), last_resid=float(resid[-1]),
        std_resid=std_resid[np.isfinite(std_resid)],
        cond_vol=pd.Series(cv / 100.0, index=y.index, name="garch_vol"),
    )


def rolling_cumulative_std(log_returns: pd.Series, fit_end: pd.Timestamp, origins: pd.DatetimeIndex,
                           horizon: int) -> pd.DataFrame:
    """Out-of-sample sigma term structure for backtesting.

    Parameters are estimated only on returns up to ``fit_end``; the conditional
    variance is then filtered forward through later observed returns, so the
    forecast made at origin ``t`` uses information up to ``t`` only.
    Returns a frame (origins x 1..horizon) of cumulative-return std (decimal).
    """
    y = log_returns.dropna() * 100.0
    am = _model(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False, last_obs=fit_end)
        fc = res.forecast(horizon=horizon, start=origins[0], reindex=False)
    var = fc.variance.reindex(origins)
    cum = np.sqrt(var.cumsum(axis=1).values / 1e4)
    return pd.DataFrame(cum, index=origins, columns=range(1, horizon + 1))


def simulate_paths(fit: GarchFit, horizon: int, n_paths: int, drift: np.ndarray | None = None,
                   seed: int = 0) -> np.ndarray:
    """Filtered historical simulation: bootstrap standardised residuals through
    the fitted GARCH recursion. Returns cumulative log returns (n_paths x horizon).
    ``drift`` (len horizon, decimal per day) replaces the constant mean."""
    rng = np.random.default_rng(seed)
    z = rng.choice(fit.std_resid, size=(n_paths, horizon), replace=True)
    var = np.full(n_paths, fit.last_variance)
    out = np.empty((n_paths, horizon))
    mu = drift * 100.0 if drift is not None else np.full(horizon, fit.mu)
    for k in range(horizon):
        sig = np.sqrt(var)
        eps = sig * z[:, k]
        out[:, k] = mu[k] + eps
        var = fit.omega + fit.alpha * eps ** 2 + fit.beta * var
    return np.cumsum(out / 100.0, axis=1)
