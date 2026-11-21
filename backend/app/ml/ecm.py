"""Baltic Dry Index spot layer: an error-correction model (ECM) on BDRY.

BDRY holds near-dated Capesize/Panamax/Supramax FFAs, so it embeds the market's
expectation of where the physical index is heading. The BDI tends to close the
gap to the level its futures imply. The daily model is

    db[t+1] = c + a1*db[t] + a2*db[t-1] + g1*df[t] + g2*df[t-1]
              - gamma * (b[t] - alpha - beta*f[t]) + e[t+1]

with b = log BDI and f = log BDRY. ``b - alpha - beta*f`` is the basis to the
BDRY-implied level, estimated by a levels regression on the training window
only.

Timing: the BDI is published at about 13:00 London and BDRY closes at 16:00
New York. The forecast origin is therefore *after* BDRY's close on day t, so
``df[t]`` (the futures move during and after the index print) is legitimately
known when forecasting ``b[t+1]``.

The same daily recursion serves three purposes:

* walk-forward evaluation (future BDRY moves set to their expectation, zero);
* a nowcast from the last BDI print to today, driven by the actual BDRY closes;
* Monte Carlo forward paths, driven by GARCH-FHS BDRY paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.backtest import diebold_mariano

COEF_NAMES = ["const", "db_0", "db_1", "df_0", "df_1", "basis"]
MIN_TRAIN = 120
REFIT_EVERY = 10
EVAL_HORIZONS = [1, 5, 10, 21]
# Pre-specified signal policy. Walk-forward skill is positive at h<=5 and not
# distinguishable from zero beyond (about 6 independent 21-day windows in one
# year of BDI history), so the ECM's deviation from a random walk is used in
# full up to FULL_SIGNAL_H and tapered linearly to zero at ZERO_SIGNAL_H.
FULL_SIGNAL_H = 5
ZERO_SIGNAL_H = 21


def signal_weight(h: np.ndarray | int) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return np.clip((ZERO_SIGNAL_H - h) / (ZERO_SIGNAL_H - FULL_SIGNAL_H), 0.0, 1.0)


def align(bdi: pd.Series, bdry: pd.Series) -> pd.DataFrame:
    """BDI print dates with the BDRY close on or before each date."""
    f = bdry.reindex(bdry.index.union(bdi.index)).ffill().reindex(bdi.index)
    return pd.DataFrame({"b": np.log(bdi), "f": np.log(f)}).dropna()


def _design(b: np.ndarray, f: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows are forecast origins t = 3..n-2; the target is db[t+1]."""
    t = np.arange(3, len(b) - 1)
    X = np.column_stack([
        np.ones(len(t)),
        b[t] - b[t - 1], b[t - 1] - b[t - 2],
        f[t] - f[t - 1], f[t - 1] - f[t - 2],
        b[t] - alpha - beta * f[t],
    ])
    return X, b[t + 1] - b[t]


@dataclass
class ECMFit:
    alpha: float
    beta: float
    coef: np.ndarray  # COEF_NAMES order; the basis coefficient is -gamma
    resid: np.ndarray
    n_obs: int
    train_start: str
    train_end: str
    r2: float

    @property
    def gamma(self) -> float:
        return float(-self.coef[-1])

    @property
    def half_life_days(self) -> float | None:
        g = self.gamma
        return float(np.log(0.5) / np.log(1 - g)) if 0 < g < 1 else None

    @property
    def resid_sd(self) -> float:
        return float(np.std(self.resid, ddof=len(self.coef)))

    def implied_level(self, log_bdry: float) -> float:
        """BDI level consistent with a BDRY price (zero basis)."""
        return float(np.exp(self.alpha + self.beta * log_bdry))

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "coef": self.coef.tolist(), "resid": self.resid.tolist(),
                "n_obs": self.n_obs, "train_start": self.train_start, "train_end": self.train_end, "r2": self.r2}

    @classmethod
    def from_dict(cls, d: dict) -> ECMFit:
        return cls(alpha=d["alpha"], beta=d["beta"], coef=np.asarray(d["coef"]), resid=np.asarray(d["resid"]),
                   n_obs=d["n_obs"], train_start=d["train_start"], train_end=d["train_end"], r2=d["r2"])

    def summary(self) -> dict:
        return {
            "model": "Error-correction model: daily BDI log change on own lags, BDRY moves "
                     "and the basis to the BDRY-implied level",
            "coefficients": {k: round(float(v), 5) for k, v in zip(COEF_NAMES, self.coef)},
            "cointegration": {"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)},
            "adjustment_speed_gamma": round(self.gamma, 4),
            "basis_half_life_days": round(self.half_life_days, 1) if self.half_life_days else None,
            "residual_daily_sd_pct": round(self.resid_sd * 100, 3),
            "r2": round(self.r2, 3),
            "n_obs": self.n_obs,
            "train_start": self.train_start,
            "train_end": self.train_end,
        }


def fit_ecm(data: pd.DataFrame, ridge: float = 1e-4) -> ECMFit:
    b, f = data["b"].values, data["f"].values
    A = np.column_stack([np.ones(len(f)), f])
    alpha, beta = np.linalg.lstsq(A, b, rcond=None)[0]
    X, y = _design(b, f, alpha, beta)
    # Tiny ridge (not on the intercept) for numerical stability on short samples.
    P = np.eye(X.shape[1]) * ridge * len(y)
    P[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + P, X.T @ y)
    resid = y - X @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return ECMFit(alpha=float(alpha), beta=float(beta), coef=coef, resid=resid, n_obs=int(len(y)),
                  train_start=data.index[0].strftime("%Y-%m-%d"), train_end=data.index[-1].strftime("%Y-%m-%d"),
                  r2=float(r2))


def simulate(fit: ECMFit, b_hist: np.ndarray, f_hist: np.ndarray, f_future: np.ndarray,
             shocks: np.ndarray | None = None) -> np.ndarray:
    """Run the daily recursion forward.

    ``b_hist``/``f_hist``: observed log levels up to the origin (last three used).
    ``f_future``: log BDRY levels for days 1..H, shape (H,) or (n_paths, H).
    ``shocks``: (n_paths, H) residual draws, or None for the conditional mean.
    Returns log BDI levels, shape (n_paths, H).
    """
    F = np.atleast_2d(np.asarray(f_future, dtype=float))
    n, H = F.shape
    c = fit.coef
    b2, b1, b0 = (np.full(n, float(v)) for v in b_hist[-3:])
    f2, f1, f0 = (np.full(n, float(v)) for v in f_hist[-3:])
    out = np.empty((n, H))
    for k in range(H):
        db = (c[0] + c[1] * (b0 - b1) + c[2] * (b1 - b2) + c[3] * (f0 - f1) + c[4] * (f1 - f2)
              + c[5] * (b0 - fit.alpha - fit.beta * f0))
        if shocks is not None:
            db = db + shocks[:, k]
        nb = b0 + db
        out[:, k] = nb
        b2, b1, b0 = b1, b0, nb
        f2, f1, f0 = f1, f0, F[:, k]
    return out


def walk_forward(data: pd.DataFrame, horizons: list[int] = EVAL_HORIZONS, min_train: int = MIN_TRAIN,
                 refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """Expanding-window pseudo-live evaluation.

    At each origin t the model is fitted only on rows <= t (refitted every
    ``refit_every`` days). The h-step forecast iterates the recursion with
    future BDRY held at its last close (its expected value under the random
    walk that BDRY itself follows, see the BDRY backtest).
    """
    b, f = data["b"].values, data["f"].values
    rows = []
    fit = None
    max_h = max(horizons)
    for t in range(min_train, len(b) - 1):
        if fit is None or (t - min_train) % refit_every == 0:
            fit = fit_ecm(data.iloc[: t + 1])
        H = min(max_h, len(b) - 1 - t)
        path = simulate(fit, b[: t + 1], f[: t + 1], np.full(H, f[t]))[0]
        for h in horizons:
            if h <= H:
                rows.append({"date": data.index[t], "h": h, "y": b[t + h] - b[t], "ecm": path[h - 1] - b[t]})
    return pd.DataFrame(rows)


def evaluate(oof: pd.DataFrame) -> list[dict]:
    out = []
    for h, g in oof.groupby("h"):
        y, yhat = g["y"].values, g["ecm"].values
        e, e_rw = y - yhat, y
        mse, mse_rw = float(np.mean(e ** 2)), float(np.mean(e_rw ** 2))
        stat, p = diebold_mariano(e, e_rw, int(h))
        nz = (y != 0) & (yhat != 0)
        out.append({
            "h": int(h),
            "n": int(len(g)),
            "rmse_ecm_pct": round(np.sqrt(mse) * 100, 3),
            "rmse_rw_pct": round(np.sqrt(mse_rw) * 100, 3),
            "mae_ecm_pct": round(float(np.mean(np.abs(e))) * 100, 3),
            "skill_vs_rw_pct": round((1 - mse / mse_rw) * 100, 2),
            "directional_accuracy_pct": round(float(np.mean(np.sign(y[nz]) == np.sign(yhat[nz]))) * 100, 1),
            "dm_stat": None if np.isnan(stat) else round(stat, 3),
            "dm_pvalue": None if np.isnan(p) else round(p, 4),
            "first_origin": g["date"].min().strftime("%Y-%m-%d"),
            "last_origin": g["date"].max().strftime("%Y-%m-%d"),
        })
    return out


def nowcast(fit: ECMFit, data: pd.DataFrame, bdry: pd.Series, n_paths: int = 4000, seed: int = 0) -> dict:
    """Carry the BDI from its last print to the latest BDRY close.

    Driven by the *actual* BDRY closes after the last BDI print, with
    bootstrapped residuals for the uncertainty that builds while the index is
    unobserved.
    """
    last_date = data.index[-1]
    fut = bdry[bdry.index > last_date]
    fut = fut[fut.index.dayofweek < 5]
    gap = len(fut)
    last_bdi = float(np.exp(data["b"].iloc[-1]))
    if gap == 0:
        return {"last_print_date": last_date.strftime("%Y-%m-%d"), "last_print": last_bdi, "as_of": last_date.strftime("%Y-%m-%d"),
                "gap_bdays": 0, "level": last_bdi, "p10": last_bdi, "p90": last_bdi, "path": []}
    rng = np.random.default_rng(seed)
    F = np.log(fut.values)
    shocks = rng.choice(fit.resid, size=(n_paths, gap), replace=True)
    sims = np.exp(simulate(fit, data["b"].values, data["f"].values, np.tile(F, (n_paths, 1)), shocks))
    mean_path = np.exp(simulate(fit, data["b"].values, data["f"].values, F)[0])
    q = np.quantile(sims, [0.1, 0.5, 0.9], axis=0)
    path = [
        {"date": d.strftime("%Y-%m-%d"), "bdi": round(float(m), 1), "p10": round(float(lo), 1),
         "p50": round(float(md), 1), "p90": round(float(hi), 1), "bdry": round(float(px), 3)}
        for d, m, lo, md, hi, px in zip(fut.index, mean_path, q[0], q[1], q[2], fut.values)
    ]
    return {
        "last_print_date": last_date.strftime("%Y-%m-%d"),
        "last_print": round(last_bdi, 1),
        "as_of": fut.index[-1].strftime("%Y-%m-%d"),
        "gap_bdays": gap,
        "level": path[-1]["bdi"],
        "p10": path[-1]["p10"],
        "p90": path[-1]["p90"],
        "implied_level": round(fit.implied_level(F[-1]), 1),
        "path": path,
    }


def forecast_paths(fit: ECMFit, data: pd.DataFrame, bdry_hist: pd.Series, bdry_future_log: np.ndarray,
                   seed: int = 0) -> tuple[np.ndarray, int]:
    """Joint nowcast + forecast simulation.

    Days between the last BDI print and the latest BDRY close are driven by the
    actual BDRY closes; days after that by ``bdry_future_log`` (n_paths x H,
    simulated log BDRY levels). Each path carries its own nowcast state, so
    uncertainty about today's unobserved index propagates into the forecast.
    The forecast segment is re-centred per the signal policy: the path's mean
    deviation from a no-change forecast is scaled by ``signal_weight(h)``.

    Returns (log BDI paths for forecast days 1..H, nowcast gap in bdays).
    """
    n, H = bdry_future_log.shape
    last = data.index[-1]
    gap_px = bdry_hist[(bdry_hist.index > last) & (bdry_hist.index.dayofweek < 5)]
    G = len(gap_px)
    F = np.hstack([np.tile(np.log(gap_px.values), (n, 1)), bdry_future_log]) if G else bdry_future_log
    rng = np.random.default_rng(seed)
    shocks = rng.choice(fit.resid, size=(n, G + H), replace=True)
    b, f = data["b"].values, data["f"].values
    sims = simulate(fit, b, f, F, shocks)
    fc = sims[:, G:]
    origin = sims[:, G - 1] if G else np.full(n, b[-1])
    dev = fc.mean(axis=0) - origin.mean()
    w = signal_weight(np.arange(1, H + 1))
    fc = fc - (1 - w) * dev
    return fc, G
