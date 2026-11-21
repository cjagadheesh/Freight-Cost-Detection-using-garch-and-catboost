"""Live forecast engine.

Two layers, each used where the walk-forward evidence supports it:

1. **BDRY (freight futures)**: stacked Ridge/CatBoost/random-walk ensemble per
   horizon, GARCH-t volatility term structure and adaptive-conformal interval
   quantiles. Out-of-sample the learners do not beat a random walk, so the
   stacking weights put (almost) all mass on it. That is the result, and the
   engine reports it.
2. **BDI spot (index points and $/day)**: the error-correction model run
   through the nowcast gap on actual BDRY closes, then forward on Monte Carlo
   GARCH-FHS BDRY paths.

Everything is computed from the saved bundle and the latest market panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.data.ingest import MarketData
from app.ml import ecm
from app.ml.backtest import INTERVAL_LEVELS
from app.ml.features import build_features
from app.ml.garch import simulate_paths
from app.ml.models import MODEL_NAMES, predict_learners
from app.ml.train import Bundle, realised_vol
from app.ml.translate import DEFAULT_CLASS_RATIOS

MAX_H = 63
N_PATHS = 5000
PATH_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass
class ForecastState:
    as_of: pd.Timestamp
    dates: pd.DatetimeIndex  # forecast business days 1..MAX_H
    bdry_last: float
    bdry_points: list[dict]  # per configured horizon
    bdry_path: list[dict]  # per day, central + bands
    bdry_paths: np.ndarray  # (N_PATHS, MAX_H) price paths
    bdi_paths: np.ndarray | None  # (N_PATHS, MAX_H) index level paths
    bdi_nowcast: dict | None
    bdi_path: list[dict] | None
    x_last: pd.DataFrame
    member_preds: dict  # h -> {model: log-return}
    seed: int

    def tce_paths(self, vessel_class: str, ratios: dict[str, float] | None = None) -> np.ndarray:
        if self.bdi_paths is None:
            raise RuntimeError("BDI layer unavailable (no Koyfin BDI history)")
        r = (ratios or DEFAULT_CLASS_RATIOS)[vessel_class]
        return self.bdi_paths * 10.0 * r

    def tce_today(self, vessel_class: str, ratios: dict[str, float] | None = None) -> float:
        r = (ratios or DEFAULT_CLASS_RATIOS)[vessel_class]
        return float(self.bdi_nowcast["level"]) * 10.0 * r


def _interp(h_known: list[int], values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation over horizon with value 0 at h=0."""
    return np.interp(grid, [0, *h_known], [0.0, *values])


def build_state(bundle: Bundle, md: MarketData, seed: int = 7, n_paths: int = N_PATHS) -> ForecastState:
    panel = md.panel
    X = build_features(panel, bundle.drivers)[bundle.features]
    x_last = X.iloc[[-1]]
    r = np.log(md.target).diff()
    vol_t = float(realised_vol(r).iloc[-1])
    p0 = float(md.target.iloc[-1])
    as_of = md.as_of
    hs = bundle.horizons

    sig_cum = bundle.garch.cumulative_std(MAX_H)  # decimal, days 1..MAX_H
    mean_h, member_preds, points = [], {}, []
    for h in hs:
        preds = {m: float(v[0]) * vol_t * np.sqrt(h) for m, v in predict_learners(bundle.learners[h], x_last).items()}
        w = bundle.weights[h]
        mu = float(np.dot([preds[m] for m in MODEL_NAMES], w))
        member_preds[h] = preds
        mean_h.append(mu)
        sig = float(sig_cum[min(h, MAX_H) - 1])
        q = bundle.score_quantiles[h]
        row = {"h": h, "mean_log_return": round(mu, 5), "point": round(p0 * np.exp(mu), 4), "sigma_pct": round(sig * 100, 2),
               "weights": dict(zip(MODEL_NAMES, np.round(w, 3).tolist())),
               "members": {m: round(p0 * np.exp(v), 4) for m, v in preds.items()}}
        for lvl in INTERVAL_LEVELS:
            row[f"lo{lvl}"] = round(p0 * np.exp(mu + q[lvl][0] * sig), 4)
            row[f"hi{lvl}"] = round(p0 * np.exp(mu + q[lvl][1] * sig), 4)
        points.append(row)

    grid = np.arange(1, MAX_H + 1)
    mu_path = _interp(hs, np.array(mean_h), grid)
    dates = pd.bdate_range(as_of + pd.offsets.BDay(1), periods=MAX_H)
    q_lo = {lvl: _interp(hs, np.array([bundle.score_quantiles[h][lvl][0] for h in hs]), grid) for lvl in INTERVAL_LEVELS}
    q_hi = {lvl: _interp(hs, np.array([bundle.score_quantiles[h][lvl][1] for h in hs]), grid) for lvl in INTERVAL_LEVELS}
    bdry_path = []
    for k, d in enumerate(dates):
        row = {"date": d.strftime("%Y-%m-%d"), "h": k + 1, "point": round(p0 * np.exp(mu_path[k]), 4)}
        for lvl in INTERVAL_LEVELS:
            row[f"lo{lvl}"] = round(p0 * np.exp(mu_path[k] + q_lo[lvl][k] * sig_cum[k]), 4)
            row[f"hi{lvl}"] = round(p0 * np.exp(mu_path[k] + q_hi[lvl][k] * sig_cum[k]), 4)
        bdry_path.append(row)

    # Monte Carlo: filtered historical simulation with the ensemble's daily drift.
    drift = np.diff(np.concatenate([[0.0], mu_path]))
    cum = simulate_paths(bundle.garch, MAX_H, n_paths, drift=drift, seed=seed)
    bdry_paths = p0 * np.exp(cum)

    bdi_paths = bdi_nowcast = bdi_path = None
    if bundle.ecm_fit is not None and len(md.bdi):
        data = ecm.align(md.bdi, md.target)
        bdi_nowcast = ecm.nowcast(bundle.ecm_fit, data, md.target, seed=seed)
        log_paths, _ = ecm.forecast_paths(bundle.ecm_fit, data, md.target, np.log(bdry_paths), seed=seed + 1)
        bdi_paths = np.exp(log_paths)
        qs = np.quantile(bdi_paths, PATH_QUANTILES, axis=0)
        mean = bdi_paths.mean(axis=0)
        bdi_path = [
            {"date": d.strftime("%Y-%m-%d"), "h": k + 1, "mean": round(float(mean[k]), 1),
             **{f"p{int(q * 100)}": round(float(qs[i, k]), 1) for i, q in enumerate(PATH_QUANTILES)}}
            for k, d in enumerate(dates)
        ]

    return ForecastState(as_of=as_of, dates=dates, bdry_last=p0, bdry_points=points, bdry_path=bdry_path,
                         bdry_paths=bdry_paths, bdi_paths=bdi_paths, bdi_nowcast=bdi_nowcast, bdi_path=bdi_path,
                         x_last=x_last, member_preds=member_preds, seed=seed)


def explain(bundle: Bundle, state: ForecastState, h: int, top: int = 12) -> dict:
    """Exact Tree SHAP for the CatBoost member at horizon ``h`` (latest row).

    Contributions are in units of the vol-standardised label and are converted
    to log-return units with today's vol scale. The ensemble weight on CatBoost
    is reported alongside: with a weight near zero the explanation describes
    what the learner sees, not what drives the published forecast.
    """
    from catboost import Pool

    cb = bundle.learners[h]["catboost"]
    sv = cb.get_feature_importance(Pool(state.x_last), type="ShapValues")[0]
    contrib, base = sv[:-1], sv[-1]
    scale = state.member_preds[h]["catboost"] / (float(cb.predict(state.x_last)[0]) or 1e-12)
    order = np.argsort(-np.abs(contrib))[:top]
    w_cb = float(bundle.weights[h][MODEL_NAMES.index("catboost")])
    return {
        "h": h,
        "catboost_weight_in_ensemble": round(w_cb, 3),
        "base_value": round(float(base * scale), 5),
        "prediction_log_return": round(float(state.member_preds[h]["catboost"]), 5),
        "contributions": [
            {"feature": bundle.features[i], "value": None if pd.isna(state.x_last.iloc[0, i]) else round(float(state.x_last.iloc[0, i]), 4),
             "shap_log_return": round(float(contrib[i] * scale), 5)}
            for i in order
        ],
        "global_importance": dict(list(bundle.feature_importance[h].items())[:top]),
    }
