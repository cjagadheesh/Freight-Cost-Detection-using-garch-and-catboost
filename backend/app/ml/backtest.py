"""Walk-forward (expanding window) evaluation with purging.

For every quarterly fold starting at position ``a``:

* learners for horizon ``h`` train only on rows ``t <= a-1-h`` so that every
  training label ``log P(t+h)`` was observed before the test period (purging);
* GARCH parameters are estimated on returns before ``a`` and filtered forward;
* stacking weights and conformal quantiles for fold ``k`` use only out-of-fold
  rows whose labels were realised before fold ``k`` starts.

The resulting out-of-fold (OOF) predictions are therefore genuine pseudo-live
forecasts, and every metric below is computed on them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from app.ml.garch import rolling_cumulative_std
from app.ml.models import MODEL_NAMES, fit_learners, fit_stacking_weights, predict_learners

logger = logging.getLogger(__name__)

INTERVAL_LEVELS = {50: (0.25, 0.75), 80: (0.10, 0.90), 95: (0.025, 0.975)}
MIN_CALIBRATION = 60
GAUSS_Z = {50: 0.674, 80: 1.2816, 95: 1.96}


@dataclass
class BacktestResult:
    oof: pd.DataFrame
    metrics: list[dict]
    folds: list[dict]


def _fold_starts(n: int, first: int, step: int) -> list[int]:
    return list(range(first, n, step))


def run_walk_forward(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    log_returns: pd.Series,
    vol: pd.Series,
    horizons: list[int],
    first_row: int,
    initial_train: int,
    step: int,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    dates = X.index
    n = len(X)
    max_h = max(horizons)
    starts = _fold_starts(n, first_row + initial_train, step)
    rows = []
    for k, a in enumerate(starts):
        t0 = time.perf_counter()
        b = min(a + step, n)
        origins = dates[a:b]
        sig = rolling_cumulative_std(log_returns, fit_end=dates[a], origins=origins, horizon=max_h)
        for h in horizons:
            train_end = a - 1 - h
            if train_end - first_row < 250:
                continue
            tr = slice(first_row, train_end + 1)
            Xtr = X.iloc[tr]
            scale_tr = vol.iloc[tr] * np.sqrt(h)
            ytr = Y[h].iloc[tr] / scale_tr  # volatility-standardised label
            mask = ytr.notna() & np.isfinite(ytr)
            learners = fit_learners(Xtr[mask], ytr[mask], iterations, seed)
            te_pos = np.arange(a, b)
            te_pos = te_pos[te_pos + h <= n - 1]
            if len(te_pos) == 0:
                continue
            Xte = X.iloc[te_pos]
            scale_te = (vol.iloc[te_pos] * np.sqrt(h)).values
            preds = {m: v * scale_te for m, v in predict_learners(learners, Xte).items()}
            frame = pd.DataFrame({
                "date": dates[te_pos], "pos": te_pos, "h": h, "fold": k, "fold_start": a,
                "y": Y[h].iloc[te_pos].values, "sigma": sig.loc[dates[te_pos], h].values,
                **preds,
            })
            rows.append(frame)
        logger.info("Fold %d/%d (%s .. %s) done in %.1fs", k + 1, len(starts),
                    dates[a].date(), dates[b - 1].date(), time.perf_counter() - t0)
    return pd.concat(rows, ignore_index=True)


def calibrate_sequentially(oof: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Fold-by-fold stacking weights and conformal intervals using only the past."""
    out = []
    weight_log = []
    for h, g in oof.groupby("h"):
        g = g.sort_values("pos").copy()
        g["ensemble"] = np.nan
        for col in [f"{s}{lvl}" for lvl in INTERVAL_LEVELS for s in ("lo", "hi")]:
            g[col] = np.nan
        g["calibrated"] = False
        for a, fold in g.groupby("fold_start", sort=True):
            past = g[(g["pos"] + h < a) & g["ensemble"].notna()]
            P = fold[MODEL_NAMES].values
            if len(past) >= MIN_CALIBRATION:
                w = fit_stacking_weights(past[MODEL_NAMES].values, past["y"].values)
                ens = P @ w
                scores = ((past["y"] - past["ensemble"]) / past["sigma"]).values
                bands = {lvl: np.quantile(scores, q) for lvl, q in INTERVAL_LEVELS.items()}
                calibrated = True
            else:
                w = np.array([1.0, 0.0, 0.0])
                ens = P @ w
                bands = {lvl: (-GAUSS_Z[lvl], GAUSS_Z[lvl]) for lvl in INTERVAL_LEVELS}
                calibrated = False
            idx = fold.index
            g.loc[idx, "ensemble"] = ens
            g.loc[idx, "calibrated"] = calibrated
            sig = fold["sigma"].values
            for lvl, (ql, qh) in bands.items():
                g.loc[idx, f"lo{lvl}"] = ens + ql * sig
                g.loc[idx, f"hi{lvl}"] = ens + qh * sig
            weight_log.append({"h": int(h), "fold_start": int(a), **dict(zip(MODEL_NAMES, np.round(w, 3)))})
        out.append(g)
    return pd.concat(out, ignore_index=True), weight_log


ACI_GAMMA = 0.002


def adaptive_conformal(oof: pd.DataFrame, gamma: float = ACI_GAMMA, warmup: int = MIN_CALIBRATION) -> tuple[pd.DataFrame, dict]:
    """Adaptive conformal inference (Gibbs & Candes, 2021) with delayed feedback.

    Scores are vol-scaled residuals ``(y - ensemble) / sigma``. For each
    horizon and level the miscoverage target ``alpha_t`` is nudged after every
    realised outcome: ``alpha <- alpha + gamma * (alpha* - miss)``. An outcome
    for origin j is only known at j + h, so the update for row i uses rows
    j <= i - h. Static split-conformal under-covers at 42-63 days because
    freight volatility regimes outlast GARCH mean reversion; ACI corrects that
    online without peeking. Returns the frame and the final alpha state per
    (h, level) for production intervals.
    """
    out, state = [], {}
    for h, g in oof.groupby("h"):
        g = g.sort_values("pos").reset_index(drop=True)
        s = ((g["y"] - g["ensemble"]) / g["sigma"]).values
        pos = g["pos"].values
        n = len(g)
        calibrated = np.zeros(n, dtype=bool)
        state[int(h)] = {}
        for lvl, (ql, qh) in INTERVAL_LEVELS.items():
            target = qh - ql
            a = 1.0 - target
            inside = np.full(n, np.nan)
            lo_col, hi_col = np.full(n, np.nan), np.full(n, np.nan)
            k = 0  # pointer: rows [0, k) have realised outcomes at time i
            for i in range(n):
                while k < n and pos[k] + h <= pos[i]:
                    if not np.isnan(inside[k]):
                        a += gamma * ((1.0 - target) - (1.0 - inside[k]))
                    k += 1
                if k < warmup:
                    continue
                ae = float(np.clip(a, 1e-3, 0.999))
                lo, hi = np.quantile(s[:k], [ae / 2, 1 - ae / 2])
                inside[i] = float(lo <= s[i] <= hi)
                lo_col[i], hi_col[i] = lo, hi
                calibrated[i] = True
            # Finish updating with every outcome realised by the end of the sample.
            while k < n:
                if not np.isnan(inside[k]):
                    a += gamma * ((1.0 - target) - (1.0 - inside[k]))
                k += 1
            sig, ens = g["sigma"].values, g["ensemble"].values
            g[f"lo{lvl}"] = ens + lo_col * sig
            g[f"hi{lvl}"] = ens + hi_col * sig
            state[int(h)][lvl] = float(np.clip(a, 1e-3, 0.999))
        g["calibrated"] = calibrated
        out.append(g)
    return pd.concat(out, ignore_index=True), state


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float]:
    """One-sided DM test (H1: model has lower squared error than benchmark),
    Newey-West variance with h-1 lags and Harvey-Leybourne-Newbold correction."""
    d = e_bench ** 2 - e_model ** 2
    n = len(d)
    if n < 20 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dm = d - d.mean()
    lags = max(h - 1, 0)
    var = dm @ dm / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dm[k:] @ dm[:-k]) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n)
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    p = 1 - stats.t.cdf(stat, df=n - 1)
    return float(stat), float(p)


def compute_metrics(oof: pd.DataFrame) -> list[dict]:
    results = []
    for h, g in oof.groupby("h"):
        y = g["y"].values
        e_rw = y - g["random_walk"].values
        mse_rw = float(np.mean(e_rw ** 2))
        for m in MODEL_NAMES + ["ensemble"]:
            yhat = g[m].values
            e = y - yhat
            mse = float(np.mean(e ** 2))
            nz = (yhat != 0) & (y != 0)
            row = {
                "h": int(h),
                "model": m,
                "n": int(len(y)),
                "mae_pct": round(float(np.mean(np.abs(e))) * 100, 3),
                "rmse_pct": round(float(np.sqrt(mse)) * 100, 3),
                "mape_price_pct": round(float(np.mean(np.abs(np.expm1(yhat - y)))) * 100, 3),
                "skill_vs_rw_pct": round((1 - mse / mse_rw) * 100, 2),
                "directional_accuracy_pct": (
                    round(float(np.mean(np.sign(yhat[nz]) == np.sign(y[nz]))) * 100, 1) if nz.sum() > 20 else None
                ),
            }
            if m != "random_walk":
                stat, p = diebold_mariano(e, e_rw, int(h))
                row["dm_stat"] = None if np.isnan(stat) else round(stat, 3)
                row["dm_pvalue"] = None if np.isnan(p) else round(p, 4)
            if m == "ensemble":
                cal = g[g["calibrated"]]
                for lvl in INTERVAL_LEVELS:
                    inside = (cal["y"] >= cal[f"lo{lvl}"]) & (cal["y"] <= cal[f"hi{lvl}"])
                    row[f"coverage{lvl}_pct"] = round(float(inside.mean()) * 100, 1) if len(cal) else None
                row["width80_pct"] = (
                    round(float(np.mean(np.expm1(cal["hi80"]) - np.expm1(cal["lo80"]))) * 100, 2) if len(cal) else None
                )
            results.append(row)
    return results


def fold_summary(oof: pd.DataFrame, dates: pd.DatetimeIndex, ref_h: int) -> list[dict]:
    out = []
    g = oof[oof["h"] == ref_h]
    for k, f in g.groupby("fold"):
        e_rw = f["y"] - f["random_walk"]
        e_en = f["y"] - f["ensemble"]
        out.append({
            "fold": int(k),
            "test_start": f["date"].min().strftime("%Y-%m-%d"),
            "test_end": f["date"].max().strftime("%Y-%m-%d"),
            "train_rows": int(f["fold_start"].iloc[0] - ref_h),
            "rmse_rw_pct": round(float(np.sqrt(np.mean(e_rw ** 2))) * 100, 2),
            "rmse_ensemble_pct": round(float(np.sqrt(np.mean(e_en ** 2))) * 100, 2),
        })
    return out
