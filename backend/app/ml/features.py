"""Leak-free feature engineering for direct multi-horizon freight forecasting.

Conventions (row ``t`` = target trading date):

* Target-derived features use prices up to and including ``t``.
* Exogenous features are lagged one business day. Brent, iron ore, FX and
  ASX-listed series close at different times from the US-listed target, so
  lagging guarantees no asynchronous-close lookahead.
* Calendar features are deterministic and known in advance.
* Labels: ``y_h(t) = log P(t+h) - log P(t)`` in trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.catalog import EQUITY_KEYS, TARGET_KEY

# Lunar New Year: Chinese industrial activity and dry-bulk demand slump around it.
LUNAR_NEW_YEAR = pd.to_datetime([
    "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01",
    "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06", "2028-01-26",
])

DRIVER_RETURN_KEYS = ["brent", "iron_ore", "coal", "copper", "dxy", "usdinr", "spx"]
WARMUP_DAYS = 253


def _robust_log_returns(s: pd.Series, clip_sd: float = 6.0) -> pd.Series:
    """Daily log returns winsorised at +/- clip_sd robust (MAD) sigmas.

    Front-month futures continuations jump at contract rolls; winsorising keeps
    those artefacts from dominating multi-day return features.
    """
    r = np.log(s).diff()
    med = r.median()
    mad = (r - med).abs().median() * 1.4826 or r.std()
    return r.clip(med - clip_sd * mad, med + clip_sd * mad)


def _days_to_lny(dates: pd.DatetimeIndex) -> np.ndarray:
    """Signed calendar days to the nearest Lunar New Year (negative = before)."""
    d = dates.values.astype("datetime64[D]").astype("int64")[:, None]
    lny = LUNAR_NEW_YEAR.values.astype("datetime64[D]").astype("int64")[None, :]
    diff = d - lny  # positive = after LNY
    nearest = diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]
    return np.clip(nearest, -60, 60).astype(float)


def build_features(panel: pd.DataFrame, drivers: list[str] | None = None) -> pd.DataFrame:
    p = panel[TARGET_KEY]
    lp = np.log(p)
    r = lp.diff()
    f = pd.DataFrame(index=panel.index)

    # --- Target dynamics: momentum, mean reversion, volatility regime
    for k in (1, 5, 10, 21, 63, 126):
        f[f"ret_{k}"] = lp - lp.shift(k)
    for k in (10, 21, 63):
        f[f"vol_{k}"] = r.rolling(k).std()
    f["vol_ratio_10_63"] = f["vol_10"] / f["vol_63"]
    for k in (63, 252):
        m, sd = lp.rolling(k).mean(), lp.rolling(k).std()
        f[f"z_{k}"] = (lp - m) / sd
    f["drawdown_252"] = lp - lp.rolling(252).max()
    f["runup_252"] = lp - lp.rolling(252).min()
    f["trend_21_63"] = lp.rolling(21).mean() - lp.rolling(63).mean()

    # --- Exogenous drivers (lagged one business day)
    drivers = drivers if drivers is not None else [c for c in DRIVER_RETURN_KEYS if c in panel]
    for key in drivers:
        dr = _robust_log_returns(panel[key]).shift(1)
        f[f"{key}_ret_5"] = dr.rolling(5).sum()
        f[f"{key}_ret_21"] = dr.rolling(21).sum()
    if "brent" in drivers:
        f["brent_ret_63"] = _robust_log_returns(panel["brent"]).shift(1).rolling(63).sum()

    if "vix" in panel:
        lv = np.log(panel["vix"]).shift(1)
        f["vix_log"] = lv
        f["vix_z_252"] = (lv - lv.rolling(252).mean()) / lv.rolling(252).std()

    eq_cols = [k for k in EQUITY_KEYS if k in panel]
    if eq_cols:
        eq = pd.concat([_robust_log_returns(panel[k]) for k in eq_cols], axis=1).mean(axis=1).shift(1)
        f["equity_ret_5"] = eq.rolling(5).sum()
        f["equity_ret_21"] = eq.rolling(21).sum()
        f["equity_ret_63"] = eq.rolling(63).sum()
        # Listed owners tend to lead physical freight: gap between the two.
        f["equity_minus_freight_21"] = f["equity_ret_21"] - r.shift(1).rolling(21).sum()

    # --- Calendar (known in advance)
    doy = panel.index.dayofyear.values
    f["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    f["days_to_lny"] = _days_to_lny(panel.index)
    f["is_sw_monsoon"] = panel.index.month.isin([6, 7, 8, 9]).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def build_labels(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    lp = np.log(panel[TARGET_KEY])
    return pd.DataFrame({h: lp.shift(-h) - lp for h in horizons}, index=panel.index)


def active_drivers(panel: pd.DataFrame, max_stale_bdays: int = 10) -> list[str]:
    """Drivers whose latest value is recent enough to be used for live forecasts."""
    out = []
    end = panel.index[-1]
    for key in DRIVER_RETURN_KEYS:
        if key not in panel:
            continue
        last_valid = panel[key].last_valid_index()
        if last_valid is not None and len(pd.bdate_range(last_valid, end)) - 1 <= max_stale_bdays:
            out.append(key)
    return out


# Compact, economically motivated feature set used by the production models.
# Fewer, more stable inputs generalise better given the small number of
# independent multi-week observations in eight years of daily data.
CORE_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126",
    "z_63", "z_252", "drawdown_252", "vol_ratio_10_63",
    "equity_ret_21", "equity_minus_freight_21",
    "brent_ret_21", "iron_ore_ret_21", "coal_ret_21", "copper_ret_21", "spx_ret_21",
    "vix_z_252",
    "sin_doy", "cos_doy", "days_to_lny",
]
