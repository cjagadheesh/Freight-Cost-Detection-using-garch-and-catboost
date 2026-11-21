"""Data source connectors.

* ``fetch_yahoo_daily`` – Yahoo Finance chart API (public, no key) for ETFs,
  futures, FX, indices and equities.
* ``load_koyfin_bdi`` – Baltic Dry Index exports from Koyfin (licensed data the
  user downloads; any number of ``koyfin_*.csv`` files are merged).
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OceanPulse/3.0)"}


class SourceError(RuntimeError):
    pass


def fetch_yahoo_daily(symbol: str, start: str, timeout: float = 20.0) -> pd.Series:
    """Daily (adjusted where available) closes indexed by exchange-local date."""
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(pd.Timestamp.now(tz="UTC").timestamp()) + 86400
    params = {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"}
    try:
        resp = httpx.get(YAHOO_URL.format(symbol=symbol), params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceError(f"Yahoo request failed for {symbol}: {exc}") from exc

    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise SourceError(f"Yahoo returned no data for {symbol}: {payload.get('chart', {}).get('error')}")
    r = result[0]
    ts = r.get("timestamp") or []
    if not ts:
        raise SourceError(f"Yahoo returned an empty series for {symbol}")

    gmtoffset = int(r.get("meta", {}).get("gmtoffset", 0))
    quote = r["indicators"]["quote"][0]
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    values = adj if adj is not None else quote["close"]

    # Convert epoch to the exchange-local calendar date so that, e.g., an Asian
    # session is not shifted onto the previous UTC day.
    dates = pd.to_datetime(np.asarray(ts, dtype="int64") + gmtoffset, unit="s").normalize()
    s = pd.Series(pd.to_numeric(values, errors="coerce"), index=dates, name=symbol, dtype="float64")
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s > 0]
    # Drop weekend stamps (some futures publish Sunday-evening sessions).
    s = s[s.index.dayofweek < 5].sort_index()
    return _drop_unfinished_session(s, r.get("meta", {}), ts)


def _drop_unfinished_session(s: pd.Series, meta: dict, ts: list[int], now: float | None = None) -> pd.Series:
    """Remove today's bar while the exchange's regular session is still open.

    During trading hours Yahoo returns the live price as the latest daily
    "close"; storing it would freeze an intraday value into the history
    (the audit found exactly this for 19-Nov-2026). Uses the exchange's own
    ``currentTradingPeriod`` so it works across time zones and asset classes.
    """
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    start, end = regular.get("start"), regular.get("end")
    if not len(s) or start is None or end is None or not ts:
        return s
    now = pd.Timestamp.now(tz="UTC").timestamp() if now is None else now
    if start <= max(ts) and now < end:
        return s.iloc[:-1]
    return s


def load_koyfin_bdi(directory: Path) -> pd.Series:
    """Merge all Koyfin BDI exports in ``directory`` into one daily close series."""
    frames = []
    for path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = [c.replace('"', "").strip() for c in df.columns]
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        close_cols = [c for c in df.columns if "close" in c.lower()]
        if not close_cols:
            logger.warning("Skipping %s: no close column", path.name)
            continue
        dates = pd.to_datetime(df[date_col], format="%m-%d-%Y", errors="coerce")
        if dates.isna().mean() > 0.5:
            dates = pd.to_datetime(df[date_col], errors="coerce")
        s = pd.Series(pd.to_numeric(df[close_cols[-1]], errors="coerce").values, index=dates)
        frames.append(s.dropna())
    if not frames:
        return pd.Series(dtype="float64", name="bdi")
    bdi = pd.concat(frames)
    bdi = bdi[~bdi.index.duplicated(keep="last")].sort_index()
    bdi = bdi[(bdi > 100) & (bdi < 20000)]
    bdi.name = "bdi"
    return bdi
