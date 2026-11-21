"""Ingestion: fetch real series, persist a raw snapshot, build the modelling panel."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import Settings
from app.data.catalog import CATALOG, CATALOG_BY_KEY, TARGET_KEY
from app.data.sources import SourceError, fetch_yahoo_daily, load_koyfin_bdi

logger = logging.getLogger(__name__)

MAX_FFILL_DAYS = 5


@dataclass
class MarketData:
    panel: pd.DataFrame  # indexed by target trading dates; one column per series
    bdi: pd.Series  # Baltic Dry Index (Koyfin exports), may be short
    raw: dict[str, pd.Series]
    manifest: dict = field(default_factory=dict)

    @property
    def target(self) -> pd.Series:
        return self.panel[TARGET_KEY]

    @property
    def as_of(self) -> pd.Timestamp:
        return self.panel.index[-1]


def refresh_raw(settings: Settings) -> dict:
    """Download every catalogue series; keep the previous snapshot for any that fail."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report = {"fetched": [], "failed": {}}
    for spec in CATALOG:
        try:
            s = fetch_yahoo_daily(spec.symbol, settings.history_start, settings.http_timeout_s)
            if len(s) < 250:
                raise SourceError(f"only {len(s)} observations")
            s.rename("value").rename_axis("date").to_csv(settings.raw_dir / f"{spec.key}.csv")
            manifest[spec.key] = {
                "symbol": spec.symbol,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "observations": int(len(s)),
                "last_date": s.index[-1].strftime("%Y-%m-%d"),
            }
            report["fetched"].append(spec.key)
        except SourceError as exc:
            logger.warning("Keeping cached %s: %s", spec.key, exc)
            report["failed"][spec.key] = str(exc)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return report


def load_raw(settings: Settings) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for spec in CATALOG:
        path = settings.raw_dir / f"{spec.key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        out[spec.key] = df.set_index("date")["value"].astype(float).sort_index()
    return out


def build_panel(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Align every series to the target's trading calendar.

    Each driver takes its last observation on or before the target date (causal
    forward-fill, at most ``MAX_FFILL_DAYS`` business days), so no future values
    leak into a row.
    """
    if TARGET_KEY not in raw:
        raise RuntimeError("Target series (BDRY) is missing from the raw snapshot")
    target = raw[TARGET_KEY]
    idx = target.index
    cols = {TARGET_KEY: target}
    for key, s in raw.items():
        if key == TARGET_KEY:
            continue
        bday = s.reindex(pd.bdate_range(min(s.index[0], idx[0]), max(s.index[-1], idx[-1])))
        bday = bday.ffill(limit=MAX_FFILL_DAYS)
        cols[key] = bday.reindex(idx)
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    return panel


def quality_report(raw: dict[str, pd.Series], bdi: pd.Series, as_of: pd.Timestamp) -> list[dict]:
    rows = []
    items = list(raw.items()) + ([("bdi", bdi)] if len(bdi) else [])
    for key, s in items:
        spec = CATALOG_BY_KEY.get(key)
        r = np.log(s).diff().dropna()
        z = (r - r.mean()) / (r.std() or 1.0)
        expected = len(pd.bdate_range(s.index[0], s.index[-1]))
        rows.append({
            "key": key,
            "name": spec.name if spec else "Baltic Dry Index (Koyfin export)",
            "symbol": spec.symbol if spec else "BDIY",
            "source": "Yahoo Finance" if spec else "Koyfin (user-supplied)",
            "unit": spec.unit if spec else "index points",
            "role": spec.role if spec else "translation",
            "first_date": s.index[0].strftime("%Y-%m-%d"),
            "last_date": s.index[-1].strftime("%Y-%m-%d"),
            "observations": int(len(s)),
            "coverage_pct": round(100.0 * len(s) / max(expected, 1), 1),
            "last_value": round(float(s.iloc[-1]), 4),
            "business_days_stale": int(max(0, len(pd.bdate_range(s.index[-1], as_of)) - 1)),
            "max_abs_daily_move_pct": round(float(np.expm1(r.abs().max()) * 100), 2) if len(r) else None,
            "extreme_moves_gt_6sd": int((z.abs() > 6).sum()),
        })
    return rows


def load_market_data(settings: Settings, refresh: bool = False) -> MarketData:
    if refresh:
        report = refresh_raw(settings)
        logger.info("Market data refresh: %d fetched, %d failed", len(report["fetched"]), len(report["failed"]))
    raw = load_raw(settings)
    if TARGET_KEY not in raw:
        logger.info("No raw snapshot found; downloading market data")
        refresh_raw(settings)
        raw = load_raw(settings)
    panel = build_panel(raw)
    bdi = load_koyfin_bdi(settings.koyfin_dir)
    manifest_path = settings.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return MarketData(panel=panel, bdi=bdi, raw=raw, manifest=manifest)


def bdi_checks(bdi: pd.Series, bdry: pd.Series) -> dict:
    """Plausibility checks on the user-supplied BDI export.

    The export cannot be authenticated, but a genuine BDI must (a) have sane
    dates and values and (b) co-move with the FFAs that BDRY holds. Weekly log
    changes of the real index correlate strongly with BDRY; an export that
    fails this is probably mislabelled, shifted or corrupt.
    """
    if len(bdi) < 30:
        return {"status": "insufficient", "checks": []}
    r = np.log(bdi).diff().dropna()
    f = bdry.reindex(bdry.index.union(bdi.index)).ffill().reindex(bdi.index)
    wk_b = np.log(bdi).resample("W-FRI").last().diff()
    wk_f = np.log(f).resample("W-FRI").last().diff()
    both = pd.concat([wk_b, wk_f], axis=1).dropna()
    corr = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 10 else float("nan")
    gap_days = bdi.index.to_series().diff().dt.days
    # The Baltic Exchange does not publish over Christmas/New Year (gap ends early January).
    holiday = (bdi.index.month == 1) & (bdi.index.day <= 10)
    gaps = ((gap_days > 7) & ~holiday).sum()
    checks = [
        {"check": "dates strictly increasing, no duplicates", "pass": bool(bdi.index.is_monotonic_increasing and bdi.index.is_unique)},
        {"check": "no weekend prints", "pass": bool((bdi.index.dayofweek < 5).all())},
        {"check": "no gaps longer than 7 days (outside the Christmas break)", "pass": bool(gaps == 0), "value": int(gaps)},
        {"check": "values within 290-12,000 (BDI record range)", "pass": bool(bdi.between(290, 12000).all())},
        {"check": "largest daily move under 15%", "pass": bool(r.abs().max() < np.log(1.15)),
         "value": round(float(np.expm1(r.abs().max())) * 100, 2)},
        {"check": "weekly changes correlate with BDRY (> 0.4)", "pass": bool(corr > 0.4), "value": round(corr, 3)},
    ]
    return {"status": "pass" if all(c["pass"] for c in checks) else "warn", "checks": checks,
            "first_date": bdi.index[0].strftime("%Y-%m-%d"), "last_date": bdi.index[-1].strftime("%Y-%m-%d")}
