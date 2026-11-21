"""Process-wide application state: market data, model bundle and forecast.

Loaded once at startup. A refresh builds a complete new snapshot off to the
side and swaps it in atomically, so requests never see a half-updated state.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.data.ingest import MarketData, load_market_data
from app.domain.voyage import vlsfo_price
from app.ml.forecast import ForecastState, build_state
from app.ml.train import Bundle, load_bundle, save_bundle, train

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    md: MarketData
    bundle: Bundle
    forecast: ForecastState
    loaded_at: str
    bunker_usd_t: float
    brent: float
    usdinr: float
    load_seconds: float
    bundle_trained_now: bool = False
    tce_cache: dict = field(default_factory=dict)

    def tce_paths(self, vessel_class: str):
        if vessel_class not in self.tce_cache:
            self.tce_cache[vessel_class] = self.forecast.tce_paths(vessel_class)
        return self.tce_cache[vessel_class]


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._snap: Snapshot | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.refreshing = False
        self.last_refresh: dict | None = None

    @property
    def ready(self) -> bool:
        return self._snap is not None

    @property
    def snap(self) -> Snapshot:
        if self._snap is None:
            raise RuntimeError("Model state not loaded")
        return self._snap

    def _build(self, refresh_data: bool, force_train: bool) -> Snapshot:
        t0 = time.perf_counter()
        s = self.settings
        md = load_market_data(s, refresh=refresh_data)
        bundle = None if force_train else load_bundle(s)
        trained = False
        if bundle is None or (refresh_data and bundle.as_of != md.as_of.strftime("%Y-%m-%d")):
            if bundle is None and not s.train_if_missing and not force_train:
                raise RuntimeError("No model bundle on disk; run `python -m app.ml.train`")
            logger.info("Training model bundle")
            bundle = train(s, md=md)
            save_bundle(bundle, s)
            trained = True
        fc = build_state(bundle, md, seed=s.random_seed)
        brent = float(md.panel["brent"].dropna().iloc[-1])
        usdinr = float(md.panel["usdinr"].dropna().iloc[-1])
        return Snapshot(
            md=md, bundle=bundle, forecast=fc,
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            bunker_usd_t=float(vlsfo_price(brent, s.brent_to_vlsfo, s.vlsfo_premium)),
            brent=brent, usdinr=usdinr, load_seconds=round(time.perf_counter() - t0, 2), bundle_trained_now=trained,
        )

    def load(self) -> None:
        snap = self._build(refresh_data=self.settings.refresh_on_startup, force_train=False)
        with self._lock:
            self._snap = snap
        logger.info("State loaded: data as of %s, bundle %s (%.1fs)", snap.md.as_of.date(), snap.bundle.created_at,
                    snap.load_seconds)

    def refresh(self, retrain: bool = True) -> dict:
        """Download fresh data, retrain if the data moved, and swap in."""
        if self.refreshing:
            return {"status": "already_running"}
        self.refreshing = True
        try:
            snap = self._build(refresh_data=True, force_train=retrain)
            with self._lock:
                self._snap = snap
            self.last_error = None
            self.last_refresh = {"status": "ok", "at": snap.loaded_at, "as_of": snap.md.as_of.strftime("%Y-%m-%d"),
                                 "trained": snap.bundle_trained_now, "seconds": snap.load_seconds}
            return self.last_refresh
        except Exception as exc:  # keep serving the previous snapshot
            logger.exception("Refresh failed")
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_refresh = {"status": "failed", "error": self.last_error,
                                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            return self.last_refresh
        finally:
            self.refreshing = False


def jsonable(obj):
    """Recursively convert numpy / pandas scalars to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
