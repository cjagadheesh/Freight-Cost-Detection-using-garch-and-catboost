"""Training pipeline: backtest, fit production models, save a versioned bundle.

    python -m app.ml.train            # use committed snapshot
    python -m app.ml.train --refresh  # download fresh market data first

Everything the API serves comes from the bundle this writes to
``data/artifacts/bundle.pkl``; metrics are the walk-forward results, not
in-sample fit statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.data.ingest import MarketData, load_market_data, quality_report
from app.ml import ecm
from app.ml.backtest import (
    INTERVAL_LEVELS, adaptive_conformal, calibrate_sequentially, compute_metrics, fold_summary, run_walk_forward,
)
from app.ml.features import CORE_FEATURES, WARMUP_DAYS, active_drivers, build_features, build_labels
from app.ml.garch import GarchFit, fit_garch
from app.ml.models import MODEL_NAMES, fit_learners, fit_stacking_weights

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 3
VOL_WINDOW = 63


@dataclass
class Bundle:
    version: int
    created_at: str
    as_of: str
    data_fingerprint: str
    horizons: list[int]
    features: list[str]
    drivers: list[str]
    learners: dict  # h -> {"ridge": ..., "catboost": ...}
    weights: dict  # h -> np.ndarray (MODEL_NAMES order)
    score_quantiles: dict  # h -> {level: (lo, hi)} on vol-scaled residuals
    aci_state: dict  # h -> {level: final adaptive miscoverage}
    garch: GarchFit
    ecm_fit: ecm.ECMFit | None
    metrics: list[dict]
    folds: list[dict]
    weight_log: list[dict]
    oof: pd.DataFrame
    ecm_metrics: list[dict]
    ecm_oof: pd.DataFrame | None
    quality: list[dict]
    feature_importance: dict  # h -> {feature: importance}
    timings: dict = field(default_factory=dict)


def _fingerprint(md: MarketData) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(md.panel.fillna(-1), index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(md.bdi, index=True).values.tobytes())
    return h.hexdigest()[:16]


def realised_vol(log_returns: pd.Series) -> pd.Series:
    """Trailing 63-day realised vol used to standardise labels (known at t)."""
    return log_returns.rolling(VOL_WINDOW, min_periods=40).std()


def prepare(md: MarketData, settings: Settings):
    drivers = active_drivers(md.panel)
    X_all = build_features(md.panel, drivers)
    features = [c for c in CORE_FEATURES if c in X_all.columns]
    X = X_all[features]
    Y = build_labels(md.panel, settings.horizons)
    r = np.log(md.target).diff()
    vol = realised_vol(r)
    return X, Y, r, vol, drivers, features


def fit_production(X, Y, vol, horizons, first_row, settings: Settings):
    learners, importance = {}, {}
    for h in horizons:
        last = len(X) - 1 - h
        tr = slice(first_row, last + 1)
        y = Y[h].iloc[tr] / (vol.iloc[tr] * np.sqrt(h))
        m = y.notna() & np.isfinite(y)
        learners[h] = fit_learners(X.iloc[tr][m], y[m], settings.catboost_iterations, settings.random_seed)
        imp = learners[h]["catboost"].get_feature_importance()
        importance[h] = {f: round(float(v), 3) for f, v in sorted(zip(X.columns, imp), key=lambda kv: -kv[1])}
    return learners, importance


def final_weights_and_bands(oof: pd.DataFrame, horizons: list[int], aci_state: dict):
    """Production stacking weights (all OOF rows) and interval quantiles at the
    final adaptive-conformal miscoverage level for each horizon and level."""
    weights, bands = {}, {}
    for h in horizons:
        g = oof[(oof["h"] == h)].sort_values("pos")
        if len(g) >= 60:
            weights[h] = fit_stacking_weights(g[MODEL_NAMES].values, g["y"].values)
            ens = g[MODEL_NAMES].values @ weights[h]
            scores = (g["y"].values - ens) / g["sigma"].values
            bands[h] = {}
            for lvl in INTERVAL_LEVELS:
                a = aci_state.get(h, {}).get(lvl, 1 - lvl / 100)
                lo, hi = np.quantile(scores, [a / 2, 1 - a / 2])
                bands[h][lvl] = (float(min(lo, 0.0)), float(max(hi, 0.0)))
        else:
            weights[h] = np.array([1.0, 0.0, 0.0])
            bands[h] = {50: (-0.674, 0.674), 80: (-1.2816, 1.2816), 95: (-1.96, 1.96)}
    return weights, bands


def train(settings: Settings | None = None, refresh: bool = False, md: MarketData | None = None) -> Bundle:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    md = md or load_market_data(settings, refresh=refresh)
    X, Y, r, vol, drivers, features = prepare(md, settings)
    first_row = max(WARMUP_DAYS, int(X.index.get_loc(vol.first_valid_index())))
    timings = {"load_s": round(time.perf_counter() - t0, 2)}

    t1 = time.perf_counter()
    oof = run_walk_forward(X, Y, r, vol, settings.horizons, first_row, settings.backtest_initial_train_days,
                           settings.backtest_step_days, settings.catboost_iterations, settings.random_seed)
    oof, weight_log = calibrate_sequentially(oof)
    oof, aci_state = adaptive_conformal(oof)
    metrics = compute_metrics(oof)
    folds = fold_summary(oof, X.index, ref_h=21 if 21 in settings.horizons else settings.horizons[0])
    timings["backtest_s"] = round(time.perf_counter() - t1, 2)

    t2 = time.perf_counter()
    learners, importance = fit_production(X, Y, vol, settings.horizons, first_row, settings)
    weights, bands = final_weights_and_bands(oof, settings.horizons, aci_state)
    garch = fit_garch(r)
    timings["production_fit_s"] = round(time.perf_counter() - t2, 2)

    ecm_fit, ecm_metrics, ecm_oof = None, [], None
    if len(md.bdi) >= ecm.MIN_TRAIN + 30:
        data = ecm.align(md.bdi, md.target)
        ecm_oof = ecm.walk_forward(data)
        ecm_metrics = ecm.evaluate(ecm_oof)
        ecm_fit = ecm.fit_ecm(data)

    bundle = Bundle(
        version=BUNDLE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        as_of=md.as_of.strftime("%Y-%m-%d"),
        data_fingerprint=_fingerprint(md),
        horizons=list(settings.horizons),
        features=features,
        drivers=drivers,
        learners=learners,
        weights=weights,
        score_quantiles=bands,
        aci_state=aci_state,
        garch=garch,
        ecm_fit=ecm_fit,
        metrics=metrics,
        folds=folds,
        weight_log=weight_log,
        oof=oof,
        ecm_metrics=ecm_metrics,
        ecm_oof=ecm_oof,
        quality=quality_report(md.raw, md.bdi, md.as_of),
        feature_importance=importance,
        timings={**timings, "total_s": round(time.perf_counter() - t0, 2)},
    )
    return bundle


def bundle_path(settings: Settings) -> Path:
    return settings.artifacts_dir / "bundle.pkl"


def save_bundle(bundle: Bundle, settings: Settings) -> Path:
    path = bundle_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic swap so a reader never sees a half-written file
    summary = {
        "version": bundle.version, "created_at": bundle.created_at, "as_of": bundle.as_of,
        "data_fingerprint": bundle.data_fingerprint, "metrics": bundle.metrics, "ecm_metrics": bundle.ecm_metrics,
        "weights": {str(h): dict(zip(MODEL_NAMES, np.round(w, 3).tolist())) for h, w in bundle.weights.items()},
        "timings": bundle.timings,
    }
    (path.parent / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return path


def load_bundle(settings: Settings) -> Bundle | None:
    path = bundle_path(settings)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        b = pickle.load(fh)
    if getattr(b, "version", None) != BUNDLE_VERSION:
        logger.warning("Ignoring bundle with version %s (expected %s)", getattr(b, "version", None), BUNDLE_VERSION)
        return None
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the OceanPulse forecasting bundle")
    ap.add_argument("--refresh", action="store_true", help="download fresh market data first")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    bundle = train(settings, refresh=args.refresh)
    path = save_bundle(bundle, settings)
    print(f"Saved {path} (as of {bundle.as_of}, {bundle.timings['total_s']}s)")
    cols = ["h", "model", "n", "rmse_pct", "skill_vs_rw_pct", "directional_accuracy_pct",
            "coverage80_pct", "coverage95_pct", "dm_pvalue"]
    print(pd.DataFrame(bundle.metrics).reindex(columns=cols).to_string(index=False))
    if bundle.ecm_metrics:
        print("\nBDI error-correction model (walk-forward):")
        print(pd.DataFrame(bundle.ecm_metrics).to_string(index=False))


if __name__ == "__main__":
    # Re-import under the package name so pickled classes resolve as
    # app.ml.train.Bundle (not __main__.Bundle) when the API loads them.
    from app.ml.train import main as _main

    _main()
