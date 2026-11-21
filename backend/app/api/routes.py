"""HTTP routes (mounted under /api/v1)."""

from __future__ import annotations

import hmac
from dataclasses import asdict

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.api.schemas import AllocationIn, LandedIn, SpeedIn, StressIn, TimingIn
from app.api.state import AppState, Snapshot, jsonable
from app.data.catalog import CATALOG, CATALOG_BY_KEY
from app.domain.allocation import Cargo, default_cargoes, solve_allocation
from app.domain.emissions import optimise_speed
from app.domain.reference import SAMPLE_FLEET, VESSEL_CLASSES, FleetVessel, reference_payload
from app.domain.risk import landed_cost, risk_signals, stress_replay
from app.domain.timing import fixing_analysis
from app.domain.voyage import voyage_cost
from app.ml.backtest import ACI_GAMMA
from app.ml.ecm import FULL_SIGNAL_H, ZERO_SIGNAL_H
from app.ml.forecast import MAX_H, explain
from app.ml.translate import DEFAULT_CLASS_RATIOS, tce_table

router = APIRouter(prefix="/api/v1")


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _snap(request: Request) -> Snapshot:
    st = _state(request)
    if not st.ready:
        raise HTTPException(503, "Model state is still loading")
    return st.snap


# ----------------------------------------------------------------- system
@router.get("/status")
def status(request: Request):
    st = _state(request)
    if not st.ready:
        return {"ready": False, "last_error": st.last_error}
    s = st.snap
    b = s.bundle
    return jsonable({
        "ready": True,
        "data_as_of": s.md.as_of.strftime("%Y-%m-%d"),
        "bdi_last_print": s.md.bdi.index[-1].strftime("%Y-%m-%d") if len(s.md.bdi) else None,
        "bundle": {"created_at": b.created_at, "as_of": b.as_of, "fingerprint": b.data_fingerprint,
                   "timings": b.timings, "version": b.version},
        "loaded_at": s.loaded_at,
        "load_seconds": s.load_seconds,
        "refreshing": st.refreshing,
        "last_error": st.last_error,
        "environment": st.settings.environment,
    })


# ----------------------------------------------------------------- market data
@router.get("/market/catalog")
def market_catalog():
    return [asdict(s) for s in CATALOG] + [{
        "key": "bdi", "symbol": "BDIY", "name": "Baltic Dry Index", "unit": "index points", "role": "spot",
        "source": "koyfin", "description": "User-supplied Koyfin export (licensed data).",
    }]


@router.get("/market/series")
def market_series(request: Request, keys: str = Query("bdry,brent", description="Comma-separated series keys"),
                  start: str | None = Query(None, description="YYYY-MM-DD")):
    s = _snap(request)
    out = {}
    for key in [k.strip() for k in keys.split(",") if k.strip()]:
        if key == "bdi":
            ser = s.md.bdi
        elif key in s.md.raw:
            ser = s.md.raw[key]
        else:
            raise HTTPException(422, f"Unknown series {key!r}")
        if start:
            try:
                start_ts = pd.Timestamp(start)
            except (ValueError, TypeError):
                raise HTTPException(422, f"start must be a date (YYYY-MM-DD), got {start!r}") from None
            ser = ser[ser.index >= start_ts]
        spec = CATALOG_BY_KEY.get(key)
        out[key] = {
            "name": spec.name if spec else "Baltic Dry Index",
            "unit": spec.unit if spec else "index points",
            "points": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in ser.items()],
        }
    return out


@router.get("/market/quality")
def market_quality(request: Request):
    s = _snap(request)
    return jsonable({"as_of": s.md.as_of.strftime("%Y-%m-%d"), "series": s.bundle.quality, "manifest": s.md.manifest})


@router.get("/market/snapshot")
def market_snapshot(request: Request):
    s = _snap(request)
    p = s.md.panel
    rows = []
    for key in s.md.raw:
        ser = p[key].dropna() if key in p else s.md.raw[key]
        last = float(ser.iloc[-1])
        prev = float(ser.iloc[-2]) if len(ser) > 1 else last
        prev21 = float(ser.iloc[-22]) if len(ser) > 22 else last
        rows.append({"key": key, "name": CATALOG_BY_KEY[key].name, "unit": CATALOG_BY_KEY[key].unit, "last": last,
                     "date": ser.index[-1].strftime("%Y-%m-%d"), "chg_1d_pct": (last / prev - 1) * 100,
                     "chg_21d_pct": (last / prev21 - 1) * 100})
    return jsonable({"as_of": s.md.as_of.strftime("%Y-%m-%d"), "bunker_vlsfo_usd_t": s.bunker_usd_t,
                     "brent": s.brent, "usdinr": s.usdinr, "series": rows})


# ----------------------------------------------------------------- forecasts
@router.get("/forecast")
def forecast(request: Request):
    s = _snap(request)
    fc = s.forecast
    nc = fc.bdi_nowcast
    out = {
        "as_of": fc.as_of.strftime("%Y-%m-%d"),
        "bdry": {"last": fc.bdry_last, "horizons": fc.bdry_points, "path": fc.bdry_path,
                 "method": "Stacked Ridge/CatBoost/random-walk ensemble; GARCH(1,1)-t volatility; "
                           f"adaptive conformal intervals (gamma={ACI_GAMMA})"},
        "bdi": None,
    }
    if nc is not None:
        out["bdi"] = {
            "nowcast": nc, "path": fc.bdi_path,
            "tce_today": tce_table(nc["level"]),
            "tce_p10_today": tce_table(nc["p10"]), "tce_p90_today": tce_table(nc["p90"]),
            "signal_policy": {"full_signal_until_h": FULL_SIGNAL_H, "zero_signal_from_h": ZERO_SIGNAL_H},
            "method": "Error-correction model on BDRY; nowcast on actual BDRY closes, then Monte Carlo on "
                      "GARCH filtered-historical-simulation BDRY paths",
        }
    return jsonable(out)


@router.get("/forecast/tce")
def forecast_tce(request: Request, vessel_class: str = Query("Panamax")):
    s = _snap(request)
    if vessel_class not in VESSEL_CLASSES:
        raise HTTPException(422, f"Unknown vessel class {vessel_class!r}")
    if s.forecast.bdi_paths is None:
        raise HTTPException(409, "BDI layer unavailable: add a Koyfin BDI export to data/koyfin")
    paths = s.tce_paths(vessel_class)
    qs = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return jsonable({
        "vessel_class": vessel_class, "ratio_to_weighted_tce": DEFAULT_CLASS_RATIOS[vessel_class],
        "today": s.forecast.tce_today(vessel_class),
        "path": [{"date": d.strftime("%Y-%m-%d"), "h": k + 1, "mean": round(float(paths[:, k].mean())),
                  "p5": round(float(qs[0, k])), "p25": round(float(qs[1, k])), "p50": round(float(qs[2, k])),
                  "p75": round(float(qs[3, k])), "p95": round(float(qs[4, k]))}
                 for k, d in enumerate(s.forecast.dates)],
    })


@router.get("/forecast/explain")
def forecast_explain(request: Request, h: int = Query(21)):
    s = _snap(request)
    if h not in s.bundle.horizons:
        raise HTTPException(422, f"h must be one of {s.bundle.horizons}")
    return jsonable(explain(s.bundle, s.forecast, h))


# ----------------------------------------------------------------- models & evaluation
@router.get("/models")
def models(request: Request):
    s = _snap(request)
    b = s.bundle
    return jsonable({
        "garch": b.garch.summary(),
        "ecm": b.ecm_fit.summary() if b.ecm_fit else None,
        "weights": {h: dict(zip(["random_walk", "ridge", "catboost"], w)) for h, w in b.weights.items()},
        "interval_quantiles": b.score_quantiles, "aci_state": b.aci_state,
        "features": b.features, "drivers": b.drivers, "feature_importance": b.feature_importance,
        "class_ratios": DEFAULT_CLASS_RATIOS,
    })


@router.get("/backtest")
def backtest(request: Request, h: int = Query(21), every: int = Query(3, ge=1, le=21)):
    s = _snap(request)
    b = s.bundle
    if h not in b.horizons:
        raise HTTPException(422, f"h must be one of {b.horizons}")
    g = b.oof[(b.oof["h"] == h) & b.oof["calibrated"]].sort_values("pos").iloc[::every]
    p0 = s.md.target.values[g["pos"].values]
    series = [{"date": d.strftime("%Y-%m-%d"), "target_date": s.md.target.index[min(pos + h, len(s.md.target) - 1)].strftime("%Y-%m-%d"),
               "actual": round(float(base * np.exp(y)), 4), "forecast": round(float(base * np.exp(e)), 4),
               "lo80": round(float(base * np.exp(lo)), 4), "hi80": round(float(base * np.exp(hi)), 4)}
              for d, pos, base, y, e, lo, hi in zip(g["date"], g["pos"], p0, g["y"], g["ensemble"], g["lo80"], g["hi80"])]
    ecm_series = None
    if b.ecm_oof is not None and h in (1, 5, 10, 21):
        e = b.ecm_oof[b.ecm_oof["h"] == h]
        ecm_series = [{"date": d.strftime("%Y-%m-%d"), "actual_log_change": round(float(y), 5), "ecm_log_change": round(float(p), 5)}
                      for d, y, p in zip(e["date"], e["y"], e["ecm"])]
    return jsonable({
        "h": h, "metrics": b.metrics, "folds": b.folds, "ecm_metrics": b.ecm_metrics, "series": series,
        "ecm_series": ecm_series, "weight_log": [w for w in b.weight_log if w["h"] == h],
        "protocol": {
            "type": "expanding-window walk-forward with purging",
            "initial_train_days": _state(request).settings.backtest_initial_train_days, "refit_every_days": _state(request).settings.backtest_step_days,
            "stacking": "convex weights fitted on prior out-of-fold predictions only",
            "intervals": f"adaptive conformal inference on GARCH-scaled residuals, gamma={ACI_GAMMA}",
            "significance": "one-sided Diebold-Mariano vs random walk (HLN-corrected, Newey-West)",
        },
    })


# ----------------------------------------------------------------- reference & risk
@router.get("/reference")
def reference():
    return reference_payload()


@router.get("/risk")
def risk(request: Request):
    s = _snap(request)
    return jsonable({"as_of": s.md.as_of.strftime("%Y-%m-%d"),
                     "signals": risk_signals(s.md, s.bundle.garch, _state(request).settings.brent_to_vlsfo)})


# ----------------------------------------------------------------- decision tools
def _require_feasible(vessel_class: str, load_port: str, discharge_port: str, cargo_t: float):
    """422 if the vessel cannot make this voyage (intake, draft, LOA)."""
    vc = voyage_cost(vessel_class, load_port, discharge_port, cargo_t, 1.0, 1.0)
    if not vc.feasible:
        raise HTTPException(422, f"Infeasible voyage: {vc.reason}")


def _require_bdi(s: Snapshot):
    if s.forecast.bdi_paths is None:
        raise HTTPException(409, "BDI layer unavailable: add a Koyfin BDI export to data/koyfin")


@router.post("/optimize/allocation")
def optimize_allocation(request: Request, body: AllocationIn):
    s = _snap(request)
    _require_bdi(s)
    cargoes = [Cargo(**c.model_dump()) for c in body.cargoes] if body.cargoes is not None else default_cargoes()
    fleet = [FleetVessel(**v.model_dump()) for v in body.fleet] if body.fleet is not None else SAMPLE_FLEET
    classes = body.spot_classes if body.spot_classes is not None else list(VESSEL_CLASSES)
    needed = {v.vessel_class for v in fleet} | set(classes)
    paths = {k: s.tce_paths(k) for k in needed}
    res = solve_allocation(cargoes, paths, body.bunker_usd_t or s.bunker_usd_t, fleet=fleet, spot_classes=classes,
                           slot_days=body.slot_days, rate_quantile=body.rate_quantile, carbon_usd_t=body.carbon_usd_t)
    res["inputs"] = {"cargoes": [asdict(c) for c in cargoes], "fleet": [asdict(v) for v in fleet],
                     "bunker_usd_t": body.bunker_usd_t or s.bunker_usd_t}
    return jsonable(res)


@router.post("/optimize/timing")
def optimize_timing(request: Request, body: TimingIn):
    s = _snap(request)
    _require_bdi(s)
    _require_feasible(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t)
    tce_now = s.forecast.tce_today(body.vessel_class)
    vc = voyage_cost(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t, tce_now, s.bunker_usd_t)
    res = fixing_analysis(s.tce_paths(body.vessel_class), s.forecast.dates, vc.total_days, body.window_days,
                          body.risk_aversion, body.cvar_level)
    res["voyage"] = vc.to_dict()
    return jsonable(res)


@router.post("/optimize/speed")
def optimize_speed_route(request: Request, body: SpeedIn):
    s = _snap(request)
    _require_feasible(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t)
    tce = body.tce_usd_day
    if tce is None:
        _require_bdi(s)
        tce = s.forecast.tce_today(body.vessel_class)
    return jsonable(optimise_speed(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t, tce,
                                   body.bunker_usd_t or s.bunker_usd_t, body.berth_available_day, body.carbon_usd_t,
                                   body.year))


@router.post("/landed-cost")
def landed(request: Request, body: LandedIn):
    s = _snap(request)
    _require_bdi(s)
    _require_feasible(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t)
    samples = s.tce_paths(body.vessel_class)[:, min(body.fixing_day, MAX_H) - 1]
    return jsonable(landed_cost(body.load_port, body.discharge_port, body.vessel_class, body.cargo_t, samples,
                                body.bunker_usd_t or s.bunker_usd_t, s.usdinr, body.fob_usd_t, body.cv_kcal_kg,
                                body.insurance_pct, body.carbon_usd_t))


@router.post("/stress")
def stress(request: Request, body: StressIn):
    s = _snap(request)
    _require_bdi(s)
    _require_feasible(body.vessel_class, body.load_port, body.discharge_port, body.cargo_t)
    tce_today = {k: s.forecast.tce_today(k) for k in VESSEL_CLASSES}
    beta = s.bundle.ecm_fit.beta if s.bundle.ecm_fit else 1.0
    return jsonable(stress_replay(s.md, tce_today, beta, s.bunker_usd_t, _state(request).settings.brent_to_vlsfo, body.vessel_class,
                                  body.load_port, body.discharge_port, body.cargo_t, body.window_days))


# ----------------------------------------------------------------- admin
@router.post("/admin/refresh")
def admin_refresh(request: Request, retrain: bool = Query(True), authorization: str | None = Header(None)):
    st = _state(request)
    token = st.settings.admin_token
    if not token:
        raise HTTPException(404, "Not found")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "Invalid token")
    return st.refresh(retrain=retrain)
