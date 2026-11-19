"""Risk signals, historical stress replay and landed cost, all from real series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.ingest import MarketData
from app.domain.reference import KCAL_PER_KG_TO_GJ_PER_T, LOAD_PORTS
from app.domain.voyage import voyage_cost
from app.ml.garch import GarchFit

# Illustrative FOB inputs ($/t). There is no free real-time seaborne coal price
# feed (Yahoo's API2 continuation went stale), so FOB is a user input.
DEFAULT_FOB_USD_T = {"newcastle": 105.0, "hay_point": 185.0, "samarinda": 48.0, "taboneo": 58.0,
                     "richards_bay": 88.0, "maputo": 85.0, "norfolk": 165.0}


def _pct_rank(hist: pd.Series, value: float) -> float:
    h = hist.dropna()
    return float((h < value).mean() * 100) if len(h) else float("nan")


def _level(pct: float, cost_up: bool) -> str:
    """Map a percentile to a level, oriented so that 'high' means cost risk for a charterer."""
    p = pct if cost_up else 100 - pct
    return "high" if p >= 90 else "elevated" if p >= 75 else "low" if p <= 25 else "normal"


def risk_signals(md: MarketData, garch: GarchFit, bunker_slope: float) -> list[dict]:
    p = md.panel
    out = []
    lr = np.log(md.target).diff()

    vol_ann = garch.cond_vol * np.sqrt(252) * 100
    cur = float(np.sqrt(garch.last_variance * 252))
    pr = _pct_rank(vol_ann, cur)
    out.append({"key": "freight_vol", "name": "Freight volatility regime (GARCH)", "value": round(cur, 1), "unit": "% ann.",
                "percentile": round(pr, 1), "level": _level(pr, True),
                "detail": "Conditional volatility of BDRY vs its 2018+ history; drives band width and CVaR."})

    r63 = np.log(md.target).diff(63)
    v = float(r63.iloc[-1])
    pr = _pct_rank(r63, v)
    out.append({"key": "freight_momentum", "name": "Freight 3-month change (BDRY)", "value": round(np.expm1(v) * 100, 1),
                "unit": "%", "percentile": round(pr, 1), "level": _level(pr, True),
                "detail": "Not a forecast (momentum showed no out-of-sample skill); context for where rates sit."})

    def chg(key: str, days: int = 21):
        s = np.log(p[key].dropna())
        c = s.diff(days)
        return float(c.iloc[-1]), c, float(p[key].dropna().iloc[-1])

    if "brent" in p:
        c, hist, last = chg("brent")
        pr = _pct_rank(hist, c)
        out.append({"key": "bunker", "name": "Bunker cost pressure (Brent 1-month)", "value": round(np.expm1(c) * 100, 1),
                    "unit": "%", "percentile": round(pr, 1), "level": _level(pr, True),
                    "detail": f"Brent ${last:.2f}/bbl; each $1/bbl moves VLSFO about ${bunker_slope:.2f}/t."})
    if "usdinr" in p:
        c, hist, last = chg("usdinr")
        pr = _pct_rank(hist, c)
        out.append({"key": "fx", "name": "Rupee weakness (USD/INR 1-month)", "value": round(np.expm1(c) * 100, 2),
                    "unit": "%", "percentile": round(pr, 1), "level": _level(pr, True),
                    "detail": f"USD/INR {last:.2f}; a weaker rupee raises the INR cost of USD freight and FOB."})
    if "vix" in p:
        lv = np.log(p["vix"].dropna())
        z = (lv - lv.rolling(252).mean()) / lv.rolling(252).std()
        val = float(z.iloc[-1])
        pr = _pct_rank(z, val)
        out.append({"key": "risk_off", "name": "Market stress (VIX z-score)", "value": round(val, 2), "unit": "z",
                    "percentile": round(pr, 1), "level": _level(pr, True),
                    "detail": f"VIX {float(p['vix'].dropna().iloc[-1]):.1f}; risk-off episodes coincide with freight volatility."})
    if "coal" in p:
        c, hist, _ = chg("coal")
        pr = _pct_rank(hist, c)
        out.append({"key": "coal", "name": "Seaborne coal proxy (Whitehaven, 1-month)", "value": round(np.expm1(c) * 100, 1),
                    "unit": "%", "percentile": round(pr, 1), "level": _level(pr, True),
                    "detail": "Listed Newcastle exporter as a proxy for coal FOB direction."})
    eq_keys = [k for k in ("gnk", "sblk", "dsx") if k in p]
    if eq_keys:
        eq = pd.concat([np.log(p[k]).diff() for k in eq_keys], axis=1).mean(axis=1)
        gap = eq.rolling(21).sum() - lr.rolling(21).sum()
        val = float(gap.iloc[-1])
        pr = _pct_rank(gap, val)
        out.append({"key": "equity_gap", "name": "Owner equities vs freight (1-month gap)", "value": round(val * 100, 1),
                    "unit": "pp", "percentile": round(pr, 1), "level": _level(pr, True),
                    "detail": "Listed dry-bulk owners outperforming freight can signal expected tightening."})
    if len(md.bdi):
        stale = len(pd.bdate_range(md.bdi.index[-1], md.as_of)) - 1
        out.append({"key": "bdi_staleness", "name": "BDI data age", "value": stale, "unit": "business days",
                    "percentile": None, "level": "high" if stale > 20 else "elevated" if stale > 5 else "low",
                    "detail": f"Last Koyfin BDI print {md.bdi.index[-1]:%Y-%m-%d}; spot level is nowcast from BDRY since then."})
    return out


def stress_windows(md: MarketData, window: int = 42, k: int = 4) -> list[dict]:
    """The k largest non-overlapping BDRY rallies and slumps over ``window`` days."""
    lp = np.log(md.target)
    r = (lp.shift(-window) - lp).dropna()
    out = []
    for kind, order in (("rally", r.sort_values(ascending=False)), ("slump", r.sort_values())):
        taken: list[pd.Timestamp] = []
        for d, v in order.items():
            if all(abs((d - t).days) > window * 1.5 for t in taken):
                taken.append(d)
                end = lp.index[lp.index.get_loc(d) + window]
                out.append({"kind": kind, "start": d, "end": end, "bdry_change_pct": round(np.expm1(v) * 100, 1)})
            if len(taken) == k:
                break
    return out


def stress_replay(md: MarketData, tce_today: dict[str, float], beta: float, bunker_today: float, bunker_slope: float,
                  vessel_class: str, load_port: str, discharge_port: str, cargo_t: float, window: int = 42) -> dict:
    """Replay historical freight and oil moves on today's voyage economics.

    BDRY's log move over each window is mapped to spot TCE with the ECM's
    long-run elasticity ``beta``; Brent's move over the same dates shifts the
    bunker price. No synthetic shocks: every scenario happened.
    """
    base = voyage_cost(vessel_class, load_port, discharge_port, cargo_t, tce_today[vessel_class], bunker_today)
    brent = md.panel["brent"].dropna() if "brent" in md.panel else None
    scenarios = []
    for w in stress_windows(md, window):
        f_move = np.log(1 + w["bdry_change_pct"] / 100) * beta
        tce = tce_today[vessel_class] * np.exp(f_move)
        bunker = bunker_today
        brent_chg = None
        if brent is not None:
            seg = brent[(brent.index >= w["start"]) & (brent.index <= w["end"])]
            if len(seg) > 1:
                brent_chg = float(seg.iloc[-1] - seg.iloc[0])
                bunker = max(150.0, bunker_today + bunker_slope * brent_chg)
        vc = voyage_cost(vessel_class, load_port, discharge_port, cargo_t, tce, bunker)
        scenarios.append({
            "kind": w["kind"], "start": w["start"].strftime("%Y-%m-%d"), "end": w["end"].strftime("%Y-%m-%d"),
            "bdry_change_pct": w["bdry_change_pct"], "spot_tce_change_pct": round(np.expm1(f_move) * 100, 1),
            "brent_change_usd": None if brent_chg is None else round(brent_chg, 2),
            "tce_usd_day": round(tce), "bunker_usd_t": round(bunker, 1),
            "voyage_cost_usd": round(vc.total_usd), "usd_per_t": round(vc.usd_per_t, 2),
            "change_usd_per_t": round(vc.usd_per_t - base.usd_per_t, 2),
            "programme_impact_usd": round((vc.usd_per_t - base.usd_per_t) * cargo_t),
        })
    return {"window_days": window, "elasticity_beta": round(beta, 3), "base": base.to_dict(), "scenarios": scenarios}


def landed_cost(load_port: str, discharge_port: str, vessel_class: str, cargo_t: float, tce_samples: np.ndarray,
                bunker_usd_t: float, usdinr: float, fob_usd_t: float | None = None, cv_kcal_kg: float | None = None,
                insurance_pct: float = 0.15, carbon_usd_t: float = 0.0) -> dict:
    """CFR + insurance + port/lightering = landed cost, with a freight band from
    the TCE distribution for the fixing day."""
    lp = LOAD_PORTS[load_port]
    fob = DEFAULT_FOB_USD_T[load_port] if fob_usd_t is None else fob_usd_t
    cv = lp.default_cv_kcal_kg if cv_kcal_kg is None else cv_kcal_kg
    gj_per_t = cv * KCAL_PER_KG_TO_GJ_PER_T
    qs = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
    out = {}
    for name, q in qs.items():
        tce = float(np.quantile(tce_samples, q))
        vc = voyage_cost(vessel_class, load_port, discharge_port, cargo_t, tce, bunker_usd_t, carbon_usd_t=carbon_usd_t)
        port_light = (vc.port_usd + vc.lightering_usd) / cargo_t
        sea_freight = vc.usd_per_t - port_light
        cfr = fob + sea_freight
        ins = cfr * insurance_pct / 100
        total = cfr + ins + port_light
        out[name] = {
            "tce_usd_day": round(tce), "fob": round(fob, 2), "sea_freight": round(sea_freight, 2),
            "insurance": round(ins, 2), "port_and_lightering": round(port_light, 2),
            "landed_usd_t": round(total, 2), "landed_inr_t": round(total * usdinr), "usd_per_gj": round(total / gj_per_t, 3),
            "freight_share_pct": round(sea_freight / total * 100, 1), "feasible": vc.feasible, "reason": vc.reason,
        }
    return {
        "load_port": load_port, "discharge_port": discharge_port, "vessel_class": vessel_class, "cargo_t": cargo_t,
        "cv_kcal_kg": cv, "gj_per_t": round(gj_per_t, 3), "usdinr": round(usdinr, 3), "bunker_usd_t": round(bunker_usd_t, 1),
        "fob_source": "user input" if fob_usd_t is not None else "illustrative default (override with a quoted FOB)",
        "bands": out,
    }
