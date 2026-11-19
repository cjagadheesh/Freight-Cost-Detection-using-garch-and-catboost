"""When to fix a spot charter: decision analysis on the Monte Carlo TCE paths.

Fixing on day d locks hire at that day's TCE for the voyage duration. Each
simulated path gives a cost per candidate day, so per day we report the
expected cost, its 5-95% range, CVaR (mean of the worst ``1 - cvar_level``
outcomes), and the probability that waiting is cheaper than fixing on day 1.

The recommendation minimises ``E[cost] + risk_aversion * (CVaR - E[cost])``
over the allowed window. With ``risk_aversion = 0`` it is risk-neutral; with
1 it minimises CVaR. Waiting is recommended only if its score beats day 1 by
more than two Monte Carlo standard errors (paired across days, since every
day is evaluated on the same simulated paths); otherwise simulation noise
could be driving the choice and the default is to fix now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixing_analysis(tce_paths: np.ndarray, dates: pd.DatetimeIndex, hire_days: float, window_days: int,
                    risk_aversion: float = 0.5, cvar_level: float = 0.95) -> dict:
    H = min(window_days, tce_paths.shape[1])
    cost = tce_paths[:, :H] * hire_days  # (n_paths, H)
    base = cost[:, 0]
    tail_n = max(1, int(round(cost.shape[0] * (1 - cvar_level))))
    rows = []
    for d in range(H):
        c = cost[:, d]
        srt = np.sort(c)
        cvar = float(srt[-tail_n:].mean())
        mean = float(c.mean())
        rows.append({
            "day": d + 1,
            "date": dates[d].strftime("%Y-%m-%d"),
            "expected_cost_usd": round(mean),
            "p5_usd": round(float(srt[int(0.05 * len(srt))])),
            "p50_usd": round(float(np.median(c))),
            "p95_usd": round(float(srt[int(0.95 * len(srt)) - 1])),
            "cvar_usd": round(cvar),
            "prob_cheaper_than_day1": round(float(np.mean(c < base)), 3) if d else None,
            "score": mean + risk_aversion * (cvar - mean),
        })
    first = rows[0]
    best = min(rows, key=lambda r: r["score"])
    se = 0.0
    if best is not first:
        d = best["day"] - 1
        diff = cost[:, d] - base
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        if first["score"] - best["score"] < 2 * se:
            best = first
    for r in rows:
        r["score"] = round(r["score"])
    return {
        "hire_days": round(hire_days, 2),
        "risk_aversion": risk_aversion,
        "cvar_level": cvar_level,
        "recommended_day": best["day"],
        "recommended_date": best["date"],
        "mc_standard_error_usd": round(se),
        "n_paths": int(cost.shape[0]),
        "expected_saving_vs_day1_usd": round(first["expected_cost_usd"] - best["expected_cost_usd"]),
        "cvar_change_vs_day1_usd": round(best["cvar_usd"] - first["cvar_usd"]),
        "days": rows,
        "interpretation": (
            "Freight-futures forecasts do not beat a random walk out of sample, so the central path is "
            "close to flat and uncertainty widens with time. Waiting mostly adds risk; it pays only when the "
            "BDI error-correction signal points lower over the next days."
        ),
    }
