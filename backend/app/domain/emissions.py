"""Speed optimisation (virtual arrival) and IMO Carbon Intensity Indicator.

Speed: with a berth not free until ``berth_available_day``, steaming at full
speed only buys anchorage time. The cost-minimising speed trades hire against
fuel (cubic law); when a berth wait exists the optimum is usually to arrive
"just in time" (virtual arrival).

CII (MARPOL Annex VI, MEPC.352-355(78)): for bulk carriers

    CII_ref      = a * DWT^-c,  a = 4745, c = 0.622 (capacity = DWT, capped at 279,000)
    CII_required = (1 - Z/100) * CII_ref, Z = 5/7/9/11 % for 2023/24/25/26
    CII_attained = CO2 (g) / (DWT * distance nm)
    rating boundaries on attained/required: d1..d4 = 0.86, 0.94, 1.06, 1.18
"""

from __future__ import annotations

import numpy as np

from app.domain.reference import DISCHARGE_PORTS, LOAD_PORTS, VESSEL_CLASSES, sea_distance_nm
from app.domain.voyage import CO2_PER_T_FUEL, fuel_tpd

CII_A, CII_C = 4745.0, 0.622
CII_DWT_CAP = 279_000
CII_Z = {2023: 5.0, 2024: 7.0, 2025: 9.0, 2026: 11.0}
CII_D = (0.86, 0.94, 1.06, 1.18)
BALLAST_FUEL_SHARE = 0.85  # ballast leg burns ~85% of laden fuel at the same speed


def cii_required(dwt: float, year: int) -> float:
    z = CII_Z.get(year, CII_Z[max(CII_Z)])
    return (1 - z / 100) * CII_A * min(dwt, CII_DWT_CAP) ** (-CII_C)


def cii_rating(attained: float, required: float) -> str:
    ratio = attained / required
    for letter, bound in zip("ABCD", CII_D):
        if ratio <= bound:
            return letter
    return "E"


def round_trip_cii(vessel_class: str, distance_nm: float, speed_kn: float, port_days: float, year: int) -> dict:
    """Attained CII for a vessel shuttling laden/ballast on one route all year.

    Port fuel counts toward CO2 but not distance, as in the IMO definition.
    """
    vc = VESSEL_CLASSES[vessel_class]
    sea_days = distance_nm / (24 * speed_kn)
    fuel = fuel_tpd(vessel_class, speed_kn) * sea_days * (1 + BALLAST_FUEL_SHARE) + vc.port_fuel_tpd * port_days
    co2_g = fuel * CO2_PER_T_FUEL * 1e6
    attained = co2_g / (min(vc.dwt, CII_DWT_CAP) * 2 * distance_nm)
    required = cii_required(vc.dwt, year)
    return {
        "attained": round(attained, 3), "required": round(required, 3), "ratio": round(attained / required, 3),
        "rating": cii_rating(attained, required), "co2_t_round_trip": round(fuel * CO2_PER_T_FUEL, 1),
    }


def optimise_speed(vessel_class: str, load_port: str, discharge_port: str, cargo_t: float, tce_usd_day: float,
                   bunker_usd_t: float, berth_available_day: float | None = None, carbon_usd_t: float = 0.0,
                   year: int = 2026) -> dict:
    """Cost of the laden passage (sea + any anchorage wait) across a speed grid.

    ``berth_available_day``: days after sailing when a berth is free. Defaults
    to the full-speed passage plus the port's average waiting time.
    """
    vc, lp, dp = VESSEL_CLASSES[vessel_class], LOAD_PORTS[load_port], DISCHARGE_PORTS[discharge_port]
    dist = sea_distance_nm(load_port, discharge_port)
    v_max = vc.max_speed_kn
    if berth_available_day is None:
        berth_available_day = dist / (24 * vc.design_speed_kn) + dp.avg_wait_days
    port_days = cargo_t / lp.load_rate_tpd + cargo_t / dp.discharge_rate_tpd + 1.0
    grid = np.round(np.arange(vc.min_speed_kn, v_max + 1e-9, 0.1), 2)
    curve = []
    for v in grid:
        sea = dist / (24 * v)
        wait = max(0.0, berth_available_day - sea)
        fuel = fuel_tpd(vessel_class, v) * sea + vc.port_fuel_tpd * wait
        co2 = fuel * CO2_PER_T_FUEL
        hire = tce_usd_day * (sea + wait)
        total = hire + fuel * bunker_usd_t + co2 * carbon_usd_t
        curve.append({"speed_kn": float(v), "sea_days": round(sea, 2), "anchorage_wait_days": round(wait, 2),
                      "fuel_t": round(fuel, 1), "co2_t": round(co2, 1), "hire_usd": round(hire),
                      "fuel_usd": round(fuel * bunker_usd_t), "total_usd": round(total),
                      "cii_rating": round_trip_cii(vessel_class, dist, float(v), port_days, year)["rating"]})
    best = min(curve, key=lambda r: r["total_usd"])
    design = min(curve, key=lambda r: abs(r["speed_kn"] - vc.design_speed_kn))
    # Unconstrained economic speed (no berth wait): d/dv [H*D/24v + P*f_d*D*v^2/(24 v_d^3)] = 0
    v_econ = (tce_usd_day * vc.design_speed_kn ** 3 / (2 * (bunker_usd_t + CO2_PER_T_FUEL * carbon_usd_t)
                                                      * vc.fuel_at_design_tpd)) ** (1 / 3)
    va_raw = dist / (24 * berth_available_day)
    va_speed = float(np.clip(va_raw, vc.min_speed_kn, v_max))
    if va_raw > v_max:
        va_note = "The berth frees up before the vessel can arrive even at maximum speed: steam at maximum and arrive late."
    elif va_raw < vc.min_speed_kn:
        va_note = "Berth time allows slower than minimum safe speed; vessel steams at minimum and waits."
    else:
        va_note = "Just-in-time arrival is achievable."
    return {
        "vessel_class": vessel_class, "distance_nm": dist, "berth_available_day": round(berth_available_day, 2),
        "virtual_arrival_speed_kn": round(va_speed, 2),
        "virtual_arrival_note": va_note,
        "economic_speed_unconstrained_kn": round(float(np.clip(v_econ, vc.min_speed_kn, v_max)), 2),
        "optimal": best, "at_design_speed": design,
        "saving_vs_design_usd": design["total_usd"] - best["total_usd"],
        "co2_saving_vs_design_t": round(design["co2_t"] - best["co2_t"], 1),
        "cii": {"optimal": round_trip_cii(vessel_class, dist, best["speed_kn"], port_days, year),
                "design": round_trip_cii(vessel_class, dist, design["speed_kn"], port_days, year), "year": year},
        "curve": curve,
    }
