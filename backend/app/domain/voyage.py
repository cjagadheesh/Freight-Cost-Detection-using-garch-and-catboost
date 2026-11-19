"""Voyage economics on a trip-time-charter basis.

The charterer hires the ship at the market TCE from delivery at the load port
to redelivery after discharge, and pays bunkers and port costs. Fuel follows
the propeller (cubic) law ``f(v) = f_d * (v / v_d)^3`` t/day, so fuel per
voyage scales with v^2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.domain.reference import DISCHARGE_PORTS, LOAD_PORTS, VESSEL_CLASSES, sea_distance_nm

CO2_PER_T_FUEL = 3.151  # IMO Cf for light fuel oil (VLSFO is commonly reported under LFO)
BALLAST_DRAFT_SHARE = 0.42  # ballast draft as a share of laden draft (typical bulker)


def vlsfo_price(brent_usd_bbl: float, slope: float, premium: float) -> float:
    """Bunker price proxy: VLSFO $/t ~= slope * Brent $/bbl + premium.

    Fitted relationship from public Singapore VLSFO and Brent quotes; override
    with a quoted bunker price when available.
    """
    return slope * brent_usd_bbl + premium


def fuel_tpd(vessel_class: str, speed_kn: float) -> float:
    vc = VESSEL_CLASSES[vessel_class]
    return vc.fuel_at_design_tpd * (speed_kn / vc.design_speed_kn) ** 3


@dataclass
class VoyageCost:
    vessel_class: str
    load_port: str
    discharge_port: str
    cargo_t: float
    distance_nm: float
    speed_kn: float
    sea_days: float
    port_days: float
    lightering_days: float
    total_days: float
    hire_usd: float
    fuel_t: float
    fuel_usd: float
    port_usd: float
    lightering_usd: float
    carbon_usd: float
    total_usd: float
    usd_per_t: float
    co2_t: float
    lightered_t: float
    feasible: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def draft_check(vessel_class: str, cargo_t: float, discharge_port: str) -> tuple[bool, float, str]:
    """Returns (feasible, tonnes to lighten, reason)."""
    vc, port = VESSEL_CLASSES[vessel_class], DISCHARGE_PORTS[discharge_port]
    if vc.loa_m > port.max_loa_m and not port.lightering_via:
        return False, 0.0, f"LOA {vc.loa_m} m exceeds {port.name} limit {port.max_loa_m} m"
    ballast = vc.laden_draft_m * BALLAST_DRAFT_SHARE
    # Draft rises roughly linearly with cargo between ballast and full laden draft.
    draft = ballast + (vc.laden_draft_m - ballast) * min(cargo_t / vc.cargo_capacity_t, 1.0)
    if draft <= port.max_draft_m and vc.loa_m <= port.max_loa_m:
        return True, 0.0, ""
    if not port.lightering_via:
        return False, 0.0, f"Arrival draft {draft:.1f} m exceeds {port.name} limit {port.max_draft_m} m"
    alongside = max(0.0, (port.max_draft_m - ballast) / (vc.laden_draft_m - ballast)) * vc.cargo_capacity_t
    if vc.loa_m > port.max_loa_m:
        alongside = 0.0  # cannot berth at all: everything moves by barge
    return True, max(cargo_t - alongside, 0.0), f"Lighten {max(cargo_t - alongside, 0):,.0f} t at {port.lightering_via}"


def voyage_cost(vessel_class: str, load_port: str, discharge_port: str, cargo_t: float, tce_usd_day: float,
                bunker_usd_t: float, speed_kn: float | None = None, carbon_usd_t: float = 0.0,
                extra_wait_days: float = 0.0) -> VoyageCost:
    vc, lp, dp = VESSEL_CLASSES[vessel_class], LOAD_PORTS[load_port], DISCHARGE_PORTS[discharge_port]
    speed = speed_kn or vc.design_speed_kn
    dist = sea_distance_nm(load_port, discharge_port)
    over_capacity = cargo_t > vc.cargo_capacity_t + 1e-6
    ok, lighten_t, reason = draft_check(vessel_class, cargo_t, discharge_port)
    if over_capacity:
        ok, reason = False, f"Cargo {cargo_t:,.0f} t exceeds {vessel_class} intake {vc.cargo_capacity_t:,.0f} t"
    if vc.laden_draft_m > lp.max_draft_m + 0.5 and cargo_t > 0.9 * vc.cargo_capacity_t:
        ok, reason = False, f"{vessel_class} cannot load fully at {lp.name} (draft {lp.max_draft_m} m)"
    sea_days = dist / (24.0 * speed)
    port_days = cargo_t / lp.load_rate_tpd + cargo_t / dp.discharge_rate_tpd + 1.0 + dp.avg_wait_days + extra_wait_days
    light_days = dp.lightering_days if lighten_t > 0 else 0.0
    total_days = sea_days + port_days + light_days
    fuel = fuel_tpd(vessel_class, speed) * sea_days + vc.port_fuel_tpd * (port_days + light_days)
    hire = tce_usd_day * total_days
    fuel_usd = fuel * bunker_usd_t
    port_usd = cargo_t * (lp.port_cost_per_t + dp.port_cost_per_t)
    light_usd = lighten_t * dp.lightering_cost_per_t
    co2 = fuel * CO2_PER_T_FUEL
    carbon = co2 * carbon_usd_t
    total = hire + fuel_usd + port_usd + light_usd + carbon
    return VoyageCost(
        vessel_class=vessel_class, load_port=load_port, discharge_port=discharge_port, cargo_t=cargo_t,
        distance_nm=dist, speed_kn=speed, sea_days=sea_days, port_days=port_days, lightering_days=light_days,
        total_days=total_days, hire_usd=hire, fuel_t=fuel, fuel_usd=fuel_usd, port_usd=port_usd,
        lightering_usd=light_usd, carbon_usd=carbon, total_usd=total, usd_per_t=total / max(cargo_t, 1.0),
        co2_t=co2, lightered_t=lighten_t, feasible=ok, reason=reason,
    )
