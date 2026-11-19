"""Fleet / cargo allocation as a mixed-integer linear programme (HiGHS).

Decision variables
    x[v, c, s] = 1 if fleet vessel v lifts cargo c starting on day s
    y[k, c, s] = 1 if a spot vessel of class k is chartered for cargo c on day s
    u[c]       = 1 if cargo c is left unserved (penalised; keeps the model feasible)

Constraints
    * every cargo is served exactly once or flagged unserved;
    * each fleet vessel lifts at most one cargo in the planning window and not
      before it opens;
    * start day lies in the cargo's laycan, on a grid of ``slot_days``;
    * intake, load-port draft, discharge draft/LOA (with lightering where the
      port supports it) are checked per candidate; infeasible candidates are
      never generated;
    * berth occupancy at each discharge port, day by day, never exceeds the
      number of berths allotted to the programme.

Objective
    total voyage cost: hire (fixed period hire, or forecast TCE for
    index-linked and spot tonnage on the fixing day), bunkers, port costs,
    lightering and optional carbon cost. ``rate_quantile`` prices the market
    at a quantile of the Monte Carlo TCE distribution instead of its mean, so
    a risk-averse planner can optimise against, say, the 75th percentile.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import highspy
import numpy as np

from app.domain.reference import DISCHARGE_PORTS, LOAD_PORTS, SAMPLE_FLEET, VESSEL_CLASSES, FleetVessel
from app.domain.voyage import voyage_cost

UNSERVED_PENALTY_USD = 50_000_000.0


@dataclass
class Cargo:
    id: str
    load_port: str
    discharge_port: str
    quantity_t: float
    laycan_start: int
    laycan_end: int


def _rate_curve(tce_paths: np.ndarray, quantile: float | None) -> np.ndarray:
    """Per-day TCE used for pricing (index 0 = next business day)."""
    return tce_paths.mean(axis=0) if quantile is None else np.quantile(tce_paths, quantile, axis=0)


def solve_allocation(
    cargoes: list[Cargo],
    tce_paths_by_class: dict[str, np.ndarray],
    bunker_usd_t: float,
    fleet: list[FleetVessel] | None = None,
    spot_classes: list[str] | None = None,
    slot_days: int = 2,
    rate_quantile: float | None = None,
    carbon_usd_t: float = 0.0,
    time_limit_s: float = 20.0,
) -> dict:
    fleet = SAMPLE_FLEET if fleet is None else fleet
    spot_classes = list(tce_paths_by_class) if spot_classes is None else spot_classes
    curves = {k: _rate_curve(p, rate_quantile) for k, p in tce_paths_by_class.items()}

    def tce_on(vclass: str, day: int) -> float:
        c = curves[vclass]
        return float(c[min(max(day - 1, 0), len(c) - 1)])

    cands = []  # (kind, owner_id, vessel_class, cargo, start_day, VoyageCost, hire_basis)
    for c in cargoes:
        days = list(range(c.laycan_start, c.laycan_end + 1, slot_days))
        if days[-1] != c.laycan_end:
            days.append(c.laycan_end)
        for v in fleet:
            for s in days:
                if s < v.open_day:
                    continue
                rate = v.hire_usd_day if v.hire_usd_day is not None else tce_on(v.vessel_class, s)
                vc = voyage_cost(v.vessel_class, c.load_port, c.discharge_port, c.quantity_t, rate, bunker_usd_t,
                                 carbon_usd_t=carbon_usd_t)
                if vc.feasible:
                    basis = "period hire" if v.hire_usd_day is not None else "index-linked (forecast TCE)"
                    cands.append(("fleet", v.id, v.vessel_class, c, s, vc, basis))
        for k in spot_classes:
            for s in days:
                vc = voyage_cost(k, c.load_port, c.discharge_port, c.quantity_t, tce_on(k, s), bunker_usd_t,
                                 carbon_usd_t=carbon_usd_t)
                if vc.feasible:
                    cands.append(("spot", f"SPOT-{k}", k, c, s, vc, "spot (forecast TCE)"))

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("time_limit", float(time_limit_s))
    h.setOptionValue("mip_rel_gap", 1e-6)
    z = [h.addBinary(obj=float(cd[5].total_usd)) for cd in cands]
    u = {c.id: h.addBinary(obj=UNSERVED_PENALTY_USD) for c in cargoes}
    n_cons = 0

    for c in cargoes:
        h.addConstr(h.qsum(z[i] for i, cd in enumerate(cands) if cd[3].id == c.id) + u[c.id] == 1)
        n_cons += 1
    for v in fleet:
        idx = [i for i, cd in enumerate(cands) if cd[0] == "fleet" and cd[1] == v.id]
        if len(idx) > 1:
            h.addConstr(h.qsum(z[i] for i in idx) <= 1)
            n_cons += 1

    # Berth occupancy by day: a candidate occupies a berth from arrival until
    # discharge completes (lightering at the anchorage does not use a berth).
    occupancy: dict[tuple[str, int], list[int]] = {}
    schedule = []
    for i, (_, _, vclass, c, s, vc, _) in enumerate(cands):
        dp = DISCHARGE_PORTS[c.discharge_port]
        load_days = c.quantity_t / _load_rate(c.load_port) + 0.5
        arrive = s + load_days + vc.sea_days + dp.avg_wait_days + vc.lightering_days
        disch = c.quantity_t / dp.discharge_rate_tpd + 0.5
        a, b = int(math.floor(arrive)), int(math.ceil(arrive + disch))
        schedule.append((arrive, arrive + disch))
        for d in range(a, b):
            occupancy.setdefault((c.discharge_port, d), []).append(i)
    for (port, d), idx in occupancy.items():
        if len(idx) > DISCHARGE_PORTS[port].berths:
            h.addConstr(h.qsum(z[i] for i in idx) <= DISCHARGE_PORTS[port].berths)
            n_cons += 1

    t0 = time.perf_counter()
    h.minimize()
    solve_s = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    sol = h.getSolution().col_value
    val = lambda var: sol[var.index]  # noqa: E731

    chosen = [i for i in range(len(cands)) if val(z[i]) > 0.5]
    assignments = []
    for i in chosen:
        kind, owner, vclass, c, s, vc, basis = cands[i]
        assignments.append({
            "cargo_id": c.id, "vessel": owner, "vessel_name": _vessel_name(owner, fleet), "kind": kind,
            "vessel_class": vclass, "start_day": s, "hire_basis": basis,
            "tce_usd_day": round(vc.hire_usd / vc.total_days), "berth_from_day": round(schedule[i][0], 1),
            "berth_to_day": round(schedule[i][1], 1), "voyage": vc.to_dict(),
        })
    assignments.sort(key=lambda a: (a["start_day"], a["cargo_id"]))
    unserved = [c.id for c in cargoes if val(u[c.id]) > 0.5]
    total = sum(cands[i][5].total_usd for i in chosen)
    tonnes = sum(cands[i][3].quantity_t for i in chosen)

    # Baseline: every cargo on the cheapest feasible spot ship fixed on laycan
    # opening at today's (day-1) rate: what a desk without a forecast does.
    baseline = 0.0
    for c in cargoes:
        opts = [voyage_cost(k, c.load_port, c.discharge_port, c.quantity_t, tce_on(k, 1), bunker_usd_t,
                            carbon_usd_t=carbon_usd_t) for k in spot_classes]
        opts = [o for o in opts if o.feasible]
        baseline += min(o.total_usd for o in opts) if opts else float("nan")

    return {
        "status": status,
        "objective_usd": round(total, 0),
        "usd_per_t": round(total / tonnes, 2) if tonnes else None,
        "tonnes": tonnes,
        "assignments": assignments,
        "unserved": unserved,
        "baseline_all_spot_today_usd": None if math.isnan(baseline) else round(baseline, 0),
        "candidates": len(cands),
        "constraints": n_cons,
        "solve_seconds": round(solve_s, 3),
        "solver": f"HiGHS {highspy.Highs().version()}" if hasattr(highspy.Highs, "version") else "HiGHS",
        "rate_basis": "mean of Monte Carlo TCE paths" if rate_quantile is None else f"P{int(rate_quantile * 100)} of Monte Carlo TCE paths",
    }


def _load_rate(load_port: str) -> float:
    return LOAD_PORTS[load_port].load_rate_tpd


def _vessel_name(owner: str, fleet: list[FleetVessel]) -> str:
    for v in fleet:
        if v.id == owner:
            return v.name
    return owner.replace("SPOT-", "Spot ")


def default_cargoes() -> list[Cargo]:
    """An illustrative monthly import programme for an east-coast steel/power buyer."""
    return [
        Cargo("C1", "hay_point", "dhamra", 160_000, 3, 12),
        Cargo("C2", "newcastle", "paradip", 75_000, 1, 8),
        Cargo("C3", "samarinda", "haldia", 55_000, 2, 10),
        Cargo("C4", "richards_bay", "gangavaram", 150_000, 8, 18),
        Cargo("C5", "taboneo", "gopalpur", 52_000, 5, 14),
        Cargo("C6", "samarinda", "vizag", 70_000, 10, 20),
    ]


def known_classes() -> list[str]:
    return list(VESSEL_CLASSES)
