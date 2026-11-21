import itertools

import numpy as np
import pandas as pd

from app.domain.allocation import Cargo, solve_allocation
from app.domain.emissions import cii_rating, cii_required, optimise_speed
from app.domain.reference import VESSEL_CLASSES, FleetVessel
from app.domain.timing import fixing_analysis
from app.domain.voyage import draft_check, fuel_tpd, voyage_cost
from app.ml.translate import BALTIC_WEIGHTS, DEFAULT_CLASS_RATIOS, bdi_to_tce, check_ratios


def test_class_ratios_reproduce_bdi_weighting():
    assert abs(check_ratios(DEFAULT_CLASS_RATIOS) - 1.0) < 1e-9
    bdi = 2000
    weighted = sum(w * bdi_to_tce(bdi, c) for c, w in BALTIC_WEIGHTS.items())
    assert abs(weighted * 0.1 - bdi) < 1e-6


def test_fuel_follows_cubic_law():
    assert np.isclose(fuel_tpd("Panamax", 10.0) / fuel_tpd("Panamax", 12.5), (10 / 12.5) ** 3)


def test_haldia_requires_lightering_for_panamax():
    ok, lighten, _ = draft_check("Panamax", 70_000, "haldia")
    assert ok and lighten > 0
    ok, lighten, _ = draft_check("Panamax", 70_000, "gangavaram")
    assert ok and lighten == 0


def test_gopalpur_rejects_capesize():
    vc = voyage_cost("Capesize", "richards_bay", "gopalpur", 150_000, 30_000, 600)
    assert not vc.feasible


def test_voyage_cost_adds_up():
    vc = voyage_cost("Panamax", "samarinda", "paradip", 70_000, 20_000, 600)
    assert np.isclose(vc.total_usd, vc.hire_usd + vc.fuel_usd + vc.port_usd + vc.lightering_usd + vc.carbon_usd)
    assert np.isclose(vc.hire_usd, 20_000 * vc.total_days)


def _flat_paths(level):
    return {k: np.full((50, 63), level * r) for k, r in DEFAULT_CLASS_RATIOS.items()}


def test_allocation_matches_brute_force():
    cargoes = [Cargo("A", "samarinda", "paradip", 70_000, 1, 3), Cargo("B", "taboneo", "vizag", 60_000, 1, 3)]
    fleet = [FleetVessel("F1", "F1", "Panamax", 1, 12_000, 2015), FleetVessel("F2", "F2", "Ultramax", 1, 11_000, 2018)]
    paths = _flat_paths(20_000)
    res = solve_allocation(cargoes, paths, 600, fleet=fleet, spot_classes=["Panamax", "Supramax"], slot_days=2)
    assert res["status"] == "Optimal" and not res["unserved"]
    # brute force over every fleet/spot choice at the laycan opening (flat rates => day irrelevant except availability)
    options = {}
    for c in cargoes:
        opts = []
        for v in fleet:
            vc = voyage_cost(v.vessel_class, c.load_port, c.discharge_port, c.quantity_t, v.hire_usd_day, 600)
            if vc.feasible:  # committed period hire is sunk: marginal cost excludes it
                opts.append((v.id, vc.total_usd - vc.hire_usd))
        for k in ["Panamax", "Supramax"]:
            vc = voyage_cost(k, c.load_port, c.discharge_port, c.quantity_t, 20_000 * DEFAULT_CLASS_RATIOS[k], 600)
            if vc.feasible:
                opts.append((f"S-{k}-{c.id}", vc.total_usd))
        options[c.id] = opts
    best = min(a[1] + b[1] for a, b in itertools.product(options["A"], options["B"]) if a[0] != b[0] or a[0].startswith("S"))
    assert abs(res["objective_usd"] - best) < 1.0


def test_allocation_respects_berths():
    # Three big cargoes into one-berth Gopalpur on the same day must be staggered or left unserved.
    cargoes = [Cargo(f"C{i}", "taboneo", "gopalpur", 52_000, 1, 1) for i in range(3)]
    res = solve_allocation(cargoes, _flat_paths(20_000), 600, fleet=[], spot_classes=["Supramax"])
    served = res["assignments"]
    for a, b in itertools.combinations(served, 2):
        assert a["berth_to_day"] <= b["berth_from_day"] + 1 or b["berth_to_day"] <= a["berth_from_day"] + 1
    assert len(served) + len(res["unserved"]) == 3


def test_cii_reference_values():
    # Bulk carrier, 76,000 DWT, 2026 (Z = 11%)
    req = cii_required(76_000, 2026)
    assert np.isclose(req, 0.89 * 4745 * 76_000 ** -0.622)
    assert cii_rating(req * 0.85, req) == "A"
    assert cii_rating(req * 1.0, req) == "C"
    assert cii_rating(req * 1.2, req) == "E"


def test_speed_optimiser_slows_down_for_berth_wait():
    res = optimise_speed("Panamax", "samarinda", "paradip", 70_000, 20_000, 600, berth_available_day=14)
    assert res["optimal"]["speed_kn"] < VESSEL_CLASSES["Panamax"].design_speed_kn
    assert res["saving_vs_design_usd"] >= 0


def test_timing_prefers_fixing_now_when_rates_rise():
    n, H = 2000, 30
    rng = np.random.default_rng(0)
    paths = 20_000 * np.exp(np.cumsum(rng.normal(0.01, 0.02, (n, H)), axis=1))
    res = fixing_analysis(paths, pd.bdate_range("2026-01-01", periods=H), 20, H)
    assert res["recommended_day"] == 1


def test_timing_waits_when_rates_fall_clearly():
    n, H = 2000, 30
    rng = np.random.default_rng(0)
    paths = 20_000 * np.exp(np.cumsum(rng.normal(-0.02, 0.005, (n, H)), axis=1))
    res = fixing_analysis(paths, pd.bdate_range("2026-01-01", periods=H), 20, H, risk_aversion=0)
    assert res["recommended_day"] > 10


def test_virtual_arrival_speed_is_clipped():
    res = optimise_speed("Panamax", "samarinda", "paradip", 70_000, 20_000, 600, berth_available_day=1)
    vc = VESSEL_CLASSES["Panamax"]
    assert vc.min_speed_kn <= res["virtual_arrival_speed_kn"] <= vc.max_speed_kn
    assert "late" in res["virtual_arrival_note"]
