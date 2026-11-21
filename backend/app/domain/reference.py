"""Static reference data: ports, routes and vessel classes.

These are planning assumptions, not market observations. Port limits come from
published port handbooks (rounded), sea distances are approximate routeings
in nautical miles, and the sample fleet is illustrative so the optimiser has
inputs to work on. Every API that uses them accepts overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

KCAL_PER_KG_TO_GJ_PER_T = 4.1868e-3  # 1 kcal/kg = 4.1868 kJ/kg = 4.1868e-3 GJ/t


@dataclass(frozen=True)
class VesselClass:
    name: str
    dwt: int
    laden_draft_m: float
    loa_m: int
    design_speed_kn: float
    fuel_at_design_tpd: float  # main + aux at sea, laden, VLSFO t/day
    port_fuel_tpd: float
    min_speed_kn: float = 9.0
    max_speed_kn: float = 14.5

    @property
    def cargo_capacity_t(self) -> float:
        """Usable cargo intake after bunkers, stores and constants (~96% of DWT)."""
        return self.dwt * 0.96


VESSEL_CLASSES: dict[str, VesselClass] = {v.name: v for v in [
    VesselClass("Newcastlemax", 208_000, 18.4, 300, 12.5, 46.0, 5.0),
    VesselClass("Capesize", 180_000, 18.1, 292, 12.5, 43.0, 5.0),
    VesselClass("Kamsarmax", 82_000, 14.4, 229, 12.5, 29.0, 3.5),
    VesselClass("Panamax", 76_000, 14.1, 225, 12.5, 28.0, 3.5),
    VesselClass("Ultramax", 63_500, 13.3, 200, 12.5, 26.0, 3.0),
    VesselClass("Supramax", 57_000, 12.8, 190, 12.5, 24.5, 3.0),
    VesselClass("Handysize", 38_000, 10.5, 180, 12.0, 20.0, 2.5),
]}


@dataclass(frozen=True)
class DischargePort:
    key: str
    name: str
    state: str
    lat: float
    lon: float
    max_draft_m: float
    max_loa_m: int
    discharge_rate_tpd: int
    port_cost_per_t: float
    avg_wait_days: float
    berths: int  # concurrent bulk berths available for this programme
    lightering_via: str | None = None  # anchorage where deep-draft vessels lighten
    lightering_cost_per_t: float = 0.0
    lightering_days: float = 0.0
    note: str = ""


DISCHARGE_PORTS: dict[str, DischargePort] = {p.key: p for p in [
    DischargePort("paradip", "Paradip", "Odisha", 20.26, 86.67, 14.5, 260, 45_000, 3.8, 3.4, 2,
                  note="Draft-limited for laden Capesize; tidal berthing above 14 m"),
    DischargePort("dhamra", "Dhamra", "Odisha", 20.79, 86.96, 17.5, 310, 65_000, 4.5, 2.5, 2,
                  note="Deep-water port with Capesize berths"),
    DischargePort("gopalpur", "Gopalpur", "Odisha", 19.31, 84.97, 13.0, 225, 28_000, 3.6, 2.3, 1,
                  note="Supramax/Ultramax only; SW-monsoon swell"),
    DischargePort("vizag", "Visakhapatnam", "Andhra Pradesh", 17.69, 83.29, 16.5, 290, 55_000, 4.1, 2.1, 2),
    DischargePort("gangavaram", "Gangavaram", "Andhra Pradesh", 17.62, 83.23, 18.5, 325, 70_000, 4.4, 1.6, 2,
                  note="All-weather deep-water port"),
    DischargePort("haldia", "Haldia", "West Bengal", 22.03, 88.06, 8.5, 195, 22_000, 5.4, 5.2, 1,
                  lightering_via="Sandheads", lightering_cost_per_t=6.5, lightering_days=4.0,
                  note="Hooghly draft limits; deep-draft vessels lighten at Sandheads anchorage"),
]}


@dataclass(frozen=True)
class LoadPort:
    key: str
    name: str
    country: str
    lat: float
    lon: float
    max_draft_m: float
    load_rate_tpd: int
    port_cost_per_t: float
    commodity: str
    default_cv_kcal_kg: int  # typical calorific value (GAR/NAR as traded)


LOAD_PORTS: dict[str, LoadPort] = {p.key: p for p in [
    LoadPort("newcastle", "Newcastle", "Australia", -32.92, 151.78, 15.2, 85_000, 2.2, "Thermal / semi-soft coal", 6000),
    LoadPort("hay_point", "Hay Point / DBCT", "Australia", -21.28, 149.30, 19.5, 110_000, 2.4, "Premium hard coking coal", 6700),
    LoadPort("samarinda", "Samarinda (Muara Berau)", "Indonesia", -0.50, 117.15, 18.0, 12_000, 1.8, "Thermal coal GAR 4200", 4200),
    LoadPort("taboneo", "Taboneo", "Indonesia", -3.75, 114.45, 17.5, 15_000, 1.7, "Thermal coal GAR 4800", 4800),
    LoadPort("richards_bay", "Richards Bay", "South Africa", -28.80, 32.08, 19.0, 70_000, 2.6, "Thermal coal NAR 5500", 5500),
    LoadPort("maputo", "Maputo (Matola)", "Mozambique", -25.97, 32.57, 14.3, 35_000, 2.8, "Thermal / met coal", 5500),
    LoadPort("norfolk", "Hampton Roads", "United States", 36.90, -76.33, 15.5, 65_000, 3.5, "Coking coal", 6800),
]}

# Approximate sea distances (nm) from load region to the Bay of Bengal entry,
# plus per-port offsets along the Indian east coast.
_BASE_NM = {"newcastle": 5_100, "hay_point": 4_700, "samarinda": 2_550, "taboneo": 2_400,
            "richards_bay": 4_600, "maputo": 4_400, "norfolk": 11_200}
_PORT_OFFSET_NM = {"gangavaram": -60, "vizag": -55, "gopalpur": 0, "paradip": 60, "dhamra": 110, "haldia": 160}


def sea_distance_nm(load: str, discharge: str) -> float:
    if load not in _BASE_NM or discharge not in _PORT_OFFSET_NM:
        raise KeyError(f"No distance for {load} -> {discharge}")
    return float(_BASE_NM[load] + _PORT_OFFSET_NM[discharge])


@dataclass(frozen=True)
class FleetVessel:
    id: str
    name: str
    vessel_class: str
    open_day: int  # first day (from today) the vessel can start loading
    hire_usd_day: float | None  # period hire already fixed; None = index-linked at forecast TCE
    built: int


SAMPLE_FLEET: list[FleetVessel] = [
    FleetVessel("V1", "MV Kalinga Star", "Capesize", 4, 24_500, 2016),
    FleetVessel("V2", "MV Utkal Pride", "Kamsarmax", 1, 15_750, 2019),
    FleetVessel("V3", "MV Bay Trader", "Panamax", 6, None, 2012),
    FleetVessel("V4", "MV Deccan Spirit", "Ultramax", 2, 14_200, 2020),
    FleetVessel("V5", "MV Hooghly Belle", "Supramax", 3, None, 2011),
    FleetVessel("V6", "MV Coromandel", "Newcastlemax", 9, None, 2021),
]


def reference_payload() -> dict:
    return {
        "vessel_classes": [{**asdict(v), "cargo_capacity_t": round(v.cargo_capacity_t)} for v in VESSEL_CLASSES.values()],
        "discharge_ports": [asdict(p) for p in DISCHARGE_PORTS.values()],
        "load_ports": [asdict(p) for p in LOAD_PORTS.values()],
        "distances_nm": {f"{lp}-{dp}": sea_distance_nm(lp, dp) for lp in LOAD_PORTS for dp in DISCHARGE_PORTS},
        "sample_fleet": [asdict(v) for v in SAMPLE_FLEET],
        "notes": "Planning assumptions (port handbooks, approximate routeing distances); sample fleet is illustrative.",
    }
