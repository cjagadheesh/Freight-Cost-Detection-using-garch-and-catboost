"""Catalogue of the real market series the platform ingests.

Every series here is observed market data. Nothing is generated or simulated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    key: str  # column name in the panel
    symbol: str  # source symbol
    name: str
    unit: str
    role: str  # target | driver | equity | macro
    source: str = "yahoo"
    description: str = ""


TARGET_KEY = "bdry"

CATALOG: list[SeriesSpec] = [
    SeriesSpec(
        "bdry", "BDRY", "Breakwave Dry Bulk Shipping ETF", "USD/share", "target",
        description=(
            "Holds near-dated Capesize (50%), Panamax (40%) and Supramax (10%) "
            "freight futures (FFAs). Longest freely available daily dry-bulk freight series."
        ),
    ),
    SeriesSpec("brent", "BZ=F", "Brent crude oil (front month)", "USD/bbl", "driver",
               description="Bunker fuel (VLSFO) cost driver."),
    SeriesSpec("iron_ore", "TIO=F", "Iron ore 62% Fe CFR China", "USD/t", "driver",
               description="Largest dry-bulk trade by volume; Capesize demand driver."),
    SeriesSpec("coal", "WHC.AX", "Whitehaven Coal (ASX)", "AUD", "driver",
               description=(
                   "Newcastle thermal/met coal exporter; listed proxy for seaborne coal prices "
                   "(Yahoo's API2 futures continuation stopped updating in Dec-2025)."
               )),
    SeriesSpec("copper", "HG=F", "Copper (COMEX)", "USD/lb", "driver",
               description="Global industrial demand proxy."),
    SeriesSpec("dxy", "DX-Y.NYB", "US Dollar Index", "index", "macro"),
    SeriesSpec("usdinr", "INR=X", "USD/INR", "INR per USD", "macro",
               description="Indian importer's currency exposure."),
    SeriesSpec("vix", "^VIX", "CBOE Volatility Index", "index", "macro"),
    SeriesSpec("spx", "^GSPC", "S&P 500", "index", "macro"),
    SeriesSpec("gnk", "GNK", "Genco Shipping & Trading", "USD", "equity"),
    SeriesSpec("sblk", "SBLK", "Star Bulk Carriers", "USD", "equity"),
    SeriesSpec("dsx", "DSX", "Diana Shipping", "USD", "equity"),
]

CATALOG_BY_KEY = {s.key: s for s in CATALOG}
EQUITY_KEYS = [s.key for s in CATALOG if s.role == "equity"]
EXOG_KEYS = [s.key for s in CATALOG if s.role != "target"]
