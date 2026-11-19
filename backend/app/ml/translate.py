"""Index points -> time-charter-equivalent (TCE) $/day by vessel class.

Since 1 March 2018 the Baltic Exchange computes

    BDI = 0.1 * (0.4 * Capesize 5TC + 0.3 * Panamax 5TC + 0.3 * Supramax 10TC)

so the weighted average TCE in $/day is ``BDI * 10``. Each class's rate is
expressed as a ratio to that weighted average. The default ratios below are
long-run averages of the published timecharter averages and satisfy
``0.4*cape + 0.3*panamax + 0.3*supramax = 1``. They are assumptions, not
observations, and can be overridden with ``OP_CLASS_RATIOS`` (JSON) or per
request. With licensed per-route TCE data they should be replaced by
regressions on that data.
"""

from __future__ import annotations

BDI_TO_WEIGHTED_TCE = 10.0
BALTIC_WEIGHTS = {"Capesize": 0.4, "Panamax": 0.3, "Supramax": 0.3}

DEFAULT_CLASS_RATIOS: dict[str, float] = {
    "Newcastlemax": 1.36,
    "Capesize": 1.30,
    "Kamsarmax": 0.84,
    "Panamax": 0.80,
    "Ultramax": 0.83,
    "Supramax": 0.80,
    "Handysize": 0.66,
}


def check_ratios(ratios: dict[str, float]) -> float:
    """Return the Baltic-weighted sum of the core ratios (should be ~1)."""
    return sum(w * ratios[c] for c, w in BALTIC_WEIGHTS.items())


def bdi_to_tce(bdi: float, vessel_class: str, ratios: dict[str, float] | None = None) -> float:
    r = (ratios or DEFAULT_CLASS_RATIOS)
    if vessel_class not in r:
        raise KeyError(f"Unknown vessel class {vessel_class!r}; known: {sorted(r)}")
    return float(bdi) * BDI_TO_WEIGHTED_TCE * r[vessel_class]


def tce_table(bdi: float, ratios: dict[str, float] | None = None) -> dict[str, float]:
    r = ratios or DEFAULT_CLASS_RATIOS
    return {c: round(bdi_to_tce(bdi, c, r)) for c in r}
