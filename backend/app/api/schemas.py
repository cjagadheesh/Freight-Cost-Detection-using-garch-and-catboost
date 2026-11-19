"""Request models. Keys are validated against the reference data."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.reference import DISCHARGE_PORTS, LOAD_PORTS, VESSEL_CLASSES


def _check(value: str, valid: dict, what: str) -> str:
    if value not in valid:
        raise ValueError(f"unknown {what} {value!r}; expected one of {sorted(valid)}")
    return value


class RouteIn(BaseModel):
    load_port: str = "samarinda"
    discharge_port: str = "paradip"
    vessel_class: str = "Panamax"
    cargo_t: float = Field(70_000, gt=0, le=250_000)

    @field_validator("load_port")
    @classmethod
    def _lp(cls, v):
        return _check(v, LOAD_PORTS, "load port")

    @field_validator("discharge_port")
    @classmethod
    def _dp(cls, v):
        return _check(v, DISCHARGE_PORTS, "discharge port")

    @field_validator("vessel_class")
    @classmethod
    def _vc(cls, v):
        return _check(v, VESSEL_CLASSES, "vessel class")


class CargoIn(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    load_port: str
    discharge_port: str
    quantity_t: float = Field(gt=0, le=250_000)
    laycan_start: int = Field(ge=1, le=60)
    laycan_end: int = Field(ge=1, le=60)

    @field_validator("load_port")
    @classmethod
    def _lp(cls, v):
        return _check(v, LOAD_PORTS, "load port")

    @field_validator("discharge_port")
    @classmethod
    def _dp(cls, v):
        return _check(v, DISCHARGE_PORTS, "discharge port")

    @model_validator(mode="after")
    def _laycan(self):
        if self.laycan_end < self.laycan_start:
            raise ValueError("laycan_end must be >= laycan_start")
        return self


class FleetVesselIn(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    name: str = Field(min_length=1, max_length=60)
    vessel_class: str
    open_day: int = Field(1, ge=1, le=60)
    hire_usd_day: float | None = Field(None, gt=0, le=200_000)
    built: int = Field(2015, ge=1990, le=2030)

    @field_validator("vessel_class")
    @classmethod
    def _vc(cls, v):
        return _check(v, VESSEL_CLASSES, "vessel class")


class AllocationIn(BaseModel):
    cargoes: list[CargoIn] | None = Field(None, max_length=30, description="Defaults to an illustrative programme")
    fleet: list[FleetVesselIn] | None = Field(None, max_length=30, description="Defaults to the sample fleet")
    spot_classes: list[str] | None = None
    slot_days: int = Field(2, ge=1, le=7)
    rate_quantile: float | None = Field(None, gt=0, lt=1, description="Price spot at this TCE quantile instead of the mean")
    bunker_usd_t: float | None = Field(None, gt=100, le=2000)
    carbon_usd_t: float = Field(0.0, ge=0, le=300)

    @field_validator("spot_classes")
    @classmethod
    def _sc(cls, v):
        if v is not None:
            for k in v:
                _check(k, VESSEL_CLASSES, "vessel class")
        return v

    @model_validator(mode="after")
    def _unique(self):
        for items, what in ((self.cargoes, "cargo"), (self.fleet, "vessel")):
            if items and len({i.id for i in items}) != len(items):
                raise ValueError(f"{what} ids must be unique")
        return self


class TimingIn(RouteIn):
    window_days: int = Field(30, ge=2, le=63)
    risk_aversion: float = Field(0.5, ge=0, le=1)
    cvar_level: float = Field(0.95, ge=0.8, le=0.99)


class SpeedIn(RouteIn):
    tce_usd_day: float | None = Field(None, gt=0, le=200_000, description="Defaults to the nowcast TCE")
    bunker_usd_t: float | None = Field(None, gt=100, le=2000)
    berth_available_day: float | None = Field(None, gt=0, le=60)
    carbon_usd_t: float = Field(0.0, ge=0, le=300)
    year: int = Field(2026, ge=2023, le=2030)


class LandedIn(RouteIn):
    fob_usd_t: float | None = Field(None, gt=0, le=1000)
    cv_kcal_kg: float | None = Field(None, ge=2500, le=8000)
    fixing_day: int = Field(1, ge=1, le=63)
    insurance_pct: float = Field(0.15, ge=0, le=2)
    bunker_usd_t: float | None = Field(None, gt=100, le=2000)
    carbon_usd_t: float = Field(0.0, ge=0, le=300)


class StressIn(RouteIn):
    window_days: int = Field(42, ge=5, le=126)
