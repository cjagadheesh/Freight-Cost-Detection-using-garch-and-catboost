"""Application settings, loaded from environment variables (prefix ``OP_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OP_", env_file=".env", extra="ignore")

    app_name: str = "OceanPulse Freight Intelligence API"
    environment: str = "development"
    log_level: str = "INFO"

    # Storage
    data_dir: Path = BACKEND_DIR / "data"

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    cors_origin_regex: str | None = r"https://.*\.onrender\.com|https://.*\.netlify\.app"

    # Data ingestion
    history_start: str = "2018-01-01"
    http_timeout_s: float = 20.0
    # Refresh market data from the network at startup. When false (or offline),
    # the committed snapshot in data/raw is used.
    refresh_on_startup: bool = False
    # Train a model bundle at startup if none exists on disk.
    train_if_missing: bool = True
    # Token for POST /api/v1/admin/refresh. Empty disables the endpoint.
    admin_token: str = ""

    # Modelling
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 21, 42, 63])
    backtest_initial_train_days: int = 504  # ~2 trading years
    backtest_step_days: int = 63  # refit each quarter
    catboost_iterations: int = 500
    random_seed: int = 7

    # Commercial assumptions (documented, override per request where relevant)
    # VLSFO $/t ~= brent_to_vlsfo * Brent $/bbl + vlsfo_premium
    brent_to_vlsfo: float = 7.33
    vlsfo_premium: float = 55.0
    carbon_price_usd_per_t: float = 0.0  # set >0 to price CO2 (e.g. EU ETS-style)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def koyfin_dir(self) -> Path:
        return self.data_dir / "koyfin"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
