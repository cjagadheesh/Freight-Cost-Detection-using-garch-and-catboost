# OceanPulse: dry-bulk freight intelligence

Freight forecasts and chartering decisions for an importer bringing coal and ore into India's east-coast
ports (Paradip, Dhamra, Gopalpur, Visakhapatnam, Gangavaram, Haldia). Everything runs on **real market
data**, and every accuracy figure is **out-of-sample**.

| | |
|---|---|
| **Data** | 12 daily Yahoo Finance series, 2018 onwards (BDRY freight-futures ETF, Brent, iron ore, coal proxy, copper, DXY, USD/INR, VIX, S&P 500, three listed dry-bulk owners), plus a Koyfin Baltic Dry Index export |
| **Models** | GARCH(1,1)-t volatility, stacked Ridge/CatBoost/random-walk ensemble, BDI error-correction model, adaptive conformal intervals, Monte Carlo (filtered historical simulation) |
| **Decisions** | MILP fleet allocation (HiGHS), fixing-time CVaR analysis, virtual-arrival speed optimisation, IMO CII rating, landed cost in $/t, ₹/t and $/GJ, historical stress replay |
| **Stack** | FastAPI · pandas · scikit-learn · CatBoost · arch · HiGHS / React 19 · Vite · Tailwind 4 · Recharts · Leaflet |

## Results (walk-forward, 2021–2026)

**Freight futures (BDRY), 1,359–1,421 forecasts per horizon.** Neither learner beats a random walk at
any horizon (Diebold–Mariano p > 0.85). The stacking weights therefore move onto the random walk, and the
ensemble ends within 2.3% MSE of it. This is the result you would expect from efficiently priced futures,
and the system reports it as found.

| Horizon | RMSE | Skill vs RW | 80% coverage | 95% coverage |
|---:|---:|---:|---:|---:|
| 1d | 3.8% | −0.5% | 80.3% | 95.4% |
| 5d | 8.2% | −0.7% | 80.3% | 94.8% |
| 21d | 18.6% | −2.3% | 79.9% | 95.4% |
| 63d | 32.6% | −0.1% | 81.2% | 89.1% |

The intervals are where the value is. A static split-conformal calibration under-covered at 42 and 63
days (about 70% against an 80% target) because freight volatility regimes outlast GARCH mean reversion.
**Adaptive conformal inference** (Gibbs & Candès, 2021) receives each outcome only once it is realised,
h days later, and brings coverage onto target at every horizon.

**Physical index (BDI), error-correction model on BDRY.** The index lags its futures, and the model
exploits that:

| Horizon | Skill vs RW | Direction hit | DM p |
|---:|---:|---:|---:|
| 1d | **+26.9%** | 74.4% | 0.03 |
| 5d | +5.3% | 60.5% | 0.21 |
| 10d | −10.2% | 56.8% | 0.78 |
| 21d | −18.8% | 43.9% | 0.77 |

One year of BDI history holds only about six independent 21-day windows, so long-horizon evidence is
weak in either direction. The production forecast follows a rule fixed in advance: use the full ECM
signal up to 5 days, then taper linearly to a random walk by 21 days.

## How it fits together

```
Yahoo + Koyfin ─► data/raw snapshot ─► leak-free features (exogenous lagged 1 bday)
                                          │
              ┌───────────────────────────┴──────────────────────────┐
     BDRY layer: Ridge + CatBoost + RW                   BDI layer: error-correction model
     per horizon, purged walk-forward,                   (own lags, BDRY moves, basis to the
     convex stacking, GARCH-t sigma,                      BDRY-implied level)
     adaptive conformal bands                                     │
              │                                                   │
              └──► 5,000 GARCH-FHS BDRY paths ──► nowcast across the BDI data gap
                                                  on actual BDRY closes, then forward
                                                          │
                                   TCE $/day by class (BDI × 10 × class ratio)
                                                          │
          ┌───────────────┬──────────────┬────────────────┼──────────────┬──────────────┐
     MILP fleet       Fixing day      Speed / CII     Landed cost     Stress replay   Risk signals
     allocation       (E, CVaR)       (cubic fuel)    ($/t, ₹/t, $/GJ) (real episodes) (percentiles)
```

## Running locally

Python 3.12 is required (CatBoost wheels).

```bash
# backend
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m app.ml.train            # ~35 s; add --refresh to download fresh market data
uvicorn app.main:app --reload     # http://127.0.0.1:8000/docs
python -m pytest -q               # 44 tests

# frontend (separate shell, repo root)
npm ci
npm run dev                       # http://localhost:5173, proxies /api to :8000
```

Docker: `docker build -t oceanpulse-api backend && docker run -p 8000:8000 oceanpulse-api`.

### Configuration (`OP_` environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `OP_REFRESH_ON_STARTUP` | `false` | Download fresh Yahoo data at startup and retrain if it moved |
| `OP_TRAIN_IF_MISSING` | `true` | Train a bundle at startup if none is on disk |
| `OP_ADMIN_TOKEN` | *(unset)* | Enables `POST /api/v1/admin/refresh` with `Authorization: Bearer <token>` |
| `OP_CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `OP_BRENT_TO_VLSFO`, `OP_VLSFO_PREMIUM` | 7.33, 55 | Bunker proxy: VLSFO ≈ slope × Brent + premium |

To update the BDI layer, drop a newer Koyfin export (`koyfin_*.csv`) into `backend/data/koyfin/` and
retrain. Files are merged, so overlapping exports are fine.

## API

`/api/v1` (interactive docs at `/docs`):

| Method | Path | |
|---|---|---|
| GET | `/status`, `/health` | readiness, data and bundle provenance |
| GET | `/market/snapshot`, `/market/series`, `/market/quality`, `/market/catalog` | data |
| GET | `/forecast`, `/forecast/tce?vessel_class=`, `/forecast/explain?h=` | forecasts and SHAP |
| GET | `/backtest?h=`, `/models` | walk-forward evidence and fitted parameters |
| GET | `/risk`, `/reference` | risk signals, ports, vessel classes |
| POST | `/optimize/allocation`, `/optimize/timing`, `/optimize/speed` | decisions |
| POST | `/landed-cost`, `/stress` | costs and scenarios |
| POST | `/admin/refresh` | token-protected data refresh and retrain |

## Deployment

`render.yaml` defines both services. The backend builds from `backend/`, trains during the build and
serves with uvicorn. The frontend is a static Vite build that reads `VITE_API_BASE_URL`. CI
(`.github/workflows/ci.yml`) trains, runs the tests, then lints and builds the frontend.

## What is assumed rather than observed

* **TCE by class.** The 2018 Baltic formula fixes the *weighted* TCE as BDI × 10. Ratios for individual
  classes are long-run averages and configurable. Per-route Baltic TCE data would replace them.
* **Port limits, distances and the sample fleet** are planning inputs from port handbooks and approximate
  routeings. All of them can be overridden through the API.
* **FOB coal prices** are a user input. No free real-time seaborne coal feed exists: Yahoo's API2
  continuation stopped updating in December 2025, so a listed exporter (Whitehaven) serves as the
  coal-price signal.
* **The BDI export ends on 2026-08-28.** The spot level after that is nowcast from BDRY, and the UI says
  so with an uncertainty band.

See [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) for design decisions and the questions to
expect.
