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

| Horizon | RMSE | Skill vs RW | 80% coverage (1st / 2nd half) | 95% coverage |
|---:|---:|---:|---:|---:|
| 1d | 3.8% | −0.5% | 80.5% (79.4 / 81.6) | 95.3% |
| 5d | 8.2% | −0.6% | 80.3% (78.1 / 82.6) | 94.8% |
| 21d | 18.6% | −2.2% | 80.0% (77.3 / 82.8) | 95.4% |
| 42d | 25.4% | −0.3% | 80.0% (73.6 / 86.4) | 91.4% |
| 63d | 32.6% | −0.2% | 81.2% (66.8 / 95.5) | 89.1% |

The intervals are where the value is, with a caveat. A static split-conformal calibration under-covered
at 42 and 63 days (about 70% against an 80% target). **Adaptive conformal inference** (Gibbs & Candès,
2021) receives each outcome only once it is realised, h days later, and brings *average* coverage onto
target at every horizon. The halves show what the average hides: at 42–63 days the bands under-covered
through the 2021–22 boom and bust and over-covered afterwards, and the 95% level hits its adaptive floor
at 63 days. Plain Gaussian GARCH bands also average about 80%. Short-horizon bands (up to 21 days) are
stable; long-horizon bands are a reasonable range, not a precise one.

**Physical index (BDI), error-correction model on BDRY.** The index lags its futures, and the model
exploits that:

| Horizon | Skill vs RW | Direction hit | DM p |
|---:|---:|---:|---:|
| 1d | **+26.9%** | 74.4% | 0.03 |
| 5d | +5.3% | 60.5% | 0.21 |
| 10d | −10.2% | 56.8% | 0.78 |
| 21d | −18.8% | 43.9% | 0.77 |

One year of BDI history holds only about six independent 21-day windows, so evidence is thin: the
block-bootstrap 90% range for skill is +12% to +40% at 1 day and −8% to +15% at 5 days. The production
forecast uses the full ECM signal up to 5 days, then tapers linearly to a random walk by 21 days. That
rule was chosen after seeing these results (it was not grid-tuned, but it is not independent of them).
The one-day edge also needs a current BDI print: with the export ending 2026-08-28, today's level is a
nowcast and the edge does not apply until a newer export is loaded.

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
python -m pytest -q               # 53 tests

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
