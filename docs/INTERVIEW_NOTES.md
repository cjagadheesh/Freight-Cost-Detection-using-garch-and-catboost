# Interview notes

A walkthrough of the decisions in this codebase, the evidence behind them, and the questions they
invite.

## 1. The one-minute pitch

An Indian steel or power company imports coal and ore through east-coast ports and pays freight in
dollars per day. OceanPulse answers four questions: *what will freight cost*, *how sure are we*, *when
should we fix*, and *which ship goes where*. It runs on real market data, and every accuracy number is
out-of-sample. The honest headline: **freight futures are not predictable beyond a random walk, but
their uncertainty is**, and the physical index is predictable one day ahead because it lags its futures.

## 2. Data choices

* **Target: the BDRY ETF.** It is the longest free, daily, tradable dry-bulk freight series, holding
  near-dated Capesize/Panamax/Supramax FFAs. The BDI itself is licensed. A one-year Koyfin export
  supplies it for the spot layer.
* **No lookahead.** Exogenous series close in other time zones (Brent in London, iron ore in Singapore,
  Whitehaven in Sydney), so every driver is lagged one business day. Panel alignment forward-fills
  causally for at most 5 days. `tests/test_features.py` shocks the future and asserts that past
  features do not change.
* **Stale sources are dropped, not hidden.** API2 coal futures on Yahoo stopped updating in Dec-2025,
  so Whitehaven Coal replaced them. `active_drivers()` excludes any driver more than 10 business days
  stale from live features.

## 3. Forecasting the futures (BDRY)

* **Direct multi-horizon models** (h = 1, 5, 10, 21, 42, 63) predict the h-day log return, with no
  recursive roll-out. Labels are divided by trailing realised volatility, so the learners see
  comparable targets across regimes.
* **Purged walk-forward.** The window expands and refits quarterly. For horizon h, training stops at row
  `a-1-h`, so every training label is realised before the fold starts. `test_walk_forward_purges_training_labels`
  checks the exact cut-off.
* **Stacking** uses convex weights (w ≥ 0, Σw = 1) fitted only on earlier out-of-fold predictions.
  Because the random walk predicts zero, its weight acts as data-driven shrinkage.
* **Result:** both learners lose to the random walk (Ridge −19%, CatBoost −38% at 21 days), and the
  stack converges to the random walk. *Why ship them?* They are the honest test. Had they beaten the
  random walk, the stack would have used them. The framework is the product; the null result is a
  finding.
* **Significance:** a one-sided Diebold–Mariano test with Newey–West variance (h−1 lags) and the
  Harvey–Leybourne–Newbold small-sample correction. Overlapping h-day errors are autocorrelated, so a
  plain t-test would overstate significance.

## 4. Getting the uncertainty right

* **GARCH(1,1) with Student-t innovations** gives the volatility term structure: persistence about
  0.989, half-life about 61 days, tail ν ≈ 10.7.
* **Conformal on GARCH-scaled residuals** makes the bands distribution-free.
* **The problem:** static calibration covered only about 70% at 42/63 days against an 80% target.
  Freight volatility regimes (2021 boom, 2022 bust) last longer than GARCH mean reversion assumes.
* **The fix:** adaptive conformal inference, `α ← α + γ(α* − miss)`, with γ = 0.002 and feedback
  delayed by h days, since an h-day outcome is only known h days later. Coverage is now 79.9–81.2% at
  every horizon. `test_adaptive_conformal_uses_only_realised_scores` proves it never peeks.
* **Alternatives tested and rejected** (on out-of-fold data): realised-vol scaling, recent-window
  calibration (252/504 days), and blended or max scalings. None got above about 73% at 63 days.

## 5. The BDI layer: an error-correction model

* **Economics:** FFAs embed the expected path of the physical index, so the BDI closes the gap (the
  "basis") to the level its futures imply. The model is
  `Δb[t+1] = c + a·Δb + g·Δf − γ·(b − α − βf) + ε`.
* **Timing is the subtle part.** The BDI publishes at about 13:00 London and BDRY closes at 16:00 New
  York. Forecasting after BDRY's close on day t, the BDRY move on day t is known, and it carries the
  biggest coefficient (0.35).
* **Evidence:** +27% MSE skill at one day (DM p = 0.03), 74% direction hit, then nothing reliable past a
  week.
* **What I did not do:** tune the horizon taper on the same 134 forecasts. I tested direct per-horizon
  regressions (worse: −68% at 10 days) and learned shrinkage weights (unstable: the 21-day result swung
  from −19% to +32% depending on the start date). I fixed a simple rule in advance instead: full signal
  to 5 days, linear taper to zero at 21.
* **Nowcasting the gap.** The export ends 2026-08-28. The model runs forward over 58 sessions on
  *actual* BDRY closes, with bootstrapped residuals, so today's spot level comes with an honest band
  (about ±15%). Each Monte Carlo path carries its own nowcast state into the forecast.

## 6. From index points to dollars

* **Baltic 2018 formula:** BDI = 0.1 × (0.4·Cape 5TC + 0.3·Panamax 5TC + 0.3·Supramax 10TC), so the
  weighted TCE is BDI × 10. Class ratios satisfy 0.4c + 0.3p + 0.3s = 1 (unit-tested). They are
  assumptions, flagged as such.
* **Bunkers:** VLSFO ≈ 7.33 × Brent + 55 $/t, or a quoted price per request.

## 7. Decision engines

* **MILP (HiGHS).** Candidate generation does the physics: intake, load-port draft, discharge draft/LOA,
  and lightering (Haldia's 8.5 m Hooghly draft forces barging at Sandheads). The solver handles the
  combinatorics: one voyage per vessel, every cargo served or flagged unserved, and day-level berth
  occupancy. The default instance solves in milliseconds (155 options, 83 constraints).
  `test_allocation_matches_brute_force` checks optimality. *Why HiGHS directly?* PuLP 4 replaced its
  API with a Rust core, and HiGHS was the solver either way, so its own modelling API is one dependency
  fewer.
* **Risk-aware pricing:** the planner can price spot tonnage at P75/P90 of the simulated TCE instead of
  the mean.
* **Fixing time:** for each candidate day, compute expected cost, CVaR95 and P(cheaper than today) on
  the same 5,000 paths. Waiting is recommended only if the risk-adjusted score beats day 1 by more than
  two paired Monte Carlo standard errors, so simulation noise never drives a recommendation.
* **Speed:** fuel ∝ v³ per day, so fuel per voyage ∝ v². The unconstrained optimum is
  `v* = (H·v_d³ / (2·P·f_d))^(1/3)`. With a known berth wait, steaming fast just buys anchorage time:
  virtual arrival saves fuel at no hire cost.
* **CII:** MEPC.353/354(78) bulk-carrier parameters (a = 4745, c = 0.622, capacity capped at 279k DWT),
  Z = 11% for 2026, rating boundaries 0.86/0.94/1.06/1.18. Tested against a hand calculation.
* **Stress replay:** the largest non-overlapping 42-day BDRY rallies and slumps since 2018, mapped to
  spot TCE through the ECM's long-run elasticity, with Brent's move over the same dates. Every scenario
  actually happened.

## 8. Engineering

* **Training is offline.** `python -m app.ml.train` writes a versioned bundle (pickle, atomic rename)
  plus `metrics.json`, in about 35 seconds. The API loads it at startup and never trains on a request.
* **Atomic refresh.** `POST /admin/refresh` (bearer token, constant-time compare) builds a new snapshot
  off to the side and swaps it in. A failure keeps serving the old one.
* **Validation:** pydantic models reject unknown ports or classes, impossible laycans and duplicate IDs
  with 422.
* **Tests (44):** leakage, purging, conformal validity and causality, ECM parameter recovery and no
  lookahead, the real-data ECM skill check, MILP against brute force, berth constraints, CII, and
  end-to-end API.

## 9. Likely questions

* **"Your model doesn't beat a random walk. Why is this useful?"** Decisions depend on the distribution,
  not the point. Calibrated bands drive CVaR fixing decisions, risk-priced allocation and budgeting.
  Knowing that the point forecast has no edge stops a desk from paying for false confidence.
* **"Isn't 254 BDI observations too few?"** For long horizons, yes, and the system says so and tapers
  the signal. For one-day-ahead, 134 out-of-sample forecasts at p = 0.03 is real evidence, and the
  mechanism (futures lead spot) is economically grounded.
* **"How would you improve it with a budget?"** Licensed Baltic route TCEs to replace class ratios; FFA
  curves by tenor to replace the ETF proxy; AIS port-congestion data for waiting times; quoted bunker
  and FOB prices; a multi-voyage MILP with ballast repositioning.
* **"How do you know there's no leakage?"** Lagged exogenous inputs, purged labels and sequential
  calibration, each with a test that perturbs the future and asserts the past is unchanged.
