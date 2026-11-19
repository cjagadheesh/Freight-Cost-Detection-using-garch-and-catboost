import numpy as np
import pandas as pd

from app.ml import ecm


def cointegrated(n=400, gamma=0.15, beta=0.8, seed=1):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 0.02, n)) + 2.5
    b = np.empty(n)
    b[:3] = 7.0 + beta * f[:3]
    for t in range(2, n - 1):
        basis = b[t] - 7.0 - beta * f[t]
        b[t + 1] = b[t] + 0.3 * (f[t] - f[t - 1]) - gamma * basis + rng.normal(0, 0.01)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"b": b, "f": f}, index=idx)


def test_fit_recovers_adjustment_speed():
    fit = ecm.fit_ecm(cointegrated())
    assert abs(fit.gamma - 0.15) < 0.05
    assert abs(fit.beta - 0.8) < 0.1
    assert fit.half_life_days is not None


def test_simulation_converges_to_implied_level():
    data = cointegrated()
    fit = ecm.fit_ecm(data)
    b = data["b"].values.copy()
    b[-3:] += 0.3  # index well above its futures-implied level
    path = ecm.simulate(fit, b, data["f"].values, np.full(200, data["f"].values[-1]))[0]
    implied = fit.alpha + fit.beta * data["f"].values[-1]
    assert abs(path[-1] - implied) < abs(b[-1] - implied) * 0.1


def test_walk_forward_has_no_lookahead():
    data = cointegrated(300)
    a = ecm.walk_forward(data, horizons=[1, 5], min_train=150)
    mod = data.copy()
    mod.iloc[250:, mod.columns.get_loc("b")] += 1.0
    b = ecm.walk_forward(mod, horizons=[1, 5], min_train=150)
    early_a = a[a["date"] < data.index[240]]["ecm"].values
    early_b = b[b["date"] < data.index[240]]["ecm"].values
    assert np.allclose(early_a, early_b)


def test_signal_policy_tapers():
    w = ecm.signal_weight(np.array([1, 5, 13, 21, 40]))
    assert list(w[:2]) == [1.0, 1.0] and 0 < w[2] < 1 and w[3] == 0 and w[4] == 0


def test_real_bdi_ecm_beats_random_walk_one_day_ahead(md):
    data = ecm.align(md.bdi, md.target)
    m = {r["h"]: r for r in ecm.evaluate(ecm.walk_forward(data))}
    assert m[1]["skill_vs_rw_pct"] > 10
