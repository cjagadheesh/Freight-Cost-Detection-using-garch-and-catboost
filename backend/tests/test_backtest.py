import numpy as np
import pandas as pd

from app.ml.backtest import adaptive_conformal, diebold_mariano, run_walk_forward, calibrate_sequentially
from app.ml.features import CORE_FEATURES, build_features, build_labels
from tests.conftest import synthetic_panel


def test_walk_forward_purges_training_labels(monkeypatch):
    """For fold start a and horizon h, the last training row is a-1-h, so its
    label log P(a-1) is observed before the first test origin a."""
    import app.ml.backtest as bt

    calls = []
    real_fit = bt.fit_learners

    def spy(X, y, iterations, seed):
        calls.append(X.index.max())
        return real_fit(X, y, 5, seed)

    monkeypatch.setattr(bt, "fit_learners", spy)
    panel = synthetic_panel(800)
    X = build_features(panel)[CORE_FEATURES]
    Y = build_labels(panel, [5, 21])
    r = np.log(panel["bdry"]).diff()
    vol = r.rolling(63).std()
    first_row, init, step = 253, 300, 63
    run_walk_forward(X, Y, r, vol, [5, 21], first_row=first_row, initial_train=init, step=step, iterations=5, seed=0)
    expected = [X.index[a - 1 - h] for a in range(first_row + init, len(X), step) for h in (5, 21)]
    assert calls == expected


def test_adaptive_conformal_hits_target_on_iid_data(rng):
    n = 2000
    rows = []
    for h in (1, 10):
        y = rng.standard_t(5, n) * 0.02
        rows.append(pd.DataFrame({"pos": np.arange(n), "h": h, "fold_start": (np.arange(n) // 63) * 63, "y": y,
                                  "ensemble": 0.0, "sigma": 0.02, "calibrated": True}))
    oof, state = adaptive_conformal(pd.concat(rows, ignore_index=True))
    for h, g in oof.groupby("h"):
        g = g[g["calibrated"]]
        cov80 = ((g["y"] >= g["lo80"]) & (g["y"] <= g["hi80"])).mean()
        assert 0.77 < cov80 < 0.83, (h, cov80)
    assert set(state) == {1, 10}


def test_adaptive_conformal_uses_only_realised_scores(rng):
    """Row m's outcome is realised at m + h: intervals before then must not depend on it."""
    n, h, m = 400, 10, 200
    base = pd.DataFrame({"pos": np.arange(n), "h": h, "fold_start": 0, "y": rng.normal(0, 0.02, n),
                         "ensemble": 0.0, "sigma": 0.02, "calibrated": True})
    a, _ = adaptive_conformal(base)
    mod = base.copy()
    mod.loc[m, "y"] = 5.0
    b, _ = adaptive_conformal(mod)
    cut = m + h
    assert np.allclose(a["lo80"].values[:cut], b["lo80"].values[:cut], equal_nan=True)
    assert not np.allclose(a["hi95"].values[cut:], b["hi95"].values[cut:], equal_nan=True)


def test_diebold_mariano_detects_better_model(rng):
    y = rng.normal(0, 1, 500)
    good = y + rng.normal(0, 0.3, 500)
    stat, p = diebold_mariano(y - good, y, h=1)
    assert stat > 3 and p < 0.01
    stat, p = diebold_mariano(y, y - good, h=1)
    assert p > 0.99


def test_stacking_gate_falls_back_to_random_walk_on_noise(rng):
    from app.ml.backtest import gated_weights

    y = rng.normal(0, 0.05, 600)
    preds = np.column_stack([np.zeros(600), rng.normal(0, 0.05, 600), rng.normal(0, 0.05, 600)])
    assert list(gated_weights(preds, y, 5)) == [1.0, 0.0, 0.0]


def test_stacking_gate_keeps_a_genuinely_useful_learner(rng):
    from app.ml.backtest import gated_weights

    signal = rng.normal(0, 0.05, 600)
    y = signal + rng.normal(0, 0.02, 600)
    preds = np.column_stack([np.zeros(600), signal, rng.normal(0, 0.05, 600)])
    assert gated_weights(preds, y, 1)[1] > 0.5
