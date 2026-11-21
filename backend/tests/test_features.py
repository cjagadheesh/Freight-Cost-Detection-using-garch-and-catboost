import numpy as np

from app.ml.features import CORE_FEATURES, build_features, build_labels
from tests.conftest import synthetic_panel


def test_features_do_not_use_future_data():
    panel = synthetic_panel()
    f0 = build_features(panel)
    t = 600
    shocked = panel.copy()
    shocked.iloc[t + 1:] *= 3.0  # change everything after t
    f1 = build_features(shocked)
    a, b = f0.iloc[: t + 1], f1.iloc[: t + 1]
    assert np.allclose(a.fillna(0).values, b.fillna(0).values), "features at <= t changed when the future changed"


def test_exogenous_inputs_are_lagged_one_day():
    panel = synthetic_panel()
    f0 = build_features(panel)
    t = 600
    shocked = panel.copy()
    shocked.iloc[t, shocked.columns.get_loc("brent")] *= 2.0  # same-day driver move
    f1 = build_features(shocked)
    assert np.isclose(f0["brent_ret_5"].iloc[t], f1["brent_ret_5"].iloc[t])
    assert not np.isclose(f0["brent_ret_5"].iloc[t + 1], f1["brent_ret_5"].iloc[t + 1])


def test_labels_are_forward_log_returns():
    panel = synthetic_panel()
    y = build_labels(panel, [1, 5])
    p = panel["bdry"]
    assert np.isclose(y[5].iloc[10], np.log(p.iloc[15] / p.iloc[10]))
    assert y[5].iloc[-5:].isna().all()


def test_core_features_exist():
    f = build_features(synthetic_panel())
    assert set(CORE_FEATURES) <= set(f.columns)


def test_truncation_invariance():
    """Features computed on data cut at T equal full-data features up to T.
    (Catches full-sample statistics such as a global winsorisation threshold.)"""
    panel = synthetic_panel(900, seed=3)
    rng = np.random.default_rng(5)
    col = panel.columns.get_loc("brent")
    r = np.diff(np.log(panel["brent"].values))
    r[400] = 0.15  # ~7.5 sd jump before the cut: clipped under calm-period statistics
    r[520:] = rng.normal(0, 0.08, len(r) - 520)  # turbulent after T widens full-sample MAD
    panel.iloc[1:, col] = 50 * np.exp(np.cumsum(r))
    full = build_features(panel)
    T = 500
    cut = build_features(panel.iloc[: T + 1])
    a, b = full.iloc[: T + 1], cut
    assert np.allclose(a.fillna(-9).values, b.fillna(-9).values)


def test_unfinished_session_bar_is_dropped():
    import pandas as pd
    from app.data.sources import _drop_unfinished_session

    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-11-18", "2026-11-19"]))
    ts = [1763485800, 1763572200]  # 2026-11-18/19 14:30 UTC session opens
    meta = {"currentTradingPeriod": {"regular": {"start": 1763562600, "end": 1763586000}}}
    assert len(_drop_unfinished_session(s, meta, ts, now=1763574000)) == 1  # mid-session
    assert len(_drop_unfinished_session(s, meta, ts, now=1763590000)) == 2  # after the close
