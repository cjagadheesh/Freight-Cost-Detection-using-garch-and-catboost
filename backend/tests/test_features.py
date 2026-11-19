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
