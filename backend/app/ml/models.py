"""Base learners for each forecast horizon.

* Random walk – the efficient-market benchmark (forecast log return = 0).
* Ridge – regularised linear model on standardised features.
* CatBoost – gradient-boosted trees capturing non-linear regime effects.
All learners predict the h-day log return directly (no recursive roll-out),
which avoids compounding errors and the inability of trees to extrapolate levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["random_walk", "ridge", "catboost"]


def make_ridge(alpha: float = 300.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def make_catboost(iterations: int, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=20.0,
        rsm=0.7,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def fit_learners(X: pd.DataFrame, y: pd.Series, iterations: int, seed: int) -> dict:
    ridge = make_ridge().fit(X, y)
    cb = make_catboost(iterations, seed).fit(X, y)
    return {"ridge": ridge, "catboost": cb}


def predict_learners(learners: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "random_walk": np.zeros(len(X)),
        "ridge": learners["ridge"].predict(X),
        "catboost": learners["catboost"].predict(X),
    }


def fit_stacking_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex combination weights (w >= 0, sum w = 1) minimising squared error.

    ``preds`` is (n_samples x n_models). Because the random walk predicts zero,
    its weight acts as data-driven shrinkage of the learned signal toward zero.
    """
    k = preds.shape[1]
    if len(y) < 30:
        w = np.zeros(k)
        w[0] = 1.0
        return w

    def loss(w):
        e = y - preds @ w
        return float(e @ e)

    res = minimize(
        loss, x0=np.full(k, 1.0 / k), method="SLSQP",
        bounds=[(0.0, 1.0)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    w = np.clip(res.x, 0, 1)
    return w / w.sum()
