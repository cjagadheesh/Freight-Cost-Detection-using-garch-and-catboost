import numpy as np
import pandas as pd
import pytest

from app.config import get_settings
from app.data.ingest import load_market_data


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def md(settings):
    return load_market_data(settings)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def synthetic_panel(n=900, seed=0):
    """Random-walk prices for every catalogue series on a business-day index."""
    from app.data.catalog import CATALOG

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {s.key: 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n))) for s in CATALOG}
    return pd.DataFrame(cols, index=idx)
