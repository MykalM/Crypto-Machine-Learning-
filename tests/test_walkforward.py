import pandas as pd
from sklearn.linear_model import LogisticRegression

from crypto_ml.ml.walkforward import walk_forward_classification


def _dummy_prices() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    # Simple upward drift with noise
    close = pd.Series(100.0 + (idx - idx[0]).days * 0.1, index=idx)
    close = close + pd.Series([((i % 10) - 5) * 0.1 for i in range(len(idx))], index=idx)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Adj Close": close,
        "Volume": 1000,
    }, index=idx)
    return df


def test_walkforward_pipeline_runs():
    ohlcv = _dummy_prices()
    model = LogisticRegression(max_iter=1000)
    result = walk_forward_classification(ohlcv, model, n_splits=4)
    assert not result.predictions.empty
    assert isinstance(result.report, str)

