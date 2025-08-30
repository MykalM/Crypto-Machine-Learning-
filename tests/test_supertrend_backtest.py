import pandas as pd

from crypto_ml.indicators.supertrend import compute_supertrend
from crypto_ml.strategy.supertrend_strategy import generate_supertrend_signals
from crypto_ml.backtesting.engine import backtest_signals


def _dummy_data() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=50, freq="D")
    price = pd.Series(range(100, 150), index=idx, dtype=float)
    df = pd.DataFrame({
        "Open": price.shift(1).fillna(price.iloc[0]),
        "High": price + 1.0,
        "Low": price - 1.0,
        "Close": price,
        "Adj Close": price,
        "Volume": 1000,
    }, index=idx)
    return df


def test_supertrend_and_backtest_runs():
    df = _dummy_data()
    st = compute_supertrend(df, atr_period=7, multiplier=2.0)
    assert not st.empty
    assert {"supertrend", "in_uptrend"}.issubset(st.columns)

    sig = generate_supertrend_signals(df, atr_period=7, multiplier=2.0)
    assert "signal" in sig.columns

    res = backtest_signals(df, sig)
    assert "equity" in res.performance.columns
    assert res.summary["final_equity"] > 0

