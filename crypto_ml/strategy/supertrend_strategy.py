from __future__ import annotations

import pandas as pd

from crypto_ml.indicators.supertrend import compute_supertrend


def generate_supertrend_signals(
    ohlcv: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Generate long-only signals based on SuperTrend trend direction.

    Returns a dataframe with columns: signal (1 long, 0 flat) and supertrend context columns.
    """
    st = compute_supertrend(
        ohlcv,
        atr_period=atr_period,
        multiplier=multiplier,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    signals = pd.DataFrame(index=ohlcv.index)
    signals["signal"] = st["in_uptrend"].astype(int)
    return signals.join(st)

