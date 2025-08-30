from __future__ import annotations

import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Compute SuperTrend and trend direction.

    Returns dataframe with columns: atr, hl2, upper_band, lower_band, supertrend, in_uptrend.
    """
    data = df[[high_col, low_col, close_col]].copy()
    data.columns = ["high", "low", "close"]

    hl2 = (data["high"] + data["low"]) / 2.0

    # True range components
    high_low = data["high"] - data["low"]
    high_close_prev = (data["high"] - data["close"].shift(1)).abs()
    low_close_prev = (data["low"] - data["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    atr = _ema(true_range, span=atr_period)

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=data.index, dtype=float)
    in_uptrend = pd.Series(index=data.index, dtype=bool)

    # Initialization
    in_uptrend.iloc[0] = True
    supertrend.iloc[0] = lower_band.iloc[0]

    for i in range(1, len(data)):
        prev_in_up = in_uptrend.iloc[i - 1]
        prev_st = supertrend.iloc[i - 1]

        # Adjust bands
        if upper_band.iloc[i] < (upper_band.iloc[i - 1] if i > 0 else upper_band.iloc[i]):
            upper = upper_band.iloc[i]
        else:
            upper = upper_band.iloc[i - 1]

        if lower_band.iloc[i] > (lower_band.iloc[i - 1] if i > 0 else lower_band.iloc[i]):
            lower = lower_band.iloc[i]
        else:
            lower = lower_band.iloc[i - 1]

        if prev_in_up:
            if data["close"].iloc[i] <= lower:
                in_uptrend.iloc[i] = False
                supertrend.iloc[i] = upper
            else:
                in_uptrend.iloc[i] = True
                supertrend.iloc[i] = max(lower, prev_st)
        else:
            if data["close"].iloc[i] >= upper:
                in_uptrend.iloc[i] = True
                supertrend.iloc[i] = lower
            else:
                in_uptrend.iloc[i] = False
                supertrend.iloc[i] = min(upper, prev_st)

    out = pd.DataFrame(
        {
            "atr": atr,
            "hl2": hl2,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "supertrend": supertrend,
            "in_uptrend": in_uptrend,
        }
    )
    return out

