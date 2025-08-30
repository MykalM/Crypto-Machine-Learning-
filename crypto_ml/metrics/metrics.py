from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = 252.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    if vol == 0:
        return 0.0
    return float(excess.mean() * periods_per_year / vol)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

