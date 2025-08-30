from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    performance: pd.DataFrame
    summary: dict


def backtest_signals(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    price_col: str = "Close",
    signal_col: str = "signal",
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    allow_short: bool = False,
    initial_equity: float = 100_000.0,
) -> BacktestResult:
    """Vectorized backtest for daily bars.

    - Positions are next-bar (signals shifted by 1) to avoid lookahead.
    - Transaction costs applied on position changes (entry/exit) as basis points.
    - Long-only by default. If allow_short, positions can be -1/0/1 if signals contain that.
    """
    df = ohlcv.copy()
    df = df.join(signals[[signal_col]].copy(), how="left").fillna({signal_col: 0})

    # Daily returns
    df["ret"] = df[price_col].pct_change().fillna(0.0)

    # Positions with no lookahead
    raw_pos = df[signal_col].copy()
    if not allow_short:
        raw_pos = raw_pos.clip(lower=0, upper=1)
    position = raw_pos.shift(1).fillna(0.0)

    # Transaction cost on position changes
    pos_change = position.diff().abs().fillna(position.abs())
    trading_cost = (fee_bps + slippage_bps) / 10_000.0
    cost_series = pos_change * trading_cost

    # Strategy returns
    strat_ret = position * df["ret"] - cost_series

    equity = (1.0 + strat_ret).cumprod() * initial_equity

    perf = pd.DataFrame(
        {
            "price": df[price_col],
            "ret": df["ret"],
            "position": position,
            "strategy_ret": strat_ret,
            "equity": equity,
        },
        index=df.index,
    )

    # Summary
    total_return = equity.iloc[-1] / initial_equity - 1.0
    ann_factor = 252.0
    vol = strat_ret.std() * np.sqrt(ann_factor)
    sharpe = 0.0 if vol == 0 else strat_ret.mean() * ann_factor / vol
    mdd = max_drawdown(equity)

    summary = {
        "total_return_pct": total_return * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd * 100.0,
        "final_equity": float(equity.iloc[-1]),
    }

    return BacktestResult(performance=perf, summary=summary)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute max drawdown from equity series. Returns as fraction (e.g., 0.2 for 20%)."""
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1.0
    return drawdown.min() * -1.0 if len(drawdown) else 0.0

