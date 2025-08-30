from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


FeatureFunc = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    report: str


def build_features_basic(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=ohlcv.index)
    df["ret_1"] = ohlcv["Close"].pct_change()
    df["ret_5"] = ohlcv["Close"].pct_change(5)
    df["vol_5"] = ohlcv["Close"].pct_change().rolling(5).std()
    df["ma_5"] = ohlcv["Close"].rolling(5).mean() / ohlcv["Close"] - 1.0
    df["ma_20"] = ohlcv["Close"].rolling(20).mean() / ohlcv["Close"] - 1.0
    df = df.dropna()
    return df


def target_from_returns(ohlcv: pd.DataFrame, horizon: int = 1) -> pd.Series:
    fwd = ohlcv["Close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[fwd.index]


def walk_forward_classification(
    ohlcv: pd.DataFrame,
    model,
    feature_fn: FeatureFunc = build_features_basic,
    n_splits: int = 5,
) -> WalkForwardResult:
    X = feature_fn(ohlcv)
    y = target_from_returns(ohlcv).reindex(X.index).dropna()
    X = X.loc[y.index]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=y.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        fitted = model.fit(X_train_scaled, y_train)
        fold_preds = fitted.predict(X_test_scaled)
        preds.iloc[test_idx] = fold_preds

    valid = preds.dropna()
    report = classification_report(y.loc[valid.index], valid, zero_division=1)
    return WalkForwardResult(predictions=valid.astype(int), report=report)

