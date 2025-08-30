import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crypto_ml.data.loader import fetch_ohlcv
from crypto_ml.strategy.supertrend_strategy import generate_supertrend_signals
from crypto_ml.backtesting.engine import backtest_signals
from crypto_ml.ml.walkforward import walk_forward_classification


st.set_page_config(page_title="Crypto ML Demo", layout="wide")
st.title("Crypto ML Backtesting Demo")

symbol = st.sidebar.text_input("Symbol", value="BTC-USD")
start = st.sidebar.date_input("Start", value=pd.to_datetime("2020-01-01"))
atr_period = st.sidebar.slider("SuperTrend ATR Period", 5, 20, 10)
multiplier = st.sidebar.slider("SuperTrend Multiplier", 1.0, 4.0, 3.0, 0.5)
fee_bps = st.sidebar.slider("Fee (bps)", 0, 50, 5)

if st.sidebar.button("Run"):
    with st.spinner("Fetching data..."):
        ohlcv = fetch_ohlcv(symbol, start=start)

    st.subheader("Data Preview")
    st.dataframe(ohlcv.tail())

    with st.spinner("Generating signals and backtesting..."):
        sig = generate_supertrend_signals(ohlcv, atr_period=atr_period, multiplier=multiplier)
        res = backtest_signals(ohlcv, sig, fee_bps=fee_bps)

    st.subheader("Performance Summary")
    st.json(res.summary)

    st.subheader("Equity Curve")
    st.line_chart(res.performance[["equity"]])

    st.subheader("Walk-Forward ML (Logistic Regression)")
    model = LogisticRegression(max_iter=1000)
    wf = walk_forward_classification(ohlcv, model)
    st.text(wf.report)

