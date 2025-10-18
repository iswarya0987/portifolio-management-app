
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📊 Advanced Portfolio Management App")

st.sidebar.header("Upload Your Portfolio Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Ensure required columns exist
    required_cols = ["Date", "ltp"]
    if all(col in df.columns for col in required_cols):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df["Returns"] = df["ltp"].pct_change()

        st.write("### Descriptive Statistics")
        st.write(df["Returns"].describe())

        # Portfolio metrics
        mean_return = df["Returns"].mean()
        volatility = df["Returns"].std()
        risk_free_rate = 0.05 / 252  # daily risk-free rate
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        beta = np.cov(df["Returns"], df["Returns"])[0][1] / np.var(df["Returns"])
        jensen_alpha = mean_return - (risk_free_rate + beta * (mean_return - risk_free_rate))
        treynor_ratio = (mean_return - risk_free_rate) / beta

        st.write("### Portfolio Metrics")
        st.metric("Mean Daily Return", f"{mean_return:.5f}")
        st.metric("Volatility", f"{volatility:.5f}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Jensen's Alpha", f"{jensen_alpha:.5f}")
        st.metric("Treynor Ratio", f"{treynor_ratio:.5f}")

        # Graphical Analysis
        st.write("### Portfolio Return Trend")
        st.line_chart(df.set_index("Date")["ltp"])

        st.write("### Rolling Mean (+3 Days)")
        df["Rolling_Mean"] = df["ltp"].rolling(window=3).mean()
        st.line_chart(df.set_index("Date")[["ltp", "Rolling_Mean"]])

        # Suggestions
        st.write("### Summary & Suggestions")
        if sharpe_ratio > 1:
            st.success("✅ Great! Your portfolio offers good returns per unit of risk.")
        elif sharpe_ratio > 0.5:
            st.info("⚖️ Moderate performance. Consider rebalancing or adding diversification.")
        else:
            st.warning("⚠️ Portfolio risk-adjusted returns are low. Try optimizing asset allocation.")
    else:
        st.error("Please ensure your CSV has 'Date' and 'ltp' columns for analysis.")
else:
    st.info("👆 Upload a CSV file to begin analysis.")
