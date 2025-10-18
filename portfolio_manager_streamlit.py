# portfolio_manager_streamlit.py
# Run with: pip install -r requirements.txt
# Then: streamlit run portfolio_manager_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Portfolio Manager (Streamlit)", layout="wide", initial_sidebar_state="expanded")

def read_price_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name
    try:
        if name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
    except Exception as e:
        st.error(f"Could not read {name}: {e}")
        return None
    df.columns = [str(c).strip() for c in df.columns]
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    date_col = date_cols[0] if date_cols else df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().all():
        st.warning(f"Could not parse dates in '{name}' (col '{date_col}'). Skipping file.")
        return None
    df = df.set_index(date_col).sort_index()
    price_candidates = [c for c in df.columns if c.lower() in ('close', 'ltp', 'last', 'close price', 'lastprice', 'closeprice')]
    if len(price_candidates) == 0:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            st.warning(f"No numeric price column found in {name}. Skipping file.")
            return None
        chosen = numeric_cols[-1]
    else:
        chosen = price_candidates[0]
    series = pd.to_numeric(df[chosen], errors='coerce').dropna()
    series.name = Path(name).stem
    return series

def compute_returns(prices_df, method='simple'):
    if method == 'log':
        return np.log(prices_df / prices_df.shift(1)).dropna()
    else:
        return prices_df.pct_change().dropna()

def annualize_return_from_series(r, periods_per_year=252, method='simple'):
    if method == 'log':
        return (1 + r.mean()) ** periods_per_year - 1
    else:
        return r.mean() * periods_per_year

def annualize_vol_from_series(r, periods_per_year=252):
    return r.std() * np.sqrt(periods_per_year)

def portfolio_performance(weights, mean_returns, cov_matrix, periods_per_year=252):
    ret = np.dot(weights, mean_returns) * periods_per_year
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * periods_per_year, weights)))
    return ret, std

def compute_beta(asset_returns, market_returns):
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])
    if cov.shape == (2,2) and cov[1,1] != 0:
        return cov[0,1] / cov[1,1]
    return np.nan

def jensen_alpha(portfolio_return, rf, portfolio_beta, market_return):
    return portfolio_return - (rf + portfolio_beta * (market_return - rf))

def random_portfolios(num_portfolios, mean_returns, cov_matrix, periods_per_year=252, rf=0.0):
    n = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        weights_record.append(w)
        r, s = portfolio_performance(w, mean_returns, cov_matrix, periods_per_year)
        sr = (r - rf) / s if s != 0 else np.nan
        results[0,i] = r
        results[1,i] = s
        results[2,i] = sr
    return results, weights_record

st.sidebar.header("Settings")
periods_per_year = int(st.sidebar.selectbox("Periods per year", [252, 365, 12], index=0))
return_method = st.sidebar.selectbox("Return calculation", ["simple", "log"], index=0)
rf_input = st.sidebar.number_input("Risk-free rate (annual, decimal)", value=0.05, format="%.4f")
rolling_window = int(st.sidebar.number_input("Rolling window (m+3 style)", min_value=1, max_value=252, value=3))
num_portfolios = int(st.sidebar.slider("Monte-Carlo samples", 500, 5000, 2000, step=500))

st.title("📊 Portfolio Manager (Streamlit)")
st.markdown("Upload one or more price files (CSV/XLSX) with a `Date` column and a `Close` or `LTP` price column. The app will auto-detect date and price columns.")

uploaded_files = st.file_uploader("Upload asset files (multiple allowed)", accept_multiple_files=True, type=['csv','xls','xlsx'])
uploaded_benchmark = st.file_uploader("Optional: upload benchmark file (single)", accept_multiple_files=False, type=['csv','xls','xlsx'])

if not uploaded_files:
    st.info("Upload at least one price file. You can also click 'Generate sample' to try the app.")
    if st.button("Generate sample dataset (4 assets)"):
        dates = pd.date_range(end=datetime.today(), periods=504, freq='B')
        import numpy as np
        np.random.seed(42)
        sample = {}
        for i in range(4):
            r = np.random.normal(0.0004 + i*0.0002, 0.01, len(dates))
            sample[f'Asset_{i+1}'] = 100 * np.exp(np.cumsum(r))
        pdf = pd.DataFrame(sample, index=dates).reset_index().rename(columns={'index':'Date'})
        csv = pdf.to_csv(index=False).encode('utf-8')
        st.download_button("Download sample CSV (single file with multiple columns)", csv, file_name="sample_prices.csv")
    st.stop()

price_series = []
for f in uploaded_files:
    s = read_price_file(f)
    if s is not None:
        price_series.append(s)
if len(price_series) == 0:
    st.error("No valid price series could be read from the uploaded files.")
    st.stop()

prices_df = pd.concat(price_series, axis=1).sort_index()
prices_df = prices_df.ffill().dropna(how='all')

st.subheader("Price preview (top rows)")
st.dataframe(prices_df.head())

benchmark_series = None
if uploaded_benchmark:
    benchmark_series = read_price_file(uploaded_benchmark)
    if benchmark_series is None:
        st.warning("Benchmark file could not be parsed. Benchmark metrics will be skipped.")
        benchmark_series = None

returns_df = compute_returns(prices_df, method=return_method)
if returns_df.shape[0] == 0:
    st.error("Not enough data to compute returns (need at least two consecutive price rows).")
    st.stop()

st.subheader("Returns preview (top rows)")
st.dataframe(returns_df.head())

st.header("Descriptive statistics")
mean_period = returns_df.mean()
std_period = returns_df.std()
ann_return = mean_period * periods_per_year if return_method=='simple' else returns_df.apply(lambda x: annualize_return_from_series(x, periods_per_year, method='log'))
ann_vol = returns_df.apply(lambda x: annualize_vol_from_series(x, periods_per_year))
skewness = returns_df.skew()
kurt = returns_df.kurt()

metrics_df = pd.DataFrame({
    "Mean(period)": mean_period,
    "Std(period)": std_period,
    "AnnReturn": ann_return,
    "AnnVol": ann_vol,
    "Skew": skewness,
    "Kurtosis": kurt
}).round(6)

st.dataframe(metrics_df)

st.subheader("Correlation matrix")
corr = returns_df.corr()
fig_corr = px.imshow(corr, text_auto=True, title="Correlation")
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Covariance matrix")
cov = returns_df.cov()
fig_cov = px.imshow(cov, text_auto=False, title="Covariance")
st.plotly_chart(fig_cov, use_container_width=True)

st.header("Portfolio construction")
cols = st.columns(len(returns_df.columns))
weights_input = {}
for i, col in enumerate(returns_df.columns):
    with cols[i]:
        w = st.number_input(f"W: {col}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
        weights_input[col] = w

weights = np.array([weights_input[c] for c in returns_df.columns])
if weights.sum() == 0:
    weights = np.array([1/len(weights)]*len(weights))
else:
    weights = weights / weights.sum()

st.write("Normalized weights:", {returns_df.columns[i]: float(round(weights[i],4)) for i in range(len(weights))})

mean_returns_arr = mean_period.values
cov_matrix = cov.values
p_ret, p_vol = portfolio_performance(weights, mean_returns_arr, cov_matrix, periods_per_year)
sharpe = (p_ret - rf_input) / p_vol if p_vol != 0 else np.nan

st.subheader("Portfolio summary")
st.write(f"Annualized Return: **{p_ret:.2%}**")
st.write(f"Annualized Volatility: **{p_vol:.2%}**")
st.write(f"Sharpe Ratio: **{sharpe:.4f}**")

if benchmark_series is not None:
    st.subheader("Benchmark-based metrics")
    bench_ret = compute_returns(benchmark_series.to_frame(), method=return_method).iloc[:,0]
    common_idx = returns_df.index.intersection(bench_ret.index)
    if len(common_idx) < 2:
        st.warning("Insufficient overlapping dates between assets and benchmark to compute market metrics.")
        p_beta = np.nan
        jensen = np.nan
        treynor = np.nan
    else:
        betas = {}
        for col in returns_df.columns:
            betas[col] = compute_beta(returns_df.loc[common_idx, col], bench_ret.loc[common_idx])
        p_beta = np.dot(weights, np.array([betas[c] for c in returns_df.columns]))
        market_ann_return = bench_ret.loc[common_idx].mean() * periods_per_year
        jensen = jensen_alpha(p_ret, rf_input, p_beta, market_ann_return)
        treynor = (p_ret - rf_input) / p_beta if p_beta != 0 else np.nan

        st.write("Asset betas:", {k: round(v,4) for k,v in betas.items()})
        st.write(f"Portfolio beta: **{p_beta:.4f}**")
        st.write(f"Jensen's alpha: **{jensen:.4%}**")
        st.write(f"Treynor ratio: **{treynor:.4f}**")
else:
    st.info("No benchmark uploaded — Beta, Jensen alpha and Treynor ratio require a benchmark file.")

st.header(f"Rolling {rolling_window}-period metrics (m+{rolling_window})")
rolling_mean = returns_df.rolling(window=rolling_window).mean()
rolling_std = returns_df.rolling(window=rolling_window).std()

st.subheader("Rolling mean (preview)")
st.dataframe(rolling_mean.dropna().head())

st.subheader("Rolling std (preview)")
st.dataframe(rolling_std.dropna().head())

st.subheader("Rolling mean chart")
st.line_chart(rolling_mean)

st.subheader("Rolling volatility chart")
st.line_chart(rolling_std)

st.header("Efficient frontier (Monte Carlo)")
results, weights_rec = random_portfolios(num_portfolios, mean_returns_arr, cov_matrix, periods_per_year, rf_input)
fig = go.Figure()
fig.add_trace(go.Scatter(x=results[1,:], y=results[0,:], mode='markers', name='Random portfolios', marker=dict(size=4, opacity=0.6)))
fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode='markers+text', name='Current portfolio', marker=dict(size=12, color='red'), text=['Current'], textposition='top center'))
max_sharpe_idx = int(np.nanargmax(results[2,:]))
fig.add_trace(go.Scatter(x=[results[1,max_sharpe_idx]], y=[results[0,max_sharpe_idx]], mode='markers+text', name='Max Sharpe', marker=dict(size=12, color='green'), text=['Max Sharpe'], textposition='bottom right'))
fig.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return', height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Approximate Max-Sharpe weights")
opt_weights = weights_rec[max_sharpe_idx]
st.write({returns_df.columns[i]: float(round(opt_weights[i],4)) for i in range(len(opt_weights))})
st.write(f"Max Sharpe Ratio (approx): {results[2,max_sharpe_idx]:.4f}")

st.header("Visualizations")
st.subheader("Price series")
fig_prices = px.line(prices_df.reset_index(), x=prices_df.reset_index().columns[0], y=prices_df.columns, title="Price series")
st.plotly_chart(fig_prices, use_container_width=True)

st.subheader("Cumulative returns")
cum_returns = (1 + returns_df).cumprod() - 1
fig_cum = px.line(cum_returns.reset_index(), x=cum_returns.reset_index().columns[0], y=cum_returns.columns, title="Cumulative returns")
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("Returns distributions")
for c in returns_df.columns:
    fig_h = px.histogram(returns_df, x=c, nbins=50, title=f"Returns: {c}")
    st.plotly_chart(fig_h, use_container_width=True)

st.header("Analysis summary & suggestions")
summary = [
    f"Portfolio Annual Return: {p_ret:.2%}",
    f"Portfolio Annual Volatility: {p_vol:.2%}",
    f"Sharpe Ratio: {sharpe:.4f}"
]
if benchmark_series is not None and not np.isnan(p_beta):
    summary += [f"Portfolio Beta: {p_beta:.4f}", f"Jensen's alpha: {jensen:.4%}", f"Treynor: {treynor:.4f}"]

if np.isnan(sharpe):
    suggestion = "Not enough data to compute Sharpe properly."
elif sharpe > 1:
    suggestion = "Good risk-adjusted returns \u2014 consider maintaining allocation and rebalancing periodically."
elif sharpe > 0.5:
    suggestion = "Moderate performance \u2014 consider diversification or trimming the highest-volatility assets."
else:
    suggestion = "Low risk-adjusted returns \u2014 consider rebalancing toward higher-Sharpe assets or reducing volatile holdings."

summary.append("Suggestion: " + suggestion)
for s in summary:
    st.write("- " + s)

st.header("Downloadable reports")
report_metrics = metrics_df.copy()
report_metrics['Weight'] = np.append(weights, [np.nan])[:len(report_metrics)]
report_metrics['OptWeight'] = np.append(opt_weights, [np.nan])[:len(report_metrics)]
csv = report_metrics.to_csv().encode('utf-8')
st.download_button("Download asset metrics CSV", csv, file_name="asset_metrics.csv")

port_df = pd.DataFrame({
    'Metric': ['PortfolioAnnualReturn','PortfolioAnnualVol','Sharpe','PortfolioBeta','JensenAlpha','Treynor'],
    'Value': [p_ret, p_vol, sharpe, (p_beta if benchmark_series is not None else np.nan), (jensen if benchmark_series is not None else np.nan), (treynor if benchmark_series is not None else np.nan)]
})
csv2 = port_df.to_csv(index=False).encode('utf-8')
st.download_button("Download portfolio summary CSV", csv2, file_name="portfolio_summary.csv")

st.success("Analysis complete \u2014 upload files and inspect results.")