import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import xgboost as xgb
import shap


# ==========================================
# 1. CORE QUANT & ADVANCED RISK METRICS
# ==========================================

@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers, start_date, end_date):
    """Fetches historical daily close prices and volumes using Yahoo Finance, including S&P 500."""
    all_tickers = list(set(tickers + ['^GSPC']))
    ticker_string = " ".join(all_tickers)
    raw = yf.download(ticker_string, start=start_date, end=end_date)

    # Modern yfinance compatibility extraction layer
    if isinstance(raw.columns, pd.MultiIndex):
        close_data = raw['Close']
        vol_data = raw['Volume']
    else:
        close_data = raw[['Close']]
        vol_data = raw[['Volume']]

    return close_data, vol_data


def calculate_advanced_metrics(returns, volumes, prices, weights, sp500_returns, benchmark_ticker='BTC-USD'):
    """Calculates professional risk, performance, and liquidity metrics."""
    # 1. Volatility (Annualized)
    cov_matrix = returns.cov() * 365
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)

    sp500_vol = sp500_returns.std() * np.sqrt(252)

    # 2. Sharpe Ratio (Assuming 4% Risk-Free Rate)
    rf_rate = 0.04
    port_annual_return = returns.dot(weights).mean() * 365
    sharpe_ratio = (port_annual_return - rf_rate) / port_volatility if port_volatility > 0 else 0

    sp500_annual_return = sp500_returns.mean() * 252
    sp500_sharpe = (sp500_annual_return - rf_rate) / sp500_vol if sp500_vol > 0 else 0

    # 3. Maximum Drawdown (Historical 2-Year Window)
    portfolio_history = returns.dot(weights)
    cum_returns = (1 + portfolio_history).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    sp_cum = (1 + sp500_returns).cumprod()
    sp_max = sp_cum.cummax()
    sp_drawdowns = (sp_cum - sp_max) / sp_max
    sp500_mdd = sp_drawdowns.min()

    # 4. Average Daily Volume & Days-to-Liquidate Proxy
    avg_volumes = volumes.mean()
    weighted_adv = np.dot(avg_volumes, weights)

    # 5. Estimated Bid/Ask Spread (Roll Covariance Model)
    spreads = []
    for col in prices.columns:
        price_diff = prices[col].diff().dropna()
        cov = np.cov(price_diff[:-1], price_diff[1:])[0, 1]
        spreads.append(2 * np.sqrt(-cov) / prices[col].mean() if cov < 0 else 0.0005)
    weighted_spread = np.dot(np.array(spreads), weights)

    # 6. Systemic Correlation Index
    corr_matrix = returns.corr()
    num_assets = len(weights)
    avg_correlation = (corr_matrix.values.sum() - num_assets) / (
            num_assets * (num_assets - 1)) if num_assets > 1 else 1.0

    # 7. Expected Shortfall (Historical 99% Tail Expectation)
    var_99_threshold = np.percentile(portfolio_history, 1)
    expected_shortfall = portfolio_history[portfolio_history <= var_99_threshold].mean()

    sp_var_99 = np.percentile(sp500_returns, 1)
    sp500_es99 = sp500_returns[sp500_returns <= sp_var_99].mean()

    # 8. Concentration Risk (Herfindahl-Hirschman Index - HHI)
    hhi = np.sum(weights ** 2)

    # 9. Factor Risk: Historical Beta relative to Bitcoin (or Equity benchmark if BTC is missing)
    factor_ref = benchmark_ticker if benchmark_ticker in returns.columns else returns.columns[0]
    factor_var = returns[factor_ref].var()
    beta_val = returns.dot(weights).cov(returns[factor_ref]) / factor_var if factor_var > 0 else 1.0

    return (port_volatility, sp500_vol, port_annual_return, sp500_annual_return, sharpe_ratio, sp500_sharpe,
            max_drawdown, sp500_mdd, weighted_adv, weighted_spread, avg_correlation,
            expected_shortfall, sp500_es99, hhi, beta_val)


# ==========================================
# 2. UI CONFIGURATION & AD INJECTION ENGINE
# ==========================================
st.set_page_config(page_title="NeufiRisk Analytics", layout="wide")

st.sidebar.title("🛡️ NeufiRisk Pro")
user_tier = st.sidebar.radio("Subscription Tier", ["Free (With Ads)", "Premium Pro"])

# --- Asset Mappings ---
crypto_asset_mapping = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Sui (SUI)": "SUI-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Hyperliquid (HYPE)": "HYPE32196-USD",
    "Toncoin (TON)": "TON10272-USD",
    "Hedera (HBAR)": "HBAR-USD"
}

equity_asset_mapping = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Alphabet (GOOGL)": "GOOGL",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "JPMorgan Chase (JPM)": "JPM"
}

# --- Equities Selection ---
st.sidebar.subheader("1. Equities Selection & Allocation")
selected_equities = st.sidebar.multiselect(
    "Select Equities in Portfolio",
    options=list(equity_asset_mapping.keys()),
    default=["Tesla (TSLA)", "Apple (AAPL)"]
)

# --- Cryptos Selection ---
st.sidebar.subheader("2. Cryptocurrencies Selection & Allocation")
selected_cryptos = st.sidebar.multiselect(
    "Select Cryptos in Portfolio",
    options=list(crypto_asset_mapping.keys()),
    default=["Bitcoin (BTC)", "Ethereum (ETH)"]
)

all_selected_assets = selected_equities + selected_cryptos

if not all_selected_assets:
    st.sidebar.error("⚠️ Please select at least one equity or cryptocurrency asset to analyze.")
    st.stop()

# --- Allocation Allocators ---
st.sidebar.subheader("3. Asset Weighting Assignment")
raw_weights = {}

if selected_equities:
    st.sidebar.caption("**Equities Allocation Weights**")
    for name in selected_equities:
        raw_weights[name] = st.sidebar.slider(f"{name} allocation (%)", 0, 100, 25)

if selected_cryptos:
    st.sidebar.caption("**Cryptocurrency Allocation Weights**")
    for name in selected_cryptos:
        raw_weights[name] = st.sidebar.slider(f"{name} allocation (%)", 0, 100, 25)

# Combine mappings
master_asset_mapping = {**equity_asset_mapping, **crypto_asset_mapping}
active_asset_mapping = {name: master_asset_mapping[name] for name in all_selected_assets}

total_w = sum(raw_weights.values())
if total_w == 0:
    weights = np.ones(len(all_selected_assets)) / len(all_selected_assets)
else:
    weights = np.array([raw_weights[name] for name in all_selected_assets]) / total_w

portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=100000, step=10000)


def inject_ad_space(location="sidebar"):
    if user_tier == "Free (With Ads)":
        if location == "sidebar":
            st.sidebar.markdown("---")
            st.sidebar.info("📌 **SPONSORED:** Protect your assets with custom offline custody. [Learn more]")
        elif location == "main_top":
            st.markdown(
                "<div style='background-color:#f0f2f6; padding:12px; border-radius:5px; text-align:center; margin-bottom:20px; border-left: 5px solid #ff4b4b;'>"
                "<strong>Ad: Manage mixed custody portfolios instantly with Neufi Prime.</strong> <a href='#'>Open Account</a>"
                "</div>",
                unsafe_allow_html=True
            )


# ==========================================
# 3. LIVE MARKET DATA INGESTION
# ==========================================
st.title("🎛️ Neufi Enterprise Risk Platform")
inject_ad_space("main_top")

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=730)
tickers = list(active_asset_mapping.values())

try:
    all_prices, all_volumes = fetch_portfolio_data(tickers, start_date, end_date)

    prices = all_prices[tickers].ffill().bfill()
    volumes = all_volumes[tickers].ffill().bfill()
    sp500_prices = all_prices['^GSPC'].ffill().bfill()

    returns = prices.pct_change().dropna()
    sp500_returns = sp500_prices.pct_change().dropna()
except Exception as e:
    st.error(f"Error fetching market data: {e}")
    st.stop()

# Align timeseries index
common_idx = returns.index.intersection(sp500_returns.index)
returns = returns.loc[common_idx]
volumes = volumes.loc[common_idx]
prices = prices.loc[common_idx]
sp500_returns = sp500_returns.loc[common_idx]

# Compute risk vectors
(vol, sp_vol, port_ret, sp_ret, sharpe, sp_sharpe, mdd, sp_mdd, adv, spread,
 avg_corr, es_99, sp_es99, hhi, factor_beta) = calculate_advanced_metrics(
    returns, volumes, prices, weights, sp500_returns, benchmark_ticker='BTC-USD'
)

# ==========================================
# 4. APP NAVIGATION & SECTIONS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Active Performance Monitor",
    "📋 Portfolio Risk Dashboard",
    "🔥 Macro Stress Testing Engine",
    "🤖 Autonomous Strategy Agent"
])

# ------------------------------------------
# SECTION 1: ACTIVE PERFORMANCE ENGINE
# ------------------------------------------
with tab1:
    st.header("Active Institutional Performance Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Portfolio Annualized Return", value=f"{port_ret * 100:.2f}%",
                delta=f"{sp_ret * 100:.2f}% S&P Benchmark")
    col2.metric(label="Annualized Portfolio Volatility", value=f"{vol * 100:.2f}%",
                delta=f"{sp_vol * 100:.2f}% S&P Benchmark", delta_color="inverse")
    col3.metric(label="Sharpe Ratio (Rf=4%)", value=f"{sharpe:.2f}", delta=f"{sp_sharpe:.2f} S&P Benchmark")

    st.markdown("---")
    st.subheader("Cumulative Growth Engine: Portfolio vs. S&P 500")

    portfolio_history = returns.dot(weights)
    cum_portfolio = (1 + portfolio_history).cumprod() - 1
    cum_sp500 = (1 + sp500_returns).cumprod() - 1

    chart_df = pd.DataFrame({
        'Your Portfolio (%)': cum_portfolio * 100,
        'S&P 500 Index (%)': cum_sp500 * 100
    }, index=common_idx)

    st.line_chart(chart_df, use_container_width=True)

# ------------------------------------------
# SECTION 2: RISK MATRIX CATEGORIES
# ------------------------------------------
with tab2:
    st.header("Risk Matrix")
    st.write(
        "Cross-sectional framework profiling portfolio risk layers directly compared side-by-side with the S&P 500 index.")

    raw_matrix = pd.DataFrame({
        "Indicator": [
            "Expected Shortfall (99%)", "Realized Volatility", "Days-to-Liquidate",
            "HHI Index / Diversification", "Market Beta (vs BTC)", "Max Drawdown",
            "Stress-Test Loss profile", "Exchange Exposure", "AI Sentiment Index", "Sentiment Momentum"
        ],
        "Risk Category": [
            "Market Risk", "Volatility Risk", "Liquidity Risk",
            "Concentration Risk", "Factor Risk", "Tail Risk",
            "Scenario Risk", "Counterparty Risk", "Forward-Looking Risk", "Narrative Risk"
        ],
        "Your Portfolio": [
            f"{es_99 * 100:.2f}%",
            f"{vol * 100:.2f}%",
            f"{max(0.1, (portfolio_value * 0.1) / adv):.2f} Days",
            f"{hhi:.3f}",
            f"{factor_beta:.2f} x",
            f"{mdd * 100:.2f}%",
            "See Tab 3",
            "0.00%" if user_tier != "Free (With Ads)" else "Locked",
            "65/100",
            "+2.1%"
        ],
        "S&P 500 Equity Index": [
            f"{sp_es99 * 100:.2f}%",
            f"{sp_vol * 100:.2f}%",
            "<0.01 Days",
            "0.002",
            "1.00 x",
            f"{sp_mdd * 100:.2f}%",
            "See Tab 3",
            "0.00%",
            "54/100",
            "+0.8%"
        ]
    }).dropna(how='all')


    def apply_dashboard_styles(style_obj):
        style_obj = style_obj.map(
            lambda
                x: 'background-color: #eaf2ff; font-weight: 800; font-size: 22px; color: #0c2340; border-left: 2px solid #b3d1ff;',
            subset=["Your Portfolio"]
        )
        style_obj = style_obj.map(
            lambda
                x: 'background-color: #f4f6f9; font-weight: 800; font-size: 22px; color: #1e293b; border-left: 2px solid #e2e8f0;',
            subset=["S&P 500 Equity Index"]
        )
        return style_obj


    styled_matrix = raw_matrix.style.pipe(apply_dashboard_styles)

    st.dataframe(
        styled_matrix,
        use_container_width=True,
        height="auto",
        hide_index=True,
        column_config={
            "Indicator": st.column_config.Column(help="The corresponding mathematical proxy calculation model used",
                                                 width="medium"),
            "Risk Category": st.column_config.Column(help="The overarching framework layer classification name"),
            "Your Portfolio": st.column_config.Column(alignment="center", width="medium"),
            "S&P 500 Equity Index": st.column_config.Column(alignment="center", width="medium"),
        }
    )

    st.markdown(
        """
        <style>
        [data-testid="stTable"] *, [data-testid="stDataFrame"] * {
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        div[data-testid="stDataFrame"] td {
            font-size: 16px !important;
            padding: 12px 10px !important;
        }
        div[data-testid="stDataFrame"] th {
            font-size: 18px !important;
            font-weight: 800 !important;
            background-color: #0f172a !important;
            color: #ffffff !important;
            padding: 14px 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------
# SECTION 3: STRESS TESTING & ML ATTRIBUTION
# ------------------------------------------
with tab3:
    st.header("Systemic Shock Scenario Simulator")

    scenario = st.selectbox(
        "Choose Stress Scenario",
        ["Select Scenario", "2022 FTX Liquidity Collapse", "Macro Correlation Corruption (Hypothetical)"]
    )

    if scenario == "2022 FTX Liquidity Collapse":
        st.subheader("Scenario: FTX Insolvency Event (Nov 2022)")

        # Determine assets that traded in 2022
        valid_historical_tickers = [t for t in tickers if t not in ["SUI-USD", "TON10272-USD", "HYPE32196-USD"]]
        valid_indices = [tickers.index(t) for t in valid_historical_tickers]

        if len(valid_historical_tickers) == 0 or sum(weights[valid_indices]) == 0:
            st.warning("The assets currently selected did not trade during Nov 2022.")
        else:
            normalized_hist_weights = weights[valid_indices] / sum(weights[valid_indices])
            ftx_p, ftx_v = fetch_portfolio_data(valid_historical_tickers, "2022-11-05", "2022-11-15")
            ftx_returns = ftx_p[valid_historical_tickers].pct_change().dropna()
            ftx_cum_returns = (1 + ftx_returns.dot(normalized_hist_weights)).cumprod() - 1

            mdd_ftx = ftx_cum_returns.min() * 100
            scol1, scol2 = st.columns(2)
            scol1.metric("Simulated Maximum Drawdown", f"{mdd_ftx:.2f}%", delta_color="inverse")
            scol2.metric("Estimated Portfolio Value at Trough", f"${portfolio_value * (1 + (mdd_ftx / 100)):,.2f}")
            st.line_chart(ftx_cum_returns * 100, use_container_width=True)

    elif scenario == "Macro Correlation Corruption (Hypothetical)":
        st.subheader("Advanced Scenario: Correlation Breakdown Factor")
        if len(all_selected_assets) < 2:
            st.warning("Correlation metrics require at least two selected assets to evaluate structural breakdowns.")
        else:
            if user_tier == "Free (With Ads)":
                st.warning(
                    "🔒 **Premium Feature Locked:** Custom macro correlation shifting is reserved for Pro accounts.")
            else:
                corruption_factor = st.slider("Target Correlation Convergence", 0.0, 1.0, 0.8)
                corr_matrix = returns.corr()
                corrupted_corr = corr_matrix * (1 - corruption_factor) + corruption_factor
                std_devs = returns.std()
                corrupted_cov = np.diag(std_devs).dot(corrupted_corr).dot(np.diag(std_devs)) * 365
                corrupted_vol = np.sqrt(np.dot(weights.T, np.dot(corrupted_cov, weights)))
                st.error(f"🚨 Portfolio volatility expands from {vol * 100:.2f}% to **{corrupted_vol * 100:.2f}%**.")

    # --- SUB-SECTION: XGBOOST + SHAP ---
    st.markdown("---")
    st.header("🧠 AI-Driven Vulnerability Discovery (XGBoost + SHAP)")

    # Feature inputs mapped solely to selected active assets
    active_returns = returns[tickers]
    y_target = (portfolio_history < np.percentile(portfolio_history, 5)).astype(int)
    X_features = active_returns.shift(1).fillna(0)
    X_features.columns = [f"{col}_Lag1" for col in X_features.columns]

    if st.button("Run ML Diagnostic Framework"):
        with st.spinner("Training XGBoost Classifier..."):
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, eval_metric='logloss',
                                      random_state=42)
            model.fit(X_features, y_target)
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_matrix = explainer.shap_values(X_features)
            if isinstance(shap_matrix, list):
                shap_matrix = shap_matrix[1] if len(shap_matrix) > 1 else shap_matrix[0]
            mean_shap = np.abs(shap_matrix).mean(axis=0)

            inv_asset_map = {v: k for k, v in active_asset_mapping.items()}
            readable_features = [f"{inv_asset_map[col.replace('_Lag1', '')]} Momentum" for col in X_features.columns]

            st.session_state['importance_df'] = pd.DataFrame({
                'Asset Momentum Feature': readable_features,
                'Vulnerability Contribution (SHAP value)': mean_shap
            }).sort_values(by='Vulnerability Contribution (SHAP value)', ascending=False)

    if 'importance_df' in st.session_state:
        df_res = st.session_state['importance_df']
        col_ml1, col_ml2 = st.columns([1, 1])
        with col_ml1:
            st.dataframe(df_res.set_index('Asset Momentum Feature'), use_container_width=True)
        with col_ml2:
            st.error(
                f"**Primary Vulnerability Vector:** Extreme non-linear vulnerability to shocks originating in **{df_res.iloc[0]['Asset Momentum Feature'].replace(' Momentum', '')}** volatility.")

# ------------------------------------------
# SECTION 4: AUTONOMOUS STRATEGY AGENT
# ------------------------------------------
with tab4:
    st.header("🤖 Autonomous Quantitative Strategy Agent")
    st.write(
        "This agent evaluates your live allocations, monitors macro trend vectors, "
        "and recommends structural risk-mitigation strategies."
    )

    # Replaced OpenAI with Google Gemini setup
    gemini_api_key = st.text_input("Enter Gemini API Key to power the Agent's cognitive layer:", type="password")

    if st.button("Deploy Strategy Recommendation Agent"):
        if not gemini_api_key:
            st.warning("Please provide a valid Gemini API Key to initialize the agent's neural logic.")
        else:
            with st.spinner("Agent initializing quantitative tools and reading portfolio matrix structures..."):
                current_allocations = ", ".join([f"{k}: {raw_weights[k]}%" for k in all_selected_assets])

                top_vulnerability = "N/A"
                if 'importance_df' in st.session_state:
                    top_vulnerability = st.session_state['importance_df'].iloc[0]['Asset Momentum Feature'].replace(
                        ' Momentum', '')

                # Internal Agent Math Tool: Optimize allocations on the fly based on momentum
                recent_returns = returns[tickers].tail(30)
                asset_momentum = recent_returns.mean() * 365
                asset_volatility = recent_returns.std() * np.sqrt(365)

                sharpe_proxies = asset_momentum / asset_volatility
                raw_rec_weights = np.clip(sharpe_proxies, 0.05, 0.60)
                rec_weights = raw_rec_weights / raw_rec_weights.sum()

                # FIXED: Implemented .iloc matching on the pandas Series to completely resolve the KeyError
                rec_allocation_list = []
                for name in all_selected_assets:
                    ticker = active_asset_mapping[name]
                    # Direct dictionary matching from Series index labels to keep sequence identical
                    if ticker in rec_weights.index:
                        rec_val = rec_weights.loc[ticker] * 100
                    else:
                        rec_val = 0.0
                    rec_allocation_list.append(f"{name}: {rec_val:.1f}%")

                rec_allocation_string = ", ".join(rec_allocation_list)

                try:
                    # Use modern google-genai SDK
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=gemini_api_key)

                    agent_prompt = f"""
                    You are a Senior Quantitative Risk Strategist Agent specializing in multi-asset class (crypto and equities) frameworks.
                    You have been called to review a user's active portfolio allocation on the Neufi Enterprise Risk Platform and recommend structural changes.

                    USER'S LIVE ENVIRONMENT INPUTS:
                    - Portfolio Total Value: ${portfolio_value:,.2f}
                    - Active Allocations: {current_allocations}
                    - Annualized Volatility: {vol * 100:.2f}%
                    - Portfolio Sharpe Ratio: {sharpe:.2f}
                    - S&P 500 Sharpe Ratio: {sp_sharpe:.2f}
                    - Isolated ML Machine Learning Vulnerability Core: {top_vulnerability}

                    MATHEMATICAL TARGET STRATEGY RECOMMENDED BY YOUR SYSTEM TOOLS:
                    - Alternative Allocation Target: {rec_allocation_string}

                    YOUR TASK:
                    Write a professional, aggressive, and highly valuable quantitative risk advisory report. 
                    1. Criticize the user's current allocation flaws (especially looking at the correlation dynamic of combining equities and crypto) based on volatility versus the S&P 500 benchmark.
                    2. Explain the rationale behind the Alternative Allocation Target generated by your tools.
                    3. Address the ML Vulnerability Vector ({top_vulnerability}) directly and provide explicit delta-hedging or rebalancing actions to nullify it.

                    Format your output beautifully using high-level financial terminology in clean markdown format. Do not use generic filler language.
                    """

                    # Using gemini-2.5-flash for real-time quantitative reasoning
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=agent_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a professional quantitative portfolio risk engine agent on the Neufi platform.",
                            temperature=0.3
                        )
                    )

                    agent_report = response.text

                    st.success("🎯 Strategy Agent Execution Complete!")
                    st.markdown("---")
                    st.subheader("📋 Tactical Allocation Advisory Brief")
                    st.markdown(agent_report)

                except Exception as error:
                    st.error(f"Agent engine failed to communicate with brain core: {error}")

inject_ad_space("sidebar")