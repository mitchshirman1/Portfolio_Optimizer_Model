import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #42a5f5 !important;
    color: white !important;
}

/* Input fields - transparent with black text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stNumberInput input {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: black !important;
    border: 1px solid white !important;
    border-radius: 6px;
    padding: 0.4rem;
    box-shadow: none !important;
    outline: none !important;
}

/* Number input wrapper - remove underline and unify border */
[data-testid="stSidebar"] .stNumberInput > div {
    border: 1px solid white !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

/* Label styling */
[data-testid="stSidebar"] label {
    color: #ffffffcc !important;
    font-weight: 500;
}

/* Stepper buttons (+ and -) */
[data-testid="stSidebar"] button[aria-label="Increment"],
[data-testid="stSidebar"] button[aria-label="Decrement"] {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    border: 1px solid white !important;
    border-radius: 6px;
    font-weight: bold;
}

/* Button hover effect */
[data-testid="stSidebar"] button[aria-label="Increment"]:hover,
[data-testid="stSidebar"] button[aria-label="Decrement"]:hover {
    background-color: rgba(255, 255, 255, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make disabled button more visible */
button[disabled] {
    background-color: #ffffff33 !important;
    color: #eeeeee !important;
    border: 1px solid #cccccc !important;
    opacity: 1 !important;
}

/* Style enabled Reset button for better contrast */
button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #999 !important;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

button[kind="secondary"]:hover {
    background-color: #f0f0f0 !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
tab1, tab2, tab3 = st.tabs(["Inputs", "Charts", "Export"])

# Initialize session state
if 'opt_weights' not in st.session_state:
    st.session_state.opt_weights = None
    st.session_state.returns = None
    st.session_state.tickers = []
    st.session_state.total_investment = 0
    st.session_state.start_str = ''
    st.session_state.end_str = ''
    st.session_state.portfolio_final_value = 0

if "optimization_ran" not in st.session_state:
    st.session_state.optimization_ran = False
if "do_reset" not in st.session_state:
    st.session_state.do_reset = False
if "show_reset_message" not in st.session_state:
    st.session_state.show_reset_message = False
if "show_success_toast" not in st.session_state:
    st.session_state.show_success_toast = False
if "show_reset_toast" not in st.session_state:
    st.session_state.show_reset_toast = False

if st.session_state.do_reset:
    st.session_state["tickers_input"] = ""
    st.session_state["investment_amount"] = 1000.0
    st.session_state["risk_free_rate"] = 2.0
    st.session_state["end_date"] = datetime.today()
    st.session_state["optimization_ran"] = False
    st.session_state["do_reset"] = False

    # Also clear session variables used in the charts
    for key in [
        "opt_weights", "returns", "tickers", "allocation_df", "performance_summary",
        "total_investment", "start_date_str", "end_date_str"
    ]:
        if key in st.session_state:
            del st.session_state[key]

if st.session_state.show_reset_message:
    st.session_state.show_reset_message = False

with tab1:
    st.title("Portfolio Optimization Dashboard (Data Input)")
    st.sidebar.header("User Inputs")

    tickers_input = st.sidebar.text_input("Enter stock tickers separated by commas: ", key="tickers_input")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

    if "end_date" not in st.session_state:
        st.session_state["end_date"] = datetime.today()

    end_date = st.sidebar.date_input("Select end date", value=st.session_state["end_date"], key="end_date")
    start_date = end_date.replace(year=end_date.year - 3)

    total_investment = st.sidebar.number_input("Total investment amount ($)", min_value=100.00, value=st.session_state.get("investment_amount", 1000.0), key="investment_amount")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, value=st.session_state.get("risk_free_rate", 2.0), key="risk_free_rate") / 100

    with st.sidebar:
        st.markdown("---")  # optional separator
        st.markdown("### Reset Options")

        reset_clicked = st.button("Reset Cache & Inputs")

        if reset_clicked:
            if st.session_state.get("optimization_ran", False):
                st.session_state["do_reset"] = True
                st.session_state["show_reset_message"] = True
                st.session_state["show_success_toast"] = False
                st.session_state["show_reset_toast"] = True
                st.rerun()
            else:
                st.markdown(
                    """
                    <style>
                    .fade-box {
                        animation: fadeIn 0.5s ease-in;
                        background-color: #FFD2D2;
                        padding: 10px 15px;
                        border-radius: 8px;
                        border: 1px solid #FF8888;
                        color: #990000;
                        font-weight: 500;
                        position: relative;
                    }

                    @keyframes fadeIn {
                        from {opacity: 0;}
                        to {opacity: 1;}
                    }

                    .fade-box.fade-out {
                        animation: fadeOut 1s ease-in forwards;
                    }

                    @keyframes fadeOut {
                        from {opacity: 1;}
                        to {opacity: 0;}
                    }
                    </style>

                    <script>
                    setTimeout(() => {
                        const box = window.parent.document.querySelectorAll('.fade-box');
                        box.forEach(el => el.classList.add('fade-out'));
                    }, 3000);
                    </script>

                    <div class="fade-box">
                        Run optimization first before resetting inputs.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    if st.button("Run Portfolio Optimization"):
        st.session_state["optimization_ran"] = True
        st.session_state["show_success_toast"] = True
        if not tickers:
            st.warning("Please enter valid")
            st.stop()
        with st.spinner("Fetching data and optimizing portfolio..."):
            sector_map = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    sector_map[ticker] = info.get('sector', 'Unknown')
                except:
                    sector_map[ticker] = 'Unknown'

            st.subheader("Sector Mapping")
            st.write(pd.DataFrame.from_dict(sector_map, orient='index', columns=['Sector']))

            sector_limits = {
                'Technology': 0.30,
                'Healthcare': 0.25,
                'Financial Services': 0.25,
                'Consumer Cyclical': 0.20,
                'Unknown': 0.15,
            }

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            price_data = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True, progress=False)['Close']
            price_data.dropna(inplace=True)
            price_data = price_data[tickers]
            returns = price_data.pct_change().dropna()

            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(tickers)
            initial_weights = np.array([1. / num_assets] * num_assets)

            sector_constraints = []
            for unique_sector in set(sector_map.values()):
                tickers_in_sector = [i for i, t in enumerate(tickers) if sector_map[t] == unique_sector]
                if not tickers_in_sector:
                    continue
                max_weight = sector_limits.get(unique_sector, 1.0)
                constraint = {
                    'type': 'ineq',
                    'fun': lambda x, idx=tickers_in_sector, max_w=max_weight: max_w - np.sum(x[idx])
                }
                sector_constraints.append(constraint)

            def neg_sharpe(weights):
                port_return = np.dot(weights, mean_returns)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -((port_return - risk_free_rate) / port_volatility)

            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}] + sector_constraints
            bounds = tuple((0.05, 0.25) for _ in range(num_assets))

            optimized = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = optimized.x

            opt_return = np.dot(opt_weights, mean_returns)
            opt_volatility = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
            opt_sharpe = (opt_return - risk_free_rate) / opt_volatility

            latest_prices = price_data.iloc[-1]
            allocations = opt_weights * total_investment
            shares = (allocations / latest_prices).apply(np.floor)
            actual_investment = (shares * latest_prices).sum()
            cash_remaining = total_investment - actual_investment

            allocation_df = pd.DataFrame({
                'Ticker': tickers,
                'Sector': [sector_map[t] for t in tickers],
                'Weight (%)': (opt_weights * 100).round(2),
                'Latest Price': latest_prices.round(2),
                'Allocated ($)': allocations.round(2),
                'Shares to Buy': shares.astype(int),
                'Total Value ($)': (shares * latest_prices).round(2)
            })

            st.subheader("Optimized Portfolio Breakdown")
            st.dataframe(allocation_df.set_index('Ticker'))

            st.subheader("Portfolio Performance Summary")
            st.write(f"**Expected Annual Return:** {opt_return * 100:.2f}%")
            st.write(f"**Annual Volatility:** {opt_volatility * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {opt_sharpe:.2f}")
            st.write(f"**Actual Invested:** ${actual_investment:,.2f}")
            st.write(f"**Unused Cash:** ${cash_remaining:,.2f}")

            st.session_state.allocation_df = allocation_df

            st.session_state.opt_weights = opt_weights
            st.session_state.returns = returns
            st.session_state.tickers = tickers
            st.session_state.total_investment = total_investment
            st.session_state.start_str = start_str
            st.session_state.end_str = end_str

            st.session_state.performance_summary = {
                "Expected Annual Return": f"{opt_return * 100:.2f}%",
                "Annual Volatility": f"{opt_volatility * 100:.2f}%",
                "Sharpe Ratio": f"{opt_sharpe:.2f}",
                "Actual Invested": f"${actual_investment:.2f}",
                "Unused Cash": f"${cash_remaining:.2f}",
            }
    if st.session_state.show_success_toast and not st.session_state.get("do_reset", False):
        st.markdown(
            """
            <style>
            #toast-container {
                position: fixed;
                top: 80px;
                right: 20px;
                z-index: 9999;
            }

            .toast {
                animation: fadeIn 0.5s ease-in, fadeOut 1s ease-out 3s forwards;
                background-color: #D4EDDA;
                color: #155724;
                border: 1px solid #A0D5AC;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: 500;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                margin-bottom: 10px;
                font-size: 15px;
                max-width: 300px;
            }

            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-10px);}
                to {opacity: 1; transform: translateY(0);}
            }

            @keyframes fadeOut {
                from {opacity: 1;}
                to {opacity: 0;}
            }
            </style>

            <div id="toast-container">
                <div class="toast"> Portfolio optimization ran successfully.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state["show_success_toast"] = False
    
    if st.session_state.get("show_reset_toast", False):
        st.markdown(
            """
            <style>
            .toast-blue {
                animation: fadeIn 0.5s ease-in;
                background-color: #D0E8FF;
                padding: 10px 15px;
                border-radius: 8px;
                border: 1px solid #3399FF;
                color: #003366;
                font-weight: 500;
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 9999;
            }

            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(10px);}
                to {opacity: 1; transform: translateY(0);}
            }

            @keyframes fadeOut {
                from {opacity: 1;}
                to {opacity: 0;}
            }

            .toast-blue.fade-out {
                animation: fadeOut 1s ease-in forwards;
            }
            </style>

            <script>
            setTimeout(() => {
                const box = window.parent.document.querySelectorAll('.toast-blue');
                box.forEach(el => el.classList.add('fade-out'));
            }, 3000);
            </script>

            <div class="toast-blue">Inputs have been reset.</div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.show_reset_toast = False


with tab2:
    st.title("Portfolio Charts")

    if st.session_state.get("opt_weights") is None:
        st.warning("Please run the optimization on the Inputs tab first.")
    else:
        st.subheader("Choose Chart Group")
        chart_category = st.radio("", ["Allocation Charts", "Performance Charts"], horizontal=True)

        tickers = st.session_state.tickers
        opt_weights = st.session_state.opt_weights
        returns = st.session_state.returns
        total_investment = st.session_state.total_investment
        start_str = st.session_state.start_str
        end_str = st.session_state.end_str

        if chart_category == "Allocation Charts":
            bar_tab, pie_tab = st.tabs(["Bar Chart", "Pie Chart"])

            with bar_tab:
                st.subheader("Bar Chart — Portfolio Allocation")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.bar(tickers, opt_weights, color='skyblue')
                ax1.set_ylabel("Weight")
                ax1.set_xlabel("Asset")
                ax1.set_ylim(0, 1)
                ax1.set_title("Optimal Portfolio Allocation")
                ax1.grid(axis='y')
                st.pyplot(fig1)

            with pie_tab:
                st.subheader("Pie Chart — Portfolio Allocation")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.pie(opt_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
                ax2.set_title("Optimal Portfolio Allocation")
                ax2.axis('equal')
                st.pyplot(fig2)

        elif chart_category == "Performance Charts":
            heatmap_tab, frontier_tab, benchmark_tab, capm_tab, stress_tab, monte_tab, drawdown_tab, rolling_tab, rebalancing_tab = st.tabs([
                "Correlation Heatmap", "Efficient Frontier", "Benchmark Comparison", "CAPM Metric Analysis", "Stress Test", "Monte Carlo Simulation", "Drawdown","Rolling Metrics", "Rebalancing Costs"])

            with heatmap_tab:
                st.subheader("Correlation Heatmap")
                corr_matrix = returns.corr()
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax3)
                ax3.set_title("Asset Correlation Matrix")
                st.pyplot(fig3)

            with frontier_tab:
                st.subheader("Efficient Frontier")
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                num_portfolios = 5000
                results = np.zeros((3, num_portfolios))

                for i in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    port_return = np.dot(weights, mean_returns)
                    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (port_return - 0.02) / port_volatility
                    results[:, i] = [port_return, port_volatility, sharpe]

                max_sharpe_idx = np.argmax(results[2])
                min_volatility_idx = np.argmin(results[1])

                fig4, ax4 = plt.subplots(figsize=(10, 6))
                scatter = ax4.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.6)
                fig4.colorbar(scatter, label='Sharpe Ratio')
                ax4.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color='orange', marker='*', s=200, label='Max Sharpe Ratio')
                ax4.scatter(results[1, min_volatility_idx], results[0, min_volatility_idx], color='limegreen', marker='o', s=100, label='Min Volatility')
                ax4.set_xlabel("Volatility")
                ax4.set_ylabel("Expected Return")
                ax4.set_title("Efficient Frontier")
                ax4.legend()
                st.pyplot(fig4)

            with benchmark_tab:
                st.subheader("Benchmark Comparison")
                benchmark_tickers = ['^GSPC', '^IXIC', '^DJI']
                benchmark_data = yf.download(benchmark_tickers, start=start_str, end=end_str, auto_adjust=True)['Close']
                benchmark_returns = benchmark_data.pct_change().dropna()

                portfolio_returns = returns @ opt_weights
                portfolio_returns.name = 'Optimized Portfolio'

                combined = pd.concat([portfolio_returns, benchmark_returns], axis=1)
                combined.columns = ['Optimized Portfolio', 'S&P 500', 'NASDAQ', 'Dow Jones']
                combined.dropna(inplace=True)
                cumulative = (1 + combined).cumprod()

                fig_bench, ax_bench = plt.subplots(figsize=(12, 6))
                for col in cumulative.columns:
                    ax_bench.plot(cumulative.index, cumulative[col], label=col, linewidth=2)
                ax_bench.set_title("Cumulative Returns")
                ax_bench.set_xlabel("Date")
                ax_bench.set_ylabel("Cumulative Return")
                ax_bench.legend()
                ax_bench.grid(True)
                st.pyplot(fig_bench)

                portfolio_final_value = total_investment * cumulative['Optimized Portfolio'].iloc[-1]
                st.markdown(f"### Projected Final Value: **${portfolio_final_value:,.2f}**")

            with capm_tab:
                st.subheader("CAPM Metric Analysis")

                benchmark_tickers = ['^GSPC', '^IXIC', '^DJI']
                benchmark_weights = {'^GSPC': 0.60, '^IXIC': 0.25, '^DJI': 0.15}

                # Download benchmark data
                capm_data = yf.download(benchmark_tickers, start=start_str, end=end_str, auto_adjust=True)['Close']
                capm_data.dropna(inplace=True)

                # Calculate daily returns
                capm_returns = capm_data.pct_change().dropna()

                # Compute weighted benchmark return
                capm_weighted_return = sum(
                    capm_returns[ticker] * weight for ticker, weight in benchmark_weights.items()
                )

                # Portfolio return
                portfolio_returns = returns.dot(opt_weights).dropna()

                # Cumulative returns
                portfolio_cum = (1 + portfolio_returns).cumprod()
                benchmark_cum = (1 + capm_weighted_return).cumprod()

                final_portfolio_value = portfolio_cum.iloc[-1]
                final_benchmark_value = benchmark_cum.iloc[-1]

                portfolio_growth_pct = (final_portfolio_value - 1) * 100
                benchmark_growth_pct = (final_benchmark_value - 1) * 100

                st.markdown("### Final Growth of $1 Investment")
                st.write(f"**Optimized Portfolio:** ${final_portfolio_value:.4f} (+{portfolio_growth_pct:.2f}%)")
                st.write(f"**Weighted Benchmark:** ${final_benchmark_value:.4f} (+{benchmark_growth_pct:.2f}%)")

                fig_capm, ax_capm = plt.subplots(figsize=(12, 6))
                ax_capm.plot(portfolio_cum, label='Optimized Portfolio', linewidth=2)
                ax_capm.plot(benchmark_cum, label='Weighted Benchmark', linestyle='--', linewidth=2)
                ax_capm.set_title('Portfolio vs Weighted Benchmark Growth')
                ax_capm.set_xlabel('Date')
                ax_capm.set_ylabel('Growth of $1 Investment')
                ax_capm.grid(True)
                ax_capm.legend()
                st.pyplot(fig_capm)

            with stress_tab:
                st.subheader("Stress Test Simulation")
                if st.session_state.opt_weights is not None and st.session_state.returns is not None:
                    returns = st.session_state.returns
                    opt_weights = st.session_state.opt_weights

                    portfolio_returns = returns.dot(opt_weights).dropna()
                    num_days = len(portfolio_returns)

                    st.markdown("Simulate random shocks to the portfolio:")
                    shock_days = st.number_input("Number of random shock days (1 to {})".format(num_days),
                                                min_value=1, max_value=num_days, value=5, step=1)

                    shock_percent = st.number_input("Shock size per day (as a negative %, e.g., 5 for -5%)",
                                                    min_value=0.0, max_value=100.0, value=5.0, step=0.5)

                    run_stress = st.button("Run Stress Test")

                    if run_stress:
                        if shock_days >= 1 and 0 <= shock_percent <= 100:
                            shock_impact = -abs(shock_percent / 100)

                            stress_returns = portfolio_returns.copy()
                            shock_indices = np.random.choice(num_days, size=shock_days, replace=False)
                            stress_returns.iloc[shock_indices] += shock_impact

                            original_growth = (1 + portfolio_returns).cumprod()
                            stress_growth = (1 + stress_returns).cumprod()

                            fig_stress, ax_stress = plt.subplots(figsize=(12, 6))
                            ax_stress.plot(original_growth, label="Original Portfolio", linewidth=2)
                            ax_stress.plot(stress_growth, label=f"Stress Test ({shock_days} shocks of -{shock_percent:.1f}%)",
                                        linestyle='--', linewidth=2)
                            ax_stress.scatter(original_growth.index[shock_indices],
                                            stress_growth.iloc[shock_indices],
                                            color='red', label="Shock Days", zorder=5)

                            ax_stress.set_title("Portfolio Stress Test vs Normal Growth")
                            ax_stress.set_xlabel("Date")
                            ax_stress.set_ylabel("Growth of $1 Investment")
                            ax_stress.legend()
                            ax_stress.grid(True)
                            st.pyplot(fig_stress)
                        else:
                            st.error("Invalid shock parameters. Please ensure inputs are within valid ranges.")
                else:
                    st.warning("Please run portfolio optimization first.")

            with monte_tab:
                st.subheader("Monte Carlo Simulation: Forecasting Future Portfolio Value")

                if "simulated_paths" not in st.session_state:
                    st.session_state.simulated_paths = None
                    
                if st.checkbox("Run Monte Carlo Simulation"):

                    mc_simulations = st.slider("Number of simulations", min_value=100, max_value=2000, value=756, step=50)
                    time_horizon = st.slider("Time Horizon (Trading Days)", min_value=50, max_value=756, value=252, step=10)

                    portfolio_returns = returns[tickers[:len(opt_weights)]].dot(opt_weights)
                    mean_return = portfolio_returns.mean()
                    volatility = portfolio_returns.std()

                    np.random.seed(random.randint(0, 1000000))

                    simulated_paths = np.zeros((time_horizon, mc_simulations))
                    simulated_paths[0] = total_investment

                    for t in range(1, time_horizon):
                        random_shocks = np.random.normal(loc=mean_return, scale=volatility, size=mc_simulations)
                        simulated_paths[t] = simulated_paths[t - 1] * (1 + random_shocks)

                    st.session_state.simulated_paths = simulated_paths


                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(simulated_paths, linewidth=0.5, alpha=0.6)
                    ax.set_title("Monte Carlo Simulation of Portfolio Value Over 1 Year")
                    ax.set_xlabel("Trading Days")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.grid(True)
                    st.pyplot(fig)

            with drawdown_tab:
                st.subheader("Drawdown Analysis")
                cumulative = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd_date = drawdown.idxmin()
                fig_dd, ax1 = plt.subplots(figsize=(12, 5))
                ax1.plot(cumulative, label='Cumulative Return', color='royalblue')
                ax1.axvline(max_dd_date, linestyle='--', color='gray')
                ax2 = ax1.twinx()
                ax2.plot(drawdown, color='crimson', linestyle='--', label='Drawdown')
                ax1.set_title('Cumulative Return vs Drawdown')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                st.pyplot(fig_dd)

            with rolling_tab:
                st.subheader("Rolling Metrics (21-Day Window)")
                rolling_return = portfolio_returns.rolling(window=21).mean() * 252
                rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
                fig_roll, ax = plt.subplots(figsize=(12, 5))
                ax.plot(rolling_return, label='Rolling Return')
                ax.plot(rolling_vol, label='Rolling Volatility')
                ax.set_title('Rolling Return & Volatility')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_roll)

            with rebalancing_tab:
                st.subheader("Rebalancing Cost Impact")
                rebalance_dates = portfolio_returns.resample('QE').first().index
                portfolio_value = (1 + portfolio_returns).cumprod()
                cost_value = portfolio_value.copy()
                for d in rebalance_dates[1:]:
                    if d in cost_value.index:
                        cost_value.loc[d:] *= (1 - 0.002)
                fig_rebal, ax = plt.subplots(figsize=(12, 5))
                ax.plot(portfolio_value, label='No Rebalancing Costs')
                ax.plot(cost_value, label='With Costs (0.2%)', linestyle='--')
                ax.set_title('Portfolio Value With vs Without Rebalancing Costs')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_rebal)
            
with tab3:
    import io
    st.title("Export Optimization Portfolio")

    if (
        "opt_weights" in st.session_state 
        and st.session_state.opt_weights is not None 
        and "allocation_df" in st.session_state 
        and "performance_summary" in st.session_state
    ):
        performance_summary =st.session_state.performance_summary

        def convert_df_to_excel(df, summary_dict):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Optimized Portfolio")
                summary_df = pd.DataFrame(summary_dict.items(), columns=["Metric", "Value"])
                summary_df.to_excel(writer, index=False, sheet_name="Performance Summary")
            return output.getvalue()

        if "filename_input" not in st.session_state:
            st.session_state.filename_input = "Optimized_Portfolio.xlsx"
        if "excel_data" not in st.session_state:
            st.session_state.excel_data = None
        if "file_ready" not in st.session_state:
            st.session_state.file_ready = False

        with st.form("filename_form"):
            st.markdown("""
            <style>
            /* Override both normal and focused states */
            form[data-testid="stForm"] input[type="text"] {
                background-color: rgba(255, 255, 255, 0.05) !important;
                color: white !important;
                border: 1px solid #ccc !important;
                padding: 0.5em !important;
                box-shadow: none !important;
            }

            form[data-testid="stForm"] input[type="text"]:focus {
                background-color: rgba(255, 255, 255, 0.05) !important;
                color: white !important;
                border: 1px solid #ccc !important;
                box-shadow: none !important;
            }

            /* Placeholder text color */
            form[data-testid="stForm"] input[type="text"]::placeholder {
                color: #dddddd !important;
            }
            </style>
            """, unsafe_allow_html=True)

            new_filename = st.text_input(
                "Enter file name (optional): ",
                value=st.session_state.filename_input
            )

            submitted = st.form_submit_button("Generate File")

            if submitted:
                if not new_filename.lower().endswith(".xlsx"):
                    new_filename += ".xlsx"
                st.session_state.filename_input = new_filename
                st.session_state.excel_data = convert_df_to_excel(
                    st.session_state.allocation_df,
                    performance_summary
                )
                st.session_state.file_ready = True

        if (
            st.session_state.file_ready 
            and st.session_state.excel_data is not None
        ):
            st.success("File is ready! Click below to download.")
            st.download_button(
                label="Download Optimized Portfolio (Excel)",
                data=st.session_state.excel_data,
                file_name=st.session_state.filename_input,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please run the optimization first from the Inputs tab.")
