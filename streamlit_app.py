# -*- coding: utf-8 -*-
"""
Smart ETF Portfolio Optimizer
Inspired by Streamlit StockPeers template
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Smart ETF Portfolio Optimizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Available ETFs with metadata
ETF_DATA = {
    "PSI": {"name": "Semiconductors", "region": "North America"},
    "IYW": {"name": "US Technology", "region": "North America"},
    "RING": {"name": "Gold Miners", "region": "Developed Markets"},
    "PICK": {"name": "Metals & Mining", "region": "Developed Markets"},
    "NLR": {"name": "Nuclear Energy", "region": "Developed Markets"},
    "UTES": {"name": "Utilities", "region": "North America"},
    "LIT": {"name": "Lithium & Battery", "region": "Developed Markets"},
    "NANR": {"name": "Natural Resources", "region": "North America"},
    "GUNR": {"name": "Global Resources", "region": "Developed Markets"},
    "XCEM": {"name": "Emerging Markets", "region": "Emerging Markets"},
    "PTLC": {"name": "Large Cap", "region": "North America"},
    "FXU": {"name": "Utilities Alpha", "region": "North America"},
}

DEFAULT_ETFS = ["IYW", "PSI", "NLR", "UTES", "PTLC", "FXU"]

# Risk level mapping to optimization strategy
RISK_STRATEGIES = {
    "Low Risk": "min_volatility",
    "Balanced": "risk_parity",
    "High Return": "max_sharpe"
}

# Title
st.title(":material/query_stats: Smart ETF Portfolio Optimizer")
st.write("Determine optimal ETF allocation using **ML-forecasted returns** and **Modern Portfolio Theory**.")
st.write("")

# Layout: Left sidebar + Main content
cols = st.columns([1, 3])

# ============== LEFT PANEL ==============
# Top left - Configuration
top_left = cols[0].container(border=True)

with top_left:
    tickers = st.multiselect(
        "Select ETFs",
        options=sorted(ETF_DATA.keys()),
        default=DEFAULT_ETFS,
        format_func=lambda x: f"{x} - {ETF_DATA[x]['name']}",
        placeholder="Choose ETFs to include",
    )

    st.write("")

    # Risk Profile using pills (like StockPeers time horizon)
    risk_level = st.pills(
        "Risk Profile",
        options=list(RISK_STRATEGIES.keys()),
        default="Balanced",
        help="Low = Min Volatility | Balanced = Risk Parity | High = Max Sharpe"
    )

    st.write("")

    # Investment Amount
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        format="%d"
    )

    st.write("")

    # Optimize Button
    optimize_clicked = st.button(
        "Optimize Portfolio",
        type="primary",
        use_container_width=True
    )


# Check if we have enough tickers
if len(tickers) < 2:
    cols[1].container(border=True).info("Pick at least 2 ETFs to compare", icon=":material/info:")
    st.stop()


# ============== MAIN CONTENT ==============

# Initialize session state
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

# Run optimization when button clicked
if optimize_clicked:
    with st.spinner("Fetching data, forecasting returns, and optimizing portfolio..."):
        try:
            payload = {
                "tickers": tickers,
                "model_path": "trained_models_LSTM_2000_epochs/trained_models_LSTM_2000_epochs",
                "risk_free_rate": 0.05,
                "strategy": RISK_STRATEGIES[risk_level],
                "investment_amount": investment_amount,
                "include_charts": True
            }

            response = requests.post(
                f"{API_BASE_URL}/smart-invest",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            st.session_state.optimization_result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the backend is running.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.optimization_result:
    result = st.session_state.optimization_result
    metrics = result["portfolio_metrics"]
    allocations = result["allocations"]

    # ============== TOP RIGHT - Section 1: Portfolio allocation and Key Porfolio Metrics ==============
    top_right = cols[1].container(border=True)

    with top_right:
        st.subheader("Portfolio Allocation")
        st.write("Optimal allocation based on ML-forecasted returns and your risk profile.")

        # Allocation pie chart

        top_row = st.columns(2)
        with top_row[0].container(border=True):
            st.markdown("**Portfolio Allocation Pie Chart**")
            pie_data = pd.DataFrame([
             {"ETF": a["ticker"], "Weight": a["weight_percent"]}
             for a in allocations if a["weight_percent"] > 0
             ])
            if not pie_data.empty:
                pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=40).encode(
                 theta=alt.Theta(field="Weight", type="quantitative"),
                 color=alt.Color(field="ETF", type="nominal", legend=alt.Legend(orient="bottom")),
                 tooltip=[alt.Tooltip("ETF:N"), alt.Tooltip("Weight:Q", format=".1f", title="Weight (%)")]
                 ).properties(height=250, title="Weight Distribution")
                st.altair_chart(pie_chart)

        left_height = 180
        right_height = 180

        # Portfolio metrics
        with top_row[1].container(border=True):
            st.markdown("**Key Metrics**")

            st.metric(
                "Expected Return",
                f"{metrics['expected_annual_return_capped']:.1f}%",
                delta=f"Sharpe: {metrics['sharpe_ratio']:.2f}",
                delta_color="off")

            st.metric(
                "Volatility",
                f"{metrics['annual_volatility_percent']:.1f}%",
                delta=f"YTD: {metrics['portfolio_ytd_return']:+.1f}%",
                delta_color="off"
                )
            st.write("")

        second_row = st.columns(2)
        # Investment summary
        with second_row[0].container(border=True):
            st.markdown("**Investment Summary**")

            expected_value = investment_amount * (1 + min(metrics["expected_annual_return"], 1.0))
            expected_gain = expected_value - investment_amount

            st.metric("Initial", f"${investment_amount:,.0f}")
            st.metric("Expected (1Y)", f"${expected_value:,.0f}")
            st.metric("Gain", f"${expected_gain:,.0f}", delta=f"{metrics['expected_annual_return_capped']:.1f}%")

        left_height = 180
        right_height = 180

        # Find best and worst performing ETFs
        with second_row[1].container(border=True):
            active_allocs = [a for a in allocations if a['weight_percent'] > 0]
            if active_allocs:
                best_etf = max(active_allocs, key=lambda x: x['predicted_return_capped'])
                worst_etf = min(active_allocs, key=lambda x: x['predicted_return_capped'])

                metric_cols = st.columns(2)
                metric_cols[0].metric(
                    "Top Pick",
                    best_etf['ticker'],
                    delta=f"+{best_etf['predicted_return_capped']:.0f}%",
                    )
                metric_cols[1].metric(
                    "Lowest",
                    worst_etf['ticker'],
                    delta=f"+{worst_etf['predicted_return_capped']:.0f}%",
                    )

        st.write("")
        st.write("")


    # ============== RIGHT PANEL - Section 2: More information about my Portfolio==============
    section_2 = cols[1].container(border=True)

    with section_2:
        st.subheader("More Information About Your Portfolio")
        st.write("")
        # Normalized Price Evolution Chart
        st.markdown("**Normalized Price Evolution Chart**")
        if result.get("normalized_prices") and result["normalized_prices"].get("dates"):
            norm_data = result["normalized_prices"]

            # Convert to DataFrame for Altair
            chart_df = pd.DataFrame({"Date": pd.to_datetime(norm_data["dates"])})
            for ticker, prices in norm_data["prices"].items():
                chart_df[ticker] = prices

            # Melt for Altair
            melted = chart_df.melt(id_vars=["Date"], var_name="ETF", value_name="Normalized price")

            price_chart = alt.Chart(melted).mark_line().encode(
                alt.X("Date:T"),
                alt.Y("Normalized price:Q", scale=alt.Scale(zero=False)),
                alt.Color("ETF:N"),
                tooltip=["Date:T", "ETF:N", alt.Tooltip("Normalized price:Q", format=".1f")]
            ).properties(height=400)

            st.altair_chart(price_chart)
        else:
            st.info("Price evolution data not available")

        bottom_row = st.columns(2)
        # Geographic allocation
        with bottom_row[0].container(border=True):
            if result.get("geographic_allocation"):
                geo_data = pd.DataFrame([
                    {"Region": g["region"], "Allocation": g["allocation_percent"]}
                    for g in result["geographic_allocation"]
                    ])

            geo_chart = alt.Chart(geo_data).mark_bar().encode(
                x=alt.X("Allocation:Q", title="Allocation (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Region:N", sort="-x", title=""),
                color=alt.Color("Region:N", legend=None),
                tooltip=[alt.Tooltip("Region:N"), alt.Tooltip("Allocation:Q", format=".1f")]
            ).properties(height=200, title="Geographic Exposure")

            st.altair_chart(geo_chart)

        # Forecast vs YTD
        with bottom_row[1].container(border=True):
            forecast_data = pd.DataFrame([
                {"ETF": a["ticker"], "Metric": "YTD", "Value": a["ytd_return"]}
                for a in allocations if a["weight_percent"] > 0
                ] + [
                    {"ETF": a["ticker"], "Metric": "Forecast", "Value": a["predicted_return_capped"]}
                    for a in allocations if a["weight_percent"] > 0
                    ])

            if not forecast_data.empty:
                forecast_chart = alt.Chart(forecast_data).mark_bar().encode(
                    x=alt.X("ETF:N", title=""),
                    y=alt.Y("Value:Q", title="Return (%)"),
                    color=alt.Color("Metric:N", scale=alt.Scale(range=["#1f77b4", "#2ca02c"])),
                    xOffset="Metric:N",
                    tooltip=["ETF:N", "Metric:N", alt.Tooltip("Value:Q", format=".1f")]
                    ).properties(height=200, title="YTD vs Forecast")

                st.altair_chart(forecast_chart)

        st.divider()
        st.write("")
        st.write("")

    # ============== Section 3: Individual ETFs vs Peer Average ==============

    section_3 = cols[1].container(border=True)
    with section_3:
        st.subheader("Individual ETFs vs Peer Average")
        st.write("For the analysis below, the peer average when analyzing ETF X always excludes X itself.")

        active_tickers = [a["ticker"] for a in allocations if a["weight_percent"] > 0]

        if len(active_tickers) > 1 and result.get("normalized_prices") and result["normalized_prices"].get("dates"):
            norm_data = result["normalized_prices"]

        # Build normalized DataFrame
        norm_df = pd.DataFrame({"Date": pd.to_datetime(norm_data["dates"])})
        for ticker in active_tickers:
            if ticker in norm_data["prices"]:
                norm_df[ticker] = norm_data["prices"][ticker]

        norm_df = norm_df.set_index("Date").dropna(axis=1)

        NUM_COLS = 4
        chart_cols = st.columns(NUM_COLS)

        for i, ticker in enumerate(active_tickers[:6]):  # Limit to 6 for display
            if ticker not in norm_df.columns:
                continue

            # Calculate peer average excluding current ticker
            peers = norm_df.drop(columns=[ticker], errors='ignore')
            if peers.empty:
                continue
            peer_avg = peers.mean(axis=1)

            # Line chart: ETF vs Peer Average
            plot_data = pd.DataFrame({
                "Date": norm_df.index,
                ticker: norm_df[ticker],
                "Peer average": peer_avg,
            }).melt(id_vars=["Date"], var_name="Series", value_name="Price")

            chart = alt.Chart(plot_data).mark_line().encode(
                alt.X("Date:T"),
                alt.Y("Price:Q", scale=alt.Scale(zero=False)),
                alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=[ticker, "Peer average"], range=["#e45756", "#72b7b2"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Date:T"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Price:Q", format=".1f")
                ],
            ).properties(title=f"{ticker} vs peer average", height=280)

            cell = chart_cols[(i * 2) % NUM_COLS].container(border=True)
            cell.write("")
            cell.altair_chart(chart)

            # Area chart: Delta from peer average
            delta_data = pd.DataFrame({
                "Date": norm_df.index,
                "Delta": norm_df[ticker] - peer_avg,
            })

            delta_chart = alt.Chart(delta_data).mark_area(
                line={'color': '#4c78a8'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='#c7e9c0', offset=0),
                        alt.GradientStop(color='#4c78a8', offset=1)
                    ],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                alt.X("Date:T"),
                alt.Y("Delta:Q", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Delta:Q", format=".2f")],
            ).properties(title=f"{ticker} minus peer average", height=280)

            cell = chart_cols[(i * 2 + 1) % NUM_COLS].container(border=True)
            cell.write("")
            cell.altair_chart(delta_chart)

    st.write("")
    st.write("")

    # ============== Section 4:  ETF DETAILS TABLE ==============
    section_4 = cols[1].container(border=True)
    with section_4:
        if result.get("top_picks"):
            st.subheader("ETF Details")

            table_data = []
            for alloc in allocations:
                table_data.append({
                    "ETF": alloc["ticker"],
                    "Sector": alloc["name"],
                    "Region": alloc["region"],
                    "Weight (%)": f"{alloc['weight_percent']:.1f}",
                    "Amount ($)": f"{alloc['amount']:,.0f}" if alloc['amount'] else "-",
                    "YTD (%)": f"{alloc['ytd_return']:+.1f}",
                    "Forecast (%)": f"{alloc['predicted_return_capped']:.1f}",
                    "Signal": alloc["recommendation"]})

            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, hide_index=True)

            st.success(f"**Top Recommendations:** {', '.join(result['top_picks'])}")
            st.info(result["recommendation_summary"])

        else:
            # Show placeholder when no optimization has been run
            st.info(
                "Configure your portfolio settings in the left panel and click **Optimize Portfolio** to get personalized recommendations.",
                icon=":material/lightbulb:"
                )

            # Show ETF overview
            st.write("")
            st.markdown("### All Available ETFs")

            overview_data = pd.DataFrame([
                {"ETF": ticker, "Sector": data["name"], "Region": data["region"]}
                for ticker, data in ETF_DATA.items()
                if ticker in tickers
                ])

            st.dataframe(overview_data, hide_index=True)

    # Bottom left placeholder
    bottom_left = cols[0].container(border=True)
    with bottom_left:
        st.write("Portfolio metrics will appear here after optimization.")

# Footer
st.write("")
st.divider()
st.caption("Smart ETF Portfolio Optimizer | Powered by Machine Learning & Modern Portfolio Theory")
