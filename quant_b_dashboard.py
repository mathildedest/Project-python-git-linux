import streamlit as st
import matplotlib.pyplot as plt

from quant_b_portfolio import (
    build_price_matrix,
    compute_returns,
    compute_portfolio_returns,
    basic_stats,
)


def run_quant_b_page():
    """
    Streamlit page for the Quant B module (multi-asset portfolio).

    I keep it in a separate module so I don't break app.py.
    In the quant-b branch, I just need to import this function and call it.
    """

    st.header("Quant B – Multi-Asset Portfolio")

    st.markdown(
        "This module extends the project to **multiple assets** in parallel: "
        "we build a simple portfolio with customizable weights."
    )

    # I stay with crypto assets to be consistent with the Quant A module
    available_assets = {
        "Bitcoin (BTC)": "bitcoin",
        "Ethereum (ETH)": "ethereum",
        "Solana (SOL)": "solana",
        "Ripple (XRP)": "ripple",
    }

    st.subheader("1. Investment universe")

    selected_labels = st.multiselect(
        "Choose the assets in the portfolio (min. 3):",
        list(available_assets.keys()),
        default=list(available_assets.keys())[:3],
    )

    if len(selected_labels) < 3:
        st.warning("⚠️ The assignment requires at least **3 assets**. Please add more.")
        return

    # Mapping label -> CoinGecko ID
    asset_mapping = {label: available_assets[label] for label in selected_labels}

    days = st.slider(
        "Historical depth (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=10,
    )

    if st.button("Load data"):
        try:
            prices = build_price_matrix(asset_mapping, days=days)
        except Exception as e:
            st.error(f"Error while loading data: {e}")
            return

        st.write("Last rows of price data (in EUR):")
        st.dataframe(prices.tail())

        # Extract short tickers, e.g. "Bitcoin (BTC)" -> "BTC"
        tickers = {}
        for label in selected_labels:
            if "(" in label and ")" in label:
                t = label.split("(")[1].split(")")[0]
            else:
                t = label.replace(" ", "_")
            tickers[label] = t

        # === Portfolio weights ===
        st.subheader("2. Portfolio weights")

        st.markdown(
            "The user chooses the **target weights**. "
            "Then I renormalize them so the sum is 100%."
        )

        rebalance_freq = st.selectbox(
            "Theoretical rebalancing frequency",
            ["None (buy & hold)", "Weekly", "Monthly"],
            help=(
                "To keep things simple I keep static weights in the code. "
                "To go further, we could resample returns at the chosen frequency "
                "and recompute the weights over time."
            ),
        )

        cols = st.columns(len(selected_labels))
        raw_weights = {}
        for col, label in zip(cols, selected_labels):
            t = tickers[label]
            with col:
                w = st.number_input(
                    f"Weight {t} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(100 / len(selected_labels)),
                    step=1.0,
                    key=f"weight_{t}",
                )
                raw_weights[t] = w / 100.0

        # Rename columns of the price DataFrame with short tickers (BTC, ETH, etc.)
        prices.columns = [tickers[label] for label in selected_labels]

        returns = compute_returns(prices)

        # Build the weight dict in the same order as the DataFrame columns
        weights = {col: raw_weights.get(col, 0.0) for col in prices.columns}

        try:
            port_ret = compute_portfolio_returns(returns, weights)
        except Exception as e:
            st.error(f"Problem with portfolio weights: {e}")

