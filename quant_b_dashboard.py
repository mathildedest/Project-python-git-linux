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

    I keep it in a separate module so I don't modify app.py.
    In the quant-b branch, I just need to import this function and call it.
    """

    st.header("Quant B – Multi-Asset Portfolio")

    st.markdown(
        "This module extends the project to **multiple assets** in parallel: "
        "we build a simple portfolio with customizable weights."
    )

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

    asset_mapping = {label: available_assets[label] for label in selected_labels}

    days = st.slider(
        "Historical depth (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=10,
    )

    # ---------------------------
    # Button to load data
    # ---------------------------
    
    if "prices" not in st.session_state:
        st.session_state["prices"] = None
        st.session_state["selected_labels"] = None

    if st.button("Load data"):
        try:
            prices = build_price_matrix(asset_mapping, days=days)
            # store in session_state so charts persist after rerun
            st.session_state["prices"] = prices
            st.session_state["selected_labels"] = selected_labels
        except Exception as e:
            st.error(f"Error while loading data: {e}")
            return

    prices = st.session_state.get("prices", None)
    stored_labels = st.session_state.get("selected_labels", selected_labels)

    if prices is None:
        st.info("Click **Load data** to retrieve prices.")
        return

    st.write("Last rows of price data (in EUR):")
    st.dataframe(prices.tail())

    tickers = {}
    for label in stored_labels:
        if "(" in label and ")" in label:
            t = label.split("(")[1].split(")")[0]
        else:
            t = label.replace(" ", "_")
        tickers[label] = t

    # ---------------------------
    # 2. Portfolio weights
    # ---------------------------
    
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

    cols = st.columns(len(stored_labels))
    raw_weights = {}
    for col, label in zip(cols, stored_labels):
        t = tickers[label]
        with col:
            w = st.number_input(
                f"Weight {t} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(100 / len(stored_labels)),
                step=1.0,
                key=f"weight_{t}",
            )
            raw_weights[t] = w / 100.0
            
    prices = prices.copy()
    prices.columns = [tickers[label] for label in stored_labels]

    returns = compute_returns(prices)
    
    weights = {col: raw_weights.get(col, 0.0) for col in prices.columns}

    try:
        port_ret = compute_portfolio_returns(returns, weights)
    except Exception as e:
        st.error(f"Problem with portfolio weights: {e}")
        return

    port_value = (1 + port_ret).cumprod()

    # ---------------------------
    # 3. Charts
    # ---------------------------
    
    st.subheader("3. Charts")

    st.markdown("**Individual price series (normalized to 1 at the start)**")
    norm_prices = prices / prices.iloc[0]
    st.line_chart(norm_prices)

    st.markdown("**Portfolio vs individual assets**")
    combined = norm_prices.copy()
    combined["Portfolio"] = port_value / port_value.iloc[0]
    st.line_chart(combined)
    
    st.markdown("**Correlation matrix of returns**")
    corr = returns.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="left")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    st.pyplot(fig)

    # ---------------------------
    # 4. Portfolio indicators
    # ---------------------------
    
    st.subheader("4. Portfolio indicators")

    stats = basic_stats(port_ret)

    if not stats:
        st.info("Not enough data to compute statistics.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Final value (base 1)", f"{stats['final_value']:.2f}")
    col2.metric("Annualized return", f"{stats['annualized_return']*100:.1f} %")
    col3.metric(
        "Annualized volatility", f"{stats['annualized_volatility']*100:.1f} %"
    )

    st.write(f"Max drawdown: {stats['max_drawdown']*100:.1f} %")

    st.caption(
        "The calculations are intentionally simple (no fees, no slippage, "
        "static weights). It is enough to illustrate the diversification effect."
    )

