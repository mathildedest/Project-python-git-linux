import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

st.title("Quant A - Single Asset Dashboard (step 5: strategies)")

coin = st.selectbox("Asset", ["bitcoin", "ethereum"])
cur = st.selectbox("Currency", ["eur", "usd"])

DATA_PATH = "data/prices.csv"

# -------------------------
# API + storage
# -------------------------

def get_price(coin_id, currency):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": currency}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 429:
        raise Exception("API limit reached. Please wait a few seconds.")
    r.raise_for_status()
    data = r.json()
    return float(data[coin_id][currency])

def save_price(price):
    os.makedirs("data", exist_ok=True)
    now = pd.Timestamp.utcnow()
    row = pd.DataFrame([{"time": now, "price": price}])

    if os.path.exists(DATA_PATH):
        old = pd.read_csv(DATA_PATH)
        df = pd.concat([old, row], ignore_index=True)
    else:
        df = row

    df.to_csv(DATA_PATH, index=False)

def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["time", "price"])
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df.sort_values("time")
    return df

# -------------------------
# Strategies
# -------------------------

def buy_and_hold(df):
    # Always invested
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)
    out["strat_ret"] = out["ret"]
    out["strat_value"] = (1 + out["strat_ret"]).cumprod()
    return out

def sma_crossover(df, fast=5, slow=10):
    # Basic SMA cross: invest when fast SMA > slow SMA
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)

    out["sma_fast"] = out["price"].rolling(fast).mean()
    out["sma_slow"] = out["price"].rolling(slow).mean()

    out["pos"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "pos"] = 1
    out["pos"] = out["pos"].fillna(0)

    # use previous position to avoid look-ahead
    out["strat_ret"] = out["pos"].shift(1).fillna(0) * out["ret"]
    out["strat_value"] = (1 + out["strat_ret"]).cumprod()
    return out

def momentum_strategy(df, window=3):
    # Very simple momentum strategy
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)

    # Momentum: past return over 'window'
    out["mom"] = out["price"].pct_change(window)

    # Position: invest if momentum > 0
    out["pos"] = 0
    out.loc[out["mom"] > 0, "pos"] = 1

    out["strat_ret"] = out["pos"].shift(1).fillna(0) * out["ret"]
    out["strat_value"] = (1 + out["strat_ret"]).cumprod()
    return out



# -------------------------
# Metrics
# -------------------------

def max_drawdown(values):
    # values is cumulative value series
    roll_max = values.cummax()
    dd = values / roll_max - 1.0
    return float(dd.min())

def sharpe_ratio(returns, periods_per_year=365):
    # very simple Sharpe with rf=0
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    if r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())

# -------------------------
# Button (anti spam)
# -------------------------

if "last_click" not in st.session_state:
    st.session_state.last_click = 0

if st.button("Fetch & save price"):
    now = time.time()
    if now - st.session_state.last_click < 10:
        st.warning("Please wait a few seconds before clicking again.")
    else:
        try:
            p = get_price(coin, cur)
            save_price(p)
            st.success(f"Saved price: {p:.2f} {cur.upper()}")
            st.session_state.last_click = now
        except Exception as e:
            st.error(str(e))

# -------------------------
# Main display
# -------------------------

df = load_data()

if len(df) < 5:
    st.info("Not enough data yet. Click 'Fetch & save price' a few times.")
else:
    st.subheader("Strategy settings")
    strat = st.selectbox(
    "Choose strategy",
    ["Buy & Hold", "SMA crossover", "Momentum"])

    fast = st.slider("SMA fast", 2, 50, 5)
    slow = st.slider("SMA slow", 3, 200, 10)

    if strat == "Buy & Hold":
        out = buy_and_hold(df)

    elif strat == "SMA crossover":
        if slow <= fast:
            st.warning("Please set SMA slow > SMA fast.")
        out = sma_crossover(df, fast=fast, slow=slow)

    else:  # Momentum
        window = st.slider("Momentum window", 2, 20, 3)
        out = momentum_strategy(df, window=window)


    st.subheader("Key metrics (simple)")
    mdd = max_drawdown(out["strat_value"])
    shrp = sharpe_ratio(out["strat_ret"])

    c1, c2 = st.columns(2)
    c1.metric("Max Drawdown", f"{mdd*100:.2f}%")
    c2.metric("Sharpe", f"{shrp:.2f}")

    st.subheader("Main chart (two curves)")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(out["time"], out["price"])
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel(f"Price ({cur.upper()})")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(out["time"], out["strat_value"])
    ax2.set_ylabel("Strategy cumulative value")

    st.pyplot(fig)

    st.subheader("Last rows")
    st.dataframe(out.tail(10))
